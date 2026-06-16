import itertools

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from rosa_gpu_jax.causal import make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.radix_events import (
    lookup_full_l_radix_events,
    lookup_full_l_radix_postings,
    pallas_available,
)


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test():
    yield
    jax.clear_caches()


def brute_force(Q, K, cap_end, successor, Lmax, tau_cap=None):
    """Pure-Python reference: longest match, rightmost tiebreak, ROSA gating."""
    Q = np.asarray(Q)
    K = np.asarray(K)
    cap_end = np.asarray(cap_end)
    successor = np.asarray(successor)
    B, R, T = Q.shape
    if tau_cap is None:
        tau_cap = np.full((B, R, T), T, dtype=np.int64)
    else:
        tau_cap = np.asarray(tau_cap)
    tau = np.full((B, R, T), -1, dtype=np.int64)
    lens = np.zeros((B, R, T), dtype=np.int32)
    for b in range(B):
        for r in range(R):
            for t in range(T):
                best_L = 0
                best_j = -1
                for L in range(1, Lmax + 1):
                    if t < L - 1:
                        continue
                    cap = int(cap_end[b, r, t])
                    for j in range(min(cap, T) - 1, L - 2, -1):
                        ok = True
                        for off in range(L):
                            if Q[b, r, t - off] != K[b, r, j - off]:
                                ok = False
                                break
                        if ok:
                            best_L = L
                            best_j = j
                            break
                if best_L > 0:
                    traw = int(successor[b, r, best_j])
                    if traw >= 0 and traw <= int(tau_cap[b, r, t]):
                        tau[b, r, t] = traw
                        lens[b, r, t] = best_L
    return tau, lens


def run_case(Q, K, Lmax, sigma, *, tau_cap=None, key_mode="base"):
    B, R, T = Q.shape
    cap, succ = make_raw_causal_aux(B, R, T)
    got_tau, got_len = lookup_full_l_radix_events(
        jnp.asarray(Q),
        jnp.asarray(K),
        cap,
        succ,
        Lmax=Lmax,
        sigma=sigma,
        tau_cap=tau_cap,
        key_mode=key_mode,
    )
    exp_tau, exp_len = brute_force(Q, K, cap, succ, Lmax, tau_cap=tau_cap)
    np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
    np.testing.assert_array_equal(np.asarray(got_len), exp_len)


@pytest.mark.parametrize("sigma", [2, 3, 7, 16])
def test_radix_events_random_base_matches_bruteforce(sigma):
    rng = np.random.default_rng(0xC0FFEE + sigma)
    for T, Lmax in [(1, 1), (5, 3)]:
        Q = rng.integers(0, sigma, size=(1, 2, T), dtype=np.int64)
        K = rng.integers(0, sigma, size=(1, 2, T), dtype=np.int64)
        run_case(Q, K, Lmax, sigma, key_mode="base")
    jax.clear_caches()


def test_bitpack_matches_base_and_bruteforce_for_binary():
    rng = np.random.default_rng(123)
    for T in [2, 9]:
        Q = rng.integers(0, 2, size=(1, 2, T), dtype=np.int64)
        K = rng.integers(0, 2, size=(1, 2, T), dtype=np.int64)
        Lmax = min(T, 5)
        B, R, _ = Q.shape
        cap, succ = make_raw_causal_aux(B, R, T)
        tau_b, len_b = lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=Lmax, sigma=2, key_mode="base"
        )
        tau_p, len_p = lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=Lmax, sigma=2, key_mode="bitpack"
        )
        np.testing.assert_array_equal(np.asarray(tau_p), np.asarray(tau_b))
        np.testing.assert_array_equal(np.asarray(len_p), np.asarray(len_b))
        exp_tau, exp_len = brute_force(Q, K, cap, succ, Lmax)
        np.testing.assert_array_equal(np.asarray(tau_p), exp_tau)
        np.testing.assert_array_equal(np.asarray(len_p), exp_len)
    jax.clear_caches()


def test_exhaustive_binary_sequences_len_3():
    T = 3
    cap, succ = make_raw_causal_aux(1, 1, T)
    for q_bits in itertools.product([0, 1], repeat=T):
        Q = np.array(q_bits, dtype=np.int64).reshape(1, 1, T)
        for k_bits in itertools.product([0, 1], repeat=T):
            K = np.array(k_bits, dtype=np.int64).reshape(1, 1, T)
            got_tau, got_len = lookup_full_l_radix_events(
                Q, K, cap, succ, Lmax=3, sigma=2, key_mode="bitpack"
            )
            exp_tau, exp_len = brute_force(Q, K, cap, succ, 3)
            np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
            np.testing.assert_array_equal(np.asarray(got_len), exp_len)


def test_equal_coordinate_is_strictly_causal_q_before_k():
    Q = np.array([[[5, 5, 5]]], dtype=np.int64)
    K = Q.copy()
    cap, succ = make_raw_causal_aux(1, 1, 3)
    got_tau, got_len = lookup_full_l_radix_events(
        Q, K, cap, succ, Lmax=1, sigma=8
    )
    # t=0 has no j < 0; t=1 sees j=0; t=2 sees j=1, not j=2.
    np.testing.assert_array_equal(np.asarray(got_tau), np.array([[[-1, 1, 2]]]))
    np.testing.assert_array_equal(
        np.asarray(got_len), np.array([[[0, 1, 1]]], dtype=np.int32)
    )


def test_prefix_padding_cannot_create_false_zero_matches():
    Q = np.array([[[0, 0, 1]]], dtype=np.int64)
    K = np.array([[[0, 0, 1]]], dtype=np.int64)
    cap, succ = make_raw_causal_aux(1, 1, 3)
    got_tau, got_len = lookup_full_l_radix_events(
        Q, K, cap, succ, Lmax=3, sigma=2, key_mode="bitpack"
    )
    exp_tau, exp_len = brute_force(Q, K, cap, succ, 3)
    np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
    np.testing.assert_array_equal(np.asarray(got_len), exp_len)


def test_official_rosa_aux_does_not_backtrack_after_successor_gate():
    K = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    Q = K.copy()
    cap, succ, tcap = make_rosa_causal_aux(K)
    got_tau, got_len = lookup_full_l_radix_events(
        Q, K, cap, succ, Lmax=4, sigma=2, tau_cap=tcap, key_mode="bitpack"
    )
    exp_tau, exp_len = brute_force(Q, K, cap, succ, 4, tau_cap=tcap)
    np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
    np.testing.assert_array_equal(np.asarray(got_len), exp_len)
    # The last raw rightmost match for symbol 0 has no currently valid successor.
    assert int(np.asarray(got_tau)[0, 0, 3]) == -1
    assert int(np.asarray(got_len)[0, 0, 3]) == 0


def test_custom_cap_successor_tau_cap():
    Q = np.array([[[1, 2, 1, 2, 1]]], dtype=np.int64)
    K = Q.copy()
    cap = np.array([[[0, 1, 2, 2, 4]]], dtype=np.int64)
    succ = np.array([[[5, 4, 4, 3, -1]]], dtype=np.int64)
    tcap = np.array([[[0, 5, 5, 5, 4]]], dtype=np.int64)
    got_tau, got_len = lookup_full_l_radix_events(
        Q, K, cap, succ, Lmax=3, sigma=3, tau_cap=tcap
    )
    exp_tau, exp_len = brute_force(Q, K, cap, succ, 3, tau_cap=tcap)
    np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
    np.testing.assert_array_equal(np.asarray(got_len), exp_len)


def test_postings_c_ge_t_matches_exact_for_base_and_bitpack():
    rng = np.random.default_rng(99)
    for sigma, key_mode in [(2, "bitpack"), (5, "base")]:
        T = 5
        Q = rng.integers(0, sigma, size=(1, 1, T), dtype=np.int64)
        K = rng.integers(0, sigma, size=(1, 1, T), dtype=np.int64)
        cap, succ = make_raw_causal_aux(1, 1, T)
        exact_tau, exact_len = lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=4, sigma=sigma, key_mode=key_mode
        )
        post_tau, post_len = lookup_full_l_radix_postings(
            Q, K, cap, succ, Lmax=4, sigma=sigma, key_mode=key_mode, C=T
        )
        np.testing.assert_array_equal(np.asarray(post_tau), np.asarray(exact_tau))
        np.testing.assert_array_equal(np.asarray(post_len), np.asarray(exact_len))


def test_nested_jit_public_wrapper():
    Q = jnp.array([[[0, 1, 0, 1, 1]]], dtype=jnp.int64)
    K = jnp.array([[[0, 1, 0, 0, 1]]], dtype=jnp.int64)
    cap, succ = make_raw_causal_aux(1, 1, 5)

    @jax.jit
    def f(Q, K, cap, succ):
        return lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=4, sigma=2, key_mode="bitpack"
        )

    got_tau, got_len = f(Q, K, cap, succ)
    exp_tau, exp_len = brute_force(np.asarray(Q), np.asarray(K), cap, succ, 4)
    np.testing.assert_array_equal(np.asarray(got_tau), exp_tau)
    np.testing.assert_array_equal(np.asarray(got_len), exp_len)


def test_validation_errors():
    Q = np.array([[[0, 1]]], dtype=np.int64)
    K = Q.copy()
    cap, succ = make_raw_causal_aux(1, 1, 2)
    with pytest.raises(ValueError, match="bitpack.*sigma=2"):
        lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=1, sigma=3, key_mode="bitpack"
        )
    with pytest.raises(ValueError, match="key_mode"):
        lookup_full_l_radix_events(
            Q, K, cap, succ, Lmax=1, sigma=2, key_mode="rank"
        )
    with pytest.raises(ValueError, match="symbols"):
        lookup_full_l_radix_events(
            np.array([[[2]]]),
            np.array([[[0]]]),
            np.array([[[0]]]),
            np.array([[[1]]]),
            Lmax=1,
            sigma=2,
        )
    with pytest.raises(ValueError, match="C"):
        lookup_full_l_radix_postings(
            Q, K, cap, succ, Lmax=1, sigma=2, C=0
        )


def test_pallas_import_probe_is_boolean():
    assert isinstance(pallas_available(), bool)
