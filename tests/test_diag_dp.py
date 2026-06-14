import numpy as np
import jax.numpy as jnp

from rosa_gpu_jax import lookup_full_l_diag_dp, make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.reference import brute_force_lookup, diag_dp_reference, rosa_batch_reference_tau


def test_diag_dp_smoke_vs_bruteforce():
    """Smoke test: diag DP should match brute force on a simple self-query."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_diag_dp(Q, K, cap_end, successor, Lmax=3)
    tau_ref, len_ref = brute_force_lookup(
        np.array(Q), np.array(K), np.array(cap_end), np.array(successor), Lmax=3
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_diag_dp_random_vs_bruteforce():
    """Random test vs brute force."""
    rng = np.random.default_rng(2)
    B, R, T = 2, 3, 8
    sigma = 6
    Lmax = 4

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]  # create some matches

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_diag_dp(Q, K, cap_end, successor, Lmax=Lmax)
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_diag_dp_random_vs_diag_reference():
    """Random test vs diag_dp_reference specifically."""
    rng = np.random.default_rng(5)
    B, R, T = 2, 2, 16
    sigma = 5
    Lmax = 6

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :6] = Q_np[:, :, :6]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_diag_dp(Q, K, cap_end, successor, Lmax=Lmax)
    tau_ref, len_ref = diag_dp_reference(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_diag_dp_rosa_counterexample_no_backtracking():
    """Diag DP must NOT backtrack to an older match when the rightmost has no valid successor."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_diag_dp(
        jnp.asarray(Z), jnp.asarray(Z), cap, succ, Lmax=Z.shape[-1], tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(Z)
    np.testing.assert_array_equal(np.array(tau), expected_tau)
    np.testing.assert_array_equal(
        expected_tau, np.array([[[-1, -1, 1, -1]]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        np.array(match_len), np.array([[[0, 0, 1, 0]]], dtype=np.int32)
    )


def test_diag_dp_enumerates_all_binary_len_6():
    """Exhaustive binary test vs reference for all length-6 sequences."""
    import itertools

    seqs = np.array(list(itertools.product([0, 1], repeat=6)), dtype=np.int64)[
        :, None, :
    ]
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(seqs))

    tau, match_len = lookup_full_l_diag_dp(
        jnp.asarray(seqs), jnp.asarray(seqs), cap, succ, Lmax=6, tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(seqs)
    expected_len = brute_force_lookup(
        seqs, seqs, np.array(cap), np.array(succ), Lmax=6, tau_cap=np.array(tau_cap)
    )[1]

    np.testing.assert_array_equal(np.array(tau), expected_tau)
    np.testing.assert_array_equal(np.array(match_len), expected_len)


def test_diag_dp_custom_cap_and_successor():
    """Diag DP with custom cap/successor matches brute force."""
    Q_np = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    cap = np.zeros((B, R, T), dtype=np.int64)
    succ = np.zeros((B, R, T), dtype=np.int64)
    cap[:] = 4
    succ[:] = np.array([10, 11, 20, 21, 30, 31], dtype=np.int64)

    tau, match_len = lookup_full_l_diag_dp(
        jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cap), jnp.asarray(succ), Lmax=3
    )
    tau_ref, len_ref = brute_force_lookup(Q_np, K_np, cap, succ, Lmax=3)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_diag_dp_lmax_larger_than_t_rejected():
    """Lmax > T must raise ValueError."""
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        lookup_full_l_diag_dp(Q, K, cap_end, successor, Lmax=4)


def test_diag_dp_jit_composable():
    """Diag DP can be nested under jax.jit."""
    import jax

    Z = jnp.array([[[0, 1, 0, 1, 0, 0]]], dtype=jnp.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cap_end, successor, tcap):
        return lookup_full_l_diag_dp(Q, K, cap_end, successor, Lmax=6, tau_cap=tcap)

    tau_jit, len_jit = jax.jit(run)(Z, Z, cap, succ, tau_cap)
    tau_ref, len_ref = lookup_full_l_diag_dp(Z, Z, cap, succ, Lmax=6, tau_cap=tau_cap)

    np.testing.assert_array_equal(np.array(tau_jit), np.array(tau_ref))
    np.testing.assert_array_equal(np.array(len_jit), np.array(len_ref))
