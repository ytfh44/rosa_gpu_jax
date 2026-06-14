import numpy as np
import jax.numpy as jnp

from rosa_gpu_jax import make_raw_causal_aux, lookup_full_l_base
from rosa_gpu_jax.reference import brute_force_lookup


def test_smoke_exact_block_table():
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_base(Q, K, cap_end, successor, Lmax=3, sigma=16)
    tau_ref, len_ref = brute_force_lookup(np.array(Q), np.array(K), np.array(cap_end), np.array(successor), Lmax=3)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_random_exact_block_table_matches_reference():
    rng = np.random.default_rng(0)
    B, R, T = 2, 3, 12
    sigma = 8
    Lmax = 4

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)

    # Make some guaranteed matches so the test covers nontrivial hits.
    K_np[:, :, :6] = Q_np[:, :, :6]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_base(Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma)
    tau_ref, len_ref = brute_force_lookup(Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_custom_cap_and_successor_match_reference():
    Q_np = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    cap = np.zeros((B, R, T), dtype=np.int64)
    succ = np.zeros((B, R, T), dtype=np.int64)
    # Permit every query to look only into K positions < 4 and use non-raw tau.
    cap[:] = 4
    succ[:] = np.array([10, 11, 20, 21, 30, 31], dtype=np.int64)

    tau, match_len = lookup_full_l_base(
        jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cap), jnp.asarray(succ), Lmax=3, sigma=4
    )
    tau_ref, len_ref = brute_force_lookup(Q_np, K_np, cap, succ, Lmax=3)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_invalid_symbol_is_rejected():
    Q = jnp.array([[[0, 1, 4]]], dtype=jnp.int64)
    K = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        lookup_full_l_base(Q, K, cap_end, successor, Lmax=2, sigma=4)


def test_lmax_larger_than_t_is_rejected():
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        lookup_full_l_base(Q, K, cap_end, successor, Lmax=4, sigma=4)


def test_exact_combined_key_overflow_is_rejected():
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    # sigma=2**63 overflows combined = key * (T + 1) + pos even for L=1.
    with np.testing.assert_raises(OverflowError):
        lookup_full_l_base(Q, K, cap_end, successor, Lmax=1, sigma=2**63)
