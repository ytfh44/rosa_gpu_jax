import numpy as np
import jax.numpy as jnp

from rosa_gpu_jax import lookup_full_l_bitset, make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.reference import brute_force_lookup, rosa_batch_reference_tau


def test_bitset_smoke_vs_bruteforce():
    """Smoke test: bitset should match brute force on a simple self-query."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_bitset(
        Q, K, cap_end, successor, Lmax=3, sigma=16
    )
    tau_ref, len_ref = brute_force_lookup(
        np.array(Q), np.array(K), np.array(cap_end), np.array(successor), Lmax=3
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_bitset_random_vs_bruteforce():
    """Random test vs brute force for small T."""
    rng = np.random.default_rng(4)
    B, R, T = 1, 1, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_bitset(
        Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_bitset_rosa_counterexample():
    """Bitset must respect ROSA no-backtrack."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_bitset(
        jnp.asarray(Z),
        jnp.asarray(Z),
        cap,
        succ,
        Lmax=Z.shape[-1],
        sigma=2,
        tau_cap=tau_cap,
    )

    expected_tau = rosa_batch_reference_tau(Z)
    np.testing.assert_array_equal(np.array(tau), expected_tau)
    np.testing.assert_array_equal(
        expected_tau, np.array([[[-1, -1, 1, -1]]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        np.array(match_len), np.array([[[0, 0, 1, 0]]], dtype=np.int32)
    )


def test_bitset_custom_cap_and_successor():
    """Bitset with custom cap/successor matches brute force."""
    Q_np = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    cap = np.zeros((B, R, T), dtype=np.int64)
    succ = np.zeros((B, R, T), dtype=np.int64)
    cap[:] = 4
    succ[:] = np.array([10, 11, 20, 21, 30, 31], dtype=np.int64)

    tau, match_len = lookup_full_l_bitset(
        jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cap), jnp.asarray(succ),
        Lmax=3, sigma=4
    )
    tau_ref, len_ref = brute_force_lookup(Q_np, K_np, cap, succ, Lmax=3)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)
