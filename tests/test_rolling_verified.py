import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import (
    lookup_full_l_rolling,
    lookup_full_l_rolling_verified,
    make_raw_causal_aux,
    make_rosa_causal_aux,
)
from rosa_gpu_jax.reference import brute_force_lookup, rosa_batch_reference_tau


def test_rolling_verified_smoke_runs():
    """Smoke test: should not crash and return valid shapes."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_rolling_verified(
        Q, K, cap_end, successor, Lmax=3, base=257, C=4
    )

    assert tau.shape == (B, R, T)
    assert match_len.shape == (B, R, T)
    assert tau.dtype == jnp.int64
    assert match_len.dtype == jnp.int32


def test_rolling_verified_produces_some_hits_on_self_query():
    """On self-query, verified rolling hash should find some matches."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, _ = lookup_full_l_rolling_verified(
        Q, K, cap_end, successor, Lmax=3, base=257, C=8
    )

    tau_np = np.array(tau)
    assert np.any(tau_np >= 0), "expected at least one valid match with self-query"


def test_rolling_verified_agrees_with_unverified_rolling_on_small_t():
    """For small T with large C, verified should match unverified rolling (which uses mask backend)."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau_unv, len_unv = lookup_full_l_rolling(
        Q, K, cap_end, successor, Lmax=3, base=257, algorithm="mask"
    )
    tau_ver, len_ver = lookup_full_l_rolling_verified(
        Q, K, cap_end, successor, Lmax=3, base=257, C=16
    )

    np.testing.assert_array_equal(np.array(tau_ver), np.array(tau_unv))
    np.testing.assert_array_equal(np.array(len_ver), np.array(len_unv))


def test_rolling_verified_random_vs_bruteforce():
    """On random data with C large enough, rolling verified should match brute force for small T."""
    rng = np.random.default_rng(3)
    B, R, T = 1, 1, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_rolling_verified(
        Q, K, cap_end, successor, Lmax=Lmax, base=257, C=32
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_rolling_verified_rosa_counterexample():
    """Verified rolling must respect ROSA no-backtrack."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_rolling_verified(
        jnp.asarray(Z),
        jnp.asarray(Z),
        cap,
        succ,
        Lmax=Z.shape[-1],
        base=257,
        C=16,
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
