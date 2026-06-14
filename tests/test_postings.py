import numpy as np
import jax.numpy as jnp

from rosa_gpu_jax import (
    lookup_full_l_base,
    lookup_full_l_base_postings,
    lookup_full_l_rolling_postings,
    make_raw_causal_aux,
    make_rosa_causal_aux,
)
from rosa_gpu_jax.reference import brute_force_lookup, rosa_batch_reference_tau


def test_base_postings_smoke_vs_exact_block_table():
    """With C large enough, postings should match exact block table."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau_exact, len_exact = lookup_full_l_base(
        Q, K, cap_end, successor, Lmax=3, sigma=16
    )
    tau_post, len_post = lookup_full_l_base_postings(
        Q, K, cap_end, successor, Lmax=3, sigma=16, C=T
    )

    np.testing.assert_array_equal(np.array(tau_post), np.array(tau_exact))
    np.testing.assert_array_equal(np.array(len_post), np.array(len_exact))


def test_base_postings_random_vs_bruteforce():
    """On random data, base postings with C=T should match brute force."""
    rng = np.random.default_rng(2)
    B, R, T = 2, 2, 8
    sigma = 5
    Lmax = 4

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_base_postings(
        Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma, C=T
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_base_postings_rosa_counterexample():
    """Postings must respect ROSA no-backtrack with official aux."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_base_postings(
        jnp.asarray(Z),
        jnp.asarray(Z),
        cap,
        succ,
        Lmax=Z.shape[-1],
        sigma=2,
        C=Z.shape[-1],
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


def test_rolling_postings_smoke_runs():
    """Rolling postings should not crash and should return valid shapes."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_rolling_postings(
        Q, K, cap_end, successor, Lmax=3, base=257, C=4
    )

    assert tau.shape == (B, R, T)
    assert match_len.shape == (B, R, T)
    assert tau.dtype == jnp.int64
    assert match_len.dtype == jnp.int32


def test_base_postings_with_small_c_nonzero_output():
    """Even with C=1, postings should produce some matches on self-query."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_base_postings(
        Q, K, cap_end, successor, Lmax=3, sigma=16, C=1
    )

    # With self-query, there should be *some* valid tau values (not all -1).
    tau_np = np.array(tau)
    assert np.any(tau_np >= 0), "expected at least one valid match with self-query"


def test_base_postings_exact_key_overflow_rejected():
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(OverflowError):
        lookup_full_l_base_postings(
            Q, K, cap_end, successor, Lmax=1, sigma=2**63, C=4
        )
