"""Tests for suffix-array based ROSA lookup."""

import numpy as np
import pytest
from rosa_gpu_jax.suffix_tree_lookup import (
    lookup_full_l_sa,
    suffix_array_batch,
    _build_sa_one,
)
from rosa_gpu_jax.causal import make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.reference import brute_force_lookup
from rosa_gpu_jax.dp import lookup_full_l_dp


class TestSuffixArrayBuild:
    def test_simple(self):
        K = np.array([2, 1, 2, 1], dtype=np.int64)
        sa = _build_sa_one(K)
        # Lexicographic order of suffixes:
        # K[1:] = [1,2,1]
        # K[3:] = [1]
        # K[0:] = [2,1,2,1]
        # K[2:] = [2,1]
        # So SA should be [1, 3, 0, 2] or [3, 1, 0, 2]
        assert len(sa) == 4
        # Verify SA is a permutation
        assert set(sa) == {0, 1, 2, 3}
        # Verify suffixes are sorted
        for i in range(len(sa) - 1):
            a = tuple(K[sa[i]:])
            b = tuple(K[sa[i + 1]:])
            assert a <= b

    def test_batch(self):
        K = np.random.randint(0, 8, (2, 3, 16)).astype(np.int64)
        sa = suffix_array_batch(K)
        assert sa.shape == (2, 3, 16)
        for b in range(2):
            for r in range(3):
                assert set(sa[b, r]) == set(range(16))


class TestSuffixArrayLookup:
    @pytest.mark.parametrize("T", [4, 7, 8, 16, 32])
    @pytest.mark.parametrize("sigma", [4, 8])
    @pytest.mark.parametrize("Lmax", [2, 3, 4])
    def test_vs_dp(self, T, sigma, Lmax):
        if Lmax > T:
            pytest.skip("Lmax > T")
        np.random.seed(123 + T + sigma + Lmax)
        Q = np.random.randint(0, sigma, (2, 2, T)).astype(np.int64)
        K = np.random.randint(0, sigma, (2, 2, T)).astype(np.int64)
        B, R, _ = Q.shape

        cap_end, successor = make_raw_causal_aux(B, R, T)

        tau_sa, ml_sa = lookup_full_l_sa(Q, K, cap_end, successor, Lmax=Lmax)
        tau_dp, ml_dp = lookup_full_l_dp(Q, K, cap_end, successor, Lmax=Lmax)

        assert np.array_equal(tau_sa, tau_dp), (
            f"tau mismatch at T={T}, sigma={sigma}, Lmax={Lmax}"
        )
        assert np.array_equal(ml_sa, ml_dp)

    @pytest.mark.parametrize("T", [4, 7, 16])
    @pytest.mark.parametrize("sigma", [4, 8])
    @pytest.mark.parametrize("Lmax", [2, 3])
    def test_vs_reference_rosa(self, T, sigma, Lmax):
        if Lmax > T:
            pytest.skip("Lmax > T")
        np.random.seed(456 + T + sigma + Lmax)
        Q = np.random.randint(0, sigma, (1, 2, T)).astype(np.int64)
        K = np.random.randint(0, sigma, (1, 2, T)).astype(np.int64)

        cap_end, successor, tau_cap = make_rosa_causal_aux(K)

        tau_sa, ml_sa = lookup_full_l_sa(
            Q, K, cap_end, successor, Lmax=Lmax, tau_cap=tau_cap
        )
        tau_ref, ml_ref = brute_force_lookup(
            Q, K, cap_end, successor, Lmax, tau_cap=tau_cap
        )

        assert np.array_equal(tau_sa, tau_ref), (
            f"ROSA tau mismatch at T={T}, sigma={sigma}, Lmax={Lmax}"
        )
        assert np.array_equal(ml_sa, ml_ref)

    def test_readme_example(self):
        """Reproduce the example from the README."""
        import jax.numpy as jnp

        Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
        K = Q
        B, R, T = Q.shape
        cap_end, successor = make_raw_causal_aux(B, R, T)

        tau, match_len = lookup_full_l_sa(Q, K, cap_end, successor, Lmax=3)
        # Compare against block table (ground truth for this case)
        from rosa_gpu_jax import lookup_full_l_base

        tau_bt, ml_bt = lookup_full_l_base(
            Q, K, cap_end, successor, Lmax=3, sigma=16
        )

        assert np.array_equal(tau, tau_bt)
        assert np.array_equal(match_len, ml_bt)

    def test_reuse_sa(self):
        """Verify that pre-built SA can be reused across calls."""
        np.random.seed(789)
        Q1 = np.random.randint(0, 4, (1, 1, 8)).astype(np.int64)
        Q2 = np.random.randint(0, 4, (1, 1, 8)).astype(np.int64)
        K = np.random.randint(0, 4, (1, 1, 8)).astype(np.int64)

        cap_end, successor = make_raw_causal_aux(1, 1, 8)
        SA = suffix_array_batch(K)

        t1, _ = lookup_full_l_sa(Q1, K, cap_end, successor, Lmax=4, SA=SA)
        t2, _ = lookup_full_l_sa(Q2, K, cap_end, successor, Lmax=4, SA=SA)

        # Both should produce valid results (compare vs DP)
        t1_dp, _ = lookup_full_l_dp(Q1, K, cap_end, successor, Lmax=4)
        t2_dp, _ = lookup_full_l_dp(Q2, K, cap_end, successor, Lmax=4)

        assert np.array_equal(t1, t1_dp)
        assert np.array_equal(t2, t2_dp)

    def test_large_sigma(self):
        """SA approach should handle sigma beyond uint64 overflow."""
        np.random.seed(999)
        Q = np.random.randint(0, 256, (1, 1, 16)).astype(np.int64)
        K = np.random.randint(0, 256, (1, 1, 16)).astype(np.int64)

        cap_end, successor = make_raw_causal_aux(1, 1, 16)

        tau_sa, _ = lookup_full_l_sa(Q, K, cap_end, successor, Lmax=4)
        tau_dp, _ = lookup_full_l_dp(Q, K, cap_end, successor, Lmax=4)

        assert np.array_equal(tau_sa, tau_dp)
