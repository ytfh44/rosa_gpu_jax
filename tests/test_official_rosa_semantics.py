import itertools

import jax
import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import (
    lookup_full_l_base,
    make_rosa_causal_aux,
    q_bit_counterfactual_tau,
    verify_cpu_candidates,
)
from rosa_gpu_jax.reference import (
    brute_force_candidate_verify,
    brute_force_lookup,
    rosa_batch_reference_tau,
)


def _full_candidates(B, R, T):
    cand = np.broadcast_to(np.arange(T, dtype=np.int64), (B, R, T, T)).copy()
    return cand


def test_rosa_aux_counterexample_does_not_backtrack_to_older_valid_match():
    # Official native recurrence returns -1 at the last token.  The tempting but
    # wrong implementation filters out the current run before matching and then
    # falls back to the older 0 at position 0, returning tau=1.
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_base(
        jnp.asarray(Z), jnp.asarray(Z), cap, succ, Lmax=Z.shape[-1], sigma=2, tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(Z)
    np.testing.assert_array_equal(np.array(tau), expected_tau)
    np.testing.assert_array_equal(expected_tau, np.array([[[-1, -1, 1, -1]]], dtype=np.int64))
    np.testing.assert_array_equal(np.array(match_len), np.array([[[0, 0, 1, 0]]], dtype=np.int32))


def test_exact_lookup_matches_official_native_sam_for_all_binary_sequences_len_6():
    seqs = np.array(list(itertools.product([0, 1], repeat=6)), dtype=np.int64)[:, None, :]
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(seqs))

    tau, match_len = lookup_full_l_base(
        jnp.asarray(seqs), jnp.asarray(seqs), cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(seqs)
    expected_len = brute_force_lookup(seqs, seqs, np.array(cap), np.array(succ), Lmax=6, tau_cap=np.array(tau_cap))[1]
    np.testing.assert_array_equal(np.array(tau), expected_tau)
    np.testing.assert_array_equal(np.array(match_len), expected_len)


def test_exact_lookup_matches_official_native_sam_on_edge_patterns():
    Z = np.array(
        [
            [[0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 1, 1, 0, 0, 1, 0]],
            [[2, 0, 2, 2, 1, 2, 0, 2]],
        ],
        dtype=np.int64,
    )
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, _ = lookup_full_l_base(
        jnp.asarray(Z), jnp.asarray(Z), cap, succ, Lmax=8, sigma=3, tau_cap=tau_cap
    )

    np.testing.assert_array_equal(np.array(tau), rosa_batch_reference_tau(Z))


def test_candidate_verifier_matches_official_aux_with_full_candidate_recall():
    Z = np.array([[[0, 1, 0, 0, 1, 0, 1]]], dtype=np.int64)
    B, R, T = Z.shape
    cand = _full_candidates(B, R, T)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, best_len = verify_cpu_candidates(
        jnp.asarray(Z),
        jnp.asarray(Z),
        jnp.asarray(cand),
        Lmax=T,
        cap_end=cap,
        successor=succ,
        tau_cap=tau_cap,
    )
    tau_ref, len_ref = brute_force_candidate_verify(
        Z, Z, cand, Lmax=T, cap_end=np.array(cap), successor=np.array(succ), tau_cap=np.array(tau_cap)
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(best_len), len_ref)
    np.testing.assert_array_equal(np.array(tau)[0, 0, 3], -1)


def test_public_lookup_wrapper_can_be_nested_under_jax_jit():
    Z = jnp.array([[[0, 1, 0, 1, 0, 0]]], dtype=jnp.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cap_end, successor, tcap):
        return lookup_full_l_base(Q, K, cap_end, successor, Lmax=6, sigma=2, tau_cap=tcap)

    tau_jit, len_jit = jax.jit(run)(Z, Z, cap, succ, tau_cap)
    tau_ref, len_ref = lookup_full_l_base(Z, Z, cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap)

    np.testing.assert_array_equal(np.array(tau_jit), np.array(tau_ref))
    np.testing.assert_array_equal(np.array(len_jit), np.array(len_ref))


def test_candidate_wrapper_can_be_nested_under_jax_jit():
    Z = jnp.array([[[0, 1, 0, 0, 1]]], dtype=jnp.int64)
    B, R, T = Z.shape
    cand = jnp.asarray(_full_candidates(B, R, T))
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cand_end, cap_end, successor, tcap):
        return verify_cpu_candidates(
            Q, K, cand_end, Lmax=5, cap_end=cap_end, successor=successor, tau_cap=tcap
        )

    tau_jit, len_jit = jax.jit(run)(Z, Z, cand, cap, succ, tau_cap)
    tau_ref, len_ref = verify_cpu_candidates(Z, Z, cand, Lmax=5, cap_end=cap, successor=succ, tau_cap=tau_cap)

    np.testing.assert_array_equal(np.array(tau_jit), np.array(tau_ref))
    np.testing.assert_array_equal(np.array(len_jit), np.array(len_ref))


def test_counterfactual_wrapper_can_be_nested_under_jax_jit():
    Z = jnp.array([[[0, 1, 2, 3, 0, 1]]], dtype=jnp.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cap_end, successor, tcap):
        return q_bit_counterfactual_tau(Q, K, cap_end, successor, Lmax=4, sigma=4, M=2, tau_cap=tcap)

    tau0_jit, tau1_jit = jax.jit(run)(Z, Z, cap, succ, tau_cap)
    tau0_ref, tau1_ref = q_bit_counterfactual_tau(Z, Z, cap, succ, Lmax=4, sigma=4, M=2, tau_cap=tau_cap)

    np.testing.assert_array_equal(np.array(tau0_jit), np.array(tau0_ref))
    np.testing.assert_array_equal(np.array(tau1_jit), np.array(tau1_ref))
