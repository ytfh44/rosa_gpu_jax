import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import lookup_full_l_base, make_raw_causal_aux, q_bit_counterfactual_tau
from rosa_gpu_jax.reference import brute_force_lookup


def test_q_bit_counterfactual_shapes_and_identity_branches():
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    M = 3
    sigma = 8
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau_true, _ = lookup_full_l_base(Q, K, cap_end, successor, Lmax=3, sigma=sigma)
    tau0, tau1 = q_bit_counterfactual_tau(Q, K, cap_end, successor, Lmax=3, sigma=sigma, M=M)

    assert tau0.shape == (M, B, R, T)
    assert tau1.shape == (M, B, R, T)

    Q_np = np.array(Q)
    tau_true_np = np.array(tau_true)
    tau0_np = np.array(tau0)
    tau1_np = np.array(tau1)

    # If the current bit is already 0, forcing 0 should match true tau at that time.
    # If the current bit is already 1, forcing 1 should match true tau at that time.
    for m in range(M):
        bit_is_one = ((Q_np >> m) & 1).astype(bool)
        np.testing.assert_array_equal(tau0_np[m][~bit_is_one], tau_true_np[~bit_is_one])
        np.testing.assert_array_equal(tau1_np[m][bit_is_one], tau_true_np[bit_is_one])


def test_q_bit_counterfactual_matches_bruteforce_for_each_current_symbol_branch():
    Q_np = np.array([[[0, 1, 2, 3, 0, 1]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    M = 2
    sigma = 4
    Lmax = 3
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau0, tau1 = q_bit_counterfactual_tau(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor, Lmax=Lmax, sigma=sigma, M=M
    )

    for m in range(M):
        for branch, tau_branch in [(0, np.array(tau0[m])), (1, np.array(tau1[m]))]:
            expected = np.full((B, R, T), -1, dtype=np.int64)
            mask = 1 << m
            for b in range(B):
                for r in range(R):
                    for t in range(T):
                        Q_forced = Q_np.copy()
                        if branch == 0:
                            Q_forced[b, r, t] &= ~mask
                        else:
                            Q_forced[b, r, t] |= mask
                        tau_ref, _ = brute_force_lookup(
                            Q_forced, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
                        )
                        expected[b, r, t] = tau_ref[b, r, t]
            np.testing.assert_array_equal(tau_branch, expected)


def test_q_bit_counterfactual_rejects_sigma_too_small_for_m():
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        q_bit_counterfactual_tau(Q, K, cap_end, successor, Lmax=2, sigma=3, M=2)
