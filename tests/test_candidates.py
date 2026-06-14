import numpy as np
import jax.numpy as jnp

from rosa_gpu_jax import verify_cpu_candidates
from rosa_gpu_jax.reference import brute_force_candidate_verify


def test_verify_cpu_candidates_matches_reference():
    Q_np = np.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    C = 4

    cand = np.full((B, R, T, C), -1, dtype=np.int64)
    for t in range(T):
        # Deliberately include both useful and useless candidates.
        vals = [t - 1, t - 2, 0, 3]
        for c, v in enumerate(vals[:C]):
            cand[0, 0, t, c] = v if v >= 0 else -1

    tau, best_len = verify_cpu_candidates(jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cand), Lmax=4)
    tau_ref, len_ref = brute_force_candidate_verify(Q_np, K_np, cand, Lmax=4)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(best_len), len_ref)


def test_verify_cpu_candidates_custom_cap_successor_matches_reference():
    Q_np = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    C = 5
    cand = np.full((B, R, T, C), -1, dtype=np.int64)
    for t in range(T):
        cand[0, 0, t, :] = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    cap = np.full((B, R, T), 4, dtype=np.int64)
    succ = np.broadcast_to(np.array([100, 101, 200, 201, 300, 301], dtype=np.int64), (B, R, T))

    tau, best_len = verify_cpu_candidates(
        jnp.asarray(Q_np),
        jnp.asarray(K_np),
        jnp.asarray(cand),
        Lmax=3,
        cap_end=jnp.asarray(cap),
        successor=jnp.asarray(succ),
    )
    tau_ref, len_ref = brute_force_candidate_verify(Q_np, K_np, cand, Lmax=3, cap_end=cap, successor=succ)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(best_len), len_ref)


def test_verify_cpu_candidates_rejects_out_of_range_candidate():
    Q_np = np.array([[[1, 2, 3]]], dtype=np.int64)
    K_np = Q_np.copy()
    cand = np.array([[[[3], [0], [1]]]], dtype=np.int64)
    with np.testing.assert_raises(ValueError):
        verify_cpu_candidates(jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cand), Lmax=2)
