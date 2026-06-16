import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax.streaming_causal import lookup_full_l_streaming_causal
from rosa_gpu_jax import make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.reference import (
    brute_force_lookup,
    rosa_batch_reference_tau,
)


def test_streaming_causal_smoke_vs_bruteforce():
    """Smoke test: streaming causal should match brute force on a simple self-query."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_streaming_causal(
        Q, K, cap_end, successor, Lmax=3, sigma=16
    )
    tau_ref, len_ref = brute_force_lookup(
        np.array(Q), np.array(K), np.array(cap_end), np.array(successor), Lmax=3
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_streaming_causal_random_vs_bruteforce():
    """Random test vs brute_force_lookup."""
    rng = np.random.default_rng(43)
    B, R, T = 1, 2, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_streaming_causal(
        Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_streaming_causal_identical_qk():
    """Q=K: streaming causal should match brute_force_lookup exactly."""
    rng = np.random.default_rng(78)
    B, R, T = 1, 2, 10
    sigma = 4
    Lmax = 4

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = Q_np.copy()
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_streaming_causal(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_streaming_causal_rosa_semantics():
    """Streaming causal must respect ROSA no-backtrack."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_streaming_causal(
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


def test_streaming_causal_repeated_run_pattern():
    """Sequences with runs like [0,0,1,1,0,0] — verify ROSA tau."""
    Z = np.array([[[0, 0, 1, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_streaming_causal(
        jnp.asarray(Z), jnp.asarray(Z), cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(Z)
    np.testing.assert_array_equal(np.array(tau), expected_tau)


def test_streaming_causal_multi_batch_and_route():
    """Streaming causal works with B=2, R=3."""
    rng = np.random.default_rng(100)
    B, R, T = 2, 3, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_streaming_causal(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_streaming_causal_lmax_larger_than_t_rejected():
    """Lmax > T must raise ValueError."""
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        lookup_full_l_streaming_causal(Q, K, cap_end, successor, Lmax=4, sigma=2)


def test_streaming_causal_jit_composable():
    """Streaming causal can be nested under jax.jit."""
    import jax

    Z = jnp.array([[[0, 1, 0, 1, 0, 0]]], dtype=jnp.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cap_end, successor, tcap):
        return lookup_full_l_streaming_causal(
            Q, K, cap_end, successor, Lmax=6, sigma=2, tau_cap=tcap
        )

    tau_jit, len_jit = jax.jit(run)(Z, Z, cap, succ, tau_cap)
    tau_ref, len_ref = lookup_full_l_streaming_causal(
        Z, Z, cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap
    )

    np.testing.assert_array_equal(np.array(tau_jit), np.array(tau_ref))
    np.testing.assert_array_equal(np.array(len_jit), np.array(len_ref))


def test_streaming_causal_matches_counting_prefix():
    """Streaming causal should match counting prefix when cap_end[t]==t."""
    rng = np.random.default_rng(55)
    B, R, T = 2, 2, 8
    sigma = 3
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)

    cap_end, successor = make_raw_causal_aux(B, R, T)

    from rosa_gpu_jax.prefix_table import lookup_full_l_counting_prefix

    tau_sc, len_sc = lookup_full_l_streaming_causal(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )
    tau_cp, len_cp = lookup_full_l_counting_prefix(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )

    np.testing.assert_array_equal(np.array(tau_sc), np.array(tau_cp))
    np.testing.assert_array_equal(np.array(len_sc), np.array(len_cp))
