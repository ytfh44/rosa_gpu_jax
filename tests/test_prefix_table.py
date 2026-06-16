import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax.prefix_table import lookup_full_l_counting_prefix
from rosa_gpu_jax import make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.reference import (
    brute_force_lookup,
    rosa_batch_reference_tau,
)


def test_counting_prefix_smoke_vs_bruteforce():
    """Smoke test: counting-prefix should match brute force on a simple self-query."""
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_counting_prefix(
        Q, K, cap_end, successor, Lmax=3, sigma=16
    )
    tau_ref, len_ref = brute_force_lookup(
        np.array(Q), np.array(K), np.array(cap_end), np.array(successor), Lmax=3
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_counting_prefix_random_vs_bruteforce():
    """Random test vs brute_force_lookup."""
    rng = np.random.default_rng(42)
    B, R, T = 1, 2, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    Q = jnp.asarray(Q_np)
    K = jnp.asarray(K_np)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_counting_prefix(
        Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_counting_prefix_identical_qk():
    """Q=K: counting-prefix should match brute_force_lookup exactly."""
    rng = np.random.default_rng(77)
    B, R, T = 1, 2, 10
    sigma = 4
    Lmax = 4

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = Q_np.copy()
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_counting_prefix(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_counting_prefix_rosa_semantics():
    """Counting-prefix must respect ROSA no-backtrack."""
    Z = np.array([[[0, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_counting_prefix(
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


def test_counting_prefix_repeated_run_pattern():
    """Sequences with runs like [0,0,1,1,0,0] — verify ROSA tau."""
    Z = np.array([[[0, 0, 1, 1, 0, 0]]], dtype=np.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(Z))

    tau, match_len = lookup_full_l_counting_prefix(
        jnp.asarray(Z), jnp.asarray(Z), cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap
    )

    expected_tau = rosa_batch_reference_tau(Z)
    np.testing.assert_array_equal(np.array(tau), expected_tau)


def test_counting_prefix_custom_cap_and_successor():
    """Counting-prefix with custom cap/successor matches brute force."""
    Q_np = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.int64)
    K_np = Q_np.copy()
    B, R, T = Q_np.shape
    cap = np.zeros((B, R, T), dtype=np.int64)
    succ = np.zeros((B, R, T), dtype=np.int64)
    cap[:] = 4
    succ[:] = np.array([10, 11, 20, 21, 30, 31], dtype=np.int64)

    tau, match_len = lookup_full_l_counting_prefix(
        jnp.asarray(Q_np), jnp.asarray(K_np), jnp.asarray(cap), jnp.asarray(succ),
        Lmax=3, sigma=4
    )
    tau_ref, len_ref = brute_force_lookup(Q_np, K_np, cap, succ, Lmax=3)

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_counting_prefix_multi_batch_and_route():
    """Counting-prefix works with B=2, R=3."""
    rng = np.random.default_rng(99)
    B, R, T = 2, 3, 8
    sigma = 4
    Lmax = 3

    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, :4] = Q_np[:, :, :4]

    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_counting_prefix(
        jnp.asarray(Q_np), jnp.asarray(K_np), cap_end, successor,
        Lmax=Lmax, sigma=sigma
    )
    tau_ref, len_ref = brute_force_lookup(
        Q_np, K_np, np.array(cap_end), np.array(successor), Lmax=Lmax
    )

    np.testing.assert_array_equal(np.array(tau), tau_ref)
    np.testing.assert_array_equal(np.array(match_len), len_ref)


def test_counting_prefix_lmax_larger_than_t_rejected():
    """Lmax > T must raise ValueError."""
    Q = jnp.array([[[0, 1, 2]]], dtype=jnp.int64)
    K = Q
    cap_end, successor = make_raw_causal_aux(1, 1, 3)
    with np.testing.assert_raises(ValueError):
        lookup_full_l_counting_prefix(Q, K, cap_end, successor, Lmax=4, sigma=2)


def test_counting_prefix_jit_composable():
    """Counting-prefix can be nested under jax.jit."""
    import jax

    Z = jnp.array([[[0, 1, 0, 1, 0, 0]]], dtype=jnp.int64)
    cap, succ, tau_cap = make_rosa_causal_aux(Z)

    def run(Q, K, cap_end, successor, tcap):
        return lookup_full_l_counting_prefix(
            Q, K, cap_end, successor, Lmax=6, sigma=2, tau_cap=tcap
        )

    tau_jit, len_jit = jax.jit(run)(Z, Z, cap, succ, tau_cap)
    tau_ref, len_ref = lookup_full_l_counting_prefix(
        Z, Z, cap, succ, Lmax=6, sigma=2, tau_cap=tau_cap
    )

    np.testing.assert_array_equal(np.array(tau_jit), np.array(tau_ref))
    np.testing.assert_array_equal(np.array(len_jit), np.array(len_ref))
