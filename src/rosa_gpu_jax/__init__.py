"""GPU-friendly JAX prototypes for ROSA-style suffix lookup.

The public API intentionally exposes only small, composable functions.  These
functions operate on already-discretized symbol streams shaped [B, R, T].
"""

import time

import jax
import jax.numpy as jnp

# Exact base keys and predecessor-search scores use int64/uint64.  Enabling x64
# at import time avoids silent truncation on default JAX installations.
jax.config.update("jax_enable_x64", True)

from rosa_gpu_jax.bitset import lookup_full_l_bitset  # experimental
from rosa_gpu_jax.block_table import (
    block_keys_base,
    lookup_full_l_base,
    lookup_one_l_base,
    lookup_one_l_from_keys,
)
from rosa_gpu_jax.candidates import verify_cpu_candidates
from rosa_gpu_jax.causal import NEG, make_raw_causal_aux, make_rosa_causal_aux
from rosa_gpu_jax.counterfactual import q_bit_counterfactual_tau
from rosa_gpu_jax.diag_dp import lookup_full_l_diag_dp  # streaming diagonal-DP
from rosa_gpu_jax.dp import lookup_full_l_dp
from rosa_gpu_jax.dp_tpu import lookup_full_l_dense_tpu  # TPU benchmark
from rosa_gpu_jax.postings import (
    lookup_full_l_base_postings,
    lookup_full_l_drp_lce,
    lookup_full_l_rolling_postings,
)
from rosa_gpu_jax.prefix_table import lookup_full_l_counting_prefix  # exact dense prefix table
from rosa_gpu_jax.rolling_hash import (
    lookup_full_l_rolling,
    lookup_one_l_rolling,
    rolling_block_keys_u64,
    rolling_prefix_u64,
)
from rosa_gpu_jax.rolling_verified import lookup_full_l_rolling_verified
from rosa_gpu_jax.shift_and import lookup_full_l_shift_and  # Shift-And bitset
from rosa_gpu_jax.streaming_causal import lookup_full_l_streaming_causal  # streaming base-key bucket
from rosa_gpu_jax.suffix_tree_lookup import lookup_full_l_sa, suffix_array_batch  # SA-based
from rosa_gpu_jax.validation import max_exact_L


def warmup(
    *,
    Lmax_values: tuple[int, ...] = (2, 4, 8),
    sigma_values: tuple[int, ...] = (8, 16, 32, 256),
    base_for_rolling: int = 257,
    B: int = 2,
    R: int = 2,
    T: int = 64,
    M: int = 3,
    verbose: bool = False,
):
    """Pre-compile JIT kernels for common parameter combinations.

    JAX traces and compiles each ``@jit`` function on its first call.  Call
    this function at service startup to pay the compilation cost upfront.

    Parameters
    ----------
    Lmax_values:
        Block lengths to warm up.
    sigma_values:
        Alphabet sizes to warm up.
    base_for_rolling:
        Base used for rolling-hash kernels (prime > max sigma).
    B, R, T:
        Dummy batch dimensions used for tracing.
    M:
        Number of bits for counterfactual warmup.
    verbose:
        When ``True``, print which combination is being compiled.
    """
    for Lmax in Lmax_values:
        for sigma in sigma_values:
            if Lmax > T:
                continue
            # Skip combinations that would overflow uint64 combined keys.
            safe_Lmax = max_exact_L(sigma, T)
            Lmax_eff = min(Lmax, safe_Lmax)
            if Lmax_eff < 1:
                continue

            Q = jnp.full((B, R, T), 0, dtype=jnp.int64)
            K = jnp.full((B, R, T), 1, dtype=jnp.int64)
            cap_end, successor = make_raw_causal_aux(B, R, T)

            label = f"base Lmax={Lmax_eff} sigma={sigma}"
            if verbose:
                print(f"  warming up {label} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_base(Q, K, cap_end, successor, Lmax=Lmax_eff, sigma=sigma)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

            # Counting-prefix warmup (same constraints as base).
            label_cp = f"counting_prefix Lmax={Lmax_eff} sigma={sigma}"
            if verbose:
                print(f"  warming up {label_cp} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_counting_prefix(Q, K, cap_end, successor, Lmax=Lmax_eff, sigma=sigma)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

            # Streaming-causal warmup (same constraints as base, causal-only).
            label_sc = f"streaming_causal Lmax={Lmax_eff} sigma={sigma}"
            if verbose:
                print(f"  warming up {label_sc} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_streaming_causal(Q, K, cap_end, successor, Lmax=Lmax_eff, sigma=sigma)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

        # Rolling-hash kernels (Lmax-dependent, not sigma-dependent).
        if Lmax <= T:
            label_r = f"rolling Lmax={Lmax} base={base_for_rolling}"
            if verbose:
                print(f"  warming up {label_r} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_rolling(Q, K, cap_end, successor, Lmax=Lmax, base=base_for_rolling)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

    # Counterfactual warmup.
    if M > 0 and len(sigma_values) > 0:
        sigma_cf = max(s for s in sigma_values if s >= (1 << M))
        Lmax_cf = min(max(Lmax_values), T, max_exact_L(sigma_cf, T))
        if Lmax_cf >= 1 and sigma_cf >= (1 << M):
            Q_cf = jnp.full((B, R, T), 0, dtype=jnp.int64)
            K_cf = jnp.full((B, R, T), 0, dtype=jnp.int64)
            cap_end, successor = make_raw_causal_aux(B, R, T)
            label_cf = f"counterfactual Lmax={Lmax_cf} sigma={sigma_cf} M={M}"
            if verbose:
                print(f"  warming up {label_cf} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            q_bit_counterfactual_tau(Q_cf, K_cf, cap_end, successor, Lmax=Lmax_cf, sigma=sigma_cf, M=M)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

    # Candidates verifier warmup.
    Lmax_cand = min(max(Lmax_values), T)
    if Lmax_cand >= 1:
        Q_cand = jnp.full((B, R, T), 0, dtype=jnp.int64)
        K_cand = jnp.full((B, R, T), 0, dtype=jnp.int64)
        # Build a trivial candidate array in numpy to avoid JAX Python-loop overhead.
        import numpy as np
        cand_np = np.full((B, R, T, 4), -1, dtype=np.int64)
        for t in range(min(T, 4)):
            cand_np[:, :, t, :t] = np.arange(t, dtype=np.int64)
        cand = jnp.asarray(cand_np)
        cap_end, successor = make_raw_causal_aux(B, R, T)
        label_cand = f"candidates Lmax={Lmax_cand}"
        if verbose:
            print(f"  warming up {label_cand} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        verify_cpu_candidates(Q_cand, K_cand, cand, Lmax=Lmax_cand)
        t1 = time.perf_counter()
        if verbose:
            print(f"{t1 - t0:.3f}s")

    # DP warmup (dense).
    Lmax_dp = min(max(Lmax_values), T)
    if Lmax_dp >= 1:
        Q_dp = jnp.full((B, R, T), 0, dtype=jnp.int64)
        K_dp = jnp.full((B, R, T), 1, dtype=jnp.int64)
        cap_end, successor = make_raw_causal_aux(B, R, T)
        label_dp = f"dp Lmax={Lmax_dp}"
        if verbose:
            print(f"  warming up {label_dp} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        lookup_full_l_dp(Q_dp, K_dp, cap_end, successor, Lmax=Lmax_dp)
        t1 = time.perf_counter()
        if verbose:
            print(f"{t1 - t0:.3f}s")

    # Diag-DP warmup (sigma-free).
    for Lmax in Lmax_values:
        if Lmax > T:
            continue
        Q_diag = jnp.full((B, R, T), 0, dtype=jnp.int64)
        K_diag = jnp.full((B, R, T), 1, dtype=jnp.int64)
        cap_end, successor = make_raw_causal_aux(B, R, T)
        label_diag = f"diag_dp Lmax={Lmax}"
        if verbose:
            print(f"  warming up {label_diag} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        lookup_full_l_diag_dp(Q_diag, K_diag, cap_end, successor, Lmax=Lmax)
        t1 = time.perf_counter()
        if verbose:
            print(f"{t1 - t0:.3f}s")

    # Shift-And warmup.
    for Lmax in Lmax_values:
        if Lmax > T:
            continue
        for sigma in sigma_values:
            Q_sa = jnp.full((B, R, T), 0, dtype=jnp.int64)
            K_sa = jnp.full((B, R, T), 1, dtype=jnp.int64)
            cap_end, successor = make_raw_causal_aux(B, R, T)
            label_sa = f"shift_and Lmax={Lmax} sigma={sigma}"
            if verbose:
                print(f"  warming up {label_sa} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_shift_and(Q_sa, K_sa, cap_end, successor, Lmax=Lmax, sigma=sigma)
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

    # SA lookup warmup.
    for Lmax in Lmax_values:
        if Lmax > T:
            continue
        Q_sa = jnp.full((B, R, T), 0, dtype=jnp.int64)
        K_sa = jnp.full((B, R, T), 1, dtype=jnp.int64)
        cap_end, successor = make_raw_causal_aux(B, R, T)
        label_sa = f"sa Lmax={Lmax}"
        if verbose:
            print(f"  warming up {label_sa} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        lookup_full_l_sa(Q_sa, K_sa, cap_end, successor, Lmax=Lmax)
        t1 = time.perf_counter()
        if verbose:
            print(f"{t1 - t0:.3f}s")

    # Postings warmup (base + rolling).
    for Lmax in Lmax_values:
        for sigma in sigma_values:
            if Lmax > T:
                continue
            safe_Lmax = max_exact_L(sigma, T)
            Lmax_eff = min(Lmax, safe_Lmax)
            if Lmax_eff < 1:
                continue
            Q_p = jnp.full((B, R, T), 0, dtype=jnp.int64)
            K_p = jnp.full((B, R, T), 1, dtype=jnp.int64)
            cap_end, successor = make_raw_causal_aux(B, R, T)
            label_bp = f"base_postings Lmax={Lmax_eff} sigma={sigma} C=4"
            if verbose:
                print(f"  warming up {label_bp} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_base_postings(
                Q_p, K_p, cap_end, successor, Lmax=Lmax_eff, sigma=sigma, C=4
            )
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

        if Lmax <= T:
            Q_rp = jnp.full((B, R, T), 0, dtype=jnp.int64)
            K_rp = jnp.full((B, R, T), 1, dtype=jnp.int64)
            cap_end, successor = make_raw_causal_aux(B, R, T)
            label_rp = f"rolling_postings Lmax={Lmax} base={base_for_rolling} C=4"
            if verbose:
                print(f"  warming up {label_rp} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            lookup_full_l_rolling_postings(
                Q_rp, K_rp, cap_end, successor, Lmax=Lmax, base=base_for_rolling, C=4
            )
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1 - t0:.3f}s")

    # DRP+LCE warmup (sigma-free, only depends on Lmax and C).
    for Lmax in Lmax_values:
        if Lmax > T:
            continue
        Q_drp = jnp.full((B, R, T), 0, dtype=jnp.int64)
        K_drp = jnp.full((B, R, T), 1, dtype=jnp.int64)
        cap_end, successor = make_raw_causal_aux(B, R, T)
        label_drp = f"drp_lce Lmax={Lmax} C=4"
        if verbose:
            print(f"  warming up {label_drp} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        lookup_full_l_drp_lce(
            Q_drp, K_drp, cap_end, successor, Lmax=Lmax, C=4
        )
        t1 = time.perf_counter()
        if verbose:
            print(f"{t1 - t0:.3f}s")


# ---- pmap-based multi-GPU wrappers ----

def _ensure_divisible(B: int, n_devices: int, name: str) -> tuple[int, int]:
    """Pad B to be divisible by n_devices, returning (B_padded, original_B)."""
    if B % n_devices == 0:
        return B, B
    padded = ((B // n_devices) + 1) * n_devices
    import warnings
    warnings.warn(
        f"{name}: B={B} is not divisible by n_devices={n_devices}; "
        f"padding to B={padded}",
        stacklevel=2,
    )
    return padded, B


def _pad_to(arr, target_B: int, original_B: int):
    """Pad arr along the leading dimension from original_B to target_B."""
    if target_B == original_B:
        return arr
    pad_size = target_B - original_B
    pad_shape = list(arr.shape)
    pad_shape[0] = pad_size
    pad_block = jnp.zeros(tuple(pad_shape), dtype=arr.dtype)
    return jnp.concatenate([arr, pad_block], axis=0)


def lookup_full_l_base_pmap(
    Q, K, cap_end, successor, Lmax: int, sigma: int, *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """``pmap``-distributed variant of :func:`lookup_full_l_base`.

    Splits the batch dimension ``B`` across available devices.  The input
    arrays are automatically padded when ``B`` is not divisible by the
    device count.
    """
    n_devices = jax.device_count()
    if n_devices <= 1:
        return lookup_full_l_base(
            Q, K, cap_end, successor, Lmax, sigma,
            tau_cap=tau_cap, validate_symbols=validate_symbols,
        )

    Q_arr = jnp.asarray(Q)
    B = Q_arr.shape[0]
    padded_B, orig_B = _ensure_divisible(B, n_devices, "lookup_full_l_base_pmap")

    def _pmap_body(Q_chunk, K_chunk, cap_chunk, succ_chunk, tcap_chunk):
        return lookup_full_l_base(
            Q_chunk, K_chunk, cap_chunk, succ_chunk, Lmax, sigma,
            tau_cap=tcap_chunk if tau_cap is not None else None,
            validate_symbols=validate_symbols,
        )

    # Reshape to [n_devices, B_per_device, R, T]
    B_per = padded_B // n_devices

    def _reshape(arr):
        arr = jnp.asarray(arr)
        arr = _pad_to(arr, padded_B, B)
        return arr.reshape((n_devices, B_per) + arr.shape[1:])

    Q_shard = _reshape(Q_arr)
    K_shard = _reshape(jnp.asarray(K))
    cap_shard = _reshape(jnp.asarray(cap_end))
    succ_shard = _reshape(jnp.asarray(successor))
    tcap = None if tau_cap is None else jnp.asarray(tau_cap)
    tcap_shard = None if tcap is None else _reshape(tcap)

    tau_shard, len_shard = jax.pmap(_pmap_body)(
        Q_shard, K_shard, cap_shard, succ_shard, tcap_shard,
    )
    # Concatenate and strip padding
    tau = tau_shard.reshape((padded_B,) + tau_shard.shape[2:])[:B]
    match_len = len_shard.reshape((padded_B,) + len_shard.shape[2:])[:B]
    return tau, match_len


def lookup_full_l_rolling_pmap(
    Q, K, cap_end, successor, Lmax: int, base: int, *,
    tau_cap=None,
    algorithm: str = "mask",
    num_buckets: int | None = None,
):
    """``pmap``-distributed variant of :func:`lookup_full_l_rolling`.

    Splits the batch dimension ``B`` across available devices.
    """
    n_devices = jax.device_count()
    if n_devices <= 1:
        return lookup_full_l_rolling(
            Q, K, cap_end, successor, Lmax, base,
            tau_cap=tau_cap, algorithm=algorithm, num_buckets=num_buckets,
        )

    Q_arr = jnp.asarray(Q)
    B = Q_arr.shape[0]
    padded_B, orig_B = _ensure_divisible(B, n_devices, "lookup_full_l_rolling_pmap")

    B_per = padded_B // n_devices

    def _reshape(arr):
        arr = jnp.asarray(arr)
        arr = _pad_to(arr, padded_B, B)
        return arr.reshape((n_devices, B_per) + arr.shape[1:])

    Q_shard = _reshape(Q_arr)
    K_shard = _reshape(jnp.asarray(K))
    cap_shard = _reshape(jnp.asarray(cap_end))
    succ_shard = _reshape(jnp.asarray(successor))
    tcap = None if tau_cap is None else jnp.asarray(tau_cap)
    tcap_shard = None if tcap is None else _reshape(tcap)

    def _pmap_body(Q_chunk, K_chunk, cap_chunk, succ_chunk, tcap_chunk):
        return lookup_full_l_rolling(
            Q_chunk, K_chunk, cap_chunk, succ_chunk, Lmax, base,
            tau_cap=tcap_chunk if tau_cap is not None else None,
            algorithm=algorithm, num_buckets=num_buckets,
        )

    tau_shard, len_shard = jax.pmap(_pmap_body)(
        Q_shard, K_shard, cap_shard, succ_shard, tcap_shard,
    )
    tau = tau_shard.reshape((padded_B,) + tau_shard.shape[2:])[:B]
    match_len = len_shard.reshape((padded_B,) + len_shard.shape[2:])[:B]
    return tau, match_len


__all__ = [
    "NEG",
    "make_raw_causal_aux",
    "make_rosa_causal_aux",
    "block_keys_base",
    "lookup_one_l_from_keys",
    "lookup_one_l_base",
    "lookup_full_l_base",
    "lookup_full_l_base_pmap",
    "lookup_full_l_dp",
    "lookup_full_l_base_postings",
    "lookup_full_l_drp_lce",
    "lookup_full_l_rolling_postings",
    "rolling_prefix_u64",
    "rolling_block_keys_u64",
    "lookup_one_l_rolling",
    "lookup_full_l_rolling",
    "lookup_full_l_rolling_pmap",
    "lookup_full_l_rolling_verified",
    "lookup_full_l_bitset",  # experimental
    "lookup_full_l_counting_prefix",  # exact dense prefix table
    "lookup_full_l_diag_dp",  # streaming diagonal-DP
    "lookup_full_l_shift_and",  # Shift-And bitset
    "lookup_full_l_streaming_causal",  # streaming base-key bucket
    "lookup_full_l_dense_tpu",  # TPU benchmark
    "lookup_full_l_sa",  # suffix-array based
    "suffix_array_batch",
    "verify_cpu_candidates",
    "q_bit_counterfactual_tau",
    "max_exact_L",
    "warmup",
]
