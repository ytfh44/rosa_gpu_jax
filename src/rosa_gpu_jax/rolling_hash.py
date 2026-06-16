"""Rolling-hash block table lookup.

This path is meant for throughput experiments with larger ``Lmax``.  It uses
uint64 overflow rolling hash, so it is probabilistic unless the caller adds
bucket backtracking and exact tuple verification.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.block_table import (
    _lookup_one_l_from_keys_hash_end_jit,
    _lookup_one_l_from_keys_mask_end_jit,
    _lookup_one_l_from_keys_mask_jit,
)
from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    require_aux,
    require_base,
    require_L_for_T,
    require_Lmax_for_T,
    require_rank3_pair,
    require_symbol_array,
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("base",))
def _rolling_prefix_u64_jit(seq, base: int):
    seq_u = seq.astype(jnp.uint64) + jnp.uint64(1)

    def step(carry, x_t):
        nxt = carry * jnp.uint64(base) + x_t
        return nxt, nxt

    init = jnp.zeros(seq.shape[:-1], dtype=jnp.uint64)
    _, tail = jax.lax.scan(step, init, jnp.moveaxis(seq_u, -1, 0))

    tail = jnp.moveaxis(tail, 0, -1)
    zero = jnp.zeros(seq.shape[:-1] + (1,), dtype=jnp.uint64)
    return jnp.concatenate([zero, tail], axis=-1)


def rolling_prefix_u64(seq, base: int):
    """Compute uint64 overflow rolling-hash prefixes.

    ``prefix[..., i+1] = prefix[..., i] * base + seq[..., i] + 1``.
    """
    base_i = require_base(base)
    arr = require_symbol_array("seq", seq, validate_symbols=False)
    if arr.ndim != 3:
        raise ValueError(f"seq must be rank-3 shaped [B, R, T]; got seq.shape={arr.shape}")
    if any(int(x) <= 0 for x in arr.shape):
        raise ValueError(f"seq must have non-empty [B, R, T] dimensions; got {arr.shape}")
    return _rolling_prefix_u64_jit(arr, base=base_i)


@partial(jax.jit, static_argnames=("L", "base"))
def _rolling_block_keys_u64_jit(seq, L: int, base: int):
    T = seq.shape[-1]
    prefix = _rolling_prefix_u64_jit(seq, base=base)

    pow_L = jnp.asarray(base, dtype=jnp.uint64) ** jnp.asarray(L, dtype=jnp.uint64)
    left = prefix[..., : T - L + 1]
    right = prefix[..., L : T + 1]
    valid = right - left * pow_L

    pad = jnp.zeros(seq.shape[:-1] + (L - 1,), dtype=jnp.uint64)
    return jnp.concatenate([pad, valid], axis=-1)


def rolling_block_keys_u64(seq, L: int, base: int):
    """Return rolling-hash keys for length-L blocks ending at each position."""
    base_i = require_base(base)
    arr = require_symbol_array("seq", seq, validate_symbols=False)
    if arr.ndim != 3:
        raise ValueError(f"seq must be rank-3 shaped [B, R, T]; got seq.shape={arr.shape}")
    if any(int(x) <= 0 for x in arr.shape):
        raise ValueError(f"seq must have non-empty [B, R, T] dimensions; got {arr.shape}")
    L_i = require_L_for_T(L, int(arr.shape[-1]))
    return _rolling_block_keys_u64_jit(arr, L=L_i, base=base_i)


@partial(jax.jit, static_argnames=("L", "base"))
def _lookup_one_l_rolling_jit(Q, K, cap_end, successor, tau_cap, L: int, base: int):
    q_keys = _rolling_block_keys_u64_jit(Q, L=L, base=base)
    k_keys = _rolling_block_keys_u64_jit(K, L=L, base=base)
    return _lookup_one_l_from_keys_mask_jit(q_keys, k_keys, cap_end, successor, tau_cap, L=L)


def lookup_one_l_rolling(Q, K, cap_end, successor, L: int, base: int, *, tau_cap=None):
    """Rolling-hash lookup for one length L.

    This function does not verify raw symbol tuples after hash match and is
    therefore probabilistic.
    """
    base_i = require_base(base)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=None, validate_symbols=False)
    L_i = require_L_for_T(L, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_one_l_rolling_jit(Q_arr, K_arr, cap, succ, tcap, L=L_i, base=base_i)


@partial(jax.jit, static_argnames=("Lmax", "base"))
def _lookup_full_l_rolling_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, base: int):
    B, R, T = Q.shape

    # Precompute block keys for each L once.  This Python loop is intentionally
    # unrolled at trace time because _rolling_block_keys_u64_jit requires static
    # L for output shape (the padding length depends on L).  For Lmax <= 8 this
    # is the best trade-off; lax.fori_loop would require dynamic-shape support.
    q_keys_by_L = []
    k_keys_by_L = []
    for L in range(1, Lmax + 1):
        q_keys_by_L.append(_rolling_block_keys_u64_jit(Q, L=L, base=base))
        k_keys_by_L.append(_rolling_block_keys_u64_jit(K, L=L, base=base))

    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for idx, L in enumerate(range(1, Lmax + 1)):
        q_keys = q_keys_by_L[idx]
        k_keys = k_keys_by_L[idx]
        _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_from_keys_mask_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L
        )
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


@partial(jax.jit, static_argnames=("Lmax", "base", "num_buckets"))
def _lookup_full_l_rolling_hash_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, base: int, num_buckets: int):
    """Full-L rolling-hash lookup using the O(T) hash-table kernel per L."""
    B, R, T = Q.shape

    q_keys_by_L = []
    k_keys_by_L = []
    for L in range(1, Lmax + 1):
        q_keys_by_L.append(_rolling_block_keys_u64_jit(Q, L=L, base=base))
        k_keys_by_L.append(_rolling_block_keys_u64_jit(K, L=L, base=base))

    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for idx, L in enumerate(range(1, Lmax + 1)):
        q_keys = q_keys_by_L[idx]
        k_keys = k_keys_by_L[idx]
        _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_from_keys_hash_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L, num_buckets=num_buckets
        )
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def _next_prime(n: int) -> int:
    """Return the smallest prime >= n."""
    import math
    if n <= 2:
        return 2
    n |= 1  # make odd
    while True:
        limit = int(math.isqrt(n)) + 1
        for p in range(3, limit, 2):
            if n % p == 0:
                break
        else:
            return n
        n += 2


def _default_num_buckets(T: int) -> int:
    """Choose a prime number of buckets given sequence length T."""
    return _next_prime(max(17, T * 4))


def lookup_full_l_rolling(
    Q, K, cap_end, successor, Lmax: int, base: int, *,
    tau_cap=None,
    algorithm: str = "mask",
    num_buckets: int | None = None,
):
    """Longest-match lookup over all lengths ``1..Lmax`` using rolling hash.

    This function is probabilistic because hash matches are not tuple-verified.

    Parameters
    ----------
    algorithm:
        ``"mask"`` (default) uses the O(T²) broadcast-mask kernel which is
        exact for uint64 keys and fast on GPU for ``T <= 256``.
        ``"hash"`` uses an O(T) hash-table kernel — faster for ``T > 512``,
        but introduces false negatives on bucket collisions.
    num_buckets:
        Hash-table bucket count when ``algorithm="hash"``.  Defaults to a
        prime > 4·T.  Larger values reduce collision probability.
    """
    base_i = require_base(base)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=None, validate_symbols=False)
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))

    if algorithm == "hash":
        nb = num_buckets if num_buckets is not None else _default_num_buckets(T)
        return _lookup_full_l_rolling_hash_jit(
            Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, base=base_i, num_buckets=nb
        )
    elif algorithm == "mask":
        return _lookup_full_l_rolling_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, base=base_i)
    else:
        raise ValueError(f"algorithm must be 'mask' or 'hash'; got {algorithm!r}")
