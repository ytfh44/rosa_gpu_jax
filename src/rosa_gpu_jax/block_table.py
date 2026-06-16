"""Exact base-encoded block table lookup.

This module turns length-L suffix lookup into sorting and predecessor search
on block keys.  The public functions validate shape, symbol ranges, uint64
safety, and optional ROSA/RLE post-successor caps before dispatching to
JIT-compiled kernels.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    ensure_exact_key_safe,
    ensure_precomputed_keys_combined_safe,
    require_aux,
    require_key_array_pair,
    require_L_for_T,
    require_Lmax_for_T,
    require_rank3_pair,
    require_sigma,
    require_symbol_array,
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("L", "sigma"))
def _block_keys_base_jit(seq, L: int, sigma: int):
    seq_u = seq.astype(jnp.uint64)
    T = seq.shape[-1]

    weights = jnp.asarray(sigma, dtype=jnp.uint64) ** jnp.arange(
        L - 1, -1, -1, dtype=jnp.uint64
    )

    valid = jnp.zeros(seq.shape[:-1] + (T - L + 1,), dtype=jnp.uint64)
    for i in range(L):
        valid = valid + seq_u[..., i : T - L + 1 + i] * weights[i]

    pad = jnp.zeros(seq.shape[:-1] + (L - 1,), dtype=jnp.uint64)
    return jnp.concatenate([pad, valid], axis=-1)


def block_keys_base(seq, L: int, sigma: int, *, validate_symbols: bool = True):
    """Encode every length-L block ending at each position.

    Positions before ``L-1`` are padded with zero and must be treated as
    invalid by lookup code.
    """
    sigma_i = require_sigma(sigma)
    arr = require_symbol_array("seq", seq, sigma=sigma_i, validate_symbols=validate_symbols)
    if arr.ndim != 3:
        raise ValueError(f"seq must be rank-3 shaped [B, R, T]; got seq.shape={arr.shape}")
    B, R, T = (int(x) for x in arr.shape)
    if B <= 0 or R <= 0 or T <= 0:
        raise ValueError(f"seq must have non-empty [B, R, T] dimensions; got {arr.shape}")
    L_i = require_L_for_T(L, T)
    ensure_exact_key_safe(sigma=sigma_i, L=L_i, T=T)
    return _block_keys_base_jit(arr, L=L_i, sigma=sigma_i)


@partial(jax.jit, static_argnames=("L",))
def _lookup_one_l_from_keys_end_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int):
    B, R, T = q_keys.shape
    del B, R
    pos_u = jnp.arange(T, dtype=jnp.uint64)
    pos_i = jnp.arange(T, dtype=jnp.int32)
    stride = jnp.asarray(T + 1, dtype=jnp.uint64)

    def line_lookup(qk, kk, cap, succ, tcap):
        # Sort by a combined key so predecessor search returns the rightmost
        # historical end position with the same block key.
        combined = kk * stride + pos_u
        order = jnp.argsort(combined, stable=False)

        combined_s = combined[order]
        key_s = kk[order]
        pos_s_i = pos_i[order]

        cap_i = jnp.clip(cap, 0, T).astype(jnp.int32)
        cap_u = cap.astype(jnp.uint64)
        query_bound = qk * stride + cap_u

        idx = (jnp.searchsorted(combined_s, query_bound, side="left") - 1).astype(jnp.int32)
        idx_clip = jnp.clip(idx, 0, T - 1)

        end_i = pos_s_i[idx_clip]

        raw_hit = (
            (idx >= 0)
            & (key_s[idx_clip] == qk)
            & (end_i < cap_i)
            & (pos_i >= (L - 1))
            & (end_i >= (L - 1))
        )

        end_safe = jnp.clip(end_i, 0, T - 1)
        tau_raw = succ[end_safe]
        valid_tau = raw_hit & (tau_raw >= 0) & (tau_raw <= tcap)
        tau = jnp.where(valid_tau, tau_raw, NEG)
        end = jnp.where(raw_hit, end_i.astype(jnp.int64), jnp.int64(-1))
        return tau.astype(jnp.int64), valid_tau, end.astype(jnp.int64), raw_hit

    return jax.vmap(
        jax.vmap(line_lookup, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(q_keys, k_keys, cap_end, successor, tau_cap)


@partial(jax.jit, static_argnames=("L",))
def _lookup_one_l_from_keys_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int):
    tau, hit, _end, _raw_hit = _lookup_one_l_from_keys_end_jit(
        q_keys, k_keys, cap_end, successor, tau_cap, L=L
    )
    return tau, hit


@partial(jax.jit, static_argnames=("L",))
def _lookup_one_l_from_keys_mask_end_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int):
    """Correct lookup for arbitrary uint64 keys without combined-key packing.

    This is slower than the combined-key kernel, but it is safe for rolling-hash
    keys whose full uint64 range cannot be multiplied by ``T + 1``.
    """
    B, R, T = q_keys.shape
    del B, R
    pos_i = jnp.arange(T, dtype=jnp.int32)

    def line_lookup(qk, kk, cap, succ, tcap):
        def query_one(qkey, cap_t, tcap_t, t_pos):
            cap_i = jnp.clip(cap_t, 0, T).astype(jnp.int32)
            valid = (kk == qkey) & (pos_i < cap_i) & (pos_i >= (L - 1)) & (t_pos >= (L - 1))
            end_i = jnp.max(jnp.where(valid, pos_i, jnp.int32(-1)))
            raw_hit = end_i >= 0
            end_safe = jnp.clip(end_i, 0, T - 1)
            tau_raw = succ[end_safe]
            valid_tau = raw_hit & (tau_raw >= 0) & (tau_raw <= tcap_t)
            tau = jnp.where(valid_tau, tau_raw, NEG)
            return tau.astype(jnp.int64), valid_tau, end_i.astype(jnp.int64), raw_hit

        return jax.vmap(query_one)(qk, cap, tcap, pos_i)

    return jax.vmap(
        jax.vmap(line_lookup, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(q_keys, k_keys, cap_end, successor, tau_cap)


@partial(jax.jit, static_argnames=("L",))
def _lookup_one_l_from_keys_mask_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int):
    tau, hit, _end, _raw_hit = _lookup_one_l_from_keys_mask_end_jit(
        q_keys, k_keys, cap_end, successor, tau_cap, L=L
    )
    return tau, hit


@partial(jax.jit, static_argnames=("L", "num_buckets"))
def _lookup_one_l_from_keys_hash_end_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int, num_buckets: int):
    """O(T) hash-table lookup for one block length L.

    Uses open addressing with a single probe (no collision resolution).  On
    collision the probe returns a false negative; the caller should ensure
    ``num_buckets`` is large enough (preferably a prime > 4·T) to make
    collisions rare.

    This kernel is designed for rolling-hash keys that cannot be safely packed
    into a single uint64 combined key.

    .. warning::
       Because keys share a single flat table, different keys that hash to
       the same bucket cause false negatives.  For exact (non-probabilistic)
       lookup use the mask-based or sort-based kernels instead.
    """
    B, R, T = q_keys.shape
    del B, R
    pos_u = jnp.arange(T, dtype=jnp.uint64)
    stride = jnp.asarray(T + 1, dtype=jnp.uint64)
    pos_i = jnp.arange(T, dtype=jnp.int32)
    nb_u = jnp.asarray(num_buckets, dtype=jnp.uint64)

    def line_lookup(qk, kk, cap, succ, tcap):
        # ---- insert K keys into hash table ----
        combined = kk * stride + pos_u + jnp.uint64(1)  # +1 so 0 = empty sentinel
        buckets = (kk % nb_u).astype(jnp.int32)
        table = jnp.zeros((num_buckets,), dtype=jnp.uint64)
        table = table.at[buckets].max(combined)

        # ---- probe Q keys ----
        query_buckets = (qk % nb_u).astype(jnp.int32)
        probe = table[query_buckets]  # [T]

        nonempty = probe > jnp.uint64(0)
        stored_key = (probe - jnp.uint64(1)) // stride
        stored_pos = ((probe - jnp.uint64(1)) % stride).astype(jnp.int32)

        cap_i = jnp.clip(cap, 0, T).astype(jnp.int32)

        raw_hit = (
            nonempty
            & (stored_key == qk)
            & (stored_pos < cap_i)
            & (pos_i >= (L - 1))
            & (stored_pos >= (L - 1))
        )
        end_safe = jnp.clip(stored_pos, 0, T - 1)
        tau_raw = succ[end_safe]
        valid_tau = raw_hit & (tau_raw >= 0) & (tau_raw <= tcap)
        tau = jnp.where(valid_tau, tau_raw, NEG)
        end = jnp.where(raw_hit, stored_pos.astype(jnp.int64), jnp.int64(-1))
        return tau.astype(jnp.int64), valid_tau, end, raw_hit

    return jax.vmap(
        jax.vmap(line_lookup, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(q_keys, k_keys, cap_end, successor, tau_cap)


@partial(jax.jit, static_argnames=("L", "num_buckets"))
def _lookup_one_l_from_keys_hash_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int, num_buckets: int):
    tau, hit, _end, _raw_hit = _lookup_one_l_from_keys_hash_end_jit(
        q_keys, k_keys, cap_end, successor, tau_cap, L=L, num_buckets=num_buckets
    )
    return tau, hit


def lookup_one_l_from_keys(q_keys, k_keys, cap_end, successor, L: int, *, tau_cap=None):
    """Lookup one block length from precomputed Q and K keys.

    ``tau_cap`` is a post-successor cap.  When supplied, the rightmost raw
    suffix match is selected first, then ``successor[end]`` is accepted only if
    it is non-negative and ``<= tau_cap[b,r,t]``.  This reproduces the official
    ROSA rule that does not backtrack to an older match when the rightmost match
    has no valid successor yet.
    """
    q_arr, k_arr, _B, _R, T = require_key_array_pair(q_keys, k_keys)
    L_i = require_L_for_T(L, T)
    ensure_precomputed_keys_combined_safe(q_arr, k_arr, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(q_arr.shape))
    return _lookup_one_l_from_keys_jit(q_arr, k_arr, cap, succ, tcap, L=L_i)


@partial(jax.jit, static_argnames=("L", "sigma"))
def _lookup_one_l_base_jit(Q, K, cap_end, successor, tau_cap, L: int, sigma: int):
    q_keys = _block_keys_base_jit(Q, L=L, sigma=sigma)
    k_keys = _block_keys_base_jit(K, L=L, sigma=sigma)
    return _lookup_one_l_from_keys_jit(q_keys, k_keys, cap_end, successor, tau_cap, L=L)


def lookup_one_l_base(
    Q,
    K,
    cap_end,
    successor,
    L: int,
    sigma: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Exact base-encoded lookup for one block length L."""
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=sigma_i, validate_symbols=validate_symbols)
    L_i = require_L_for_T(L, T)
    ensure_exact_key_safe(sigma=sigma_i, L=L_i, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_one_l_base_jit(Q_arr, K_arr, cap, succ, tcap, L=L_i, sigma=sigma_i)


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _lookup_full_l_base_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int):
    B, R, T = Q.shape

    # Precompute block keys for each L once.  This Python loop is intentionally
    # unrolled at trace time because _block_keys_base_jit requires static L for
    # output shape.  For Lmax <= 8 this is the best trade-off.
    q_keys_by_L = []
    k_keys_by_L = []
    for L in range(1, Lmax + 1):
        q_keys_by_L.append(_block_keys_base_jit(Q, L=L, sigma=sigma))
        k_keys_by_L.append(_block_keys_base_jit(K, L=L, sigma=sigma))

    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for idx, L in enumerate(range(1, Lmax + 1)):
        q_keys = q_keys_by_L[idx]
        k_keys = k_keys_by_L[idx]
        _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_from_keys_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L
        )
        # Store the raw longest/rightmost match.  Do not filter by successor
        # validity here, otherwise a longer but successor-invalid official ROSA
        # match could incorrectly fall back to a shorter valid match.
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_base(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    sigma: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Find longest suffix match for all lengths ``1..Lmax``.

    Longer raw matches overwrite shorter raw matches.  Equal-length ties are
    resolved by rightmost predecessor search.  The selected raw match is then
    mapped through ``successor`` and optionally filtered by ``tau_cap``.
    """
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=sigma_i, validate_symbols=validate_symbols)
    Lmax_i = require_Lmax_for_T(Lmax, T)
    for L in range(1, Lmax_i + 1):
        ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_base_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i)
