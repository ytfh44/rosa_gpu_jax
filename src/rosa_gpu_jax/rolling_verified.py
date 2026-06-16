"""Verified rolling-hash lookup with multi-slot hash table and tuple verification.

This module extends the probabilistic rolling-hash path (``rolling_hash.py``)
by collecting *C* candidates per hash bucket instead of only the single
rightmost, then verifying them symbol-by-symbol.  With a sufficiently large
``C``, the result is exact up to the rolling-hash collision rate; every
returned match is verified against raw K symbols so hash collisions cannot
produce false positives (only false negatives from bucket collisions).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.rolling_hash import _rolling_block_keys_u64_jit, _default_num_buckets
from rosa_gpu_jax.validation import (
    require_aux,
    require_base,
    require_Lmax_for_T,
    require_rank3_pair,
    require_tau_cap,
)


# ---------------------------------------------------------------------------
# Single-line multi-slot hash table builders / probes.
# These operate on flat ``[T]`` arrays and are vmapped over (B,R) by the
# full-L JIT function.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("L", "C", "num_buckets"))
def _build_rolling_table_line(k_keys_line, L: int, C: int, num_buckets: int):
    """Build ``[num_buckets, C]`` table from one K line ``[T]``.

    Each position inserts its ``(combined = key * (T+1) + pos + 1)`` into
    a pseudo-random slot within its bucket.  Conflicts are resolved by
    ``.max`` so the rightmost entry per slot is retained.
    """
    T = k_keys_line.shape[-1]
    pos_u = jnp.arange(T, dtype=jnp.uint64)
    stride = jnp.asarray(T + 1, dtype=jnp.uint64)
    # +1 so 0 = empty sentinel (matching the single-slot hash-table convention).
    combined = k_keys_line * stride + pos_u + jnp.uint64(1)

    nb_u = jnp.asarray(num_buckets, dtype=jnp.uint64)
    buckets = (k_keys_line % nb_u).astype(jnp.int32)

    # Slot = pos % C  — deterministic, evenly distributes positions within
    # a bucket so that the C rightmost entries are retained across slots.
    slot_offsets = (pos_u % jnp.asarray(C, dtype=jnp.uint64)).astype(jnp.int32)

    table = jnp.zeros((num_buckets, C), dtype=jnp.uint64)
    table = table.at[buckets, slot_offsets].max(combined)
    return table


@partial(jax.jit, static_argnames=("L", "C", "num_buckets"))
def _probe_rolling_table_line(q_keys_line, table, C: int, num_buckets: int, L: int):
    """Probe one ``[num_buckets, C]`` table for ``[T]`` query keys.

    Returns ``cand`` of shape ``[T, C]`` int32 (``-1`` for empty slots).
    """
    T = q_keys_line.shape[-1]
    stride = jnp.asarray(T + 1, dtype=jnp.uint64)
    nb_u = jnp.asarray(num_buckets, dtype=jnp.uint64)

    def _query_one(qk_val, t_pos):
        bucket = (qk_val % nb_u).astype(jnp.int32)
        slots = table[bucket]  # [C] uint64

        nonempty = slots > jnp.uint64(0)
        stored_key = (slots - jnp.uint64(1)) // stride
        stored_pos = ((slots - jnp.uint64(1)) % stride).astype(jnp.int32)

        valid = (
            nonempty
            & (stored_key == qk_val)
            & (stored_pos >= (L - 1))
            & (t_pos >= (L - 1))
        )
        return jnp.where(valid, stored_pos, jnp.int32(-1))

    return jax.vmap(_query_one)(q_keys_line, jnp.arange(T, dtype=jnp.int32))


# ---------------------------------------------------------------------------
# Inline single-line candidate verifier (mirrors candidates.py logic).
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax",))
def _verify_candidates_line(q_line, k_line, cand_line, cap_line, succ_line, tcap_line, Lmax: int):
    """Verify ``[T,C]`` candidates, select best (length, rightmost), ROSA gate.

    Returns ``(tau[T], best_len[T])`` — ``int64`` and ``int32``.
    """
    T = q_line.shape[-1]
    C = cand_line.shape[-1]
    offsets = jnp.arange(Lmax, dtype=jnp.int32)

    t_idx = jnp.arange(T, dtype=jnp.int32)[:, None, None]  # [T,1,1]
    j_idx = cand_line.astype(jnp.int32)[:, :, None]  # [T,C,1]
    cap_t = cap_line[:, None, None]
    o = offsets[None, None, :]  # [1,1,Lmax]

    q_idx = t_idx - o
    k_idx = j_idx - o

    valid = (
        (cand_line[:, :, None] >= 0)
        & (q_idx >= 0)
        & (k_idx >= 0)
        & (j_idx < cap_t)
    )

    q_tok = q_line[jnp.clip(q_idx, 0, T - 1)]
    k_tok = k_line[jnp.clip(k_idx, 0, T - 1)]

    eq = valid & (q_tok == k_tok)  # [T,C,Lmax]

    all_match = jnp.all(eq, axis=-1)  # [T,C]
    first_false = jnp.argmin(eq.astype(jnp.int32), axis=-1).astype(jnp.int32)
    lens = jnp.where(all_match, jnp.int32(Lmax), first_false)  # [T,C]

    j_nonneg = jnp.maximum(cand_line.astype(jnp.int32), 0)
    score = (
        lens.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64)
        + j_nonneg
    )

    best_c = jnp.argmax(score, axis=-1).astype(jnp.int32)
    best_j = jnp.take_along_axis(cand_line, best_c[:, None], axis=-1)[:, 0].astype(jnp.int32)
    best_len_raw = jnp.take_along_axis(lens, best_c[:, None], axis=-1)[:, 0]
    best_j_safe = jnp.clip(best_j, 0, T - 1)

    tau_raw = succ_line[best_j_safe]
    valid_tau = (best_len_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    tau = jnp.where(valid_tau, tau_raw, NEG)
    best_len = jnp.where(valid_tau, best_len_raw, jnp.int32(0))
    return tau.astype(jnp.int64), best_len.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Full-L verified rolling-hash lookup (vmap over B,R, loop over L).
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "base", "C", "num_buckets"))
def _lookup_full_l_rolling_verified_line(
    q_line, k_line, cap_line, succ_line, tcap_line, Lmax: int, base: int, C: int, num_buckets: int
):
    """Full-L verified lookup for a single ``[T]`` line."""
    T = q_line.shape[-1]
    best_end = jnp.full((T,), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((T,), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        q_keys = _rolling_block_keys_u64_jit(
            q_line[None, None, :], L=L, base=base
        )[0, 0]  # [T]
        k_keys = _rolling_block_keys_u64_jit(
            k_line[None, None, :], L=L, base=base
        )[0, 0]  # [T]

        # Build table from K keys, probe for Q keys.
        table = _build_rolling_table_line(k_keys, L=L, C=C, num_buckets=num_buckets)
        cand_L = _probe_rolling_table_line(
            q_keys, table, C=C, num_buckets=num_buckets, L=L
        )  # [T, C]

        # Verify candidates and extract raw best for this L.
        tau_L, len_L = _verify_candidates_line(
            q_line, k_line, cand_L, cap_line, succ_line, tcap_line, Lmax=Lmax
        )
        raw_hit_L = len_L > 0

        # For the raw accumulation we need the end position.  Derive it from
        # the candidate that gave the best match.
        # Reconstruct score to find best candidate index, then extract end.
        offsets = jnp.arange(Lmax, dtype=jnp.int32)
        t_idx = jnp.arange(T, dtype=jnp.int32)[:, None, None]  # [T,1,1]
        j_idx = cand_L.astype(jnp.int32)[:, :, None]  # [T,C,1]
        cap_t = cap_line[:, None, None]
        o = offsets[None, None, :]

        q_idx = t_idx - o
        k_idx = j_idx - o
        valid = (
            (cand_L[:, :, None] >= 0)
            & (q_idx >= 0)
            & (k_idx >= 0)
            & (j_idx < cap_t)
        )
        q_tok = q_line[jnp.clip(q_idx, 0, T - 1)]
        k_tok = k_line[jnp.clip(k_idx, 0, T - 1)]
        eq = valid & (q_tok == k_tok)
        all_match = jnp.all(eq, axis=-1)
        first_false = jnp.argmin(eq.astype(jnp.int32), axis=-1).astype(jnp.int32)
        lens = jnp.where(all_match, jnp.int32(Lmax), first_false)
        j_nonneg = jnp.maximum(cand_L.astype(jnp.int32), 0)
        score = lens.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64) + j_nonneg
        best_c_idx = jnp.argmax(score, axis=-1).astype(jnp.int32)
        end_L = jnp.take_along_axis(cand_L, best_c_idx[:, None], axis=-1)[:, 0].astype(jnp.int64)

        # Use the actual verified length (len_L) rather than the block-key
        # length L, since the verifier may have confirmed a longer match.
        better = raw_hit_L & (
            (len_L > best_L_raw)
            | ((len_L == best_L_raw) & (end_L > best_end))
        )
        best_end = jnp.where(better, end_L, best_end)
        best_L_raw = jnp.where(better, len_L, best_L_raw)

    # ROSA gating on accumulated best.
    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = succ_line[end_safe]
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


# ---------------------------------------------------------------------------
# Batched entry point.
# ---------------------------------------------------------------------------


def lookup_full_l_rolling_verified(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    base: int,
    C: int = 8,
    *,
    tau_cap=None,
    num_buckets: int | None = None,
):
    """Verified rolling-hash lookup with multi-slot hash table.

    Unlike ``lookup_full_l_rolling``, this function retains up to *C*
    candidates per hash bucket and verifies them symbol-by-symbol.  Every
    returned match is verified against raw K symbols — there are no false
    positives from hash collisions.  False negatives are still possible
    from bucket collisions when ``C`` is too small.

    Parameters
    ----------
    C:
        Candidates retained per hash bucket.  Larger values reduce false
        negatives.
    num_buckets:
        Number of hash buckets.  Defaults to a prime > 4·T.
    """
    base_i = require_base(base)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=None, validate_symbols=False
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    C_i = max(1, int(C))
    nb = num_buckets if num_buckets is not None else _default_num_buckets(T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))

    # vmap over (B,R): each call handles one [T] line.
    def _single_line(q, k, ce, s, tc):
        return _lookup_full_l_rolling_verified_line(
            q, k, ce, s, tc, Lmax=Lmax_i, base=base_i, C=C_i, num_buckets=nb
        )

    tau, match_len = jax.vmap(
        jax.vmap(_single_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q_arr, K_arr, cap, succ, tcap)
    return tau, match_len
