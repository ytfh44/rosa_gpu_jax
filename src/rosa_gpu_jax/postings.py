"""Fixed-C postings candidate generator for ROSA suffix lookup.

This module extends the exact block-table approach (``block_table.py``)
by collecting the *C* rightmost matching positions per query key instead
of only the single rightmost.  The resulting candidates are then verified
symbol-by-symbol and gated through the standard ROSA successor/tau_cap
pipeline.

Two public entry points are provided:

* ``lookup_full_l_base_postings`` — exact base-encoded keys, precise
  within uint64 safety bounds.
* ``lookup_full_l_rolling_postings`` — rolling-hash keys, larger
  ``Lmax`` / ``sigma``, probabilistic unless combined with full tuple
  verification.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.block_table import _block_keys_base_jit
from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.rolling_hash import _rolling_block_keys_u64_jit
from rosa_gpu_jax.validation import (
    ensure_exact_key_safe,
    require_aux,
    require_base,
    require_Lmax_for_T,
    require_rank3_pair,
    require_sigma,
    require_tau_cap,
)

# ---------------------------------------------------------------------------
# Single-L postings helper: collect C rightmost candidates per query.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("L", "C"))
def _postings_one_l_from_keys_end_jit(q_keys, k_keys, cap_end, successor, tau_cap, L: int, C: int):
    """Return ``(cand_end, raw_hit_mask)`` for one block length L.

    ``cand_end`` is ``int32[B, R, T, C]`` with the C rightmost matching
    K end positions, or ``-1`` for empty slots.  ``raw_hit_mask`` is
    ``bool[B, R, T]`` — True when *any* of the C slots is a valid raw
    match (same key, within cap, valid block boundaries).
    """
    B, R, T = q_keys.shape
    del B, R
    pos_u = jnp.arange(T, dtype=jnp.uint64)
    pos_i = jnp.arange(T, dtype=jnp.int32)
    stride = jnp.asarray(T + 1, dtype=jnp.uint64)
    c_offsets = jnp.arange(C, dtype=jnp.int32)

    def line_postings(qk, kk, cap, succ, tcap):
        # Sort K keys for predecessor search.
        combined = kk * stride + pos_u
        order = jnp.argsort(combined, stable=False)

        combined_s = combined[order]
        key_s = kk[order]
        pos_s_i = pos_i[order]

        cap_i = jnp.clip(cap, 0, T).astype(jnp.int32)
        cap_u = cap.astype(jnp.uint64)
        query_bound = qk * stride + cap_u  # [T]

        def query_one(qk_val, bound, cap_t):
            # Rightmost predecessor index (existing logic).
            idx = (jnp.searchsorted(combined_s, bound, side="left") - 1).astype(
                jnp.int32
            )
            idx_clip = jnp.clip(idx, 0, T - 1)

            # Collect C predecessors: idx_clip, idx_clip-1, ..., idx_clip-(C-1).
            idx_c = idx_clip - c_offsets  # [C]
            idx_c_clip = jnp.clip(idx_c, 0, T - 1)

            valid = (
                (idx_c >= 0)
                & (key_s[idx_c_clip] == qk_val)
            )
            return jnp.where(valid, pos_s_i[idx_c_clip], jnp.int32(-1))  # [C]

        cand_end = jax.vmap(query_one)(qk, query_bound, cap_i)  # [T, C]

        # Build per-query raw-hit mask: the first candidate (c=0) is the
        # rightmost and determines whether *any* valid match exists.
        best_j = cand_end[:, 0]  # [T] — rightmost candidate
        pos_t = jnp.arange(T, dtype=jnp.int32)
        raw_hit = (
            (best_j >= 0)
            & (best_j < cap_i)
            & (pos_t >= (L - 1))
            & (best_j >= (L - 1))
        )
        return cand_end.astype(jnp.int32), raw_hit  # raw_hit not used by caller

    cand, _raw = jax.vmap(
        jax.vmap(line_postings, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(q_keys, k_keys, cap_end, successor, tau_cap)
    return cand  # [B, R, T, C] int32


# ---------------------------------------------------------------------------
# Inline candidate verification (reuses the score/gating logic from
# ``candidates.py`` without requiring a separate module call).
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax",))
def _verify_postings_jit(Q, K, cand_end, cap_end, successor, tau_cap, Lmax: int):
    """Verify postings candidates symbol-by-symbol, select best, and gate.

    Returns
    -------
    tau : int64[B,R,T]
        ROSA-gated successor, or ``NEG`` when no valid match exists.
    best_len : int32[B,R,T]
        Verified match length after gating (0 when gated out).
    best_j_raw : int32[B,R,T]
        Raw best K end position *before* successor/tau_cap gating,
        or -1 when no raw candidate passes verification.
    best_len_raw : int32[B,R,T]
        Verified match length *before* gating.

    This is a slightly adapted copy of ``_verify_cpu_candidates_jit``
    so that the postings pipeline stays self-contained and avoids extra
    vmap nesting.
    """
    B, R, T = Q.shape
    del B, R
    offsets = jnp.arange(Lmax, dtype=jnp.int32)

    def line_verify(q, k, cand, cap, succ, tcap):
        t_idx = jnp.arange(T, dtype=jnp.int32)[:, None, None]  # [T,1,1]
        j_idx = cand.astype(jnp.int32)[:, :, None]  # [T,C,1]
        cap_t = cap[:, None, None]  # [T,1,1]
        o = offsets[None, None, :]  # [1,1,Lmax]

        q_idx = t_idx - o  # [T,1,Lmax]
        k_idx = j_idx - o  # [T,C,Lmax]

        valid = (
            (cand[:, :, None] >= 0)
            & (q_idx >= 0)
            & (k_idx >= 0)
            & (j_idx < cap_t)
        )

        q_tok = q[jnp.clip(q_idx, 0, T - 1)]
        k_tok = k[jnp.clip(k_idx, 0, T - 1)]

        eq = valid & (q_tok == k_tok)  # [T,C,Lmax]

        all_match = jnp.all(eq, axis=-1)  # [T,C]
        first_false = jnp.argmin(eq.astype(jnp.int32), axis=-1).astype(jnp.int32)
        lens = jnp.where(all_match, jnp.int32(Lmax), first_false)  # [T,C]

        # Score: length first, then rightmost j.
        j_nonneg = jnp.maximum(cand.astype(jnp.int32), 0)
        score = (
            lens.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64)
            + j_nonneg
        )

        best_c = jnp.argmax(score, axis=-1).astype(jnp.int32)
        best_j = jnp.take_along_axis(cand, best_c[:, None], axis=-1)[:, 0].astype(
            jnp.int32
        )
        best_len_raw = jnp.take_along_axis(lens, best_c[:, None], axis=-1)[:, 0]
        best_j_safe = jnp.clip(best_j, 0, T - 1)

        tau_raw = succ[best_j_safe]
        valid_tau = (best_len_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap)
        tau = jnp.where(valid_tau, tau_raw, NEG)
        best_len = jnp.where(valid_tau, best_len_raw, jnp.int32(0))
        return (
            tau.astype(jnp.int64),
            best_len.astype(jnp.int32),
            best_j.astype(jnp.int32),
            best_len_raw.astype(jnp.int32),
        )

    return jax.vmap(
        jax.vmap(line_verify, in_axes=(0, 0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0, 0),
    )(Q, K, cand_end, cap_end, successor, tau_cap)


# ---------------------------------------------------------------------------
# Full-L postings lookup: exact base keys
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "sigma", "C"))
def _lookup_full_l_base_postings_jit(
    Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int, C: int
):
    B, R, T = Q.shape

    # Accumulate best raw match across L.
    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        q_keys = _block_keys_base_jit(Q, L=L, sigma=sigma)
        k_keys = _block_keys_base_jit(K, L=L, sigma=sigma)
        cand_L = _postings_one_l_from_keys_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L, C=C
        )  # [B,R,T,C] int32

        # Verify with the inline verifier (uses Lmax, not L — the verifier
        # checks symbols to the full Lmax depth regardless of which L
        # produced the candidates).
        _tau_L, _len_L, end_raw_L, len_raw_L = _verify_postings_jit(
            Q, K, cand_L, cap_end, successor, tau_cap, Lmax=Lmax
        )
        # Guard against padded (zero) block keys for positions t < L-1:
        # a raw match is only valid when both the query and key positions
        # are at least L-1 (matching the boundary check in
        # _lookup_one_l_from_keys_end_jit).
        pos_t = jnp.arange(T, dtype=jnp.int32)
        raw_hit_L = (
            (len_raw_L > 0)
            & (pos_t[None, None, :] >= (L - 1))
            & (end_raw_L >= (L - 1))
        )

        # Accumulate raw best across L: the verifier already selected the
        # best (longest, rightmost) candidate and verified it
        # symbol-by-symbol.  Use its raw end position and verified match
        # length before gating — ROSA gating is applied once at the end,
        # matching the pattern in _lookup_full_l_base_jit.
        better = raw_hit_L & (
            (len_raw_L > best_L_raw)
            | ((len_raw_L == best_L_raw) & (end_raw_L > best_end))
        )
        best_end = jnp.where(
            better, end_raw_L.astype(jnp.int64), best_end
        )
        best_L_raw = jnp.where(better, len_raw_L, best_L_raw)

    # ROSA gating on the accumulated best raw match.
    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_base_postings(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    sigma: int,
    C: int = 8,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Full-L exact base-encoded postings lookup.

    Builds per-key postings of the C rightmost positions and verifies
    candidates symbol-by-symbol.  Equivalent to ``lookup_full_l_base``
    when ``C`` is large enough to capture all same-key positions;
    trades a small risk of candidate-recall loss for more regular GPU
    memory patterns at large ``T``.

    Parameters
    ----------
    C:
        Number of candidates to retain per unique block key.  Larger
        values reduce recall loss.  ``C >= T`` guarantees exactness
        (but defeats the purpose of postings).
    """
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=sigma_i, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    for L in range(1, Lmax_i + 1):
        ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    C_i = max(1, int(C))
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_base_postings_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i, C=C_i
    )


# ---------------------------------------------------------------------------
# Full-L postings lookup: rolling-hash keys
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "base", "C"))
def _lookup_full_l_rolling_postings_jit(
    Q, K, cap_end, successor, tau_cap, Lmax: int, base: int, C: int
):
    B, R, T = Q.shape

    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        q_keys = _rolling_block_keys_u64_jit(Q, L=L, base=base)
        k_keys = _rolling_block_keys_u64_jit(K, L=L, base=base)
        cand_L = _postings_one_l_from_keys_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L, C=C
        )

        # Verify the C candidates symbol-by-symbol against raw K symbols.
        # This is the key difference from the earlier rolling-hash path:
        # hash collisions cannot produce false positives — every returned
        # match is validated against the actual token stream.
        _tau_L, _len_L, end_raw_L, len_raw_L = _verify_postings_jit(
            Q, K, cand_L, cap_end, successor, tau_cap, Lmax=Lmax
        )
        pos_t = jnp.arange(T, dtype=jnp.int32)
        raw_hit_L = (
            (len_raw_L > 0)
            & (pos_t[None, None, :] >= (L - 1))
            & (end_raw_L >= (L - 1))
        )

        better = raw_hit_L & (
            (len_raw_L > best_L_raw)
            | ((len_raw_L == best_L_raw) & (end_raw_L > best_end))
        )
        best_end = jnp.where(
            better, end_raw_L.astype(jnp.int64), best_end
        )
        best_L_raw = jnp.where(better, len_raw_L, best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_rolling_postings(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    base: int,
    C: int = 8,
    *,
    tau_cap=None,
):
    """Full-L rolling-hash postings lookup.

    Probabilistic (hash collisions) unless ``C`` is large and verification
    is extended to raw-tuple comparison.  Suitable for throughput studies
    where exact block-table ``sigma`` / ``Lmax`` constraints are violated.

    Parameters
    ----------
    C:
        Number of postings candidates per unique hash key.
    """
    base_i = require_base(base)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=None, validate_symbols=False
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    C_i = max(1, int(C))
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_rolling_postings_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, base=base_i, C=C_i
    )


# ---------------------------------------------------------------------------
# Dyadic-Rank Postings + binary-lifting LCE lookup
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "C"))
def _lookup_drp_lce_line_jit(q, k, cap_end, successor, tau_cap, *, Lmax: int, C: int):
    """Dyadic-Rank Postings + exact LCE verifier for one [T] line.

    Builds dyadic (power-of-2) joint ranks over Q and K, collects *C*
    candidate positions per dyadic level via predecessor search, then
    verifies via binary lifting over the same dyadic ranks.

    This is exact — no false positives.  The only source of false
    negatives is ``C`` being smaller than the number of same-rank
    positions at some level.

    Parameters
    ----------
    q, k : int64[T]
        Raw symbol arrays.
    cap_end, successor, tau_cap : int64[T]
        ROSA auxiliary tensors.

    Returns
    -------
    tau : int64[T]
    match_len : int32[T]
    """
    T = q.shape[0]
    max_level = int(Lmax).bit_length() - 1
    rank_stride = jnp.asarray(2 * T + 2, dtype=jnp.int64)

    def _rank_base(q_line, k_line):
        q_key = q_line.astype(jnp.int64) + 1
        k_key = k_line.astype(jnp.int64) + 1
        both = jnp.concatenate([q_key, k_key], axis=0)
        sorted_both = jnp.sort(both)

        q_rank = (jnp.searchsorted(sorted_both, q_key, side="left") + 1).astype(jnp.int64)
        k_rank = (jnp.searchsorted(sorted_both, k_key, side="left") + 1).astype(jnp.int64)
        return q_rank, k_rank

    q_ids = []
    k_ids = []

    q0, k0 = _rank_base(q, k)
    q_ids.append(q0)
    k_ids.append(k0)

    prev_q = q0
    prev_k = k0
    prev_len = 1

    # Build dyadic ranks: length 1, 2, 4, ...
    for _level in range(1, max_level + 1):
        block_len = prev_len * 2
        idx = jnp.arange(T, dtype=jnp.int32)
        left_idx = idx - prev_len
        valid = idx >= (block_len - 1)

        q_left = prev_q[jnp.clip(left_idx, 0, T - 1)]
        k_left = prev_k[jnp.clip(left_idx, 0, T - 1)]

        q_pair_key = jnp.where(
            valid,
            q_left * rank_stride + prev_q + 1,
            jnp.int64(0),
        )
        k_pair_key = jnp.where(
            valid,
            k_left * rank_stride + prev_k + 1,
            jnp.int64(0),
        )

        both = jnp.concatenate([q_pair_key, k_pair_key], axis=0)
        sorted_both = jnp.sort(both)

        q_rank = jnp.where(
            q_pair_key > 0,
            (jnp.searchsorted(sorted_both, q_pair_key, side="left") + 1).astype(jnp.int64),
            jnp.int64(0),
        )
        k_rank = jnp.where(
            k_pair_key > 0,
            (jnp.searchsorted(sorted_both, k_pair_key, side="left") + 1).astype(jnp.int64),
            jnp.int64(0),
        )

        q_ids.append(q_rank)
        k_ids.append(k_rank)
        prev_q = q_rank
        prev_k = k_rank
        prev_len = block_len

    def _candidates_for_level(qid, kid, anchor_len: int):
        pos = jnp.arange(T, dtype=jnp.int64)
        pos_stride = jnp.asarray(T + 1, dtype=jnp.int64)

        combined = kid * pos_stride + pos
        order = jnp.argsort(combined, stable=False)

        combined_s = combined[order]
        key_s = kid[order]
        pos_s = pos[order]

        cap = jnp.clip(cap_end, 0, T).astype(jnp.int64)
        bound = qid * pos_stride + cap

        base_idx = (jnp.searchsorted(combined_s, bound, side="left") - 1).astype(jnp.int64)
        offsets = jnp.arange(C, dtype=jnp.int64)
        idxs = base_idx[:, None] - offsets[None, :]
        idxs_clip = jnp.clip(idxs, 0, T - 1)

        cand_pos = pos_s[idxs_clip]
        ok = (
            (idxs >= 0)
            & (qid[:, None] > 0)
            & (key_s[idxs_clip] == qid[:, None])
            & (cand_pos < cap[:, None])
            & (cand_pos >= anchor_len - 1)
        )

        return jnp.where(ok, cand_pos, jnp.int64(-1))

    # Collect candidate positions from every dyadic level.
    cand_blocks = []
    for level in range(max_level + 1):
        anchor_len = 1 << level
        cand_blocks.append(_candidates_for_level(q_ids[level], k_ids[level], anchor_len))

    cand = jnp.concatenate(cand_blocks, axis=1)  # [T, (max_level+1) * C]
    NC = cand.shape[1]

    # Exact LCE via binary lifting on dyadic ranks.
    t0 = jnp.arange(T, dtype=jnp.int64)[:, None]
    j0 = cand.astype(jnp.int64)
    cap_t = jnp.clip(cap_end, 0, T).astype(jnp.int64)[:, None]

    active = (j0 >= 0) & (j0 < cap_t)

    tq = jnp.broadcast_to(t0, (T, NC))
    jk = j0
    length = jnp.zeros((T, NC), dtype=jnp.int32)

    for level in range(max_level, -1, -1):
        step_len = 1 << level

        qid = q_ids[level]
        kid = k_ids[level]

        tq_clip = jnp.clip(tq, 0, T - 1).astype(jnp.int32)
        jk_clip = jnp.clip(jk, 0, T - 1).astype(jnp.int32)

        same = qid[tq_clip] == kid[jk_clip]
        can_take = (
            active
            & (tq >= step_len - 1)
            & (jk >= step_len - 1)
            & ((length.astype(jnp.int64) + step_len) <= Lmax)
            & same
        )

        length = length + jnp.where(can_take, jnp.int32(step_len), jnp.int32(0))
        delta = jnp.where(can_take, jnp.int64(step_len), jnp.int64(0))

        tq = tq - delta
        jk = jk - delta

    raw_len = jnp.where(active, length, jnp.int32(0))

    # Select longest, then rightmost raw match.
    j_nonneg = jnp.maximum(j0, 0)
    score = raw_len.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64) + j_nonneg
    score = jnp.where(raw_len > 0, score, NEG)

    best_idx = jnp.argmax(score, axis=1).astype(jnp.int32)
    best_score = jnp.take_along_axis(score, best_idx[:, None], axis=1)[:, 0]
    best_j = jnp.take_along_axis(cand, best_idx[:, None], axis=1)[:, 0].astype(jnp.int64)
    best_len_raw = jnp.take_along_axis(raw_len, best_idx[:, None], axis=1)[:, 0]

    raw_hit = best_score > NEG
    best_j_safe = jnp.clip(best_j, 0, T - 1).astype(jnp.int32)

    tau_raw = successor[best_j_safe]
    final_hit = raw_hit & (tau_raw >= 0) & (tau_raw <= tau_cap)

    tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    match_len = jnp.where(final_hit, best_len_raw, jnp.int32(0))

    return tau, match_len


@partial(jax.jit, static_argnames=("Lmax", "C"))
def _lookup_drp_lce_jit(Q, K, cap_end, successor, tau_cap, *, Lmax: int, C: int):
    """Batched [B,R,T] wrapper around ``_lookup_drp_lce_line_jit``."""

    def one_line(q, k, ce, succ, tc):
        return _lookup_drp_lce_line_jit(q, k, ce, succ, tc, Lmax=Lmax, C=C)

    return jax.vmap(
        jax.vmap(one_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q, K, cap_end, successor, tau_cap)


def lookup_full_l_drp_lce(
    Q,
    K,
    cap_end,
    successor,
    *,
    Lmax: int,
    C: int = 8,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Dyadic-Rank Postings + binary-lifting LCE suffix lookup.

    Builds a hierarchy of joint dyadic ranks (length 1, 2, 4, ... up to
    the largest power of two ≤ ``Lmax``), collects *C* candidate
    positions per dyadic level via predecessor search, then verifies
    match lengths via binary lifting on the rank hierarchy.

    Unlike ``lookup_full_l_base_postings`` this method does **not**
    require a ``sigma`` parameter and has no uint64 overflow constraint.
    Unlike ``lookup_full_l_rolling_postings`` it is exact — no hash
    collisions.  The only source of false negatives is ``C`` being
    smaller than the number of same-rank positions.

    Complexity is ``O(log Lmax · T log T + log Lmax · C · T)`` per line,
    compared to ``O(Lmax · T log T + Lmax · C · T)`` for the per-L
    postings path.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        ROSA auxiliary tensors (see :func:`make_rosa_causal_aux`).
    Lmax:
        Maximum suffix length to consider.  Must satisfy ``1 <= Lmax <= T``.
    C:
        Number of candidates retained per dyadic level.  Larger values
        reduce false negatives.  ``C >= T`` guarantees exactness.
    tau_cap:
        Optional post-successor cap for official ROSA/RLE semantics.
    validate_symbols:
        When ``True`` (default), validate that symbols are in range.

    Returns
    -------
    tau:
        ``int64[B, R, T]`` — ROSA tau (next-run start), or ``-1``.
    match_len:
        ``int32[B, R, T]`` — length of the accepted raw suffix match.
    """
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=None, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    C_i = max(1, int(C))
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_drp_lce_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, C=C_i
    )
