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

from rosa_gpu_jax.block_table import _block_keys_base_jit, _lookup_one_l_from_keys_end_jit
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
        best_j_safe = jnp.clip(best_j, 0, T - 1)
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
        return tau.astype(jnp.int64), best_len.astype(jnp.int32)

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
        tau_L, len_L = _verify_postings_jit(
            Q, K, cand_L, cap_end, successor, tau_cap, Lmax=Lmax
        )
        raw_hit_L = len_L > 0

        # For accumulating raw best: we need the raw end position, not tau.
        # Extract best_j from cand_L using argmax logic mirroring the verifier.
        # Simpler: use _lookup_one_l_from_keys_end_jit to get the single
        # rightmost raw match for this L, then accumulate.
        _tau_single, _valid, end_single, raw_hit_single = (
            _lookup_one_l_from_keys_end_jit(
                q_keys, k_keys, cap_end, successor, tau_cap, L=L
            )
        )
        best_end = jnp.where(raw_hit_single, end_single, best_end)
        best_L_raw = jnp.where(raw_hit_single, jnp.int32(L), best_L_raw)

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

        # Use mask-based lookup for the single-best raw match (exact for
        # uint64 keys).  The postings provide candidates; we fall back to
        # mask for the raw accumulator.
        from rosa_gpu_jax.block_table import _lookup_one_l_from_keys_mask_end_jit

        _tau_single, _valid, end_single, raw_hit_single = (
            _lookup_one_l_from_keys_mask_end_jit(
                q_keys, k_keys, cap_end, successor, tau_cap, L=L
            )
        )
        best_end = jnp.where(raw_hit_single, end_single, best_end)
        best_L_raw = jnp.where(raw_hit_single, jnp.int32(L), best_L_raw)

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
