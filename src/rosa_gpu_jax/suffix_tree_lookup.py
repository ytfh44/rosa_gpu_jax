"""Exact ROSA suffix lookup via suffix-array binary search.

Builds a suffix array for K on CPU, then uses binary search on the GPU
to find the SA range for each query suffix.  Within that range the
rightmost valid position is selected.

This is a *ninth* lookup path that complements the existing eight.
It is exact for all L <= Lmax, has no sigma/Lmax overflow issue,
and replaces per‑L sorting with per‑L binary searches on a
pre‑sorted suffix array.

Complexity
----------
Build:  O(T²) worst‑case (SA via tuple sort), once per route on CPU.
Query:  O(B·R·Lmax·T·log T) — Lmax binary searches per position.

The binary‑search hot‑loop is unrolled at trace time (Lmax is static),
so XLA can fuse operations across L values.  For small Lmax (≤ 8)
this is competitive with the block‑table approach while avoiding its
uint64 overflow constraints.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    require_aux,
    require_Lmax_for_T,
    require_rank3_pair,
    require_tau_cap,
)

# ---------------------------------------------------------------------------
# CPU / NumPy: build suffix array for one K line
# ---------------------------------------------------------------------------


def _build_sa_one(k_line: np.ndarray) -> np.ndarray:
    """Build suffix array for one 1-D int64 sequence via tuple sort."""
    T = len(k_line)
    lst = k_line.tolist()
    sa = sorted(range(T), key=lambda i: tuple(lst[i:]))
    return np.asarray(sa, dtype=np.int32)


def suffix_array_batch(K: np.ndarray) -> np.ndarray:
    """Build suffix arrays for K ``[B, R, T]``.

    Returns int32 ``[B, R, T]`` — SA[b,r,i] is the starting position
    (in K) of the i‑th lexicographically smallest suffix.
    """
    K = np.asarray(K, dtype=np.int64)
    if K.ndim != 3:
        raise ValueError(f"K must be rank-3 [B,R,T]; got shape {K.shape}")
    B, R, T = K.shape
    out = np.zeros((B, R, T), dtype=np.int32)
    for b in range(B):
        for r in range(R):
            out[b, r] = _build_sa_one(K[b, r])
    return out


# ---------------------------------------------------------------------------
# JAX kernel: binary search in SA for pattern matching
# ---------------------------------------------------------------------------

# We need a helper that, given a sorted array of suffix *keys*, finds the
# leftmost and rightmost indices where the query key appears.
#
# The suffix key for position j and length L is the tuple K[j:j+L].
# In the suffix array, suffixes are sorted lexicographically, so the
# sub‑array SA[lo:hi+1] contains all positions whose length‑L prefix
# equals the query pattern.
#
# To locate lo/hi efficiently we compare the query pattern against the
# L‑length prefix of the suffix at SA[mid].  This is done via a helper
# ``_compare_suffix`` that returns -1/0/+1.


@partial(jax.jit, static_argnames=("L",))
def _sa_range_for_pattern(
    q_pattern,   # [L] — query suffix symbols (oldest…newest)
    K_line,      # [T] — the K symbol line
    SA_line,     # [T] — suffix array for this K line
    L: int,
):
    """Return (lo, hi) SA indices where the L‑length prefix equals q_pattern.

    If the pattern does not appear, lo > hi is returned.
    Uses binary search on the suffix array — O(log T) comparisons.
    """
    T = K_line.shape[-1]

    # Compare L‑length prefix of suffix at SA[mid] against q_pattern.
    # Returns -1 (suffix < query), 0 (equal), +1 (suffix > query).
    def _cmp_at(mid):
        pos = SA_line[mid]  # starting position of this suffix
        # We need to compare K[pos:pos+L] against q_pattern
        # Build indices for comparison
        offsets = jnp.arange(L, dtype=jnp.int32)
        k_idx = pos + offsets  # [L]
        k_idx_clip = jnp.clip(k_idx, 0, T - 1)
        k_syms = K_line[k_idx_clip]  # [L]

        # Pad: if pos + L > T, suffix is shorter, should compare as less
        valid = k_idx < T

        # Find first position where symbols differ
        neq = (k_syms != q_pattern) | (~valid)
        # neq encodes: True = difference found or out of bounds
        # The first True gives the decision point
        first_diff = jnp.argmax(neq.astype(jnp.int32))  # first differing index
        # If all equal and valid, first_diff will point to a position where
        # neq=False, but argmax returns first max; since all False, it returns 0.
        # We need to check if ANY neq is True.
        any_diff = jnp.any(neq)

        def _do_cmp():
            k_sym = k_syms[first_diff]
            q_sym = q_pattern[first_diff]
            k_valid = valid[first_diff]
            # If k is out of bounds, it's "shorter" → smaller
            # If both valid, compare symbols
            return jnp.where(
                k_valid,
                jnp.where(k_sym < q_sym, -1, jnp.where(k_sym > q_sym, 1, 0)),
                -1,  # K suffix is shorter → smaller
            )

        return jnp.where(any_diff, _do_cmp(), 0)

    # Binary search for lower bound (first >= query)
    def _lower_bound():
        lo = jnp.int32(0)
        hi = jnp.int32(T)
        # Unrolled binary search (log₂(T) ≤ 10 for T ≤ 1024)
        # We use a simple while-like approach via jax.lax.while_loop
        def _cond(state):
            lo_, hi_ = state
            return lo_ < hi_

        def _body(state):
            lo_, hi_ = state
            mid = (lo_ + hi_) // 2
            cmp = _cmp_at(mid)
            lo_new = jnp.where(cmp < 0, mid + 1, lo_)
            hi_new = jnp.where(cmp < 0, hi_, mid)
            return (lo_new, hi_new)

        lo_final, _ = jax.lax.while_loop(_cond, _body, (lo, hi))
        return lo_final

    # Binary search for upper bound (first > query)
    def _upper_bound():
        lo = jnp.int32(0)
        hi = jnp.int32(T)

        def _cond(state):
            lo_, hi_ = state
            return lo_ < hi_

        def _body(state):
            lo_, hi_ = state
            mid = (lo_ + hi_) // 2
            cmp = _cmp_at(mid)
            lo_new = jnp.where(cmp <= 0, mid + 1, lo_)
            hi_new = jnp.where(cmp <= 0, hi_, mid)
            return (lo_new, hi_new)

        lo_final, _ = jax.lax.while_loop(_cond, _body, (lo, hi))
        return lo_final

    lo = _lower_bound()  # first SA index with prefix >= pattern
    hi = _upper_bound() - 1  # last SA index with prefix == pattern
    return lo, hi


@partial(jax.jit, static_argnames=("Lmax",))
def _sa_lookup_one_line(
    q_line,       # [T]
    K_line,       # [T]
    SA_line,      # [T]
    cap_line,     # [T]
    succ_line,    # [T]
    tcap_line,    # [T]
    Lmax: int,
):
    """Full‑L suffix‑array lookup for one ``[T]`` line.

    For each L in 1..Lmax, performs binary search on the suffix array
    to find matching K positions, accumulates the longest/rightmost
    raw match, then applies ROSA successor + tau_cap gating.
    """
    T = q_line.shape[-1]
    pos_valid = jnp.arange(T, dtype=jnp.int32)

    best_j = jnp.full(T, -1, dtype=jnp.int32)
    best_L_raw = jnp.zeros(T, dtype=jnp.int32)

    # Pre-compute a full SA-index range for the bounded scan inside
    # _find_one.  We need this outside the vmap so jnp.arange sees
    # the concrete T from the outer JIT scope.
    scan_idx = jnp.arange(T, dtype=jnp.int32)  # [T]

    # Loop over L (unrolled at trace time because Lmax is static)
    for L in range(1, Lmax + 1):
        # Build query pattern for each position t: Q[t-L+1:t+1]
        # For each t, q_patterns[t, :] is the L-length suffix
        offsets = jnp.arange(L, dtype=jnp.int32)
        t_idx = pos_valid[:, None] - (L - 1) + offsets[None, :]  # [T, L]
        t_valid = (pos_valid >= L - 1)[:, None]
        t_idx_clip = jnp.clip(t_idx, 0, T - 1)
        q_pats = q_line[t_idx_clip]  # [T, L]
        q_pats = jnp.where(t_valid, q_pats, jnp.int32(0))  # mask invalid

        # For each t, find SA range, then rightmost valid position.
        def _find_one(q_pat, cap_t, t_pos):
            lo, hi = _sa_range_for_pattern(q_pat, K_line, SA_line, L=L)
            # Bounded scan: check SA indices lo..hi.
            # SA contains *starting* positions; ROSA needs *end* positions.
            in_range = (scan_idx >= lo) & (scan_idx <= hi)
            start_positions = SA_line[jnp.clip(scan_idx, 0, T - 1)]
            end_positions = start_positions + jnp.int32(L - 1)  # convert start→end
            valid_pos = (
                in_range
                & (start_positions >= 0)
                & (end_positions < cap_t)
                & (end_positions < T)
            )
            # Rightmost END position (for tie-breaking), but we also need to
            # score by rightmost START for the raw match?
            # ROSA rule: among same-length matches, pick rightmost END position.
            best_end = jnp.max(jnp.where(valid_pos, end_positions, jnp.int32(-1)))
            hit = best_end >= 0
            return best_end, hit

        j_L, hit_L = jax.vmap(_find_one)(
            q_pats, cap_line, pos_valid
        )  # [T], [T]

        # Mask out positions where t < L-1: the query pattern was
        # zero-padded for those positions, and zero is a valid symbol
        # that could match actual K symbols, creating false positives.
        hit_L = hit_L & (pos_valid >= (L - 1))

        # If this L gives a valid match, it always beats shorter L
        # (since we process in increasing L, and we want longest).
        best_j = jnp.where(hit_L & (jnp.int32(L) > best_L_raw), j_L, best_j)
        best_L_raw = jnp.where(hit_L & (jnp.int32(L) > best_L_raw), jnp.int32(L), best_L_raw)

    # ROSA gate
    best_j_safe = jnp.clip(best_j, 0, T - 1)
    tau_raw = succ_line[best_j_safe]
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    tau = jnp.where(final_hit, tau_raw, NEG)
    match_len = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return tau.astype(jnp.int64), match_len.astype(jnp.int32)


# (Inlined into _sa_lookup_one_line — see _find_one below)


# ---------------------------------------------------------------------------
# Batched entry point
# ---------------------------------------------------------------------------


def lookup_full_l_sa(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
    SA: np.ndarray | None = None,
):
    """Exact ROSA suffix lookup via suffix‑array binary search.

    Builds suffix arrays for K on CPU (unless ``SA`` is provided from
    a previous call) and performs binary search on GPU.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        ROSA auxiliary tensors.
    Lmax:
        Maximum suffix length to consider.
    tau_cap:
        Optional post‑successor cap for official ROSA/RLE semantics.
    validate_symbols:
        When True, validate symbol ranges.
    SA:
        Optional pre‑built suffix array ``[B, R, T]`` from
        :func:`suffix_array_batch`.  Reuse when K is shared.

    Returns
    -------
    tau:
        ``int64[B, R, T]`` — ROSA tau, or -1.
    match_len:
        ``int32[B, R, T]`` — accepted match length.
    """
    Q_arr, K_arr, B, R, T = require_rank3_pair(
        Q, K, sigma=None, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))

    if SA is None:
        SA = suffix_array_batch(np.asarray(K_arr, dtype=np.int64))
    SA_j = jnp.asarray(SA)

    def _one_route(q, k, sa_l, ce, s, tc):
        return _sa_lookup_one_line(
            q, k, sa_l, ce, s, tc, Lmax=Lmax_i,
        )

    tau, match_len = jax.vmap(
        jax.vmap(_one_route, in_axes=(0, 0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0, 0),
    )(Q_arr, K_arr, SA_j, cap, succ, tcap)
    return tau, match_len
