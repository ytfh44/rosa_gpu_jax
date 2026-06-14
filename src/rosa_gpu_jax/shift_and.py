"""Shift-And bitset exact finite-L ROSA suffix lookup.

This module implements the classic bit-parallel Shift-And string-matching
algorithm adapted for ROSA suffix predecessor queries.  It represents
matching positions as multi-word uint64 bitsets and uses bitwise operations
(shift, AND, highest-set-bit) instead of sorting or hashing.

Key properties
--------------
- **Exact** — no hash collisions, no false positives, no false negatives
  (up to the ``C=T`` guarantee).
- **No base-key overflow** — no ``sigma^L`` term and no uint64 combined-key
  constraint.  The only parameter limits are ``T`` (bitset width) and
  ``Lmax`` (loop depth).
- **Streaming-friendly** — each new symbol needs only ``O(Lmax * W)``
  bitwise operations to update the per-L bitsets.
- **Rightmost predecessor** is a single ``highest_set_bit`` query on the
  masked bitset, which aligns perfectly with ROSA tie-breaking.

Compared to the existing ``bitset.py`` (Method 8) this path does **not**
construct a ``[T, T]`` boolean matrix.  It maintains only ``Lmax`` bitsets
of width ``W = ceil(T/64)`` words.

Complexity
----------
Time:   O(B·R·Lmax·T·ceil(T/64))
Memory: O(B·R·Lmax·ceil(T/64))
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    require_aux,
    require_Lmax_for_T,
    require_rank3_pair,
    require_sigma,
    require_tau_cap,
)

# ---------------------------------------------------------------------------
# Word-level helpers
# ---------------------------------------------------------------------------

WORD_BITS = 64
ALL_ONES = jnp.uint64(0xFFFFFFFFFFFFFFFF)


def _words_for_T(T: int) -> int:
    """Number of uint64 words needed to represent T bits."""
    return (T + WORD_BITS - 1) // WORD_BITS


def _build_symbol_masks(k_line, sigma: int, W: int):
    """Build per-symbol occurrence masks ``[sigma, W]`` uint64.

    Bit ``j`` is 1 in ``P[sym, j // 64]`` iff ``k_line[j] == sym``.
    """
    T = k_line.shape[0]
    j_idx = jnp.arange(T, dtype=jnp.int32)
    word_idx = j_idx // WORD_BITS
    bit_idx = j_idx % WORD_BITS
    ones = jnp.left_shift(jnp.uint64(1), bit_idx.astype(jnp.uint64))

    # scatter: accumulate 1-bits into P[sym, word]
    P = jnp.zeros((sigma, W), dtype=jnp.uint64)
    P = P.at[k_line.astype(jnp.int32), word_idx].add(ones)
    return P


def _build_cap_masks(cap_line, W: int):
    """Precompute per-t cap masks ``[T, W]`` uint64.

    ``cap_masks[t, w]`` has bits 0..cap_end[t]-1 set to 1 in the
    corresponding word.  Words entirely within the cap are ``ALL_ONES``;
    the partial word gets ``(1 << remaining) - 1``.
    """
    T = cap_line.shape[0]
    cap_i32 = jnp.clip(cap_line, 0, T).astype(jnp.int32)  # [T]

    cap_t = cap_i32[:, None].astype(jnp.uint64)  # [T, 1]
    ws = (jnp.arange(W, dtype=jnp.uint64) * jnp.uint64(WORD_BITS))[None, :]  # [1, W]
    we = ws + jnp.uint64(WORD_BITS)  # [1, W]

    # Full mask: cap_t >= we  →  all bits of this word are in range.
    full_mask = jnp.where(cap_t >= we, ALL_ONES, jnp.uint64(0))

    # Partial mask: ws < cap_t < we → (1 << (cap_t - ws)) - 1
    remaining = cap_t - ws
    partial = jnp.where(
        (cap_t > ws) & (cap_t < we),
        jnp.left_shift(jnp.uint64(1), remaining) - jnp.uint64(1),
        jnp.uint64(0),
    )

    return full_mask | partial  # [T, W] uint64


def _word_shift_left(M):
    """Shift a ``[T, W]`` or ``[W]`` bitset left by 1 across word boundaries.

    Bit 63 of word ``w-1`` becomes bit 0 of word ``w``.
    """
    # M: [T, W]
    shifted = jnp.left_shift(M, jnp.uint64(1))  # [T, W]
    carry = jnp.right_shift(M[..., :-1], jnp.uint64(WORD_BITS - 1))  # [T, W-1]
    shifted = shifted.at[..., 1:].add(carry)
    return shifted


def _highest_set_bit_u64(x):
    """Return 0-based index of highest set bit in a uint64 scalar, or -1 if zero.

    Uses binary search over bit ranges — exact and JIT-friendly (no float64 log2).
    ``x`` must be a ``uint64[...]`` array.
    """
    # Binary-search for the highest set bit in a uint64 word.
    # Step 1: check top 32 bits.
    hi32 = jnp.right_shift(x, jnp.uint64(32))
    has_hi32 = hi32 > jnp.uint64(0)
    result = jnp.where(has_hi32, jnp.int32(32), jnp.int32(0))
    val = jnp.where(has_hi32, hi32, x & jnp.uint64(0xFFFFFFFF))

    # Step 2: check bit 16 within the selected 32-bit half.
    hi16 = jnp.right_shift(val, jnp.uint64(16))
    has_hi16 = hi16 > jnp.uint64(0)
    result = result + jnp.where(has_hi16, jnp.int32(16), jnp.int32(0))
    val = jnp.where(has_hi16, hi16, val & jnp.uint64(0xFFFF))

    # Step 3: check bit 8.
    hi8 = jnp.right_shift(val, jnp.uint64(8))
    has_hi8 = hi8 > jnp.uint64(0)
    result = result + jnp.where(has_hi8, jnp.int32(8), jnp.int32(0))
    val = jnp.where(has_hi8, hi8, val & jnp.uint64(0xFF))

    # Step 4: check bit 4.
    hi4 = jnp.right_shift(val, jnp.uint64(4))
    has_hi4 = hi4 > jnp.uint64(0)
    result = result + jnp.where(has_hi4, jnp.int32(4), jnp.int32(0))
    val = jnp.where(has_hi4, hi4, val & jnp.uint64(0xF))

    # Step 5: check bit 2.
    hi2 = jnp.right_shift(val, jnp.uint64(2))
    has_hi2 = hi2 > jnp.uint64(0)
    result = result + jnp.where(has_hi2, jnp.int32(2), jnp.int32(0))
    val = jnp.where(has_hi2, hi2, val & jnp.uint64(0x3))

    # Step 6: check bit 1.
    hi1 = jnp.right_shift(val, jnp.uint64(1))
    has_hi1 = hi1 > jnp.uint64(0)
    result = result + jnp.where(has_hi1, jnp.int32(1), jnp.int32(0))

    # When x == 0, result stays 0 — mask externally.
    return result


def _highest_set_bit(masked):
    """Return the highest set bit position (0-based) for each row of ``[T, W]``.

    Returns ``int32[T]`` with values in ``[0, T-1]``, or ``-1`` for all-zero rows.
    """
    # For each row, find the highest non-zero word.
    word_nonzero = masked > jnp.uint64(0)
    highest_word = jnp.argmax(
        jnp.where(word_nonzero, jnp.arange(word_nonzero.shape[-1], dtype=jnp.int32), -1),
        axis=-1,
    ).astype(jnp.int32)

    # Extract the value in the highest non-zero word.
    T = masked.shape[0]
    highest_word_clip = jnp.clip(highest_word, 0, masked.shape[-1] - 1)
    val = masked[jnp.arange(T, dtype=jnp.int32), highest_word_clip]  # [T] uint64

    # Find the highest set bit within that word (exact binary-search, no float64).
    bit_in_word = _highest_set_bit_u64(val)

    global_bit = highest_word * jnp.int32(WORD_BITS) + bit_in_word

    # Mask out rows that were all-zero.
    any_hit = jnp.any(word_nonzero, axis=-1)  # [T] bool
    return jnp.where(any_hit, global_bit, jnp.int32(-1))


# ---------------------------------------------------------------------------
# Single-line Shift-And kernel
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _shift_and_lookup_line(q_line, k_line, cap_line, succ_line, tcap_line, Lmax: int, sigma: int):
    """Shift-And bitset ROSA lookup for one ``[T]`` line.

    Returns ``(tau[T], match_len[T])`` — ``int64`` and ``int32``.
    """
    T = q_line.shape[-1]
    W = _words_for_T(T)

    # Precompute symbol masks and cap masks.
    P = _build_symbol_masks(k_line, sigma, W)  # [sigma, W]
    cap_masks = _build_cap_masks(cap_line, W)  # [T, W]

    # Accumulate raw best per position.
    raw_best_L = jnp.zeros((T,), dtype=jnp.int32)
    raw_best_j = jnp.full((T,), jnp.int32(-1), dtype=jnp.int32)

    # M_prev[t] for the previous length level.
    # For L=1: M[t] = P[Q[t]] directly.
    M_prev = P[q_line.astype(jnp.int32)]  # [T, W]

    # --- L = 1 ---
    masked = M_prev & cap_masks  # [T, W]
    J = _highest_set_bit(masked)  # [T] int32
    raw_best_j = jnp.where(J >= 0, J, raw_best_j)
    raw_best_L = jnp.where(J >= 0, jnp.int32(1), raw_best_L)

    # --- L = 2 .. Lmax ---
    for L in range(2, Lmax + 1):
        # M_L(t) = shift_left(M_{L-1}(t-1)) & P[Q[t]]
        # For t=0: M_prev[-1] is invalid — use all-zeros.
        M_prev_shifted_t_minus_1 = jnp.concatenate(
            [jnp.zeros((1, W), dtype=jnp.uint64), M_prev[:-1]], axis=0
        )
        M_prev_shifted = _word_shift_left(M_prev_shifted_t_minus_1)  # [T, W]

        P_q = P[q_line.astype(jnp.int32)]  # [T, W]
        M_curr = M_prev_shifted & P_q  # [T, W]

        # Rightmost valid j for this L.
        masked = M_curr & cap_masks
        J = _highest_set_bit(masked)  # [T] int32

        # Update: longer L wins; for equal L, _highest_set_bit already
        # returns the rightmost j within the masked bitset.
        L_i32 = jnp.int32(L)
        upgrade = (J >= 0) & (L_i32 > raw_best_L)
        raw_best_L = jnp.where(upgrade, L_i32, raw_best_L)
        raw_best_j = jnp.where(upgrade, J, raw_best_j)

        M_prev = M_curr

    # ROSA successor / tau_cap gating on accumulated best.
    best_j_safe = jnp.clip(raw_best_j, 0, T - 1)
    tau_raw = succ_line[best_j_safe]
    valid_tau = (raw_best_L > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    tau = jnp.where(valid_tau, tau_raw, NEG).astype(jnp.int64)
    match_len = jnp.where(valid_tau, raw_best_L, jnp.int32(0))

    return tau, match_len


# ---------------------------------------------------------------------------
# Batched entry point
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _shift_and_lookup_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int):
    """Batched ``[B,R,T]`` wrapper around ``_shift_and_lookup_line``."""

    def one_line(q, k, ce, succ, tc):
        return _shift_and_lookup_line(q, k, ce, succ, tc, Lmax=Lmax, sigma=sigma)

    return jax.vmap(
        jax.vmap(one_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q, K, cap_end, successor, tau_cap)


def lookup_full_l_shift_and(
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
    """Shift-And bitset exact finite-L ROSA suffix lookup.

    Uses the classic bit-parallel Shift-And algorithm adapted for ROSA
    suffix predecessor queries.  Maintains per-length bitsets of matching
    K positions and resolves rightmost-predecessor via highest-set-bit.

    This method is **exact** — no hash collisions, no false positives,
    no uint64 base-key overflow.  Compared to the existing ``bitset.py``
    (Method 8) it does not construct a ``[T, T]`` boolean matrix; it
    maintains only ``Lmax`` multi-word bitsets.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        ROSA auxiliary tensors (see :func:`make_rosa_causal_aux`).
    Lmax:
        Maximum suffix length to consider.  Must satisfy ``1 <= Lmax <= T``.
    sigma:
        Alphabet size.  Must be ≥ 2.  Used to size the per-symbol mask
        table (``sigma × ceil(T/64)`` words).
    tau_cap:
        Optional post-successor cap for official ROSA/RLE semantics.
    validate_symbols:
        When ``True`` (default), validate that symbols are within range.

    Returns
    -------
    tau:
        ``int64[B, R, T]`` — ROSA tau (next-run start), or ``-1`` when no
        valid match exists.
    match_len:
        ``int32[B, R, T]`` — length of the accepted raw suffix match.

    Notes
    -----
    Complexity is ``O(B·R·Lmax·T·ceil(T/64))`` time and
    ``O(B·R·Lmax·ceil(T/64))`` memory.  This is better than the
    existing ``bitset.py`` (``O(B·R·T²·Lmax²)``) and is competitive
    with base block-table sort for larger ``Lmax`` or when sorting
    overhead dominates.

    At small ``T`` and small ``Lmax``, the base-encoded block table
    (``lookup_full_l_base``) may still be faster due to lower constant
    factors.  At larger ``Lmax``, Shift-And's per-level work is
    constant-cost bitwise operations per word, giving it an advantage.
    """
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=sigma_i, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _shift_and_lookup_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i)
