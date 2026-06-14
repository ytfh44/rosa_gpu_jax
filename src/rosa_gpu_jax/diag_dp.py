"""Streaming diagonal-DP for exact finite-L ROSA suffix lookup.

This module implements a memory-efficient streaming DP that maintains
only the previous row ``D_prev[T]`` instead of the full ``[T, T]``
matrix used by ``dp.py``.  It is exact and sigma-free, with no uint64
overflow constraint.

Complexity
----------
Time:   O(B·R·T²) — same as dense DP.
Memory: O(B·R·T)  — reduced from O(B·R·T²).

Recommended for ``T <= 1024`` as a correctness oracle or small-context
benchmark.  For larger ``T``, prefer the block-table or Shift-And paths.
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
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("Lmax",))
def _diag_dp_lookup_line(q_line, k_line, cap_line, succ_line, tcap_line, Lmax: int):
    """Streaming diagonal-DP for one ``[T]`` line.

    Returns ``(tau[T], match_len[T])`` — ``int64`` and ``int32``.
    """
    T = q_line.shape[-1]
    pos_j = jnp.arange(T, dtype=jnp.int32)
    Lmax_i32 = jnp.int32(Lmax)

    def step(carry, t):
        """carry = (D_prev[T], tau_acc[T], len_acc[T])"""
        D_prev, tau_acc, len_acc = carry

        q_sym = q_line[t]
        eq = (q_sym == k_line).astype(jnp.int32)  # [T]

        # D_curr[j] = D_prev[j-1] + 1 if eq[j] else 0
        shifted = jnp.pad(D_prev[:-1], (1, 0), constant_values=0)
        D_curr = jnp.where(eq, shifted + 1, 0)

        # Clamp to Lmax.
        D_curr = jnp.minimum(D_curr, Lmax_i32)

        # Select raw best: longest length, then rightmost j.
        cap_t = jnp.clip(cap_line[t], 0, T).astype(jnp.int32)
        valid = (pos_j < cap_t) & (D_curr > 0)

        # Score: (match_len, j) lexicographic — multiply len by (T+1), add j.
        score = jnp.where(
            valid,
            D_curr.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64)
            + pos_j.astype(jnp.int64),
            NEG,
        )

        best_j_idx = jnp.argmax(score).astype(jnp.int32)
        best_score_val = score[best_j_idx]
        actually_matched = best_score_val > NEG

        best_len_raw = jnp.where(
            actually_matched,
            D_curr[best_j_idx],
            jnp.int32(0),
        )

        # ROSA successor gating.
        best_j_safe = jnp.clip(best_j_idx, 0, T - 1)
        tau_raw = succ_line[best_j_safe]
        valid_tau = (best_len_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line[t])
        tau_t = jnp.where(valid_tau, tau_raw, NEG).astype(jnp.int64)
        len_t = jnp.where(valid_tau, best_len_raw, jnp.int32(0))

        # Accumulate (we overwrite per t, not a global accumulation).
        tau_acc = tau_acc.at[t].set(tau_t)
        len_acc = len_acc.at[t].set(len_t)

        return (D_curr, tau_acc, len_acc), None

    # Initial carry.
    D_init = jnp.zeros(T, dtype=jnp.int32)
    tau_init = jnp.full((T,), NEG, dtype=jnp.int64)
    len_init = jnp.zeros((T,), dtype=jnp.int32)

    t_seq = jnp.arange(T, dtype=jnp.int32)
    (_, tau_out, len_out), _ = jax.lax.scan(step, (D_init, tau_init, len_init), t_seq)

    return tau_out, len_out


@partial(jax.jit, static_argnames=("Lmax",))
def _diag_dp_lookup_jit(Q, K, cap_end, successor, tau_cap, Lmax: int):
    """Batched ``[B,R,T]`` wrapper around ``_diag_dp_lookup_line``."""

    def one_line(q, k, ce, succ, tc):
        return _diag_dp_lookup_line(q, k, ce, succ, tc, Lmax=Lmax)

    return jax.vmap(
        jax.vmap(one_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q, K, cap_end, successor, tau_cap)


def lookup_full_l_diag_dp(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Streaming diagonal-DP exact finite-L ROSA suffix lookup.

    Maintains only the previous row ``D_prev[T]``, reducing memory from
    ``O(T²)`` (dense DP) to ``O(T)``.  The semantic contract (longest →
    rightmost → successor → no backtracking) is identical to every other
    ``lookup_full_l_*`` function.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        ROSA auxiliary tensors (see :func:`make_rosa_causal_aux`).
    Lmax:
        Maximum suffix length to consider.  Must satisfy ``1 <= Lmax <= T``.
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
    """
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=None, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _diag_dp_lookup_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i)
