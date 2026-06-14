"""Dense equality DP for exact finite-L ROSA lookup.

This module implements a simple O(B·R·T²) dynamic-programming exact
baseline.  It is not a production accelerator but serves as:

1. A correctness oracle for all other lookup paths (small ``T``).
2. A TPU-friendly reference (dense matmul + scan).
3. A benchmark for small-context studies.

It is intentionally independent of the block-table and rolling-hash
modules — the DP works directly on raw symbol arrays.
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
def _dp_lookup_full_l_jit(Q, K, cap_end, successor, tau_cap, Lmax: int):
    """Dense DP: compute longest suffix match per (t,j) pair, then gate.

    Returns ``(tau, match_len)`` each ``[B, R, T]``, identical in contract
    to every other ``lookup_full_l_*`` function.
    """
    B, R, T = Q.shape
    del B, R
    pos_j = jnp.arange(T, dtype=jnp.int32)

    def line_dp(q, k, cap, succ, tcap):
        # q, k, cap, succ, tcap: [T]
        eq = q[:, None] == k[None, :]  # [T, T] bool

        def scan_step(carry, eq_row):
            # carry: [T] int32 = previous D row
            shifted = jnp.pad(carry[:-1], (1, 0), constant_values=0)
            new_row = jnp.where(eq_row, shifted + 1, 0)
            return new_row, new_row

        init = jnp.zeros(T, dtype=jnp.int32)
        _, D = jax.lax.scan(scan_step, init, eq)  # D: [T, T] int32

        # Clamp to Lmax and compute score.
        D_clamped = jnp.minimum(D, jnp.int32(Lmax))
        score = jnp.where(
            D > 0,
            D_clamped.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64)
            + pos_j.astype(jnp.int64),
            NEG,
        )  # [T, T]

        # Mask by cap_end.
        cap_i = jnp.clip(cap, 0, T).astype(jnp.int32)
        valid = pos_j[None, :] < cap_i[:, None]  # [T, T]
        score_masked = jnp.where(valid, score, NEG)

        best_j_idx = jnp.argmax(score_masked, axis=-1).astype(jnp.int32)  # [T]
        best_score = jnp.take_along_axis(
            score_masked, best_j_idx[:, None], axis=-1
        )[:, 0]  # [T]
        actually_matched = best_score > NEG  # [T] bool

        best_j = best_j_idx  # positions are direct indices, not candidate slots
        best_len_raw = jnp.where(
            actually_matched,
            jnp.take_along_axis(D_clamped, best_j[:, None], axis=-1)[:, 0],
            jnp.int32(0),
        )  # [T]

        # ROSA successor gating (no backtracking).
        best_j_safe = jnp.clip(best_j, 0, T - 1)
        tau_raw = succ[best_j_safe]
        valid_tau = (best_len_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap)
        tau = jnp.where(valid_tau, tau_raw, NEG)
        best_len = jnp.where(valid_tau, best_len_raw, jnp.int32(0))
        return tau.astype(jnp.int64), best_len.astype(jnp.int32)

    return jax.vmap(
        jax.vmap(line_dp, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q, K, cap_end, successor, tau_cap)


def lookup_full_l_dp(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Exact dense-DP longest suffix lookup for all lengths ``1..Lmax``.

    This is a reference / oracle path.  It computes a full ``[T, T]``
    equality matrix and then runs a row-by-row scan to obtain match
    lengths.  Complexity is ``O(B·R·T²)``, so it is only recommended
    for ``T <= 1024``.

    The semantic contract (longest → rightmost → successor → no
    backtracking) is identical to :func:`lookup_full_l_base`.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        Lookup auxiliaries (see :func:`make_rosa_causal_aux`).
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
    return _dp_lookup_full_l_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i)
