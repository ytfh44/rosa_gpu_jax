"""GPU suffix verification for CPU/coarse-proposed candidates."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.causal import NEG, make_raw_causal_aux
from rosa_gpu_jax.validation import (
    require_Lmax_for_T,
    require_aux,
    require_int_array,
    require_rank3_pair,
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("Lmax",))
def _verify_cpu_candidates_jit(Q, K, cand_end, cap_end, successor, tau_cap, Lmax: int):
    B, R, T = Q.shape
    del B, R
    offsets = jnp.arange(Lmax, dtype=jnp.int32)

    def line_verify(q, k, cand, cap, succ, tcap):
        # q,k,cap,succ,tcap: [T], cand: [T,C]
        t = jnp.arange(T, dtype=jnp.int32)[:, None, None]
        j = cand.astype(jnp.int32)[:, :, None]
        cap_t = cap[:, None, None]
        o = offsets[None, None, :]

        q_idx = t - o
        k_idx = j - o

        valid = (
            (cand[:, :, None] >= 0)
            & (q_idx >= 0)
            & (k_idx >= 0)
            & (j < cap_t)
        )

        q_tok = q[jnp.clip(q_idx, 0, T - 1)]
        k_tok = k[jnp.clip(k_idx, 0, T - 1)]

        eq = valid & (q_tok == k_tok)  # [T, C, Lmax]

        # Match length = number of leading True offsets before the first False.
        # Equivalent to cumprod+sum but friendlier to XLA (no prefix product).
        all_match = jnp.all(eq, axis=-1)  # [T, C]
        first_false = jnp.argmin(eq.astype(jnp.int32), axis=-1).astype(jnp.int32)
        lens = jnp.where(all_match, jnp.int32(Lmax), first_false)  # [T, C]

        # Select the raw best candidate first.  Official ROSA maps this raw
        # rightmost/longest match through successor and only then checks whether
        # the successor is currently valid.  It does not backtrack to an older
        # occurrence merely because the rightmost one has no valid successor.
        j_nonneg = jnp.maximum(cand.astype(jnp.int32), 0)
        score = lens.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64) + j_nonneg

        best_c = jnp.argmax(score, axis=-1).astype(jnp.int32)
        best_j = jnp.take_along_axis(cand, best_c[:, None], axis=-1)[:, 0].astype(jnp.int32)
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


def verify_cpu_candidates(
    Q,
    K,
    cand_end,
    Lmax: int,
    cap_end=None,
    successor=None,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
    validate_candidates: bool = True,
):
    """Verify CPU/coarse candidates on device.

    ``cap_end`` pre-filters raw candidate K end positions.  ``tau_cap`` is an
    optional post-successor cap used by official ROSA/RLE semantics.
    """
    Q_arr, K_arr, B, R, T = require_rank3_pair(Q, K, sigma=None, validate_symbols=validate_symbols)
    Lmax_i = require_Lmax_for_T(Lmax, T)

    cand = jnp.asarray(cand_end)
    if cand.ndim != 4:
        raise ValueError(f"cand_end must be rank-4 shaped [B, R, T, C]; got {cand.shape}")
    if tuple(cand.shape[:3]) != tuple(Q_arr.shape):
        raise ValueError(
            f"cand_end leading dimensions must match Q/K shape {Q_arr.shape}; got {cand.shape}"
        )
    C = int(cand.shape[-1])
    if C <= 0:
        raise ValueError(f"cand_end must contain at least one candidate slot; got C={C}")
    cand = require_int_array(
        "cand_end",
        cand,
        shape=tuple(cand.shape),
        min_value=-1,
        max_value=T - 1,
        validate_values=validate_candidates,
    )

    if cap_end is None or successor is None:
        if cap_end is not None or successor is not None:
            raise ValueError("cap_end and successor must be provided together")
        cap_end, successor = make_raw_causal_aux(B, R, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _verify_cpu_candidates_jit(Q_arr, K_arr, cand, cap, succ, tcap, Lmax=Lmax_i)
