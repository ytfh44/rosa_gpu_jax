"""Boolean-array exact finite-L ROSA suffix lookup (experimental).

Directly tests suffix matches by comparing symbols for each query
position and suffix length.  Complexity O(B·R·T²·Lmax).  Included
as an experimental correctness reference, not a production path.

.. warning::
   Only suitable for very small ``T`` (e.g. ``T <= 32``).
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


@partial(jax.jit, static_argnames=("Lmax",))
def _lookup_full_l_bitset_line(
    q_line, k_line, cap_line, succ_line, tcap_line, Lmax: int
):
    """Direct symbol comparison for one [T] line.

    For each (t, L): check K[j-off] == Q[t-off] for off in 0..L-1.
    """
    T = q_line.shape[-1]
    pos_j = jnp.arange(T, dtype=jnp.int32)

    best_end = jnp.full((T,), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((T,), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        # R[t, j] = True iff K[j-off] == Q[t-off] for all off in 0..L-1
        R = jnp.ones((T, T), dtype=bool)  # [T, T]
        for offset in range(L):
            q_idx = jnp.clip(jnp.arange(T, dtype=jnp.int32) - offset, 0, T - 1)
            k_idx = pos_j[None, :] - offset  # [1, T]
            k_idx_clip = jnp.clip(k_idx, 0, T - 1)
            valid_off = k_idx >= 0
            eq = (q_line[q_idx, None] == k_line[k_idx_clip]) & valid_off
            R = R & eq

        cap_i = jnp.clip(cap_line, 0, T).astype(jnp.int32)
        valid = R & (pos_j[None, :] < cap_i[:, None]) & (pos_j[None, :] >= (L - 1))
        t_mask = jnp.arange(T, dtype=jnp.int32) >= (L - 1)
        valid = valid & t_mask[:, None]

        score = jnp.where(valid, pos_j.astype(jnp.int64)[None, :], jnp.int64(-1))
        best_j = jnp.max(score, axis=-1)
        hit_L = best_j >= 0

        best_end = jnp.where(hit_L & (jnp.int32(L) > best_L_raw), best_j, best_end)
        best_L_raw = jnp.where(hit_L & (jnp.int32(L) > best_L_raw), jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = succ_line[end_safe]
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    tau = jnp.where(final_hit, tau_raw, NEG)
    best_len = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return tau.astype(jnp.int64), best_len.astype(jnp.int32)


def lookup_full_l_bitset(
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
    """Direct-comparison exact finite-L ROSA suffix lookup (experimental).

    Directly tests Q[t-off]==K[j-off] for each position and suffix length.
    Complexity O(B·R·T²·Lmax²).  Only suitable for T <= 32.

    Parameters
    ----------
    sigma:
        Alphabet size (used only for input validation).
    """
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=sigma_i, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))

    def _one_line(q, k, ce, s, tc):
        return _lookup_full_l_bitset_line(q, k, ce, s, tc, Lmax=Lmax_i)

    tau, match_len = jax.vmap(
        jax.vmap(_one_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q_arr, K_arr, cap, succ, tcap)
    return tau, match_len
