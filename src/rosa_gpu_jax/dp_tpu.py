"""TPU-friendly dense one-hot equality DP benchmark.

Computes the ``[T, T]`` equality matrix via ``one_hot(Q) @ one_hot(K).T``,
then runs a per-line DP scan for suffix match lengths.  Designed for TPU
systolic-array throughput experiments; not recommended for GPU or
``T > 2048``.

.. note::
   This path is separate from the main lookup methods because the one-hot
   matmul is only efficient on TPU hardware.  On GPU it is slower than
   the direct broadcast used by ``dp.py``.
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


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _dp_tpu_lookup_full_l_line(
    q_line, k_line, cap_line, succ_line, tcap_line, Lmax: int, sigma: int
):
    """Dense one-hot DP for one line.  Returns (tau, best_len)."""
    T = q_line.shape[-1]

    # One-hot + matmul to get equality matrix.
    Q_oh = jax.nn.one_hot(q_line, sigma, dtype=jnp.int32)  # [T, sigma]
    K_oh = jax.nn.one_hot(k_line, sigma, dtype=jnp.int32)  # [T, sigma]
    E = Q_oh @ K_oh.T  # [T, T] — E[t,j] = 1 iff Q[t]==K[j]

    # DP scan identical to dp.py.
    def scan_step(carry, eq_row):
        shifted = jnp.pad(carry[:-1], (1, 0), constant_values=0)
        new_row = jnp.where(eq_row > 0, shifted + 1, 0)
        return new_row, new_row

    init = jnp.zeros(T, dtype=jnp.int32)
    _, D = jax.lax.scan(scan_step, init, E)  # [T, T] int32

    D_clamped = jnp.minimum(D, jnp.int32(Lmax))
    pos_j = jnp.arange(T, dtype=jnp.int32)
    score = jnp.where(
        D > 0,
        D_clamped.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64)
        + pos_j.astype(jnp.int64),
        NEG,
    )

    cap_i = jnp.clip(cap_line, 0, T).astype(jnp.int32)
    valid = pos_j[None, :] < cap_i[:, None]
    score_masked = jnp.where(valid, score, NEG)

    best_j_idx = jnp.argmax(score_masked, axis=-1).astype(jnp.int32)
    best_score = jnp.take_along_axis(score_masked, best_j_idx[:, None], axis=-1)[:, 0]
    actually_matched = best_score > NEG

    best_len_raw = jnp.where(
        actually_matched,
        jnp.take_along_axis(D_clamped, best_j_idx[:, None], axis=-1)[:, 0],
        jnp.int32(0),
    )

    best_j_safe = jnp.clip(best_j_idx, 0, T - 1)
    tau_raw = succ_line[best_j_safe]
    valid_tau = (best_len_raw > 0) & (tau_raw >= 0) & (tau_raw <= tcap_line)
    tau = jnp.where(valid_tau, tau_raw, NEG)
    best_len = jnp.where(valid_tau, best_len_raw, jnp.int32(0))
    return tau.astype(jnp.int64), best_len.astype(jnp.int32)


def lookup_full_l_dense_tpu(
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
    """TPU-friendly dense one-hot DP baseline.

    Uses ``one_hot(Q) @ one_hot(K).T`` for the equality matrix, which
    leverages TPU systolic arrays.  Not recommended for GPU or
    ``T > 2048``.

    Parameters
    ----------
    sigma:
        Alphabet size.  Large sigma increases one-hot dimension.
    """
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=sigma_i, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))

    def _one_line(q, k, ce, s, tc):
        return _dp_tpu_lookup_full_l_line(
            q, k, ce, s, tc, Lmax=Lmax_i, sigma=sigma_i
        )

    tau, match_len = jax.vmap(
        jax.vmap(_one_line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(Q_arr, K_arr, cap, succ, tcap)
    return tau, match_len
