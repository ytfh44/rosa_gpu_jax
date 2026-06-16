"""Q-symbol bit counterfactual destination lookup."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.block_table import _block_keys_base_jit, _lookup_one_l_from_keys_end_jit
from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    ensure_exact_key_safe,
    require_aux,
    require_Lmax_for_T,
    require_M,
    require_rank3_pair,
    require_sigma,
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("Lmax", "sigma", "M"))
def _q_bit_counterfactual_tau_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int, M: int):
    bits = jnp.arange(M, dtype=jnp.int64)
    B, R, T = Q.shape
    pos_valid_base = jnp.arange(T)[None, None, :]
    old_q_u = Q.astype(jnp.uint64)

    # Precompute Q-keys and K-keys once per L — they do not depend on m or
    # branch_value and would otherwise be recomputed 2 * M * Lmax times.
    q_keys_by_L = []
    k_keys_by_L = []
    for L in range(1, Lmax + 1):
        q_keys_by_L.append(_block_keys_base_jit(Q, L=L, sigma=sigma))
        k_keys_by_L.append(_block_keys_base_jit(K, L=L, sigma=sigma))

    def lookup_for_forced_current_symbol(m, branch_value):
        mask = jnp.asarray(1, dtype=jnp.int64) << m
        if branch_value == 0:
            forced = (Q & (~mask)).astype(jnp.uint64)
        else:
            forced = (Q | mask).astype(jnp.uint64)

        best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
        best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

        for idx, L in enumerate(range(1, Lmax + 1)):
            q_keys = q_keys_by_L[idx]
            k_keys = k_keys_by_L[idx]

            # In base encoding, the current symbol is the last digit of every
            # block ending at the current query position and has weight 1.
            # Prefix-padded positions are kept unchanged to avoid meaningless
            # uint64 wraparound on invalid query positions.
            valid_query_pos = pos_valid_base >= (L - 1)
            q_keys_forced = jnp.where(valid_query_pos, q_keys - old_q_u + forced, q_keys)

            _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_from_keys_end_jit(
                q_keys_forced, k_keys, cap_end, successor, tau_cap, L=L
            )
            best_end = jnp.where(raw_hit_L, end_L, best_end)
            best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

        end_safe = jnp.clip(best_end, 0, T - 1)
        tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
        final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
        return jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)

    def one_bit(m):
        return lookup_for_forced_current_symbol(m, 0), lookup_for_forced_current_symbol(m, 1)

    return jax.vmap(one_bit)(bits)


def q_bit_counterfactual_tau(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    sigma: int,
    M: int,
    *,
    tau_cap=None,
    validate_symbols: bool = True,
):
    """Force each current query-symbol bit to 0/1 and return tau.

    This function alters only the current query symbol for each query time.  It
    does not rebuild run-length encoding after the bit flip.  Pass ``tau_cap``
    from ``make_rosa_causal_aux`` to reproduce official ROSA/RLE successor
    validity.
    """
    sigma_i = require_sigma(sigma)
    M_i = require_M(M)
    if sigma_i < (1 << M_i):
        raise ValueError(f"sigma must be at least 2**M so forced symbols stay valid; got sigma={sigma_i}, M={M_i}")
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=(1 << M_i), validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    for L in range(1, Lmax_i + 1):
        ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _q_bit_counterfactual_tau_jit(Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i, M=M_i)
