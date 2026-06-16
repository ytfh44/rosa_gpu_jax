"""Exact counting-prefix-table finite-L ROSA suffix lookup.

This module builds a dense per‑L table ``table[key, pos+1]`` that records
the rightmost occurrence of each base‑encoded block key.  A single
``cummax`` pass turns it into a prefix‑max table so that
``prefix[qk, cap]`` returns the rightmost K end position < cap with the
same block key.

Key properties
--------------
- **Exact** — no hash collisions, no false positives, no false negatives.
- **O(sigma^L · T) memory** — only viable for small alphabets and moderate
  ``Lmax`` (e.g. sigma ≤ 4, Lmax ≤ 4).
- **O(B·R·Lmax·T) time** — asymptotically optimal for exact base‑key
  lookup, but with a large constant from the dense table.

Compared to ``block_table.py``
-------------------------------
The block‑table sort approach uses ``O(T log T)`` per (b,r) line with
``O(T)`` auxiliary memory.  The counting‑prefix table replaces the sort
with a dense direct‑address table, trading memory for lower per‑position
work when ``sigma^L`` is tiny.

Complexity
----------
Time:   O(B·R·Lmax·(T + sigma^L))
Memory: O(B·R·sigma^L · T)   (the table is built per (b,r) line inside vmap)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rosa_gpu_jax.block_table import _block_keys_base_jit
from rosa_gpu_jax.causal import NEG
from rosa_gpu_jax.validation import (
    ensure_exact_key_safe,
    require_aux,
    require_Lmax_for_T,
    require_rank3_pair,
    require_sigma,
    require_tau_cap,
)


@partial(jax.jit, static_argnames=("L", "sigma"))
def _lookup_one_l_counting_prefix_end_jit(
    q_keys, k_keys, cap_end, successor, tau_cap, L: int, sigma: int
):
    """One-L exact lookup via prefix-max table.

    ``table[key, pos+1] = rightmost j with block key == key``, then
    ``prefix = cummax(table)`` so ``prefix[qk, cap]`` gives the answer.
    """
    _B, _R, T = q_keys.shape
    keyspace = sigma**L
    pos = jnp.arange(T, dtype=jnp.int32)

    def line(qk, kk, cap, succ, tcap):
        rows = jnp.where(pos >= L - 1, kk.astype(jnp.int32), jnp.int32(0))
        cols = pos + 1
        vals = jnp.where(pos >= L - 1, pos, jnp.int32(-1))

        table = jnp.full((keyspace, T + 1), -1, dtype=jnp.int32)
        table = table.at[rows, cols].max(vals)
        prefix = jnp.maximum.accumulate(table, axis=1)

        cap_i = jnp.clip(cap, 0, T).astype(jnp.int32)
        end = prefix[qk.astype(jnp.int32), cap_i]
        raw = (end >= 0) & (pos >= L - 1)

        tau_raw = succ[jnp.clip(end, 0, T - 1)]
        valid_tau = raw & (tau_raw >= 0) & (tau_raw <= tcap)
        tau = jnp.where(valid_tau, tau_raw, NEG)
        end_out = jnp.where(raw, end.astype(jnp.int64), jnp.int64(-1))
        return tau.astype(jnp.int64), valid_tau, end_out, raw

    return jax.vmap(
        jax.vmap(line, in_axes=(0, 0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    )(q_keys, k_keys, cap_end, successor, tau_cap)


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _lookup_full_l_counting_prefix_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int):
    """Full-L counting-prefix lookup kernel."""
    B, R, T = Q.shape

    q_keys_by_L = []
    k_keys_by_L = []
    for L in range(1, Lmax + 1):
        q_keys_by_L.append(_block_keys_base_jit(Q, L=L, sigma=sigma))
        k_keys_by_L.append(_block_keys_base_jit(K, L=L, sigma=sigma))

    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for idx, L in enumerate(range(1, Lmax + 1)):
        q_keys = q_keys_by_L[idx]
        k_keys = k_keys_by_L[idx]
        _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_counting_prefix_end_jit(
            q_keys, k_keys, cap_end, successor, tau_cap, L=L, sigma=sigma,
        )
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_counting_prefix(
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
    """Exact counting-prefix-table finite-L ROSA suffix lookup.

    Builds a dense ``[sigma^L, T+1]`` prefix-max table for each block
    length and answers rightmost-predecessor queries in O(1) per position.

    This method is **exact** and has no hash collisions or false positives,
    but memory is ``O(sigma^L · T)`` per (b,r) line.  It is only practical
    when the alphabet and ``Lmax`` are very small (e.g. sigma ≤ 4,
    Lmax ≤ 4).

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end, successor:
        ROSA auxiliary tensors (see :func:`make_rosa_causal_aux`).
    Lmax:
        Maximum suffix length to consider.  Must satisfy ``1 <= Lmax <= T``.
    sigma:
        Alphabet size.  Must be ≥ 2.
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
    sigma_i = require_sigma(sigma)
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(
        Q, K, sigma=sigma_i, validate_symbols=validate_symbols
    )
    Lmax_i = require_Lmax_for_T(Lmax, T)
    for L in range(1, Lmax_i + 1):
        ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_counting_prefix_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i,
    )
