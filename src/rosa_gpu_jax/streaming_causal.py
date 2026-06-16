"""Exact streaming causal bucket finite-L ROSA suffix lookup.

This module implements an online, O(T) per (b,r) bucket lookup that is
exact for the special case ``cap_end[t] == t`` (strict causal).  Query
and insert happen in sequence via ``lax.scan``, guaranteeing ``j < t``
without explicit masking.

Key properties
--------------
- **Exact** — no hash collisions, no false positives, no false negatives.
- **Streaming / online** — processes positions left-to-right with a single
  ``lax.scan``, making it suitable as a blueprint for autoregressive
  inference kernels.
- **Causal only** — only valid when ``cap_end[t] == t``.  Arbitrary
  ``cap_end`` is **not** supported.
- **O(sigma^L) memory** — a single 1-D bucket array indexed by base-
  encoded block key.  Practical for small alphabets and moderate Lmax
  (e.g. sigma ≤ 8, Lmax ≤ 4).

Compared to ``diag_dp.py``
---------------------------
Both handle ``cap_end[t] == t`` and are streaming.  ``diag_dp.py`` uses
diagonal dynamic programming (sigma-free, O(T) memory).  This module
uses base-key bucketing, trading sigma-generality for faster per-symbol
work when the key space is small.

Complexity
----------
Time:   O(B·R·Lmax·T)
Memory: O(B·R·sigma^L)    (per-line 1-D bucket, much smaller than prefix_table)
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
def _lookup_one_l_streaming_causal_end_jit(
    q_keys, k_keys, successor, tau_cap, L: int, sigma: int
):
    """One-L exact online bucket lookup for cap_end[t] == t.

    Uses ``lax.scan``: for each position t, first query the bucket
    (enforcing j < t), then insert the current K block key.

    ``cap_end`` is intentionally absent — this kernel is only valid
    when ``cap_end[t] == t``.
    """
    _B, _R, T = q_keys.shape
    keyspace = sigma**L
    pos = jnp.arange(T, dtype=jnp.int32)

    def line(qk, kk, succ, tcap):
        def step(table, x):
            t, qkey, kkey, tcap_t = x

            # Query before insert: enforces j < t.
            end = table[qkey.astype(jnp.int32)]
            raw = (end >= 0) & (t >= L - 1)

            tau_raw = succ[jnp.clip(end, 0, T - 1)]
            valid_tau = raw & (tau_raw >= 0) & (tau_raw <= tcap_t)
            tau = jnp.where(valid_tau, tau_raw, NEG)

            # Insert current K block after answering current query.
            table2 = jnp.where(
                t >= L - 1,
                table.at[kkey.astype(jnp.int32)].set(t),
                table,
            )
            end_out = jnp.where(raw, end.astype(jnp.int64), jnp.int64(-1))
            return table2, (tau.astype(jnp.int64), valid_tau, end_out, raw)

        init = jnp.full((keyspace,), -1, dtype=jnp.int32)
        _, outs = jax.lax.scan(step, init, (pos, qk, kk, tcap))
        return outs

    return jax.vmap(
        jax.vmap(line, in_axes=(0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0),
    )(q_keys, k_keys, successor, tau_cap)


@partial(jax.jit, static_argnames=("Lmax", "sigma"))
def _lookup_full_l_streaming_causal_jit(Q, K, successor, tau_cap, Lmax: int, sigma: int):
    """Full-L streaming causal bucket lookup kernel.

    ``cap_end`` is absent — this kernel is only valid when ``cap_end[t] == t``.
    """
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
        _tau_L, _valid_hit_L, end_L, raw_hit_L = _lookup_one_l_streaming_causal_end_jit(
            q_keys, k_keys, successor, tau_cap, L=L, sigma=sigma,
        )
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_streaming_causal(
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
    """Exact streaming causal bucket finite-L ROSA suffix lookup.

    Processes positions left-to-right with ``lax.scan``: queries the
    per-key bucket before inserting the current K block, naturally
    enforcing ``j < t``.

    .. warning::
       This function is **only valid when ``cap_end[t] == t``** (strict
       causal).  The ``cap_end`` argument is accepted for API consistency
       but is **not used** by the kernel.  Pass the result of
       :func:`make_raw_causal_aux` or :func:`make_rosa_causal_aux`.

    Parameters
    ----------
    Q, K:
        Integer ``[B, R, T]`` symbol streams.
    cap_end:
        ROSA cap_end tensor.  Must satisfy ``cap_end[t] == t`` for all
        positions for the result to be correct.  (Accepted for API
        consistency; not used internally.)
    successor:
        ROSA successor tensor (see :func:`make_rosa_causal_aux`).
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
    return _lookup_full_l_streaming_causal_jit(
        Q_arr, K_arr, succ, tcap, Lmax=Lmax_i, sigma=sigma_i,
    )
