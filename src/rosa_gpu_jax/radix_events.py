"""Exact ROSA lookup via sort-scan event streams.

This module implements the radix-sort family sketched by the block-table and
postings paths: every K position is an update event, every Q position is a
query event, and sorting by ``(block_key, causal_coordinate, event_type)`` turns
rightmost-predecessor lookup into a segmented prefix scan.

The implementation deliberately keeps the sort backend as ``jax.numpy.lexsort``.
On GPU, XLA lowers fixed-width integer sorts to device sort kernels; a custom
Pallas radix sort can replace only this sorting primitive without changing the
semantic scan/reduction code below.  A Pallas import/probe helper is provided so
callers can decide whether to route future kernels through Pallas.  On CPU,
Pallas only supports interpret mode in current JAX builds, so correctness tests
exercise the JAX path.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

try:  # pragma: no cover - environment capability probe
    from jax.experimental import pallas as pl  # noqa: F401

    PALLAS_IMPORTABLE = True
except Exception:  # pragma: no cover
    PALLAS_IMPORTABLE = False

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


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("L",))
def _block_keys_bitpack_jit(seq, L: int):
    """Pack binary length-L suffix blocks into uint64 keys.

    The newest/current symbol is the least significant bit, matching the base
    encoder's convention that Q[t] has weight sigma**0.  Positions ``t < L-1``
    are padded with zero and must be masked by lookup code.
    """
    seq_u = seq.astype(jnp.uint64)
    T = seq.shape[-1]
    valid = jnp.zeros(seq.shape[:-1] + (T - L + 1,), dtype=jnp.uint64)
    for i in range(L):
        # seq[..., start+i] is offset L-1-i from the current end, so it gets
        # bit weight L-1-i.  Current symbol i=L-1 has weight 0.
        shift = jnp.asarray(L - 1 - i, dtype=jnp.uint64)
        valid = valid | (seq_u[..., i : T - L + 1 + i] << shift)
    pad = jnp.zeros(seq.shape[:-1] + (L - 1,), dtype=jnp.uint64)
    return jnp.concatenate([pad, valid], axis=-1)


def pallas_available() -> bool:
    """Return whether Pallas can be imported in the current JAX runtime."""

    return bool(PALLAS_IMPORTABLE)


# ---------------------------------------------------------------------------
# Event scans for one line [T]
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("L",))
def _event_predecessor_one_l_line_jit(q_keys, k_keys, cap, L: int):
    """Return rightmost raw K end for each Q position for one line and L.

    This is the semantic target for a future custom Pallas radix sort:

    * K events use coordinate ``j``.
    * Q events use coordinate ``cap_end[t]``.
    * Sort key is ``(block_key, coordinate, event_type)`` with Q before K.
      Therefore a Q event at coordinate c sees exactly K events with j < c.
    * A segmented prefix max over K payloads returns the rightmost predecessor.
    """

    T = q_keys.shape[-1]
    pos = jnp.arange(T, dtype=jnp.int32)

    event_key = jnp.concatenate([k_keys, q_keys], axis=0)
    event_coord = jnp.concatenate([pos.astype(jnp.int64), cap.astype(jnp.int64)], axis=0)
    # Q must sort before K at equal coordinate so the causal relation is j < cap.
    event_type = jnp.concatenate(
        [jnp.ones((T,), dtype=jnp.int32), jnp.zeros((T,), dtype=jnp.int32)], axis=0
    )
    event_payload = jnp.concatenate([pos, pos], axis=0)
    event_is_k = event_type == 1
    event_is_q = event_type == 0
    event_k_valid = event_is_k & (event_payload >= (L - 1))
    event_q_valid = event_is_q & (event_payload >= (L - 1))

    # lexsort uses the last key as the primary key, so this sorts by
    # key -> coord -> type.
    order = jnp.lexsort((event_type, event_coord, event_key))
    key_s = event_key[order]
    payload_s = event_payload[order]
    is_k_s = event_is_k[order]
    is_q_s = event_is_q[order]
    k_valid_s = event_k_valid[order]
    q_valid_s = event_q_valid[order]

    single_val = jnp.where(is_k_s & k_valid_s, payload_s, jnp.int32(-1))

    def combine(a, b):
        key_a, val_a = a
        key_b, val_b = b
        same = key_a == key_b
        return key_b, jnp.where(same, jnp.maximum(val_a, val_b), val_b)

    _scan_key, scan_val = jax.lax.associative_scan(combine, (key_s, single_val))
    out_val = jnp.where(is_q_s & q_valid_s, scan_val, jnp.int32(-1))
    out_ext = jnp.full((T + 1,), jnp.int32(-1))
    scatter_idx = jnp.where(is_q_s, payload_s, jnp.int32(T))
    scatter_val = jnp.where(is_q_s, out_val, jnp.int32(-1))
    out_ext = out_ext.at[scatter_idx].max(scatter_val)
    return out_ext[:T]


@partial(jax.jit, static_argnames=("L", "C"))
def _event_postings_one_l_line_jit(q_keys, k_keys, cap, L: int, C: int):
    """Return C rightmost raw K ends per Q position for one line and L."""

    T = q_keys.shape[-1]
    pos = jnp.arange(T, dtype=jnp.int32)
    event_key = jnp.concatenate([k_keys, q_keys], axis=0)
    event_coord = jnp.concatenate([pos.astype(jnp.int64), cap.astype(jnp.int64)], axis=0)
    event_type = jnp.concatenate(
        [jnp.ones((T,), dtype=jnp.int32), jnp.zeros((T,), dtype=jnp.int32)], axis=0
    )
    event_payload = jnp.concatenate([pos, pos], axis=0)
    event_is_k = event_type == 1
    event_is_q = event_type == 0
    event_k_valid = event_is_k & (event_payload >= (L - 1))
    event_q_valid = event_is_q & (event_payload >= (L - 1))

    order = jnp.lexsort((event_type, event_coord, event_key))
    key_s = event_key[order]
    payload_s = event_payload[order]
    is_k_s = event_is_k[order]
    is_q_s = event_is_q[order]
    k_valid_s = event_k_valid[order]
    q_valid_s = event_q_valid[order]

    base_vec = jnp.full((2 * T, C), jnp.int32(-1))
    base_vec = base_vec.at[:, 0].set(
        jnp.where(is_k_s & k_valid_s, payload_s, jnp.int32(-1))
    )

    def combine(a, b):
        key_a, vec_a = a
        key_b, vec_b = b
        same = key_a == key_b
        merged = jnp.sort(jnp.concatenate([vec_a, vec_b], axis=-1), axis=-1)[:, ::-1][:, :C]
        return key_b, jnp.where(same[:, None], merged, vec_b)

    _scan_key, scan_vec = jax.lax.associative_scan(combine, (key_s, base_vec))
    out_vec = jnp.where((is_q_s & q_valid_s)[:, None], scan_vec, jnp.int32(-1))
    out_ext = jnp.full((T + 1, C), jnp.int32(-1))
    scatter_idx = jnp.where(is_q_s, payload_s, jnp.int32(T))
    scatter_val = jnp.where(is_q_s[:, None], out_vec, jnp.int32(-1))
    out_ext = out_ext.at[scatter_idx].max(scatter_val)
    return out_ext[:T]


@partial(jax.jit, static_argnames=("L",))
def _event_predecessor_one_l_jit(q_keys, k_keys, cap_end, L: int):
    return jax.vmap(
        jax.vmap(_event_predecessor_one_l_line_jit, in_axes=(0, 0, 0, None)),
        in_axes=(0, 0, 0, None),
    )(q_keys, k_keys, cap_end, L)


@partial(jax.jit, static_argnames=("L", "C"))
def _event_postings_one_l_jit(q_keys, k_keys, cap_end, L: int, C: int):
    return jax.vmap(
        jax.vmap(_event_postings_one_l_line_jit, in_axes=(0, 0, 0, None, None)),
        in_axes=(0, 0, 0, None, None),
    )(q_keys, k_keys, cap_end, L, C)


# ---------------------------------------------------------------------------
# Public exact predecessor path
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax", "sigma", "key_mode"))
def _lookup_full_l_radix_events_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int, key_mode: str):
    B, R, T = Q.shape
    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        if key_mode == "bitpack":
            q_keys = _block_keys_bitpack_jit(Q, L=L)
            k_keys = _block_keys_bitpack_jit(K, L=L)
        else:
            q_keys = _block_keys_base_jit(Q, L=L, sigma=sigma)
            k_keys = _block_keys_base_jit(K, L=L, sigma=sigma)
        end_L = _event_predecessor_one_l_jit(q_keys, k_keys, cap_end, L=L).astype(jnp.int64)
        raw_hit_L = end_L >= 0
        # Since L increases monotonically, any raw hit at this L beats shorter L.
        best_end = jnp.where(raw_hit_L, end_L, best_end)
        best_L_raw = jnp.where(raw_hit_L, jnp.int32(L), best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_radix_events(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    sigma: int,
    *,
    tau_cap=None,
    key_mode: str = "base",
    validate_symbols: bool = True,
):
    """Exact finite-L ROSA lookup via sorted K/Q event streams.

    Parameters
    ----------
    key_mode:
        ``"base"`` uses exact base-sigma keys and inherits the same uint64
        safety bound as ``lookup_full_l_base``.  ``"bitpack"`` is for binary
        streams only and supports ``Lmax <= 63``.
    """

    sigma_i = require_sigma(sigma)
    if key_mode not in {"base", "bitpack"}:
        raise ValueError(f"key_mode must be 'base' or 'bitpack'; got {key_mode!r}")
    if key_mode == "bitpack" and sigma_i != 2:
        raise ValueError("key_mode='bitpack' requires sigma=2")
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=sigma_i, validate_symbols=validate_symbols)
    Lmax_i = require_Lmax_for_T(Lmax, T)
    if key_mode == "bitpack" and Lmax_i > 63:
        raise OverflowError("bitpack radix events support Lmax <= 63 in one uint64 key")
    if key_mode == "base":
        for L in range(1, Lmax_i + 1):
            ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_radix_events_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i, key_mode=key_mode
    )


# ---------------------------------------------------------------------------
# Public C-postings path
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("Lmax",))
def _verify_candidates_jit(Q, K, cand_end, cap_end, Lmax: int):
    """Verify candidate K end positions symbol-by-symbol and select best.

    Unlike ``candidates._verify_cpu_candidates_jit``, this function does **not**
    apply successor/tau_cap gating.  Gating is deferred to the full-L loop
    so that a longer raw match at one L is not discarded before shorter L
    values are considered.
    """
    B, R, T = Q.shape
    offsets = jnp.arange(Lmax, dtype=jnp.int32)

    def line_verify(q, k, cand, cap):
        t_idx = jnp.arange(T, dtype=jnp.int32)[:, None, None]
        j_idx = cand.astype(jnp.int32)[:, :, None]
        cap_t = cap[:, None, None]
        o = offsets[None, None, :]
        q_idx = t_idx - o
        k_idx = j_idx - o
        valid = (cand[:, :, None] >= 0) & (q_idx >= 0) & (k_idx >= 0) & (j_idx < cap_t)
        q_tok = q[jnp.clip(q_idx, 0, T - 1)]
        k_tok = k[jnp.clip(k_idx, 0, T - 1)]
        eq = valid & (q_tok == k_tok)
        all_match = jnp.all(eq, axis=-1)
        first_false = jnp.argmin(eq.astype(jnp.int32), axis=-1).astype(jnp.int32)
        lens = jnp.where(all_match, jnp.int32(Lmax), first_false)
        j_nonneg = jnp.maximum(cand.astype(jnp.int32), 0)
        score = lens.astype(jnp.int64) * jnp.asarray(T + 1, dtype=jnp.int64) + j_nonneg
        score = jnp.where(cand >= 0, score, jnp.int64(-1))
        best_c = jnp.argmax(score, axis=-1).astype(jnp.int32)
        best_j = jnp.take_along_axis(cand, best_c[:, None], axis=-1)[:, 0].astype(jnp.int32)
        best_len_raw = jnp.take_along_axis(lens, best_c[:, None], axis=-1)[:, 0]
        return best_j.astype(jnp.int64), best_len_raw.astype(jnp.int32)

    return jax.vmap(
        jax.vmap(line_verify, in_axes=(0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0),
    )(Q, K, cand_end, cap_end)


@partial(jax.jit, static_argnames=("Lmax", "sigma", "key_mode", "C"))
def _lookup_full_l_radix_postings_jit(Q, K, cap_end, successor, tau_cap, Lmax: int, sigma: int, key_mode: str, C: int):
    B, R, T = Q.shape
    best_end = jnp.full((B, R, T), jnp.int64(-1), dtype=jnp.int64)
    best_L_raw = jnp.zeros((B, R, T), dtype=jnp.int32)

    for L in range(1, Lmax + 1):
        if key_mode == "bitpack":
            q_keys = _block_keys_bitpack_jit(Q, L=L)
            k_keys = _block_keys_bitpack_jit(K, L=L)
        else:
            q_keys = _block_keys_base_jit(Q, L=L, sigma=sigma)
            k_keys = _block_keys_base_jit(K, L=L, sigma=sigma)
        cand_L = _event_postings_one_l_jit(q_keys, k_keys, cap_end, L=L, C=C).astype(jnp.int64)
        end_raw_L, len_raw_L = _verify_candidates_jit(
            Q, K, cand_L, cap_end, Lmax=Lmax
        )
        pos_t = jnp.arange(T, dtype=jnp.int32)
        raw_hit_L = (
            (len_raw_L > 0)
            & (pos_t[None, None, :] >= (L - 1))
            & (end_raw_L >= (L - 1))
        )
        better = raw_hit_L & (
            (len_raw_L > best_L_raw) | ((len_raw_L == best_L_raw) & (end_raw_L > best_end))
        )
        best_end = jnp.where(better, end_raw_L, best_end)
        best_L_raw = jnp.where(better, len_raw_L, best_L_raw)

    end_safe = jnp.clip(best_end, 0, T - 1)
    tau_raw = jnp.take_along_axis(successor, end_safe, axis=-1)
    final_hit = (best_L_raw > 0) & (tau_raw >= 0) & (tau_raw <= tau_cap)
    best_tau = jnp.where(final_hit, tau_raw, NEG).astype(jnp.int64)
    best_L = jnp.where(final_hit, best_L_raw, jnp.int32(0))
    return best_tau, best_L


def lookup_full_l_radix_postings(
    Q,
    K,
    cap_end,
    successor,
    Lmax: int,
    sigma: int,
    *,
    C: int = 4,
    tau_cap=None,
    key_mode: str = "base",
    validate_symbols: bool = True,
):
    """C-postings radix-event lookup with exact raw-symbol verification.

    This path is exact when the true raw predecessor is retained in the C
    postings collected for at least one anchor length.  ``C >= T`` is a simple
    correctness setting for tests and small algorithmic tasks.
    """

    sigma_i = require_sigma(sigma)
    if not isinstance(C, int) or isinstance(C, bool) or C < 1:
        raise ValueError(f"C must be a positive Python int; got {C!r}")
    if key_mode not in {"base", "bitpack"}:
        raise ValueError(f"key_mode must be 'base' or 'bitpack'; got {key_mode!r}")
    if key_mode == "bitpack" and sigma_i != 2:
        raise ValueError("key_mode='bitpack' requires sigma=2")
    Q_arr, K_arr, _B, _R, T = require_rank3_pair(Q, K, sigma=sigma_i, validate_symbols=validate_symbols)
    Lmax_i = require_Lmax_for_T(Lmax, T)
    if key_mode == "bitpack" and Lmax_i > 63:
        raise OverflowError("bitpack radix postings support Lmax <= 63 in one uint64 key")
    if key_mode == "base":
        for L in range(1, Lmax_i + 1):
            ensure_exact_key_safe(sigma=sigma_i, L=L, T=T)
    cap, succ = require_aux(cap_end, successor, shape=tuple(Q_arr.shape))
    tcap = require_tau_cap(tau_cap, shape=tuple(Q_arr.shape))
    return _lookup_full_l_radix_postings_jit(
        Q_arr, K_arr, cap, succ, tcap, Lmax=Lmax_i, sigma=sigma_i, key_mode=key_mode, C=int(C)
    )
