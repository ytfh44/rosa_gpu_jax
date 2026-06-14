"""Auxiliary tensors for ROSA-style causal lookup."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

NEG = jnp.int64(-1)


def make_raw_causal_aux(B: int, R: int, T: int):
    """Create raw-token causal fallback tensors.

    ``cap_end[b,r,t] = t`` and ``successor[b,r,j] = j + 1``.  This is a simple
    token-level fallback, not the official ROSA/RLE successor rule.
    """
    cap_end = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int64), (B, R, T))
    successor = jnp.broadcast_to(jnp.arange(1, T + 1, dtype=jnp.int64), (B, R, T))
    return cap_end, successor


def make_rosa_causal_aux(K):
    """Create official ROSA/RLE lookup auxiliaries from raw K symbols.

    Parameters
    ----------
    K:
        Integer ``[B, R, T]`` K-symbol stream.

    Returns
    -------
    cap_end:
        ``int64[B,R,T]`` with raw online causality: candidate K end ``j`` must
        satisfy ``j < t``.
    successor:
        ``int64[B,R,T]`` mapping any matched raw K end position to the start
        position of the next K run, or ``-1`` if no next run exists offline.
    tau_cap:
        ``int64[B,R,T]`` equal to the start position of the current K run at
        time ``t``.  After choosing the longest/rightmost match, ROSA accepts
        ``tau = successor[j]`` only when ``0 <= tau <= tau_cap[b,r,t]``.

    Notes
    -----
    This matches the public ROSA rule ``nxt = rpos + 1`` and
    ``tau = s[nxt]`` only if ``nxt <= rcap(t)``.  The post-successor cap is
    essential: ROSA does not backtrack to an older occurrence when the
    rightmost matched occurrence has no valid successor yet.
    """
    arr = np.asarray(K)
    if arr.ndim != 3:
        raise ValueError(f"K must be rank-3 shaped [B, R, T]; got K.shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"K must have an integer dtype; got {arr.dtype}")
    B, R, T = arr.shape
    if B <= 0 or R <= 0 or T <= 0:
        raise ValueError(f"K must have non-empty [B, R, T] dimensions; got {arr.shape}")

    cap_end = np.broadcast_to(np.arange(T, dtype=np.int64), (B, R, T)).copy()
    successor = np.full((B, R, T), -1, dtype=np.int64)
    tau_cap = np.zeros((B, R, T), dtype=np.int64)

    for b in range(B):
        for r in range(R):
            z = arr[b, r]
            starts: list[int] = []
            run_id = np.empty(T, dtype=np.int64)
            current = -1
            last = None
            for t, x in enumerate(z):
                if t == 0 or x != last:
                    starts.append(t)
                    current += 1
                    last = x
                run_id[t] = current
            starts_np = np.asarray(starts, dtype=np.int64)
            tau_cap[b, r] = starts_np[run_id]
            for j in range(T):
                rid = int(run_id[j])
                if rid + 1 < len(starts):
                    successor[b, r, j] = starts[rid + 1]

    return jnp.asarray(cap_end), jnp.asarray(successor), jnp.asarray(tau_cap)
