"""CPU-side candidate generators for use with ``verify_cpu_candidates``.

These functions run on CPU (NumPy) and produce ``cand_end[B,R,T,C]``
arrays that can be fed to the GPU verifier.  They are not JIT-compiled
and are intentionally kept simple.
"""

from __future__ import annotations

import numpy as np


def suffix_array_candidates(
    K: np.ndarray,
    Lmax: int,
    sigma: int,
    C: int = 8,
) -> np.ndarray:
    """Generate candidates via a naive suffix-array-like approach.

    Builds a sorted list of all length-``Lmax`` (or shorter) K suffixes
    and uses binary search to propose candidate end positions for each
    query suffix.

    This is a reference implementation, not optimized for speed.

    Parameters
    ----------
    K:
        ``int64[B, R, T]`` K-symbol stream.
    Lmax:
        Maximum suffix length to consider.
    sigma:
        Alphabet size (unused; kept for API consistency).
    C:
        Maximum candidates per query position.

    Returns
    -------
    cand_end:
        ``int64[B, R, T, C]`` — candidate K end positions, or -1 for
        empty slots.
    """
    K = np.asarray(K, dtype=np.int64)
    B, R, T = K.shape
    cand = np.full((B, R, T, C), -1, dtype=np.int64)

    for b in range(B):
        for r in range(R):
            k_line = K[b, r]
            # Build suffix array entries: (suffix_tuple, end_position)
            entries: list[tuple[tuple, int]] = []
            for j in range(T):
                start = max(0, j - Lmax + 1)
                tup = tuple(int(k_line[p]) for p in range(start, j + 1))
                entries.append((tup, j))
            entries.sort(key=lambda x: x[0])

            keys = [e[0] for e in entries]
            ends = [e[1] for e in entries]

            for t in range(T):
                query_start = max(0, t - Lmax + 1)
                query_tup = tuple(int(k_line[p]) for p in range(query_start, t + 1))

                # Binary search for matching range.
                import bisect
                lo = bisect.bisect_left(keys, query_tup)
                hi = bisect.bisect_right(keys, query_tup)

                count = 0
                for idx in range(hi - 1, lo - 1, -1):
                    j = ends[idx]
                    if j < t and count < C:
                        cand[b, r, t, count] = j
                        count += 1
                    if count >= C:
                        break

    return cand


def brute_force_candidates(
    K: np.ndarray,
    Lmax: int,
    sigma: int = 0,
    C: int = -1,
) -> np.ndarray:
    """Generate all valid candidates via brute-force enumeration.

    Every position j < t where K[j-Lmax+1..j] == Q[t-Lmax+1..t] for
    at least L=1 is included.  With ``C=-1`` (default), all candidates
    are returned (``C=T``).  This gives full recall for the verifier.

    Parameters
    ----------
    K:
        ``int64[B, R, T]`` K-symbol stream.
    Lmax:
        Maximum suffix length.
    sigma:
        Unused.
    C:
        Max candidates per position; -1 means all (T).

    Returns
    -------
    cand_end:
        ``int64[B, R, T, C_eff]``.
    """
    K = np.asarray(K, dtype=np.int64)
    B, R, T = K.shape
    C_eff = T if C < 0 else min(C, T)
    cand = np.full((B, R, T, C_eff), -1, dtype=np.int64)

    for b in range(B):
        for r in range(R):
            k_line = K[b, r]
            for t in range(T):
                count = 0
                for j in range(t):
                    max_len = min(Lmax, t, j + 1)
                    match = True
                    for off in range(max_len):
                        if k_line[t - off] != k_line[j - off]:
                            match = False
                            break
                    if match:
                        cand[b, r, t, count] = j
                        count += 1
                        if count >= C_eff:
                            break
    return cand
