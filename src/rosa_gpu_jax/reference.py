"""Slow reference implementations for tests and debugging.

These functions use NumPy and Python loops.  They are intentionally simple and
should not be used in benchmarks.
"""

from __future__ import annotations

import numpy as np


def _default_tau_cap(shape):
    return np.full(shape, int(shape[-1]), dtype=np.int64)


def brute_force_lookup(Q, K, cap_end, successor, Lmax: int, tau_cap=None):
    """Reference longest-suffix lookup.

    Parameters are NumPy arrays shaped ``[B, R, T]``.  The raw matched end ``j``
    must satisfy ``j < cap_end[b,r,t]``.  The longest/rightmost raw match is
    selected before ``successor[j]`` is checked against ``tau_cap``.  This order
    is required by official ROSA/RLE semantics.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    cap_end = np.asarray(cap_end)
    successor = np.asarray(successor)
    if tau_cap is None:
        tau_cap = _default_tau_cap(Q.shape)
    else:
        tau_cap = np.asarray(tau_cap)

    B, R, T = Q.shape
    tau = np.full((B, R, T), -1, dtype=np.int64)
    best_L = np.zeros((B, R, T), dtype=np.int32)

    for b in range(B):
        for r in range(R):
            for t in range(T):
                max_end = max(0, min(int(cap_end[b, r, t]), T))
                raw_best_L = 0
                raw_best_j = -1
                for L in range(1, Lmax + 1):
                    if t - L + 1 < 0:
                        continue
                    best_j_for_L = -1
                    q_block = Q[b, r, t - L + 1 : t + 1]
                    for j in range(max_end):
                        if j - L + 1 < 0:
                            continue
                        if np.array_equal(q_block, K[b, r, j - L + 1 : j + 1]):
                            best_j_for_L = j
                    if best_j_for_L >= 0:
                        raw_best_L = L
                        raw_best_j = best_j_for_L
                if raw_best_L > 0:
                    tau_raw = int(successor[b, r, raw_best_j])
                    if tau_raw >= 0 and tau_raw <= int(tau_cap[b, r, t]):
                        best_L[b, r, t] = raw_best_L
                        tau[b, r, t] = tau_raw

    return tau, best_L


def brute_force_candidate_verify(Q, K, cand_end, Lmax: int, cap_end=None, successor=None, tau_cap=None):
    """Reference candidate verification with optional custom causal tensors.

    If ``cap_end`` and ``successor`` are omitted, raw-token semantics are used:
    ``j < t`` and ``tau = j + 1``.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    cand_end = np.asarray(cand_end)

    B, R, T = Q.shape
    C = cand_end.shape[-1]

    if cap_end is None or successor is None:
        if cap_end is not None or successor is not None:
            raise ValueError("cap_end and successor must be provided together")
        cap_end = np.broadcast_to(np.arange(T, dtype=np.int64), (B, R, T))
        successor = np.broadcast_to(np.arange(1, T + 1, dtype=np.int64), (B, R, T))
    else:
        cap_end = np.asarray(cap_end)
        successor = np.asarray(successor)
    if tau_cap is None:
        tau_cap = _default_tau_cap(Q.shape)
    else:
        tau_cap = np.asarray(tau_cap)

    tau = np.full((B, R, T), -1, dtype=np.int64)
    best_L = np.zeros((B, R, T), dtype=np.int32)

    for b in range(B):
        for r in range(R):
            for t in range(T):
                raw_best_len = 0
                raw_best_j = -1
                cap = max(0, min(int(cap_end[b, r, t]), T))
                for c in range(C):
                    j = int(cand_end[b, r, t, c])
                    if j < 0 or j >= cap or j >= T:
                        continue
                    length = 0
                    for off in range(Lmax):
                        qi = t - off
                        ki = j - off
                        if qi < 0 or ki < 0:
                            break
                        if Q[b, r, qi] != K[b, r, ki]:
                            break
                        length += 1
                    if length > raw_best_len or (length == raw_best_len and j > raw_best_j):
                        raw_best_len = length
                        raw_best_j = j
                if raw_best_len > 0:
                    tau_raw = int(successor[b, r, raw_best_j])
                    if tau_raw >= 0 and tau_raw <= int(tau_cap[b, r, t]):
                        best_L[b, r, t] = raw_best_len
                        tau[b, r, t] = tau_raw
    return tau, best_L


class _SAM:
    __slots__ = ("next", "link", "length", "last", "e")

    def __init__(self):
        self.next = [{}]
        self.link = [-1]
        self.length = [0]
        self.last = 0
        self.e = [-1]

    def _new_state(self, length: int) -> int:
        self.next.append({})
        self.link.append(-1)
        self.length.append(length)
        self.e.append(-1)
        return len(self.next) - 1

    def extend(self, x: int, pos: int):
        cur = self._new_state(self.length[self.last] + 1)
        p = self.last
        while p != -1 and x not in self.next[p]:
            self.next[p][x] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][x]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self._new_state(self.length[p] + 1)
                self.next[clone] = self.next[q].copy()
                self.link[clone] = self.link[q]
                self.e[clone] = self.e[q]
                while p != -1 and self.next[p].get(x, None) == q:
                    self.next[p][x] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        v = cur
        while v != -1 and self.e[v] != pos:
            self.e[v] = pos
            v = self.link[v]
        self.last = cur

    def match_next(self, x: int) -> int:
        p = self.last
        while p != -1 and x not in self.next[p]:
            p = self.link[p]
        if p == -1:
            return -1
        return self.next[p][x]


def rosa_one_sequence_reference_tau(z) -> np.ndarray:
    """Official native ROSA recurrence for one raw sequence, returning tau.

    This mirrors the public Native ROSA code but returns the start position of
    the successor run instead of the successor symbol value.
    """
    z = [int(x) for x in z]
    y = np.full((len(z),), -1, dtype=np.int64)
    sam = _SAM()
    run_starts: list[int] = []
    last_token = None
    for t, x in enumerate(z):
        v = sam.match_next(x)
        if v != -1 and sam.e[v] >= 0:
            j_c = sam.e[v]
            if j_c + 1 < len(run_starts):
                y[t] = run_starts[j_c + 1]
        if last_token is None or x != last_token:
            run_starts.append(t)
            last_token = x
        sam.extend(x, pos=len(run_starts) - 1)
    return y


def rosa_batch_reference_tau(Z) -> np.ndarray:
    """Apply ``rosa_one_sequence_reference_tau`` to ``[B,R,T]`` input."""
    Z = np.asarray(Z)
    B, R, T = Z.shape
    out = np.full((B, R, T), -1, dtype=np.int64)
    for b in range(B):
        for r in range(R):
            out[b, r] = rosa_one_sequence_reference_tau(Z[b, r])
    return out


# ---------------------------------------------------------------------------
# Reference: streaming diagonal-DP
# ---------------------------------------------------------------------------


def diag_dp_reference(Q, K, cap_end, successor, Lmax: int, tau_cap=None):
    """Streaming diagonal-DP reference for tests.

    Maintains only the previous row ``D_prev[T]``, reducing memory from
    ``O(T²)`` (dense DP) to ``O(T)``.  Semantics are identical to
    ``brute_force_lookup``.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    cap_end = np.asarray(cap_end)
    successor = np.asarray(successor)
    if tau_cap is None:
        tau_cap = _default_tau_cap(Q.shape)
    else:
        tau_cap = np.asarray(tau_cap)

    B, R, T = Q.shape
    tau = np.full((B, R, T), -1, dtype=np.int64)
    best_L_out = np.zeros((B, R, T), dtype=np.int32)

    for b in range(B):
        for r in range(R):
            q_line = Q[b, r]
            k_line = K[b, r]
            cap_line = cap_end[b, r]
            succ_line = successor[b, r]
            tc_line = tau_cap[b, r]

            # D_prev[j] = length of longest suffix of Q[0..t-1] ending at K[j]
            D_prev = np.zeros(T, dtype=np.int32)

            for t in range(T):
                # Shift D_prev right by 1, pad with 0 at position 0.
                D_curr = np.zeros(T, dtype=np.int32)
                D_curr[1:] = D_prev[:-1]
                eq = (q_line[t] == k_line).astype(np.int32)
                D_curr = np.where(eq, D_curr + 1, 0)

                # Clamp to Lmax.
                D_curr = np.minimum(D_curr, np.int32(Lmax))

                # Select raw best: longest length, then rightmost j.
                cap_t = max(0, min(int(cap_line[t]), T))
                best_len = 0
                best_j = -1
                for j in range(cap_t):
                    lj = int(D_curr[j])
                    if lj > best_len:
                        best_len = lj
                        best_j = j
                    elif lj == best_len and lj > 0 and j > best_j:
                        best_j = j

                if best_len > 0:
                    tau_raw = int(succ_line[best_j])
                    if tau_raw >= 0 and tau_raw <= int(tc_line[t]):
                        best_L_out[b, r, t] = best_len
                        tau[b, r, t] = tau_raw

                D_prev = D_curr

    return tau, best_L_out


# ---------------------------------------------------------------------------
# Reference: Shift-And bitset ROSA (using Python integers as bitsets)
# ---------------------------------------------------------------------------


def shift_and_reference(Q, K, cap_end, successor, Lmax: int, sigma: int, tau_cap=None):
    """Shift-And bitset reference for tests.

    Uses Python arbitrary-precision integers as bitsets.  For each symbol
    ``a`` builds ``P_a`` (bit j is 1 iff ``K[j] == a``).  Then iterates
    ``L = 1..Lmax`` maintaining ``M_L(t)`` — the bitset of K positions
    whose length-L suffix matches Q ending at position ``t``.

    The rightmost set bit of ``M_L(t) & prefix_mask(cap_end[t])`` is the
    raw match end ``j``.  Every semantic detail (longest first, rightmost
    tie-break, ROSA successor gating, no backtracking) is reproduced.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    cap_end = np.asarray(cap_end)
    successor = np.asarray(successor)
    if tau_cap is None:
        tau_cap = _default_tau_cap(Q.shape)
    else:
        tau_cap = np.asarray(tau_cap)

    B, R, T = Q.shape
    tau = np.full((B, R, T), -1, dtype=np.int64)
    best_L_out = np.zeros((B, R, T), dtype=np.int32)

    for b in range(B):
        for r in range(R):
            q_line = Q[b, r]
            k_line = K[b, r]
            cap_line = cap_end[b, r]
            succ_line = successor[b, r]
            tc_line = tau_cap[b, r]

            # Build per-symbol position masks.
            P = [0] * sigma
            for j in range(T):
                sym = int(k_line[j])
                if 0 <= sym < sigma:
                    P[sym] |= 1 << j

            # cap_masks[t] = bits 0..cap_end[t]-1 set to 1.
            cap_masks = []
            for t in range(T):
                c = max(0, min(int(cap_line[t]), T))
                cap_masks.append((1 << c) - 1 if c > 0 else 0)

            # M_prev[t] = bitset for length-L match ending at t.
            # We store only the previous-t M vector; current t depends on
            # M_prev[t-1].
            M_prev = [0] * T  # length-0: no match (all zeros)

            raw_best_L = np.zeros(T, dtype=np.int32)
            raw_best_j = np.full(T, -1, dtype=np.int32)

            for L in range(1, Lmax + 1):
                M_curr = [0] * T
                for t in range(T):
                    q_sym = int(q_line[t])
                    if q_sym < 0 or q_sym >= sigma:
                        continue
                    P_q = P[q_sym]
                    if L == 1:
                        M_curr[t] = P_q
                    else:
                        if t > 0 and M_prev[t - 1] != 0:
                            # Shift left by 1: position j in M_prev corresponds
                            # to a length-(L-1) match ending at K[j-1],
                            # so a length-L match can end at K[j].
                            shifted = (M_prev[t - 1] << 1)
                            M_curr[t] = shifted & P_q

                    if M_curr[t] != 0:
                        # Rightmost valid j.
                        masked = M_curr[t] & cap_masks[t]
                        if masked != 0:
                            j = masked.bit_length() - 1  # highest set bit
                            if L > raw_best_L[t]:
                                raw_best_L[t] = L
                                raw_best_j[t] = j
                            elif L == raw_best_L[t] and j > raw_best_j[t]:
                                raw_best_j[t] = j

                M_prev = M_curr

            # ROSA gating on accumulated best.
            for t in range(T):
                if raw_best_L[t] > 0 and raw_best_j[t] >= 0:
                    tau_raw = int(succ_line[raw_best_j[t]])
                    if tau_raw >= 0 and tau_raw <= int(tc_line[t]):
                        best_L_out[b, r, t] = raw_best_L[t]
                        tau[b, r, t] = tau_raw

    return tau, best_L_out
