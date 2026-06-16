# rosa-gpu-jax

JAX research prototypes for GPU-friendly ROSA-style suffix lookup.

This package is not a full ROSA implementation. It isolates several GPU-feasible indexing paths for symbol-level retrieval:

1. full-L exact block table lookup within `L <= Lmax` (base-σ encoding, sort + binary search);
2. counting-prefix-table exact lookup (dense direct-address, tiny alphabet only);
3. streaming causal bucket exact lookup (online `lax.scan`, causal-only);
4. rolling-hash block table lookup for larger `Lmax` throughput experiments;
5. verified rolling-hash lookup with multi-slot hash table;
6. fixed-C postings candidate generator (exact base and rolling-hash variants);
7. CPU-candidate + GPU suffix verification;
8. dyadic-rank postings + binary-lifting LCE lookup (exact, sigma-free);
9. suffix-array binary search lookup (exact, sigma-free);
10. dense equality DP exact baseline (small-`T` oracle);
11. streaming diagonal-DP lookup (reduced-memory exact oracle);
12. TPU dense one-hot DP (systolic-array throughput experiments);
13. bitset exact suffix lookup (experimental, small-`T` only);
14. Shift-And bitset exact suffix lookup (no sort, no hash, no overflow);
15. Q-bit counterfactual destination lookup for the current query symbol.

The package enables JAX x64 at import time, because exact base keys and predecessor-search scores use int64/uint64. The code assumes the core input is already a run-level or token-level symbol stream:

```python
Q, K: int64[B, R, T]
```

where `B` is batch, `R` is route, and `T` is sequence or run length. If your tensors are `[B, T, R]`, transpose them first:

```python
Q = jnp.transpose(Q_sym, (0, 2, 1))
K = jnp.transpose(K_sym, (0, 2, 1))
```

## Installation

CPU-only development:

```bash
pip install -e .[dev]
```

CUDA install depends on the local driver/CUDA stack. For a standard CUDA 12 JAX setup:

```bash
pip install -e .[dev,cuda12]
```

When using a cluster-provided JAX build, install the package without dependencies and keep the cluster JAX environment:

```bash
pip install -e . --no-deps
```

## Semantic contract

The public ROSA-Tuning description defines ROSA as an online suffix-automaton retrieval rule over discrete symbols.  For each route, K symbols induce run starts:

```text
s_0 = 0
a^K_t != a^K_{t-1}  =>  s_{l+1} = t
rcap(t) = max l such that s_l <= t
nxt = rpos + 1
tau = s_nxt if match succeeds and nxt <= rcap(t), else -1
```

This order matters.  ROSA first selects the longest suffix match and, for equal length, the rightmost matched occurrence.  It then maps the matched occurrence to the next K run.  If that successor is not yet available under `rcap(t)`, the result is `-1`; it must not backtrack to an older occurrence with a valid successor.

This package represents the rule with three auxiliary tensors:

```python
cap_end[b, r, t]
```

A raw matched K end position `j` must satisfy `j < cap_end[b,r,t]`.  For official online ROSA over raw K tokens, this is simply `t`.

```python
successor[b, r, j]
```

This maps a raw matched K end position to the start position of the next K run.  If no next run exists offline, it is `-1`.

```python
tau_cap[b, r, t]
```

This is the post-successor cap.  Official ROSA accepts `tau = successor[b,r,j]` only if `0 <= tau <= tau_cap[b,r,t]`.  For raw K tokens, `tau_cap` is the start position of the current K run.

Use `make_rosa_causal_aux(K)` to build these tensors from raw K symbols:

```python
from rosa_gpu_jax import make_rosa_causal_aux, lookup_full_l_base

cap_end, successor, tau_cap = make_rosa_causal_aux(K)
tau, match_len = lookup_full_l_base(
    Q, K, cap_end, successor, Lmax=8, sigma=16, tau_cap=tau_cap
)
```

`make_raw_causal_aux(B, R, T)` is still available as a token-level fallback.  It uses `j < t` and `tau = j + 1`.  It is not the official ROSA/RLE successor rule.

---

## Base-σ encoding methods

These methods encode length-*L* suffix blocks as base-σ integers. They are **exact** (no hash collisions, no false positives/negatives), but are limited to `Lmax` values where σ^L · (T + 1) + T ≤ 2^64 − 1 (uint64 safe). They require a known alphabet size `sigma`.

### Method 1: full-L exact block table

Use `lookup_full_l_base`.

#### Encoding

Every length-*L* suffix block ending at position *t* is encoded as a base-σ
integer via polynomial evaluation:

```
key_L(t) = Σ_{k=0}^{L−1}  seq[t−k] · σ^{L−1−k}
```

This is a bijection from length-*L* symbol tuples to integers in
[0, σ^L − 1], assuming 0 ≤ symbol < σ.  Precomputed powers σ^{L−1}, …, σ^0
make evaluation a dot-product over the last *L* symbols.

#### Predecessor search

To locate the rightmost K position whose block key matches the query, each
key is combined with its position:

```
combined(t) = key_L(t) · (T + 1) + t
```

The K-side combined keys are sorted.  For each query position *t*, a binary
search (`searchsorted`) finds the rightmost K combined key below
`key_L(t) · (T + 1) + cap_end[t]`.  The corresponding K position is the raw
match end *j*.

#### ROSA tie-breaking

The ROSA rule prefers the longest suffix match; for equal length, the
rightmost occurrence.  The block-table path iterates *L* = 1, 2, …, Lmax,
accumulating the raw match with largest *L* (breaking ties by largest *j*).
ROSA successor/tau_cap gating is applied once at the end.

#### Properties

- **Exact** for all *L* ≤ Lmax when σ^L · (T + 1) + T ≤ 2^64 − 1 (uint64
  safe).  This caps Lmax at roughly ⌊64 / log₂ σ⌋.
- **Complexity** O(B·R·Lmax·T log T) — per-level sort plus binary search
  over T positions.
- Public APIs validate uint64 safety and raise `OverflowError` before JIT
  when constraints are violated.
- For *M* = 4, σ = 16, Lmax = 4 is a more informative starting point than
  sparse grids like {1, 2, 4, 8}, because length-3 matches are common at
  long context length.

### Method 12: counting-prefix-table lookup

Use `lookup_full_l_counting_prefix`.

#### Principle

Instead of sorting combined keys (Method 1), this method builds a dense
direct-address table `table[key, pos+1]` that records the rightmost
occurrence of each base-encoded block key.  A single `cummax` pass across
the position axis turns it into a prefix-max table, so that:

```
prefix[q_key, cap] = rightmost K end position < cap with the same block key
```

Each query is then a single O(1) table lookup — no binary search required.

#### Build

For each K position *t* (with *t* ≥ L−1):

```
key   = key_L(t)                          ∈ [0, σ^L − 1]
col   = t + 1                             (1-based column)
table[key, col]  ←  max(table[key, col], t)
```

Then `prefix = cummax(table, axis=1)` so that `prefix[k, c]` gives the
rightmost position with key *k* whose column is ≤ *c*.

#### Query

For a query position *t* with key *q_key* and cap *cap_end[t]*:

```
j = prefix[q_key, cap_end[t]]
```

The match is valid if `j >= 0` and `t >= L-1`.

#### Properties

- **Exact** — no hash collisions, no false positives, no false negatives.
- **O(σ^L · T) memory** — only viable for very small alphabets and moderate
  Lmax (e.g. σ ≤ 4, Lmax ≤ 4).  The table is built per (b,r) line inside
  `vmap`.
- **O(B·R·Lmax·(T + σ^L)) time** — asymptotically optimal for exact
  base-key lookup, replacing the O(T log T) sort with a dense table build.
- Compared to Method 1: trades memory for lower per-position work when σ^L
  is tiny.  At σ = 4, Lmax = 4, the keyspace is only 256 entries.

### Method 13: streaming causal bucket lookup

Use `lookup_full_l_streaming_causal`.

#### Principle

An online, O(T) per (b,r) bucket lookup that processes positions
left-to-right with `lax.scan`.  It queries the per-key bucket **before**
inserting the current K block, naturally enforcing `j < t` without explicit
masking.

#### Scan step

For each position *t*:

1. **Query:** look up `table[q_key]` — this returns the rightmost K position
   inserted so far (all < *t*), giving the raw match end *j*.
2. **Gate:** apply ROSA successor/tau_cap at position *t*.
3. **Insert:** write *t* into `table[k_key]` for future queries.

The scan state is a single 1-D array of size σ^L, initialized to −1.

#### Constraint

This kernel is **only valid when `cap_end[t] == t`** (strict causal).
Arbitrary `cap_end` is not supported — the `cap_end` argument is accepted
for API consistency but is not used internally.  Pass the result of
`make_raw_causal_aux` or `make_rosa_causal_aux`.

#### Properties

- **Exact** — no hash collisions, no false positives, no false negatives.
- **Streaming / online** — suitable as a blueprint for autoregressive
  inference kernels.
- **O(σ^L) memory** — a single 1-D bucket array per (b,r) line, much
  smaller than the prefix table (Method 12).
- **Complexity** O(B·R·Lmax·T) time, O(B·R·σ^L) memory.
- Compared to `diag_dp.py` (Method 10): both handle `cap_end[t] == t` and
  are streaming.  This method uses base-key bucketing, trading
  sigma-generality for faster per-symbol work when the key space is small.

---

## Hash-based methods

These methods use polynomial rolling hashes modulo 2^64 instead of base-σ encoding. They support large Lmax (not limited by σ^L < 2^64) but are probabilistic unless combined with exact verification.

### Method 2: rolling-hash block table

Use `lookup_full_l_rolling`.

#### Encoding

Instead of base-σ encoding, length-*L* blocks are identified by a polynomial
rolling hash with wrap-around at 2^64:

```
P[0]   = 0
P[t+1] = (P[t] · base + seq[t] + 1)  mod 2^64
```

The hash of the length-*L* block ending at *t* is then:

```
H_L(t) = P[t+1] − P[t−L+1] · base^L   (mod 2^64)
```

where subtraction exploits unsigned 64-bit overflow for the modulo.  The
+1 offset on each symbol avoids degenerate zero prefixes.

#### Lookup

The same predecessor-search pipeline as Method 1 is applied, with rolling
hashes replacing base-σ keys.  Two backends select the matching strategy:

| backend | complexity | mechanism |
|---------|-----------|-----------|
| `"mask"` (default) | O(T²) | mask-multiply over all (t, j) pairs |
| `"hash"` | O(T) | hash-table probe, O(1) per query |

#### Properties

- **Probabilistic** — hash collisions can produce false positives.  No false
  negatives from the encoding itself.
- **Large Lmax feasible** — not limited by σ^L < 2^64; can use Lmax = 16, 32, …
- To make exact, combine with bucket backtracking and raw-symbol tuple
  verification (see Method 7).
- The `"hash"` backend achieves ~9× speedup over `"mask"` at T = 1024.

### Method 7: verified rolling-hash lookup

Use `lookup_full_l_rolling_verified`.

#### Multi-slot hash table

Unlike Method 2 (single-slot hash table, one candidate per bucket), this
method builds a multi-slot table with *C* slots per bucket.

**Insert.** For each K position *t* with rolling hash *h* = H_L(t):

```
bucket        = h  mod  N_buckets
slot          = t  mod  C
combined      = h · (T + 1) + t + 1     (0 = empty sentinel)
table[bucket, slot]  ←  max(table[bucket, slot], combined)
```

The per-slot `max` ensures the rightmost position per slot is retained.
The modulo-based slot assignment distributes positions within a bucket
deterministically, making it likely that the *C* rightmost entries survive.

**Query.** For each Q position with hash *h*:

```
bucket = h  mod  N_buckets
candidates = { table[bucket, 0], …, table[bucket, C−1] }
```

Each non-zero combined value is decoded to recover the K position *j*.
All *C* candidates are then verified symbol-by-symbol against raw K symbols
(identical to the Method 3 verifier).

#### Properties

- **No false positives** — every returned match is verified against raw
  symbols; hash collisions cannot produce spurious matches.
- **False negatives** possible from bucket collisions when more than *C*
  positions hash to the same bucket (bucket overflow).
- **Exact** when *C* ≥ max bucket occupancy; larger *C* trades memory for
  recall.
- **Complexity** O(B·R·Lmax·(T + C·T)) — O(T) table build + O(C) probe
  and verify per position per level.

---

## Candidate-based methods

These methods decouple candidate generation from GPU verification. A candidate generator proposes K end positions; the GPU verifier checks each candidate symbol-by-symbol and selects the best match under ROSA semantics.

### Method 6: fixed-C postings lookup

Use `lookup_full_l_base_postings` (exact base keys) or
`lookup_full_l_rolling_postings` (rolling-hash keys).

#### Principle

Methods 1–2 retain only the single rightmost K position per block key.
This method collects the *C* rightmost matching positions via a multi-offset
predecessor lookup, then verifies all C candidates symbol-by-symbol.

#### Candidate collection

For a query key *q*, let *idx* be the rightmost predecessor index in the
sorted K combined-key array (same `searchsorted` as Method 1).  The C
candidates are the positions at sorted indices:

```
idx,  idx−1,  idx−2,  …,  idx−(C−1)
```

Each candidate is kept only if its block key equals *q* (key match) and its
position is within the causal cap.  Candidates that fail either check are
set to −1.

#### Verification

The C candidates are verified symbol-by-symbol against raw Q and K symbols
(identical to the Method 3 verifier).  The best candidate is selected by
`(match_len, rightmost_j)` scoring and gated through the ROSA
successor/tau_cap pipeline.

#### Variants

- **Base-encoded** (`lookup_full_l_base_postings`) — exact when *C* ≥ the
  maximum number of positions sharing any single block key.  Smaller *C*
  trades a bounded recall risk for more regular GPU memory patterns.
- **Rolling-hash** (`lookup_full_l_rolling_postings`) — probabilistic due to
  hash collisions; combine with full tuple verification for exactness.

#### Properties

- **Complexity** O(B·R·Lmax·(T log T + C·T)) — sorting plus C-offset
  lookup and C-way verification per level.
- **C ≥ T** guarantees exactness (but defeats the purpose of postings).

### Method 3: CPU candidate + GPU verification

Use `verify_cpu_candidates`.

#### Principle

A coarse retriever (CPU, heuristic, or external index) proposes a set of
candidate K end positions `cand_end[b,r,t,:]` for each query position *t*.
The GPU verifier then checks each candidate symbol-by-symbol and selects the
best match under ROSA semantics — it does not generate candidates itself.

#### Verification

For each candidate position *j* and each offset *k* ∈ {0, …, Lmax−1}:

```
match(L, t, j)  ⇔  ∀k < L:  Q[t−k] = K[j−k]
```

All candidates are evaluated in parallel via broadcasted tensor operations.
The match length is the number of leading *k* offsets where equality holds
before the first mismatch (or Lmax if all match).

#### Selection and gating

Candidates are scored by `(match_length, rightmost_j)`:

```
score(j) = match_len · (T + 1) + j
```

The best-scoring candidate is selected per query position, then gated through
the ROSA successor/tau_cap pipeline (identical to every other lookup path).

#### Properties

- **Exact** when the candidate set has full recall — i.e., the true ROSA
  raw match is among the candidates for every position.
- **Error attribution** — when candidates are top-k or bucket-truncated, all
  semantic error comes from recall loss in the candidate generator, never from
  the GPU verifier.
- **Complexity** O(B·R·T·C·Lmax) where C is the number of candidates per
  position.

---

## Sigma-free exact methods

These methods work directly on raw symbols without a `sigma` parameter. They have no uint64 overflow constraint and no hash collisions. They are exact for all L ≤ Lmax.

### Method 9: dyadic-rank postings + binary-lifting LCE

Use `lookup_full_l_drp_lce`.

#### Dyadic rank encoding

Instead of base-σ encoding (Method 1), this method builds a hierarchy of
joint ranks over Q and K symbols that identifies matching blocks of lengths
1, 2, 4, …, 2^k ≤ Lmax.

**Level 0 (length 1).** Sort all T symbols from Q and T symbols from K
together; assign each symbol its 1-based position in the sorted order:

```
rank₀(x)  =  |{ y ∈ Q∪K : y < x }| + 1
```

Two length-1 blocks are equal iff their rank₀ values are equal.

**Level ℓ (length 2^ℓ).** Given ranks for length 2^{ℓ−1}, encode each
length-2^ℓ block as a pair:

```
pair(t)  =  (rank_{ℓ−1}(t − 2^{ℓ−1}),  rank_{ℓ−1}(t))
```

Sort all such pairs from Q and K jointly to assign rank_ℓ.  Two length-2^ℓ
blocks are equal iff their rank_ℓ values are equal (by induction).

Invalid positions (*t* < 2^ℓ − 1) are assigned rank 0.

#### Candidate collection

For each dyadic level ℓ (anchor length 2^ℓ), *C* candidate K positions are
collected via predecessor search on `combined = rank_ℓ · (T+1) + pos`,
identical to the postings mechanism in Method 6.  Candidates from all levels
are concatenated, yielding (max_level + 1) · C candidates per query position.

#### Binary-lifting LCE verification

The longest common extension (LCE) from each candidate is computed via
binary lifting on the rank hierarchy — from the highest power of 2 down:

```
ℓ = max_level, max_level−1, …, 0:
    if  rank_ℓ[Q, t_q] == rank_ℓ[K, j_k]  AND  bounds ok  AND  length + 2^ℓ ≤ Lmax:
        length  += 2^ℓ
        t_q     −= 2^ℓ
        j_k     −= 2^ℓ
```

This checks O(log Lmax) dyadic levels instead of O(Lmax) symbol offsets.
After LCE computation, the best candidate is selected by `(length, rightmost_j)`
scoring and gated through ROSA successor/tau_cap.

#### Properties

- **Exact** — no hash collisions, no false positives.  The dyadic rank
  hierarchy is a deterministic, collision-free encoding.
- **Sigma-free** — no `sigma` parameter; no uint64 overflow constraint.
  Only depends on Lmax and C.
- **O(log Lmax) verification** — binary lifting over dyadic ranks reduces
  verification from O(Lmax) to O(log Lmax) per candidate.
- **C-controlled recall** — C ≥ T guarantees exactness; smaller C trades
  bounded recall for memory regularity.
- **Complexity** O(B·R·(log Lmax · T log T + log Lmax · C · T)) — sorting
  at each of log Lmax levels plus C-way probe and binary-lifting verify.

### Method 14: suffix-array binary search

Use `lookup_full_l_sa`. Build suffix arrays with `suffix_array_batch`.

#### Principle

A suffix array (SA) for K is built on CPU via tuple sort.  On GPU, each
query suffix is located in the SA via binary search, yielding a contiguous
range `[lo, hi]` of SA indices whose length-L prefix matches the query.
The rightmost valid K end position within that range is selected.

#### SA construction (CPU)

For each (b, r) line of K, all T suffixes are sorted lexicographically:

```
SA[b, r, i] = starting position of the i-th smallest suffix
```

This is a one-time O(T² log T) CPU cost (tuple sort), amortized over all
queries that share the same K.

#### GPU binary search

For each query position *t* and each length *L* ∈ {1, …, Lmax}:

1. Form the query pattern `Q[t−L+1 … t]` (L symbols).
2. Binary-search the SA (O(log T) comparisons) to find `lo` (first SA entry
   ≥ pattern) and `hi` (last SA entry ≤ pattern).
3. Within SA[lo…hi], convert starting positions to end positions
   (`end = start + L − 1`), gate by `cap_end[t]`, and select the rightmost.

The binary-search loop is unrolled at trace time because Lmax is static,
letting XLA fuse operations across L values.

#### ROSA tie-breaking and gating

As with all other methods, longer matches beat shorter ones, and the
rightmost K end position breaks ties.  ROSA successor/tau_cap gating is
applied at the end.

#### Properties

- **Exact** — no hash collisions, no false positives/negatives.
- **Sigma-free** — no `sigma` parameter, no uint64 overflow constraint.
- **Build** O(T²) worst-case per route on CPU (one-time, amortized).
- **Query** O(B·R·Lmax·T·log T) — Lmax binary searches per position.
- For small Lmax (≤ 8), competitive with the block-table approach while
  avoiding its uint64 overflow constraints.
- SA can be precomputed and reused via the `SA=` parameter.

---

## DP-based methods (small T)

These methods compute the full [T × T] suffix-match matrix via dynamic programming. They are exact and sigma-free but have O(T²) time complexity, making them suitable only for T ≤ 1024 as correctness oracles or small-context benchmarks.

### Method 5: dense equality DP exact baseline

Use `lookup_full_l_dp`.

#### Recurrence

Build the [T × T] boolean equality matrix:

```
eq[t, j]  =  (Q[t] == K[j])
```

Define D[t, j] as the length of the longest suffix of Q[0…t] that is also a
suffix of K[0…j].  The classic DP recurrence:

```
           ⎧ D[t−1, j−1] + 1   if eq[t, j]
D[t, j] =  ⎨
           ⎩ 0                 otherwise
```

with boundary D[−1, ·] = D[·, −1] = 0.  This is computed via `lax.scan`,
which applies the row-by-row update in O(T) sequential steps on GPU.

#### Selection

Match lengths are clamped to Lmax, then scored:

```
score(t, j) = min(D[t,j], Lmax) · (T + 1) + j
```

For each query row *t*, the *j* with maximum score (longest, then rightmost)
is selected.  ROSA successor/tau_cap gating is applied as a final step,
identical to every other lookup path.

#### Properties

- **Exact** for all *L* ≤ Lmax — works directly on raw symbols, no encoding
  or hash collisions.
- **Sigma-free** — no base/sigma parameter, no uint64 overflow constraint.
- **Complexity** O(B·R·T²) — recommended only for T ≤ 1024 as a correctness
  oracle or small-context benchmark.
- **TPU-friendly** — dense matmul + scan avoids sparse gather/scatter.

### Method 10: streaming diagonal-DP lookup

Use `lookup_full_l_diag_dp`.

#### Principle

Identical recurrence to the dense DP (Method 5), but only the previous row
``D_prev[T]`` is kept:

```
D_curr[j] = D_prev[j-1] + 1   if Q[t] == K[j]
            0                  otherwise
```

Memory drops from O(T²) to O(T).  A `lax.scan` over time steps carries the
single-row state, and for each query position the best raw match (longest,
then rightmost) is selected and gated through the ROSA successor/tau_cap
pipeline.

#### Properties

- **Exact** for all L ≤ Lmax — works directly on raw symbols, no encoding
  or hash collisions.
- **Sigma-free** — no base/sigma parameter, no uint64 overflow constraint.
- **Complexity** O(B·R·T²) time, O(B·R·T) memory.  Same asymptotics as
  dense DP but with drastically lower memory.  Recommended for T ≤ 1024
  as a correctness oracle.
- **TPU-friendly** — dense scan operations avoid sparse gather/scatter.

### Method 15: TPU dense one-hot DP

Use `lookup_full_l_dense_tpu`.

#### Principle

Instead of broadcasting `Q[t] == K[j]` (Method 5), this method computes the
[T × T] equality matrix via one-hot matmul:

```
E = one_hot(Q, σ) @ one_hot(K, σ)ᵀ
```

where `E[t, j] = 1` iff `Q[t] == K[j]`.  This leverages TPU systolic arrays
for the matrix multiplication.  The DP recurrence and selection are
otherwise identical to Method 5.

#### Properties

- **Exact** — same correctness as Method 5.
- **TPU-optimized** — the one-hot matmul is efficient on TPU hardware but
  slower than direct broadcast on GPU.
- **Sigma-dependent** — requires `sigma`; large σ increases the one-hot
  dimension, raising memory and compute cost.
- **Complexity** O(B·R·(T·σ + T²)) — not recommended for T > 2048 or
  large σ.
- This path is separate from the main lookup methods because the one-hot
  matmul is only efficient on TPU.  On GPU, prefer Method 5 or Method 10.

---

## Bitset methods

These methods use bit-parallel operations to track matching positions. They are exact and require no sorting or hashing.

### Method 8: bitset exact suffix lookup (experimental)

Use `lookup_full_l_bitset`.

#### Principle

For each suffix length *L* ∈ {1, …, Lmax}, build a [T × T] boolean matrix:

```
R_L[t, j]  =  ⋀_{k=0}^{L−1}  ( Q[t−k] == K[j−k]  ∧  t−k ≥ 0  ∧  j−k ≥ 0 )
```

That is, R_L[t, j] is true if and only if the length-L suffix ending at Q[t]
exactly matches the length-L suffix ending at K[j], with all indices in bounds.

#### Selection

R_L is gated by the causal cap (*j* < cap_end[t]) and valid block boundaries
(*j* ≥ L−1, *t* ≥ L−1).  Valid *j* positions are scored by their position
(rightmost preference), and the best raw match across all *L* is accumulated
identically to Method 1.

ROSA successor/tau_cap gating is applied as a final step.

#### Properties

- **Exact** for all *L* ≤ Lmax — direct symbol comparison, no encoding or
  hashing.
- **Complexity** O(B·R·T²·Lmax) — only suitable for tiny T (≤ 32) as a
  correctness reference.

### Method 11: Shift-And bitset exact lookup

Use `lookup_full_l_shift_and`.

#### Principle

The classic bit-parallel Shift-And string-matching algorithm is adapted to
ROSA suffix predecessor queries.  For each symbol ``a``, a multi-word bitset
``P_a`` records every K position where the symbol occurs.  A per-length
bitset ``M_L(t)`` tracks K positions whose length-L suffix matches Q ending
at time ``t``:

```
M_1(t)    = P_{Q[t]}
M_L(t)    = (M_{L-1}(t-1) << 1)  &  P_{Q[t]}      (L > 1)
```

The `<< 1` shifts match positions forward by one (crossing word boundaries),
and `& P_{Q[t]}` requires the current symbol to match.

ROSA rightmost predecessor becomes a single `highest_set_bit` query on the
masked bitset ``M_L(t) & cap_mask(cap_end[t])``.  No sorting, no hashing,
and no base-σ encoding are needed.

#### Selection and gating

For each time ``t``, the longest length ``L`` with a non-empty masked bitset
is selected.  The highest set bit in that bitset is the rightmost matching
K end position ``j``.  ROSA successor/tau_cap gating is applied identically
to every other lookup path.

#### Comparison with existing `bitset.py` (Method 8)

The existing ``bitset.py`` constructs a ``[T, T]`` boolean matrix per ``L``
and has complexity O(B·R·T²·Lmax²).  Shift-And instead maintains only
``Lmax`` multi-word bitsets and has complexity O(B·R·Lmax·T·ceil(T/64)).

#### Properties

- **Exact** — no hash collisions, no false positives, no base-key overflow.
- **No sort required** — uses only bitwise operations (`&`, `<<`, binary
  search for highest-set-bit).
- **Streaming-friendly** — each new symbol needs O(Lmax·W) bitwise ops.
- **Complexity** O(B·R·Lmax·T·ceil(T/64)) time,
  O(B·R·Lmax·ceil(T/64)) memory.
- **Best at**: moderate T, large Lmax, small alphabet, exactness required,
  and when sorting/hashing overhead should be avoided.
- **Compared to base block table**: at small T and Lmax the base-sort path
  may still be faster due to lower constants; at larger Lmax Shift-And's
  per-level constant-cost bitwise operations begin to dominate.

---

## Special-purpose

### Method 4: Q-bit counterfactual lookup

Use `q_bit_counterfactual_tau`.

#### Principle

For each bit position *m* ∈ {0, …, M−1} of an M-bit symbol alphabet, this
method asks: *what would τ be if the current query symbol's m-th bit were
forced to 0 (resp. 1)?*  It returns two tensors `(tau0, tau1)` where
`tau0[b,r,t,m]` is the counterfactual τ under bit-forced-to-0 and
`tau1[b,r,t,m]` under bit-forced-to-1.

#### Encoding adjustment

The base-σ block key (Method 1) at position *t* has the current symbol as its
least significant digit (weight σ^0 = 1).  To force the m-th bit without
rebuilding the entire block:

```
forced[t]  = (Q[t] & ~(1<<m))   or   (Q[t] | (1<<m))
q_key'(t)  = q_key(t) − Q[t] + forced[t]
```

Only valid query positions (*t* ≥ L−1) are modified; prefix-padded positions
keep their original (invalid) keys unchanged.

#### Reuse

Q-keys and K-keys are precomputed once per *L* and reused across all 2·M
branch evaluations.  Each branch invokes the exact block-table predecessor
search (Method 1) on the adjusted Q-keys.

#### Properties

- **Exact** under the same uint64 constraints as Method 1.
- **Does not rebuild RLE** after the bit flip — if the target ROSA
  implementation treats run merge/split specially under counterfactuals,
  integrate that run-level representation first.
- **Complexity** O(B·R·Lmax·M·T log T) — 2·M lookups with precomputed keys.

---

## Internal helpers: Bloom filter

```python
from rosa_gpu_jax.filters import bloom_filter_keys, bloom_query_keys
```

A simple 2-hash Bloom filter for pre-screening block keys.  Designed as an
inexpensive pre-check for the postings or rolling-verified paths.

## Performance optimization guide

### 1. Warm up JIT kernels

The first call to each `@jit` function triggers XLA compilation (hundreds of ms).  Call
`rosa_gpu_jax.warmup()` at service startup to pay this cost upfront:

```python
from rosa_gpu_jax import warmup
warmup()  # compiles all kernels for common Lmax/sigma combos
```

Set `JAX_COMPILATION_CACHE_DIR` to persist compiled kernels across restarts:

```bash
export JAX_COMPILATION_CACHE_DIR=/path/to/jax_cache
```

### 2. Choose the right algorithm for rolling hash

`lookup_full_l_rolling` supports two backends via the `algorithm` parameter:

| algorithm | complexity | best for | limitations |
|---|---|---|---|
| `"mask"` (default) | O(T^2) | T <= 256 | exact for all uint64 keys |
| `"hash"` | O(T) | T >= 512 | false negatives on bucket collisions |

For large `T` the hash backend is significantly faster:

```
     T        mask        hash     speedup
    64      0.696ms      0.603ms       1.15x
   128      1.507ms      0.668ms       2.25x
   256      3.414ms      0.750ms       4.55x
   512      6.784ms      0.910ms       7.46x
  1024     20.071ms      2.216ms       9.06x
```

### 3. Multi-GPU with pmap

Use the `_pmap` variants to split the batch dimension across devices:

```python
from rosa_gpu_jax import lookup_full_l_base_pmap, lookup_full_l_rolling_pmap

tau, match_len = lookup_full_l_base_pmap(
    Q, K, cap_end, successor, Lmax=4, sigma=256, tau_cap=tau_cap
)
```

These automatically pad `B` to be divisible by the device count and fall back to
single-device execution when only one device is available.

### 4. dtype optimizations

- The public API accepts and returns `int64` arrays.
- All intermediate index arithmetic (positions, offsets, match lengths) uses `int32`
  internally, halving memory bandwidth for key lookup buffers.
- `uint64` combined-key arithmetic is preserved for safety.

### 5. Counterfactual reuse

`q_bit_counterfactual_tau` precomputes Q-keys and K-keys once per `L` and
reuses them across all `2*M` branch evaluations, avoiding redundant `O(L*T)`
work that was present in earlier versions.

## Package layout

```text
src/rosa_gpu_jax/
  __init__.py
  causal.py              raw fallback and official ROSA/RLE auxiliary tensors
  block_table.py         exact full-L base-encoded lookup (Method 1)
  prefix_table.py        counting-prefix-table exact lookup (Method 12)
  streaming_causal.py    streaming causal bucket exact lookup (Method 13)
  rolling_hash.py        probabilistic rolling-hash lookup (Method 2)
  rolling_verified.py    verified rolling-hash with multi-slot tables (Method 7)
  postings.py            fixed-C postings + dyadic-rank LCE lookup (Methods 6, 9)
  candidates.py          GPU suffix verification for CPU candidates (Method 3)
  cpu_candidates.py      CPU-side candidate generators (helpers for Method 3)
  suffix_tree_lookup.py  suffix-array binary search lookup (Method 14)
  dp.py                  dense equality DP exact baseline (Method 5)
  diag_dp.py             streaming diagonal-DP exact lookup (Method 10)
  dp_tpu.py              TPU dense one-hot DP benchmark (Method 15)
  bitset.py              boolean-array exact suffix lookup (Method 8, experimental)
  shift_and.py           Shift-And bitset exact lookup (Method 11)
  counterfactual.py      Q-bit counterfactual lookup (Method 4)
  filters.py             Bloom negative filter (internal helper)
  reference.py           slow NumPy reference used by tests
  validation.py          input validation helpers
examples/
  smoke_test.py
tests/
  test_bitset.py
  test_block_table.py
  test_candidates.py
  test_counterfactual.py
  test_diag_dp.py
  test_dp.py
  test_official_rosa_semantics.py
  test_postings.py
  test_prefix_table.py
  test_rolling_verified.py
  test_shift_and.py
  test_streaming_causal.py
  test_suffix_array.py
```

## Important limitations

The code is a research scaffold.

The full-L base version rejects parameters that would overflow uint64 combined keys. Use small `Lmax`, or switch to rolling hash plus verification for larger blocks.

The rolling-hash version is probabilistic unless extended with exact tuple verification. Its lookup avoids exact-key combined packing so full-width hash keys do not corrupt predecessor ordering, but hash collisions are still possible.

The candidate verifier cannot recover candidates that the CPU or coarse retriever never returns.

For exact ROSA/RLE reproduction, use `make_rosa_causal_aux(K)` or pass equivalent `cap_end`, `successor`, and `tau_cap` tensors.  Omitting `tau_cap` preserves the older token-level fallback and can differ from official ROSA on repeated-run edge cases.

## Running tests

```bash
pytest
```

The tests compare the exact block table path, candidate verifier, and counterfactual path against slow NumPy references on small randomized and adversarial sequences. Regression tests cover custom `cap_end/successor`, invalid symbols, invalid lengths, exact-key overflow, and candidate bounds.
