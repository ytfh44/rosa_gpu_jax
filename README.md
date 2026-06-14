# rosa-gpu-jax

JAX research prototypes for GPU-friendly ROSA-style suffix lookup.

This package is not a full ROSA implementation. It isolates several GPU-feasible indexing paths for symbol-level retrieval:

1. full-L exact block table lookup within `L <= Lmax`;
2. rolling-hash block table lookup for larger `Lmax` throughput experiments;
3. CPU-candidate + GPU suffix verification;
4. Q-bit counterfactual destination lookup for the current query symbol.

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

## Method 1: full-L exact block table

Use `lookup_full_l_base`.

```python
import jax.numpy as jnp
from rosa_gpu_jax import make_raw_causal_aux, lookup_full_l_base

Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
K = Q
B, R, T = Q.shape
cap_end, successor = make_raw_causal_aux(B, R, T)

tau, match_len = lookup_full_l_base(Q, K, cap_end, successor, Lmax=3, sigma=16)
print(tau)
print(match_len)
```

This method is exact for `L <= Lmax` if all of the following hold:

- `L_set` is complete: `{1, 2, ..., Lmax}`;
- exact base encoding and the combined key `block_key * (T + 1) + pos` fit in uint64;
- `cap_end`, `successor`, and `tau_cap` reproduce the target ROSA/RLE semantics;
- the symbol alphabet is actually bounded by `0 <= symbol < sigma`.

The public exact lookup APIs now validate these conditions where they can. Unsafe exact-key parameters raise `OverflowError` instead of silently returning corrupted matches. Illegal symbols, bad shapes, invalid `Lmax`, and out-of-range causal caps raise clear errors before JIT execution.

It is usually the best first experiment. For `M=4`, `sigma=16`, `Lmax=4` is a more meaningful starting point than sparse grids such as `{1,2,4,8}`, because length-3 matches are common at long context length.

## Method 2: rolling-hash block table

Use `lookup_full_l_rolling`.

```python
from rosa_gpu_jax import lookup_full_l_rolling

tau, match_len = lookup_full_l_rolling(
    Q, K, cap_end, successor, Lmax=16, base=11400714819323198485, tau_cap=tau_cap
)

# For T >= 512, the O(T) hash backend is faster:
tau, match_len = lookup_full_l_rolling(
    Q, K, cap_end, successor, Lmax=16, base=257,
    tau_cap=tau_cap, algorithm="hash"
)
```

This method uses uint64 overflow rolling hash. It is useful for throughput studies with larger `Lmax`, but it is not a proof of exact ROSA equivalence. To make it exact, add bucket backtracking and raw-symbol tuple verification. Verification must continue to earlier candidates in the same hash bucket if the rightmost hash candidate fails.

## Method 3: CPU candidate + GPU verification

Use `verify_cpu_candidates`.

```python
from rosa_gpu_jax import verify_cpu_candidates

# cand_end[b,r,t,c] is a candidate K end position, or -1 for an empty slot.
cand_end = jnp.array([[[[0, -1], [1, 0], [2, 1]]]], dtype=jnp.int64)
tau, best_len = verify_cpu_candidates(Q[:, :, :3], K[:, :, :3], cand_end, Lmax=3)

# For non-raw ROSA/RLE semantics, pass the same auxiliary tensors used by
# exact lookup:
tau, best_len = verify_cpu_candidates(
    Q[:, :, :3],
    K[:, :, :3],
    cand_end,
    Lmax=3,
    cap_end=cap_end[:, :, :3],
    successor=successor[:, :, :3],
    tau_cap=tau_cap[:, :, :3],
)
```

This method is exact if the candidate set has full recall and the same `cap_end/successor/tau_cap` tensors are supplied as the exact lookup path. If candidates are top-k or bucket-truncated, all semantic error comes from candidate recall loss, not from the GPU verifier.

## Method 4: Q-bit counterfactual lookup

Use `q_bit_counterfactual_tau`.

```python
from rosa_gpu_jax import q_bit_counterfactual_tau

tau0, tau1 = q_bit_counterfactual_tau(
    Q, K, cap_end, successor, Lmax=4, sigma=16, M=4, tau_cap=tau_cap
)
```

This forces the current query symbol bit to 0 or 1 and reuses the exact full-L block lookup. It does not rebuild RLE after the flip. If the target ROSA implementation defines counterfactuals at run level and treats run merge/split specially, integrate that run-level representation before calling this function.

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
  aux.py             raw fallback and official ROSA/RLE auxiliary tensors
  block_table.py     exact full-L base-encoded lookup
  rolling_hash.py    probabilistic rolling-hash lookup
  candidates.py      GPU suffix verification for CPU candidates
  counterfactual.py  Q-bit counterfactual lookup
  reference.py       slow NumPy reference used by tests
examples/
  smoke_test.py
tests/
  test_block_table.py
  test_candidates.py
  test_counterfactual.py
  test_official_rosa_semantics.py
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
