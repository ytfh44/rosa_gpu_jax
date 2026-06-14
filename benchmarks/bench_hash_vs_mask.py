"""Benchmark hash-table vs mask-based rolling-hash lookup.

Run from the repository root:

    python benchmarks/bench_hash_vs_mask.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import lookup_full_l_rolling, make_raw_causal_aux, warmup


def _make_data(B: int, R: int, T: int, sigma: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    # Create some overlap so both kernels have nontrivial work.
    K_np[:, :, : T // 2] = Q_np[:, :, : T // 2]
    return jnp.asarray(Q_np), jnp.asarray(K_np)


def bench_one(name: str, fn, warmup_iters: int = 3, bench_iters: int = 20):
    # Warm up
    for _ in range(warmup_iters):
        result = fn()
        jax.block_until_ready(result)
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        result = fn()
        jax.block_until_ready(result)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / bench_iters
    print(f"  {name:>8s}: {elapsed * 1000:8.3f} ms/iter")
    return elapsed


def main():
    # Pre-compile all kernels first so benchmark measures runtime only.
    print("--- Pre-warming all kernels ---")
    warmup(verbose=False)

    base = 257
    Lmax = 4
    sigma = 256
    B, R = 2, 3

    print()
    print(f"--- Benchmark: B={B}, R={R}, Lmax={Lmax}, base={base} ---")
    print(f"{'T':>6s}  {'mask':>10s}  {'hash':>10s}  {'speedup':>10s}")
    print("-" * 48)

    for T in [64, 128, 256, 512, 1024]:
        Q, K = _make_data(B, R, T, sigma)
        cap_end, successor = make_raw_causal_aux(B, R, T)

        # Measure mask
        def run_mask():
            return lookup_full_l_rolling(
                Q, K, cap_end, successor, Lmax=Lmax, base=base, algorithm="mask"
            )

        t_mask = bench_one("mask", run_mask)

        # Measure hash
        def run_hash():
            return lookup_full_l_rolling(
                Q, K, cap_end, successor, Lmax=Lmax, base=base, algorithm="hash"
            )

        t_hash = bench_one("hash", run_hash)

        speedup = t_mask / t_hash if t_hash > 0 else float("inf")
        print(
            f"{T:>6d}  {t_mask * 1000:>9.3f}ms  {t_hash * 1000:>9.3f}ms  {speedup:>9.2f}x"
        )


if __name__ == "__main__":
    main()
