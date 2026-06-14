"""Benchmark postings vs exact block table across different C values.

Run from the repository root:

    python benchmarks/bench_postings.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import (
    lookup_full_l_base,
    lookup_full_l_base_postings,
    make_raw_causal_aux,
    warmup,
)


def _make_data(B: int, R: int, T: int, sigma: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np = rng.integers(0, sigma, size=(B, R, T), dtype=np.int64)
    K_np[:, :, : T // 2] = Q_np[:, :, : T // 2]
    return jnp.asarray(Q_np), jnp.asarray(K_np)


def bench_one(name: str, fn, warmup_iters: int = 3, bench_iters: int = 10):
    for _ in range(warmup_iters):
        result = fn()
        jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        result = fn()
        jax.block_until_ready(result)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / bench_iters
    print(f"  {name:>20s}: {elapsed * 1000:8.3f} ms/iter")
    return elapsed


def main():
    print("--- Pre-warming ---")
    warmup(verbose=False)

    sigma = 16
    Lmax = 3
    B, R, T = 2, 2, 64

    Q, K = _make_data(B, R, T, sigma)
    cap_end, successor = make_raw_causal_aux(B, R, T)

    print(f"\n--- Benchmark: B={B}, R={R}, T={T}, Lmax={Lmax}, sigma={sigma} ---")

    # Exact block table baseline.
    bench_one("exact block table", lambda: lookup_full_l_base(Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma))

    # Base postings at different C values.
    for C in [1, 2, 4, 8, 16, 32]:
        bench_one(
            f"base postings C={C}",
            lambda C=C: lookup_full_l_base_postings(
                Q, K, cap_end, successor, Lmax=Lmax, sigma=sigma, C=C
            ),
        )


if __name__ == "__main__":
    main()
