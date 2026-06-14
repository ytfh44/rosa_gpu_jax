"""Cross-validate dp, postings, bitset vs brute force on small-T enumeration.

Run from the repository root:

    python benchmarks/bench_dp_accuracy.py
"""

import itertools

import jax.numpy as jnp
import numpy as np

from rosa_gpu_jax import (
    lookup_full_l_base,
    lookup_full_l_base_postings,
    lookup_full_l_bitset,
    lookup_full_l_dp,
    lookup_full_l_rolling_verified,
    make_rosa_causal_aux,
)
from rosa_gpu_jax.reference import brute_force_lookup


def main():
    T = 5
    sigma = 2
    Lmax = 3

    print(f"Enumerating all {sigma}^{T} = {sigma**T} binary sequences of length {T}...")

    all_seqs = np.array(
        list(itertools.product(range(sigma), repeat=T)), dtype=np.int64
    )[:, None, :]  # [N, 1, T]

    cap, succ, tau_cap = make_rosa_causal_aux(jnp.asarray(all_seqs))
    cap_np = np.array(cap)
    succ_np = np.array(succ)
    tau_cap_np = np.array(tau_cap)

    tau_ref, _ = brute_force_lookup(
        all_seqs, all_seqs, cap_np, succ_np, Lmax=Lmax, tau_cap=tau_cap_np
    )

    methods = {
        "exact block table": lambda: lookup_full_l_base(
            jnp.asarray(all_seqs), jnp.asarray(all_seqs),
            cap, succ, Lmax=Lmax, sigma=sigma, tau_cap=tau_cap
        ),
        "dp": lambda: lookup_full_l_dp(
            jnp.asarray(all_seqs), jnp.asarray(all_seqs),
            cap, succ, Lmax=Lmax, tau_cap=tau_cap
        ),
        "base postings C=T": lambda: lookup_full_l_base_postings(
            jnp.asarray(all_seqs), jnp.asarray(all_seqs),
            cap, succ, Lmax=Lmax, sigma=sigma, C=T, tau_cap=tau_cap
        ),
        "rolling verified C=T": lambda: lookup_full_l_rolling_verified(
            jnp.asarray(all_seqs), jnp.asarray(all_seqs),
            cap, succ, Lmax=Lmax, base=257, C=T, tau_cap=tau_cap,
            num_buckets=37,
        ),
        "bitset": lambda: lookup_full_l_bitset(
            jnp.asarray(all_seqs), jnp.asarray(all_seqs),
            cap, succ, Lmax=Lmax, sigma=sigma, tau_cap=tau_cap
        ),
    }

    all_pass = True
    for name, fn in methods.items():
        tau_method = np.array(fn()[0])
        match = np.array_equal(tau_method, tau_ref)
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
            mismatches = np.sum(tau_method != tau_ref)
            print(f"  {name}: {status} ({mismatches} mismatches)")
        else:
            print(f"  {name}: {status}")

    if all_pass:
        print("\nAll methods agree with brute-force reference on full enumeration.")
    else:
        print("\nSome methods disagree — check failures above.")


if __name__ == "__main__":
    main()
