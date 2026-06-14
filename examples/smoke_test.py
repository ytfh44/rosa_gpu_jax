"""Minimal smoke test for exact full-L lookup.

Run from the repository root:

    python examples/smoke_test.py
"""

import jax.numpy as jnp

from rosa_gpu_jax import make_raw_causal_aux, lookup_full_l_base, q_bit_counterfactual_tau


def main():
    Q = jnp.array([[[1, 2, 3, 1, 2, 3, 4, 1]]], dtype=jnp.int64)
    K = Q
    B, R, T = Q.shape
    cap_end, successor = make_raw_causal_aux(B, R, T)

    tau, match_len = lookup_full_l_base(Q, K, cap_end, successor, Lmax=3, sigma=16)
    print("tau:")
    print(tau)
    print("match_len:")
    print(match_len)

    tau0, tau1 = q_bit_counterfactual_tau(Q, K, cap_end, successor, Lmax=3, sigma=16, M=4)
    print("q-bit tau0 shape:", tau0.shape)
    print("q-bit tau1 shape:", tau1.shape)


if __name__ == "__main__":
    main()
