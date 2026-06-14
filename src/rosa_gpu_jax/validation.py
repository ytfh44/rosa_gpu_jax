"""Validation helpers for public ROSA GPU JAX APIs."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

MAX_U64 = (1 << 64) - 1


def _is_integer_dtype(dtype: Any) -> bool:
    try:
        return np.issubdtype(np.dtype(dtype), np.integer)
    except TypeError:
        return False


def _host_array_or_none(arr):
    """Return a NumPy view/copy when values are concrete, else None under tracing."""
    try:
        return np.asarray(arr)
    except Exception:
        return None


def require_python_int(name: str, value: int, *, min_value: int | None = None) -> int:
    """Require a static Python integer-like value and return it as ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a Python integer; got {type(value).__name__}")
    value_i = int(value)
    if min_value is not None and value_i < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {value_i}")
    return value_i


def require_symbol_array(
    name: str,
    value,
    *,
    sigma: int | None = None,
    shape: tuple[int, ...] | None = None,
    validate_symbols: bool = True,
):
    """Return an int64 JAX array after validating rank, shape, and symbol range."""
    arr = jnp.asarray(value)
    if not _is_integer_dtype(arr.dtype):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype}")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}; got {tuple(arr.shape)}")
    if sigma is not None and validate_symbols:
        host = _host_array_or_none(arr)
        if host is not None:
            bad = (host < 0) | (host >= sigma)
            if np.any(bad):
                idx = tuple(int(i) for i in np.argwhere(bad)[0])
                got = int(host[idx])
                raise ValueError(
                    f"{name} symbols must satisfy 0 <= symbol < sigma={sigma}; "
                    f"got {got} at index {idx}"
                )
    return arr.astype(jnp.int64)


def require_int_array(
    name: str,
    value,
    *,
    shape: tuple[int, ...] | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
    validate_values: bool = True,
):
    """Return an int64 JAX array after validating integer dtype and optional bounds."""
    arr = jnp.asarray(value)
    if not _is_integer_dtype(arr.dtype):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype}")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}; got {tuple(arr.shape)}")
    if validate_values and (min_value is not None or max_value is not None):
        host = _host_array_or_none(arr)
        if host is not None:
            bad = np.zeros(host.shape, dtype=bool)
            if min_value is not None:
                bad |= host < min_value
            if max_value is not None:
                bad |= host > max_value
            if np.any(bad):
                idx = tuple(int(i) for i in np.argwhere(bad)[0])
                got = int(host[idx])
                lo = "-inf" if min_value is None else str(min_value)
                hi = "inf" if max_value is None else str(max_value)
                raise ValueError(f"{name} values must be in [{lo}, {hi}]; got {got} at index {idx}")
    return arr.astype(jnp.int64)


def require_rank3_pair(Q, K, *, sigma: int | None, validate_symbols: bool):
    """Validate Q and K as same-shaped [B, R, T] symbol streams."""
    Q_arr = require_symbol_array("Q", Q, sigma=sigma, validate_symbols=validate_symbols)
    K_arr = require_symbol_array("K", K, sigma=sigma, validate_symbols=validate_symbols)
    if Q_arr.ndim != 3:
        raise ValueError(f"Q and K must be rank-3 arrays shaped [B, R, T]; got Q.ndim={Q_arr.ndim}")
    if K_arr.ndim != 3:
        raise ValueError(f"Q and K must be rank-3 arrays shaped [B, R, T]; got K.ndim={K_arr.ndim}")
    if tuple(Q_arr.shape) != tuple(K_arr.shape):
        raise ValueError(f"Q and K must have the same shape; got Q.shape={Q_arr.shape}, K.shape={K_arr.shape}")
    B, R, T = (int(x) for x in Q_arr.shape)
    if B <= 0 or R <= 0 or T <= 0:
        raise ValueError(f"Q and K must have non-empty [B, R, T] dimensions; got {Q_arr.shape}")
    return Q_arr, K_arr, B, R, T


def require_key_array_pair(q_keys, k_keys):
    """Validate precomputed key arrays as same-shaped uint64 [B, R, T]."""
    q_arr = jnp.asarray(q_keys)
    k_arr = jnp.asarray(k_keys)
    if q_arr.dtype != jnp.uint64:
        raise TypeError(f"q_keys must have dtype uint64; got {q_arr.dtype}")
    if k_arr.dtype != jnp.uint64:
        raise TypeError(f"k_keys must have dtype uint64; got {k_arr.dtype}")
    if q_arr.ndim != 3 or k_arr.ndim != 3:
        raise ValueError(f"q_keys and k_keys must be rank-3 arrays shaped [B, R, T]; got {q_arr.shape}, {k_arr.shape}")
    if tuple(q_arr.shape) != tuple(k_arr.shape):
        raise ValueError(f"q_keys and k_keys must have the same shape; got {q_arr.shape}, {k_arr.shape}")
    B, R, T = (int(x) for x in q_arr.shape)
    if B <= 0 or R <= 0 or T <= 0:
        raise ValueError(f"q_keys and k_keys must have non-empty [B, R, T] dimensions; got {q_arr.shape}")
    return q_arr, k_arr, B, R, T


def require_aux(cap_end, successor, *, shape: tuple[int, int, int], validate_values: bool = True):
    """Validate cap_end and successor tensors for lookup semantics."""
    T = int(shape[-1])
    cap = require_int_array(
        "cap_end",
        cap_end,
        shape=shape,
        min_value=0,
        max_value=T,
        validate_values=validate_values,
    )
    succ = require_int_array("successor", successor, shape=shape, validate_values=False)
    return cap, succ


def default_tau_cap(shape: tuple[int, int, int]):
    """Default post-successor cap: allow tau up to T.

    This preserves the older raw-token behavior when no official ROSA run cap is
    supplied.  Official ROSA/RLE use should pass the run-start cap returned by
    ``make_rosa_causal_aux``.
    """
    return jnp.full(shape, int(shape[-1]), dtype=jnp.int64)


def require_tau_cap(tau_cap, *, shape: tuple[int, int, int], validate_values: bool = True):
    """Validate optional post-successor tau cap."""
    if tau_cap is None:
        return default_tau_cap(shape)
    T = int(shape[-1])
    return require_int_array(
        "tau_cap",
        tau_cap,
        shape=shape,
        min_value=0,
        max_value=T,
        validate_values=validate_values,
    )


def require_L_for_T(L: int, T: int) -> int:
    L_i = require_python_int("L", L, min_value=1)
    if L_i > T:
        raise ValueError(f"L must be <= T={T}; got L={L_i}")
    return L_i


def require_Lmax_for_T(Lmax: int, T: int) -> int:
    Lmax_i = require_python_int("Lmax", Lmax, min_value=1)
    if Lmax_i > T:
        raise ValueError(f"Lmax must be <= T={T}; got Lmax={Lmax_i}")
    return Lmax_i


def require_sigma(sigma: int) -> int:
    return require_python_int("sigma", sigma, min_value=2)


def require_base(base: int) -> int:
    base_i = require_python_int("base", base, min_value=1)
    if base_i > MAX_U64:
        raise ValueError(f"base must fit in uint64; got {base_i}")
    return base_i


def require_M(M: int) -> int:
    return require_python_int("M", M, min_value=1)


def ensure_exact_key_safe(*, sigma: int, L: int, T: int) -> None:
    """Reject exact base encoding when the combined lookup key may overflow uint64."""
    key_space = int(sigma) ** int(L)
    max_key = key_space - 1
    max_combined = max_key * (int(T) + 1) + (int(T) - 1)
    if max_combined > MAX_U64:
        max_L = 0
        while True:
            candidate = max_L + 1
            candidate_key_space = int(sigma) ** candidate
            candidate_combined = (candidate_key_space - 1) * (int(T) + 1) + (int(T) - 1)
            if candidate_combined > MAX_U64:
                break
            max_L = candidate
        raise OverflowError(
            "exact base lookup would overflow uint64 combined keys: "
            f"sigma={sigma}, L={L}, T={T}. "
            f"Maximum safe L for this sigma and T is {max_L}."
        )


def ensure_precomputed_keys_combined_safe(q_keys, k_keys, *, T: int) -> None:
    """Reject precomputed keys that cannot be safely combined with positions."""
    max_key_safe = (MAX_U64 - (int(T) - 1)) // (int(T) + 1)
    for name, arr in (("q_keys", q_keys), ("k_keys", k_keys)):
        host = _host_array_or_none(arr)
        if host is None or host.size == 0:
            continue
        max_seen = int(host.max())
        if max_seen > max_key_safe:
            raise OverflowError(
                f"{name} contains key {max_seen}, which would overflow when combined as "
                f"key * (T + 1) + pos for T={T}; maximum safe key is {max_key_safe}"
            )


def max_exact_L(sigma: int, T: int) -> int:
    """Return the largest L safe for exact single-uint64 combined keys."""
    sigma_i = require_sigma(sigma)
    T_i = require_python_int("T", T, min_value=1)
    max_L = 0
    while True:
        candidate = max_L + 1
        candidate_combined = (sigma_i**candidate - 1) * (T_i + 1) + (T_i - 1)
        if candidate_combined > MAX_U64:
            return max_L
        max_L = candidate
