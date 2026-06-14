"""Bloom negative filter for block-key pre-screening (internal helper).

A simple Bloom filter with two hash functions.  It is designed to be used
as an inexpensive pre-check before more expensive lookup (postings or
verification).  False positives add computation but never produce wrong
answers when the downstream path is exact.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("num_buckets",))
def _bloom_insert_jit(keys, num_buckets: int):
    """Insert keys into a Bloom filter, returning a bool table [num_buckets].

    Uses two hash functions: h1 = key % m, h2 = (key // m) % m.
    """
    m_u = jnp.asarray(num_buckets, dtype=jnp.uint64)
    h1 = (keys % m_u).astype(jnp.int32)
    h2 = ((keys // m_u) % m_u).astype(jnp.int32)

    table = jnp.zeros((num_buckets,), dtype=bool)
    table = table.at[h1].set(True)
    table = table.at[h2].set(True)
    return table


@partial(jax.jit, static_argnames=("num_buckets",))
def _bloom_query_jit(query_keys, table, num_buckets: int):
    """Query a Bloom filter.  Returns bool mask: True = possibly present."""
    m_u = jnp.asarray(num_buckets, dtype=jnp.uint64)
    h1 = (query_keys % m_u).astype(jnp.int32)
    h2 = ((query_keys // m_u) % m_u).astype(jnp.int32)

    return table[h1] & table[h2]


def bloom_filter_keys(k_keys, num_buckets: int):
    """Build a Bloom filter from K block keys.

    Parameters
    ----------
    k_keys:
        uint64 ``[B, R, T]`` K block keys.
    num_buckets:
        Number of filter buckets (should be a prime > 4·T).

    Returns
    -------
    table:
        bool ``[B, R, num_buckets]`` Bloom filter.
    """
    return _bloom_insert_jit(k_keys, num_buckets)


def bloom_query_keys(q_keys, table, num_buckets: int):
    """Query Bloom filter for Q block keys.

    Parameters
    ----------
    q_keys:
        uint64 ``[B, R, T]`` Q block keys.
    table:
        bool ``[B, R, num_buckets]`` from :func:`bloom_filter_keys`.
    num_buckets:
        Must match the value used during insertion.

    Returns
    -------
    mask:
        bool ``[B, R, T]`` — True where the key *might* be present in K
        (False = definitely absent).
    """
    return _bloom_query_jit(q_keys, table, num_buckets)
