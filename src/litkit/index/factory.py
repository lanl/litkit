# litkit/index/factory.py
"""FAISS index factory functions for litkit."""

from __future__ import annotations

import sys

import faiss

from litkit.index.constants import (
    PQ_BITS,
    DEFAULT_HNSW_M,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
)


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def flat_ip_index(dim: int) -> faiss.Index:
    """Create an exact inner-product (cosine when normalized) flat index.
    
    Args:
        dim: Vector dimension
    
    Returns:
        IndexFlatIP instance
    """
    return faiss.IndexFlatIP(dim)


def hnsw_index(
    dim: int,
    M: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> faiss.Index:
    """Create an HNSW index with inner product metric.
    
    Tries multiple construction methods for compatibility across FAISS versions:
    1. Explicit IP constructor (3-arg)
    2. index_factory
    3. 2-arg constructor with metric_type attribute
    
    Args:
        dim: Vector dimension
        M: Number of connections per layer (higher = better quality, more memory)
        ef_construction: Search depth during construction
        ef_search: Search depth during queries
    
    Returns:
        IndexHNSWFlat instance configured for inner product
    
    Raises:
        RuntimeError: If no compatible HNSW construction method is available
    """
    # 1) Prefer explicit IP constructor (faiss >= 1.7.4)
    try:
        idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        # 2) Try index_factory
        try:
            idx = faiss.index_factory(
                dim, f"HNSW{M}", faiss.METRIC_INNER_PRODUCT
            )
        except Exception as e:
            # 3) Last resort: 2-arg constructor + attribute
            idx = faiss.IndexHNSWFlat(dim, M)
            if hasattr(idx, "metric_type"):
                idx.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise RuntimeError(
                    "FAISS build does not support IP HNSW "
                    "(metric_type unset and 3-arg ctor unavailable). "
                    "Install a newer faiss (>=1.7.4, CPU or GPU) "
                    "or run with --papers-index flat."
                ) from e
    
    # Set construction and search parameters
    if hasattr(idx, "hnsw"):
        idx.hnsw.efConstruction = ef_construction
        idx.hnsw.efSearch = ef_search
    
    return idx


def ivfpq_index(
    dim: int,
    nlist: int = 16384,
    m: int = 64,
    bits: int = PQ_BITS,
) -> faiss.Index:
    """Create IVF-PQ index with inner product metric.
    
    Code size = `m` bytes per vector (with 8 bits/subquantizer).
    
    Args:
        dim: Vector dimension
        nlist: Number of Voronoi cells (IVF clusters)
        m: Number of PQ subquantizers (code size in bytes)
        bits: Bits per subquantizer (usually 8)
    
    Returns:
        IndexIVFPQ instance (untrained)
    """
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
    if hasattr(idx, "metric_type"):
        idx.metric_type = faiss.METRIC_INNER_PRODUCT
    return idx


def safe_pq_m(dim: int, requested_m: int) -> int:
    """Compute a safe PQ subquantizer count that divides dimension evenly.
    
    PQ requires dim to be divisible by m. This function finds the largest
    valid m <= requested_m.
    
    Args:
        dim: Vector dimension
        requested_m: Desired number of subquantizers
    
    Returns:
        Safe m value that divides dim evenly
    """
    if dim % requested_m == 0:
        return requested_m
    
    # Find largest divisor <= requested_m
    for m in range(requested_m, 0, -1):
        if dim % m == 0:
            if m != requested_m:
                _eprint(
                    f"[index] PQ m={requested_m} doesn't divide dim={dim}; "
                    f"using m={m}"
                )
            return m
    
    # Fallback (should never happen for dim > 0)
    return 1
