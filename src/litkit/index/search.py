# litkit/index/search.py
"""FAISS search operations for litkit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from litkit.index.introspection import extract_ivf, kind_and_core
from litkit.index.io import faiss_load_cached
from litkit.index.constants import INDEX_KIND_HNSW, INDEX_KIND_IVF


def pick_nprobe(nlist: int, user: int | None) -> int:
    """Compute nprobe for IVF index.
    
    If user provides a value, use it. Otherwise, use sqrt(nlist) clamped
    to [8, 512].
    
    Args:
        nlist: Number of Voronoi cells in the IVF index
        user: User-specified nprobe (or None for auto)
    
    Returns:
        nprobe value to use
    """
    if user is not None:
        return user
    
    # Auto: sqrt(nlist) clamped to reasonable range
    auto = int(math.sqrt(nlist))
    return max(8, min(512, auto))


def auto_set_nprobe(
    index,
    user_nprobe: int | None = None,
    min_probe: int = 8,
    max_probe: int = 512,
) -> int | None:
    """Set nprobe on IVF indices.
    
    If user_nprobe is None, chooses approximately sqrt(nlist) clamped
    to [min_probe, max_probe].
    
    Args:
        index: FAISS index (possibly wrapped)
        user_nprobe: User-specified nprobe (or None for auto)
        min_probe: Minimum nprobe for auto selection
        max_probe: Maximum nprobe for auto selection
    
    Returns:
        The nprobe value set, or None if not an IVF index
    """
    ivf = extract_ivf(index)
    if ivf is None:
        return None
    
    nlist = int(ivf.nlist)
    target = pick_nprobe(nlist, user_nprobe)
    target = max(min_probe, min(max_probe, target))
    ivf.nprobe = target
    return target


def faiss_search(
    index_path: Path,
    qvec: np.ndarray,
    k: int,
    **kwargs: Any,
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Search a FAISS index for k nearest neighbors.
    
    Handles index-type-specific parameters (efSearch for HNSW, nprobe
    for IVF) and returns metadata about the search.
    
    Args:
        index_path: Path to the FAISS index file
        qvec: Query vector(s), shape (n, dim) or (dim,)
        k: Number of neighbors to return
        **kwargs: Index-specific parameters:
            - efSearch: For HNSW indices
            - nprobe: For IVF indices
    
    Returns:
        Tuple of (ids, distances, metadata) where:
        - ids: List of neighbor IDs
        - distances: List of distances/scores
        - metadata: Dict with search parameters used (e.g., nprobe, efSearch)
    """
    index = faiss_load_cached(index_path)
    kind, core = kind_and_core(index)
    
    info: dict[str, Any] = {}
    
    # Configure index-specific parameters
    if kind == INDEX_KIND_HNSW:
        ef = kwargs.get("efSearch", 128)
        if hasattr(core, "hnsw"):
            core.hnsw.efSearch = ef
        info["efSearch"] = ef
    
    elif kind == INDEX_KIND_IVF:
        target = pick_nprobe(int(core.nlist), kwargs.get("nprobe", None))
        core.nprobe = target
        info["nprobe"] = target
    
    # Ensure query is 2D
    q = np.ascontiguousarray(qvec.reshape(1, -1) if qvec.ndim == 1 else qvec)
    q = q.astype("float32", copy=False)
    
    # Search
    D, I = index.search(q, k)
    
    # Flatten results (assumes single query)
    ids = [int(x) for x in I[0] if x >= 0]
    dists = [float(D[0][i]) for i, x in enumerate(I[0]) if x >= 0]
    
    return ids, dists, info
