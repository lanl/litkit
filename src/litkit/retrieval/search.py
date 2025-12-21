# src/litkit/retrieval/search.py
"""FAISS search utilities for retrieval.

Contains FAISS index search with temporary parameter adjustment.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from litkit.index import faiss_load_cached, kind_and_core, pick_nprobe


@contextmanager
def temporary_search_params(kind: str, core, *, efSearch=None, nprobe=None):
    """Temporarily adjust FAISS search parameters.
    
    Context manager that modifies search parameters (efSearch for HNSW,
    nprobe for IVF) and restores them on exit.
    
    Args:
        kind: Index kind ("hnsw" or "ivf")
        core: Core FAISS index object
        efSearch: HNSW efSearch parameter (optional)
        nprobe: IVF nprobe parameter (optional)
    
    Yields:
        None
    """
    saved = {}
    try:
        if kind == "hnsw" and hasattr(core, "hnsw"):
            if efSearch is not None:
                saved["efSearch"] = int(core.hnsw.efSearch)
                core.hnsw.efSearch = int(efSearch)
        elif kind == "ivf":
            if nprobe is not None and hasattr(core, "nprobe"):
                saved["nprobe"] = int(core.nprobe)
                core.nprobe = int(nprobe)
        yield
    finally:
        try:
            if kind == "hnsw" and "efSearch" in saved:
                core.hnsw.efSearch = saved["efSearch"]
            elif kind == "ivf" and "nprobe" in saved:
                core.nprobe = saved["nprobe"]
        except Exception:
            pass


def faiss_search(
    index_path: Path,
    qvec: np.ndarray,
    k: int,
    **kwargs,
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Search a FAISS index with automatic parameter tuning.
    
    Loads the index (cached), determines its type, sets appropriate
    search parameters, and returns results.
    
    Args:
        index_path: Path to FAISS index file
        qvec: Query vector (1, dim) shape
        k: Number of results to return
        **kwargs: Additional parameters (efSearch for HNSW, nprobe for IVF)
    
    Returns:
        (ids, distances, meta) where:
        - ids: List of result IDs (excluding -1 entries)
        - distances: List of distances for each result
        - meta: Dict with effective search params (e.g., efSearch, nprobe)
    """
    index = faiss_load_cached(index_path)
    kind, core = kind_and_core(index)
    info: dict[str, Any] = {}

    if kind == "hnsw":
        ef = int(kwargs.get("efSearch") or 128)
        info["efSearch"] = ef
        ctx = temporary_search_params(kind, core, efSearch=ef)
    elif kind == "ivf":
        target = pick_nprobe(int(core.nlist), kwargs.get("nprobe", None))
        info["nprobe"] = target
        info["nlist"] = int(core.nlist)
        ctx = temporary_search_params(kind, core, nprobe=target)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        D, indices = index.search(qvec.astype("float32"), k)

    ids = [int(x) for x in indices[0] if x != -1]
    ds = [float(d) for (d, x) in zip(D[0], indices[0], strict=False) if x != -1]
    return ids, ds, info
