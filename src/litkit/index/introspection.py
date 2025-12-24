# litkit/index/introspection.py
"""FAISS index introspection utilities for litkit."""

from __future__ import annotations

import sys
from pathlib import Path

import faiss

from litkit.index.constants import (
    USE_DOWNCAST_FALLBACK,
    INDEX_KIND_FLAT,
    INDEX_KIND_HNSW,
    INDEX_KIND_IVF,
    INDEX_KIND_UNKNOWN,
)


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def unwrap_core_and_kind(
    idx: faiss.Index,
    max_depth: int = 12,
    allow_downcast_fallback: bool = USE_DOWNCAST_FALLBACK,
) -> tuple[str, faiss.Index, list[faiss.Index]]:
    """Unwrap FAISS index wrappers to find core index and determine type.
    
    Unwraps common wrappers (IndexIDMap, IndexIDMap2). Does NOT follow the
    .index attribute of IVF-type indices (which points to their quantizer).
    Prefers duck-typing and faiss.extract_index_ivf; optionally falls back
    to downcast_index for read-only introspection.
    
    Args:
        idx: FAISS index (possibly wrapped)
        max_depth: Maximum unwrap depth to prevent infinite loops
        allow_downcast_fallback: Whether to use faiss.downcast_index
    
    Returns:
        Tuple of (kind, core_index, wrappers) where:
        - kind: One of "flat", "hnsw", "ivf", "unknown"
        - core_index: The unwrapped base index
        - wrappers: List of wrapper indices traversed
    """
    wrappers = []
    base = idx
    
    # Unwrap common wrappers (IndexIDMap, IndexIDMap2, etc.)
    # IMPORTANT: Do NOT follow .index on IVF-type indices - that points to their
    # quantizer (e.g., IndexFlatIP), not a wrapper. Check for IVF BEFORE unwrapping.
    for _ in range(max_depth):
        # Check if current base is already an IVF-type index - if so, stop unwrapping
        # IVF indices have .index pointing to their quantizer, not a wrapper
        if hasattr(base, "nlist") and hasattr(base, "nprobe"):
            break  # This is an IVF index, don't follow .index
        if "IVF" in type(base).__name__:
            break  # IVF, IVFPQ, etc.
        
        # Check for HNSW before unwrapping (HNSW doesn't typically have wrappers)
        if hasattr(base, "hnsw") or "HNSW" in type(base).__name__:
            break
        
        # Only unwrap known wrapper types (IndexIDMap, IndexIDMap2)
        if isinstance(base, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            inner = getattr(base, "index", None)
            if inner is not None and inner is not base:
                wrappers.append(base)
                base = inner
                continue
        
        # Try .base_index attribute (some other wrappers)
        inner = getattr(base, "base_index", None)
        if inner is not None and inner is not base:
            wrappers.append(base)
            base = inner
            continue
        
        break
    
    # Detect index type
    
    # Check for HNSW
    if hasattr(base, "hnsw") or "HNSW" in type(base).__name__:
        return INDEX_KIND_HNSW, base, wrappers
    
    # Check for IVF via attributes (most reliable)
    if hasattr(base, "nlist") and hasattr(base, "nprobe"):
        return INDEX_KIND_IVF, base, wrappers
    
    # Check for IVF via extract_index_ivf (handles some edge cases)
    try:
        ivf = faiss.extract_index_ivf(base)
        if ivf is not None:
            return INDEX_KIND_IVF, ivf, wrappers
    except Exception:
        pass
    
    # Check for Flat
    if "Flat" in type(base).__name__:
        return INDEX_KIND_FLAT, base, wrappers
    
    # Optional downcast fallback for SWIG base objects
    if allow_downcast_fallback:
        try:
            obj = faiss.downcast_index(base)
            if hasattr(obj, "nlist"):
                return INDEX_KIND_IVF, obj, wrappers
            if hasattr(obj, "hnsw") or "HNSW" in type(obj).__name__:
                return INDEX_KIND_HNSW, obj, wrappers
            if "Flat" in type(obj).__name__:
                return INDEX_KIND_FLAT, obj, wrappers
        except Exception:
            pass
    
    return INDEX_KIND_UNKNOWN, base, wrappers


def kind_and_core(idx: faiss.Index) -> tuple[str, faiss.Index]:
    """Get index kind and core index (simplified unwrap).
    
    Args:
        idx: FAISS index
    
    Returns:
        Tuple of (kind, core_index)
    """
    kind, core, _ = unwrap_core_and_kind(idx)
    return kind, core


def extract_ivf(index: faiss.Index) -> faiss.Index | None:
    """Extract IVF/IVFPQ core from index if present.
    
    Args:
        index: FAISS index
    
    Returns:
        IVF core index if found, None otherwise
    """
    kind, core, _ = unwrap_core_and_kind(index)
    if kind == INDEX_KIND_IVF and hasattr(core, "nlist"):
        return core
    return None


def faiss_present_ids(index: faiss.Index) -> set[int] | None:
    """Return the set of external IDs present in an IndexIDMap2-wrapped index.
    
    Args:
        index: FAISS index (should be IndexIDMap2 or have one as inner)
    
    Returns:
        Set of integer IDs, or None if IDs cannot be extracted
    """
    try:
        if isinstance(index, faiss.IndexIDMap2):
            return set(int(x) for x in faiss.vector_to_array(index.id_map))
        
        # Try to unwrap one layer
        base = getattr(index, "index", None)
        if isinstance(base, faiss.IndexIDMap2):
            return set(int(x) for x in faiss.vector_to_array(base.id_map))
    except Exception:
        pass
    
    return None


def report_faiss_index(label: str, path: Path) -> None:
    """Print a summary of a FAISS index to stderr.
    
    Identifies and prints the FAISS core index type, robust to wrappers
    and SWIG base-class objects.
    
    Args:
        label: Label for the index (e.g., "papers", "chunks")
        path: Path to the index file
    """
    try:
        idx = faiss.read_index(str(path))
    except Exception as e:
        _eprint(
            f"[faiss] {label}: (could not load index: {e.__class__.__name__})"
        )
        return
    
    kind, core, wrappers = unwrap_core_and_kind(idx)
    
    # IVF family
    if kind == INDEX_KIND_IVF:
        nlist = getattr(core, "nlist", None)
        nprobe = getattr(core, "nprobe", None)
        
        # Check if IVFPQ
        m_val = None
        if hasattr(core, "pq") and hasattr(core.pq, "M"):
            m_val = int(core.pq.M)
        elif hasattr(core, "code_size"):
            m_val = int(core.code_size)
        
        if m_val is not None:
            msg = (
                f"[faiss] {label}: IVF-PQ "
                f"nlist={int(nlist) if nlist is not None else 'N/A'} "
                f"m={m_val}"
            )
            if nprobe is not None:
                msg += f" nprobe={int(nprobe)}"
            _eprint(msg)
        else:
            msg = (
                f"[faiss] {label}: IVF "
                f"nlist={int(nlist) if nlist is not None else 'N/A'}"
            )
            if nprobe is not None:
                msg += f" nprobe={int(nprobe)}"
            _eprint(msg)
        return
    
    # HNSW
    if kind == INDEX_KIND_HNSW:
        M = None
        ef = None
        if hasattr(core, "hnsw"):
            M = getattr(core.hnsw, "M", None)
            ef = getattr(core.hnsw, "efSearch", None)
        
        if M is not None:
            _eprint(
                f"[faiss] {label}: HNSW M={M} "
                f"efSearch={int(ef) if ef is not None else 'N/A'}"
            )
        else:
            _eprint(
                f"[faiss] {label}: HNSW "
                f"efSearch={int(ef) if ef is not None else 'N/A'}"
            )
        return
    
    # FLAT
    if kind == INDEX_KIND_FLAT:
        _eprint(f"[faiss] {label}: FLAT")
        return
    
    # Unknown
    _eprint(f"[faiss] {label}: {type(core).__name__}")
