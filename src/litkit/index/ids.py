# litkit/index/ids.py
"""FAISS ID selector and removal utilities for litkit."""

from __future__ import annotations

import sys
from typing import Any

import faiss
import numpy as np


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def make_id_selector(ids_like: list[int] | np.ndarray) -> Any:
    """Create a FAISS IDSelector for the given IDs.
    
    Compatible with both newer (IDSelectorArray) and older (IDSelectorBatch)
    FAISS versions. Falls back to returning the raw int64 array if no
    selector class is available.
    
    Args:
        ids_like: List or array of integer IDs
    
    Returns:
        FAISS IDSelector object, or numpy int64 array as fallback
    """
    ids_arr = np.asarray(ids_like, dtype="int64")
    
    # Try newer name first, then older name
    Sel = getattr(faiss, "IDSelectorArray", None)
    if Sel is None:
        Sel = getattr(faiss, "IDSelectorBatch", None)
    
    if Sel is not None:
        return Sel(ids_arr)
    
    # Fallback: return raw array (caller must handle differently)
    return ids_arr


def safe_remove_ids(index: faiss.Index, sel: Any) -> int:
    """Best-effort removal of IDs from a FAISS index.
    
    Works with IDMap2/HNSW/FLAT/IVF indices. Some index types don't
    support removal, in which case this is a no-op.
    
    Args:
        index: FAISS index
        sel: ID selector from make_id_selector(), or numpy array
    
    Returns:
        Number of IDs removed (0 if removal not supported)
    """
    # Handle raw array fallback (when no IDSelector available)
    if isinstance(sel, np.ndarray):
        # Try remove_ids with array directly (some FAISS versions support it)
        try:
            return index.remove_ids(sel)
        except (TypeError, AttributeError, RuntimeError):
            return 0
    
    # Standard path with IDSelector
    try:
        return index.remove_ids(sel)
    except (TypeError, AttributeError, RuntimeError) as e:
        # Some index types don't support removal
        _eprint(f"[faiss] remove_ids not supported: {e}")
        return 0
