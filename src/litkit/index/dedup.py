# litkit/index/dedup.py
"""Deduplicated vector addition for FAISS indices in litkit."""

from __future__ import annotations

import faiss
import numpy as np

from litkit.index.introspection import faiss_present_ids
from litkit.index.ids import make_id_selector, safe_remove_ids


def add_with_ids_dedup(
    index: faiss.Index,
    ids: list[int],
    X: np.ndarray,
    skip_dedup: bool = False,
) -> tuple[int, np.ndarray]:
    """Add vectors with IDs, optionally removing stale IDs first (dedup).
    
    Normalizes vectors and (unless skip_dedup=True) removes any existing
    IDs before adding. Ensures the index is wrapped in IndexIDMap2 for
    safe external ID semantics.
    
    Args:
        index: FAISS index (should be IndexIDMap2-wrapped)
        ids: List of integer IDs for the vectors
        X: Vector array, shape (n, dim)
        skip_dedup: If True, skip the O(N) removal step. Use this for
                    --rebuild mode where the index starts empty and no
                    dedup is needed. Default False for safety.
    
    Returns:
        Tuple of (num_added, ids_added_array) where:
        - num_added: Number of vectors actually added
        - ids_added_array: numpy array of IDs that were added
    """
    # Ensure float32 and contiguous
    X = np.ascontiguousarray(X.astype("float32"))
    
    # Normalize for inner product = cosine similarity
    faiss.normalize_L2(X)
    
    # Convert to numpy array
    ids_arr = np.asarray(ids, dtype="int64")
    
    # Ensure IndexIDMap2 for safe external ID semantics
    if not isinstance(index, faiss.IndexIDMap2):
        index = faiss.IndexIDMap2(index)
    
    try:
        # Remove existing IDs first (dedup) unless explicitly skipped
        # skip_dedup=True is safe for --rebuild mode (index starts empty)
        if not skip_dedup:
            sel = make_id_selector(ids_arr)
            safe_remove_ids(index, sel)
        
        # Add new vectors
        index.add_with_ids(X, ids_arr)
        return len(ids_arr), ids_arr
        
    except RuntimeError:
        # Fallback: skip IDs already present
        present = faiss_present_ids(index) or set()
        mask = np.array([int(i) not in present for i in ids_arr], dtype=bool)
        
        if not mask.any():
            # All IDs already present
            return 0, np.array([], dtype="int64")
        
        # Add only new IDs
        X_new = X[mask]
        ids_new = ids_arr[mask]
        index.add_with_ids(X_new, ids_new)
        return len(ids_new), ids_new
