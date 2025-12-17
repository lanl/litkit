# litkit/index/io.py
"""FAISS index I/O operations for litkit."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import faiss

from litkit.concurrent.locking import in_faiss_lock

# Thread-local cache for loaded indices
_TL_FAISS_CACHE = threading.local()


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _assert_faiss_locked() -> None:
    """Raise AssertionError if FAISS lock is not held."""
    if not in_faiss_lock():
        raise AssertionError(
            "FAISS save called without holding FAISS_LOCK. "
            "If you also update SQLite, acquire DB_LOCK first."
        )


def faiss_save(index: faiss.Index, path: Path) -> bool:
    """Save FAISS index atomically (temp file + rename).
    
    Requires FAISS lock to be held. Uses atomic write pattern to prevent
    corruption on crash.
    
    Args:
        index: FAISS index to save
        path: Destination path
    
    Returns:
        True if save succeeded, False otherwise
    """
    _assert_faiss_locked()
    
    label = "papers" if Path(path).name.startswith("papers") else "chunks"
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    try:
        faiss.write_index(index, str(tmp))
        try:
            os.replace(tmp, path)
            _eprint(f"[faiss] {label}: saved {path.name}")
            return True
        except OSError as e:
            _eprint(f"[faiss] {label}: rename failed: {e}")
            return False
    except Exception as e:
        _eprint(f"[faiss] {label}: write failed: {e}")
        return False


def faiss_save_force(index: faiss.Index, path: Path) -> bool:
    """Save FAISS index atomically, always logging success.
    
    Similar to faiss_save but used when save must happen (e.g., after
    training or explicit flush).
    
    Args:
        index: FAISS index to save
        path: Destination path
    
    Returns:
        True if save succeeded, False otherwise
    """
    _assert_faiss_locked()
    
    tmp = path.with_suffix(path.suffix + ".tmp")
    label = "papers" if Path(path).name.startswith("papers") else "chunks"
    
    try:
        faiss.write_index(index, str(tmp))
        try:
            os.replace(tmp, path)
            _eprint(f"[faiss] {label}: force-saved {path.name}")
            return True
        except OSError as e:
            _eprint(f"[faiss] {label}: rename failed: {e}")
            return False
    except Exception as e:
        _eprint(f"[faiss] {label}: write failed: {e}")
        return False


def faiss_load(path: Path) -> faiss.Index:
    """Load a FAISS index from disk.
    
    Args:
        path: Path to the index file
    
    Returns:
        Loaded FAISS index
    
    Raises:
        FileNotFoundError: If the index file doesn't exist
    """
    return faiss.read_index(str(path))


def faiss_load_cached(path: Path) -> faiss.Index:
    """Load a FAISS index with thread-local caching.
    
    Caches indices by (path, mtime) to avoid repeated disk I/O during
    search operations. Cache is invalidated when file is modified.
    
    Args:
        path: Path to the index file
    
    Returns:
        Loaded FAISS index (possibly from cache)
    """
    p = Path(path)
    
    try:
        st = p.stat()
    except FileNotFoundError:
        return faiss_load(path)
    
    mt_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    key = (str(p.resolve()), mt_ns)
    
    cache = getattr(_TL_FAISS_CACHE, "faiss", None)
    if cache is None:
        cache = {}
        _TL_FAISS_CACHE.faiss = cache
    
    if key not in cache:
        cache.clear()  # Clear on mtime change to avoid memory growth
        cache[key] = faiss.read_index(str(p))
    
    return cache[key]


def clear_faiss_cache() -> None:
    """Clear the thread-local FAISS index cache."""
    cache = getattr(_TL_FAISS_CACHE, "faiss", None)
    if cache is not None:
        cache.clear()
