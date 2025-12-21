# litkit/index/io.py
"""FAISS index I/O operations for litkit."""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path

import faiss

from litkit.concurrent.locking import in_faiss_lock

# Thread-local cache for loaded indices
_TL_FAISS_CACHE = threading.local()

# Throttle index saves to reduce I/O on shared filesystems (HPC/NFS)
_SAVE_MIN_SEC = int(os.environ.get("LITKIT_SAVE_EVERY_SEC", "120"))
_last_save_ts: dict[str, float] = {"papers": 0.0, "chunks": 0.0}


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _maybe_fsync_dir(p: Path) -> None:
    """Optional: fsync the containing directory for extra safety on NFS."""
    if os.environ.get("LITKIT_SEGMENT_FSYNC_DIR", "1") != "1":
        return
    try:
        dfd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
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
    """Save FAISS index atomically with throttling and fsync.
    
    Requires FAISS lock to be held. Uses atomic write pattern to prevent
    corruption on SIGINT/SIGTERM:
    
    1. Write to <path>.tmp in the same directory
    2. fsync() the temp file (data on disk)
    3. os.replace(tmp, path) - atomic rename on POSIX
    4. fsync() the parent directory (metadata durable)
    
    This guarantees the index file is either fully old or fully new,
    never half-written. Critical for the os._exit(1) signal strategy
    in cli.py which bypasses normal shutdown.
    
    Throttles saves to reduce I/O on shared filesystems (configurable
    via LITKIT_SAVE_EVERY_SEC, default 120s).
    
    Args:
        index: FAISS index to save
        path: Destination path
    
    Returns:
        True if save was performed, False if throttled/skipped
    """
    _assert_faiss_locked()
    
    label = "papers" if Path(path).name.startswith("papers") else "chunks"
    now = time.time()
    
    # Throttle: skip if saved recently
    if (now - _last_save_ts.get(label, 0.0)) < _SAVE_MIN_SEC:
        return False
    
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    
    # fsync the temp file before rename for durability
    try:
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass
    
    os.replace(tmp, path)
    _maybe_fsync_dir(path)
    _last_save_ts[label] = now
    return True


def faiss_save_force(index: faiss.Index, path: Path) -> bool:
    """Save FAISS index atomically, bypassing throttle.
    
    Uses same atomic write pattern as faiss_save():
    1. Write to <path>.tmp in the same directory
    2. fsync() the temp file (data on disk)
    3. os.replace(tmp, path) - atomic rename on POSIX
    4. fsync() the parent directory (metadata durable)
    
    Used when save must happen immediately (e.g., after training,
    explicit flush, or before exit). The atomicity guarantee is
    critical for interrupt safety - see cli.py signal handler comments.
    
    Args:
        index: FAISS index to save
        path: Destination path
    
    Returns:
        True (always saves)
    """
    _assert_faiss_locked()
    
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    
    # fsync the temp file before rename for durability
    try:
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass
    
    os.replace(tmp, path)
    _maybe_fsync_dir(path)
    
    label = "papers" if Path(path).name.startswith("papers") else "chunks"
    _last_save_ts[label] = time.time()
    return True


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
