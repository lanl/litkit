"""Checkpoint management for resumable builds.

This module provides multi-process-safe checkpoint operations for tracking
build progress across tar shards. Each producer shard writes to its own
checkpoint file to avoid the "last writer wins" problem.

Usage:
    # Single-node mode (backward compatible)
    ckpt = load_checkpoint(ckpt_path)
    save_checkpoint(ckpt, ckpt_path, lock_path)
    
    # Multi-node mode (per-shard checkpoints)
    ckpt = load_checkpoint(ckpt_path, shard_id=0)
    save_checkpoint(ckpt, ckpt_path, lock_path, shard_id=0)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def shard_ckpt_path(sqlite_dir: Path, shard_id: int) -> Path:
    """Return per-shard checkpoint path.
    
    Args:
        sqlite_dir: Base directory for SQLite files and checkpoints
        shard_id: Shard ID (0-indexed)
    
    Returns:
        Path to the shard-specific checkpoint file
    """
    return Path(sqlite_dir) / f"build_checkpoint_shard_{shard_id:02d}.json"


def load_checkpoint(ckpt_path: Path, shard_id: int | None = None) -> dict:
    """Load JSON checkpoint (if exists) for resumable workflows; else {}.
    
    Args:
        ckpt_path: Path to the shared checkpoint file (used when shard_id is None,
                   or as base directory reference when shard_id is provided)
        shard_id: If provided, load per-shard checkpoint from sqlite_dir
    
    Returns:
        Checkpoint dict or {} if not found
    """
    if shard_id is not None:
        # Per-shard checkpoint: derive path from ckpt_path's parent (sqlite_dir)
        path = shard_ckpt_path(ckpt_path.parent, shard_id)
    else:
        path = Path(ckpt_path)
    
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_checkpoint(
    obj: dict,
    ckpt_path: Path,
    lock_path: Path,
    shard_id: int | None = None,
    *,
    fsync_dir: bool = True,
) -> None:
    """Atomically save checkpoint to disk.
    
    Args:
        obj: Checkpoint data to save
        ckpt_path: Path to the shared checkpoint file (used when shard_id is None,
                   or as base directory reference when shard_id is provided)
        lock_path: Path to lock file (used for single-node mode)
        shard_id: If provided, save to per-shard checkpoint file
        fsync_dir: Whether to fsync the directory after rename
    """
    # Import FileLock lazily to avoid circular imports
    from litkit.concurrent.locking import FileLock
    
    if shard_id is not None:
        # Per-shard checkpoint: each shard owns its own file
        path = shard_ckpt_path(ckpt_path.parent, shard_id)
        lock = path.with_suffix(".lock")
    else:
        path = Path(ckpt_path)
        lock = Path(lock_path)
    
    with FileLock(lock):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            fh.write(json.dumps(obj, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        
        if fsync_dir:
            _maybe_fsync_dir(path)


def clear_shard_checkpoints(sqlite_dir: Path, num_shards: int) -> int:
    """Remove all per-shard checkpoint files.
    
    Called when resharding is detected (different num_shards than previous build).
    
    Args:
        sqlite_dir: Directory containing checkpoint files
        num_shards: Number of shards from previous build
    
    Returns:
        Number of checkpoint files removed
    """
    removed = 0
    sqlite_dir = Path(sqlite_dir)
    
    # Remove numbered shard checkpoints (up to some reasonable max)
    for i in range(max(num_shards, 100)):
        path = shard_ckpt_path(sqlite_dir, i)
        if path.exists():
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    
    return removed


def _maybe_fsync_dir(p: Path) -> None:
    """Fsync the containing directory for NFS/Lustre safety."""
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
