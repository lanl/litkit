# litkit/segments/metadata.py
"""Build metadata management for segment directories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from litkit.segments.constants import (
    BUILD_META_FILENAME,
    PAPER_SEGMENT_PREFIX,
    CHUNK_SEGMENT_PREFIX,
    SEGMENT_EXTENSION,
)


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def has_segment_files(seg_dir: Path) -> bool:
    """Check if directory contains any segment files.
    
    Args:
        seg_dir: Directory to check
    
    Returns:
        True if any paper or chunk segment files exist
    """
    seg_dir = Path(seg_dir)
    if not seg_dir.exists():
        return False
    
    # Check for paper segments
    if list(seg_dir.glob(f"{PAPER_SEGMENT_PREFIX}_*{SEGMENT_EXTENSION}")):
        return True
    
    # Check for chunk segments
    if list(seg_dir.glob(f"{CHUNK_SEGMENT_PREFIX}_*{SEGMENT_EXTENSION}")):
        return True
    
    return False


def write_build_meta(
    seg_dir: Path,
    num_shards: int,
    manifest_path: str | None = None,
    *,
    mode: str = "single",
) -> None:
    """Write build metadata to segment directory.
    
    This metadata is used for shard consistency checking to ensure
    resume operations use the same configuration as the original build.
    
    Args:
        seg_dir: Segment directory
        num_shards: Number of producer shards
        manifest_path: Path to manifest file (optional)
        mode: Build mode ("single" or "multi")
    """
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "num_shards": num_shards,
        "mode": mode,
    }
    if manifest_path:
        meta["manifest"] = manifest_path
    
    meta_path = seg_dir / BUILD_META_FILENAME
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    _eprint(f"[meta] Wrote build metadata: {meta}")


def read_build_meta(seg_dir: Path) -> dict | None:
    """Read build metadata from segment directory.
    
    Args:
        seg_dir: Segment directory
    
    Returns:
        Metadata dictionary, or None if not found
    """
    meta_path = Path(seg_dir) / BUILD_META_FILENAME
    
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _eprint(f"[meta] WARNING: Failed to read build metadata: {e}")
        return None


def validate_shard_consistency(
    seg_dir: Path,
    current_num_shards: int,
    *,
    current_mode: str = "single",
    sqlite_dir: Path | None = None,
) -> dict:
    """Validate shard config and handle resharding if detected.
    
    When shard count changes (resharding), we WARN and clear old checkpoints
    rather than ERROR. The DB-based already_processed() provides fallback
    resume capability.
    
    Args:
        seg_dir: Segment directory
        current_num_shards: Number of shards for this run
        current_mode: Build mode for this run ("single" or "multi")
        sqlite_dir: SQLite directory for clearing shard checkpoints
    
    Returns:
        dict with keys:
          - "status": "fresh" | "resume" | "reshard"
          - "previous_shards": int | None
          - "checkpoints_cleared": int (only if resharding)
    
    Raises:
        ValueError: If mode changes (single↔multi) - this is not supported
    """
    seg_dir = Path(seg_dir)
    meta = read_build_meta(seg_dir)
    has_segs = has_segment_files(seg_dir)
    
    if meta is None and not has_segs:
        # Fresh start - nothing to validate
        return {"status": "fresh", "previous_shards": None}
    
    if meta is None and has_segs:
        # Orphan segments without metadata - warn but continue
        _eprint(
            "[meta] WARNING: Segment files exist without build_meta.json. "
            "Cannot validate shard consistency."
        )
        return {"status": "resume", "previous_shards": None}
    
    # Validate against existing metadata
    assert meta is not None  # Narrowing for mypy (handled above)
    existing_shards = meta.get("num_shards", 1)
    existing_mode = meta.get("mode", "single")
    
    # Mode changes are NOT supported (would corrupt data)
    if existing_mode != current_mode:
        raise ValueError(
            f"Build mode mismatch: existing build used '{existing_mode}' "
            f"mode, but current run specifies '{current_mode}'. "
            f"To start fresh, remove {seg_dir}."
        )
    
    # Shard count changes: WARN and clear checkpoints (DB provides resume)
    if existing_shards != current_num_shards:
        _eprint(
            f"[meta] WARNING: Shard count changed from {existing_shards} "
            f"to {current_num_shards}."
        )
        _eprint(
            "[meta] Checkpoints invalidated. Resume will use DB-based "
            "already_processed() checks (slower but correct)."
        )
        
        cleared = 0
        if sqlite_dir is not None:
            from litkit.segments.checkpoint import clear_shard_checkpoints
            cleared = clear_shard_checkpoints(sqlite_dir, existing_shards)
            if cleared:
                _eprint(f"[meta] Cleared {cleared} old checkpoint file(s).")
        
        # Update metadata to reflect new shard count
        write_build_meta(seg_dir, current_num_shards, mode=current_mode)
        
        return {
            "status": "reshard",
            "previous_shards": existing_shards,
            "checkpoints_cleared": cleared,
        }
    
    _eprint(
        f"[meta] Validated: resuming {existing_mode} build "
        f"with {existing_shards} shards"
    )
    return {"status": "resume", "previous_shards": existing_shards}
