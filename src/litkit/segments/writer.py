# litkit/segments/writer.py
"""Segment writers for embedding data in litkit.

Two-phase durability: segments are written to .npz.pending files first,
then atomically renamed to .npz on finalize(). This ensures:
- If chunk write fails after paper write: cleanup_pending() removes orphans
- Consumer only sees finalized (.npz) segments
- Crashed mid-write leaves only .pending files (cleaned up on restart)
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from litkit.segments.constants import (
    DEFAULT_EMBED_SEGMENT_SIZE,
    DEFAULT_EMBED_SEGMENT_DTYPE,
    PAPER_SEGMENT_PREFIX,
    CHUNK_SEGMENT_PREFIX,
    SEGMENT_EXTENSION,
)

# Suffix for pending (not-yet-finalized) segment files
# NOTE: This goes BEFORE .npz, not after, because np.savez_compressed
# auto-appends .npz if the filename doesn't already end with it.
PENDING_SUFFIX = "_pending"


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def cleanup_orphan_pending_files(outdir: Path, kinds: list[str] | None = None) -> int:
    """Remove orphan _pending.npz files from a previous crashed run.
    
    Call this at producer startup to clean up incomplete segments.
    
    Args:
        outdir: Segment output directory
        kinds: List of prefixes to clean (default: ["papers", "chunks"])
    
    Returns:
        Number of pending files removed
    """
    if kinds is None:
        kinds = [PAPER_SEGMENT_PREFIX, CHUNK_SEGMENT_PREFIX]
    
    outdir = Path(outdir)
    if not outdir.exists():
        return 0
    
    removed = 0
    for kind in kinds:
        # Pattern: chunk_seg_*_pending.npz (PENDING_SUFFIX before extension)
        for p in outdir.glob(f"{kind}_*{PENDING_SUFFIX}{SEGMENT_EXTENSION}"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    
    if removed:
        _eprint(f"[segment] Cleaned up {removed} orphan pending files")
    
    return removed


@dataclass
class SegmentWriterConfig:
    """Configuration for segment writers."""
    
    segment_dir: Path
    segment_size: int = DEFAULT_EMBED_SEGMENT_SIZE
    dtype: str = DEFAULT_EMBED_SEGMENT_DTYPE
    shard_id: int = 0


class SegmentWriter:
    """Writes paper embedding segments to disk with two-phase durability.
    
    Uses doc_id (globally unique file path) for identification.
    Buffers embeddings and flushes to disk when segment_size is reached.
    
    Two-phase write protocol:
    1. write() -> flushes to .npz.pending files (not visible to consumer)
    2. finalize() -> atomic rename .pending -> .npz (visible to consumer)
    
    If write fails, call cleanup_pending() to remove orphan .pending files.
    
    Segment file format (.npz):
    - doc_ids: array of document IDs (strings)
    - embeddings: array of embedding vectors (float16 or float32)
    """
    
    def __init__(
        self,
        outdir: Path,
        segment_size: int = DEFAULT_EMBED_SEGMENT_SIZE,
        dtype: str = DEFAULT_EMBED_SEGMENT_DTYPE,
        shard_id: int = 0,
        kind: str = "papers",
    ):
        """Initialize the segment writer.
        
        Args:
            outdir: Directory to write segment files
            segment_size: Number of embeddings per segment file
            dtype: Storage dtype ("fp16" or "fp32")
            shard_id: Shard identifier for file naming
            kind: Segment kind for file prefix (default: "papers")
        """
        self.outdir = Path(outdir)
        self.segment_size = segment_size
        self.dtype = dtype
        self.shard_id = shard_id
        self.kind = kind
        
        # Buffers
        self._doc_ids: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._segment_counter = 0
        
        # Pending files awaiting finalization
        self._pending_paths: list[Path] = []
        
        # Ensure output directory exists
        self.outdir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: SegmentWriterConfig, kind: str = "papers") -> "SegmentWriter":
        """Create a SegmentWriter from a config dataclass."""
        return cls(
            outdir=config.segment_dir,
            segment_size=config.segment_size,
            dtype=config.dtype,
            shard_id=config.shard_id,
            kind=kind,
        )
    
    def add(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add a paper embedding to the buffer.
        
        Args:
            doc_id: Unique document identifier
            embedding: Embedding vector
        """
        self._doc_ids.append(doc_id)
        self._embeddings.append(embedding)
        
        if len(self._doc_ids) >= self.segment_size:
            self.flush()
    
    def write(self, *, doc_ids: list[str], vecs: np.ndarray) -> Path | None:
        """Batch write: add multiple embeddings and flush.
        
        This is the primary interface for cli.py batch flushes.
        Writes to .pending file (call finalize() after commit to make visible).
        
        Args:
            doc_ids: List of document IDs
            vecs: Embedding matrix (n_docs x dim)
        
        Returns:
            Path to the written .pending segment file, or None if empty
        """
        if len(doc_ids) == 0:
            return None
        
        # Add all to buffer and flush
        for doc_id, vec in zip(doc_ids, vecs):
            self._doc_ids.append(doc_id)
            self._embeddings.append(vec.reshape(1, -1))
        
        return self.flush()
    
    def flush(self) -> Path | None:
        """Flush buffered embeddings to a .pending file.
        
        The file is NOT visible to consumers until finalize() is called.
        
        Returns:
            Path to the written .pending segment file, or None if buffer was empty
        """
        if not self._doc_ids:
            return None
        
        # Stack embeddings and convert dtype
        X = np.vstack(self._embeddings)
        if self.dtype == "fp16":
            X = X.astype("float16")
        else:
            X = X.astype("float32")
        
        # Generate unique filename with _pending suffix BEFORE .npz
        # (np.savez_compressed auto-appends .npz if not present)
        seg_id = f"{self.shard_id}_{self._segment_counter}_{uuid.uuid4().hex[:8]}"
        prefix = PAPER_SEGMENT_PREFIX if self.kind == "papers" else self.kind
        filename = f"{prefix}_{seg_id}{PENDING_SUFFIX}{SEGMENT_EXTENSION}"
        path = self.outdir / filename
        
        # Save to disk with fsync for durability
        np.savez_compressed(
            path,
            doc_ids=np.array(self._doc_ids, dtype=object),
            vecs=X,  # Use 'vecs' key for consistency with ingest.py
            kind=np.array(self.kind),
        )
        
        # fsync to ensure durability before we consider it "written"
        try:
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except Exception:
            pass
        
        n = len(self._doc_ids)
        _eprint(f"[segment] Wrote {n} {self.kind} embeddings to {filename} (pending)")
        
        # Track pending path for finalization
        self._pending_paths.append(path)
        
        # Clear buffers
        self._doc_ids.clear()
        self._embeddings.clear()
        self._segment_counter += 1
        
        return path
    
    def finalize(self) -> int:
        """Atomically rename all _pending.npz files to final .npz names.
        
        Call this AFTER conn.commit() succeeds to make segments visible to consumers.
        
        Renames: paper_seg_0_0_abc_pending.npz -> paper_seg_0_0_abc.npz
        
        Returns:
            Number of segments finalized
        """
        finalized = 0
        for pending_path in self._pending_paths:
            if not pending_path.exists():
                continue
            
            # Remove _pending from the stem (before .npz)
            # e.g., paper_seg_0_0_abc_pending.npz -> paper_seg_0_0_abc.npz
            final_path = pending_path.with_name(
                pending_path.name.replace(f"{PENDING_SUFFIX}{SEGMENT_EXTENSION}", SEGMENT_EXTENSION)
            )
            
            try:
                os.replace(pending_path, final_path)
                finalized += 1
            except Exception as e:
                _eprint(f"[segment] WARNING: failed to finalize {pending_path.name}: {e}")
        
        if finalized:
            _eprint(f"[segment] Finalized {finalized} {self.kind} segment(s)")
        
        self._pending_paths.clear()
        return finalized
    
    def cleanup_pending(self) -> int:
        """Remove all pending files (for rollback scenarios).
        
        Call this if chunk write fails after paper write succeeded,
        to avoid orphan paper segments.
        
        Returns:
            Number of .pending files removed
        """
        removed = 0
        for pending_path in self._pending_paths:
            try:
                if pending_path.exists():
                    pending_path.unlink()
                    removed += 1
            except Exception:
                pass
        
        if removed:
            _eprint(f"[segment] Cleaned up {removed} {self.kind} pending file(s)")
        
        self._pending_paths.clear()
        return removed
    
    @property
    def pending_count(self) -> int:
        """Number of pending (not yet finalized) segment files."""
        return len(self._pending_paths)
    
    def close(self) -> None:
        """Flush any remaining data and close the writer.
        
        NOTE: Does NOT call finalize(). Caller must explicitly finalize()
        after DB commit to make segments visible.
        """
        self.flush()
    
    def __enter__(self) -> "SegmentWriter":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ChunkSegmentWriter:
    """Writes chunk embedding segments to disk with two-phase durability.
    
    Uses (paper_doc_id, ord) for identification.
    Buffers embeddings and flushes to disk when segment_size is reached.
    
    Two-phase write protocol:
    1. write() -> flushes to .npz.pending files (not visible to consumer)
    2. finalize() -> atomic rename .pending -> .npz (visible to consumer)
    
    If write fails, call cleanup_pending() to remove orphan .pending files.
    
    Segment file format (.npz):
    - paper_doc_ids: array of document IDs (strings)
    - ords: array of chunk ordinals (int32)
    - vecs: array of embedding vectors (float16 or float32)
    """
    
    def __init__(
        self,
        outdir: Path,
        segment_size: int = DEFAULT_EMBED_SEGMENT_SIZE,
        dtype: str = DEFAULT_EMBED_SEGMENT_DTYPE,
        shard_id: int = 0,
    ):
        """Initialize the chunk segment writer.
        
        Args:
            outdir: Directory to write segment files
            segment_size: Number of embeddings per segment file
            dtype: Storage dtype ("fp16" or "fp32")
            shard_id: Shard identifier for file naming
        """
        self.outdir = Path(outdir)
        self.segment_size = segment_size
        self.dtype = dtype
        self.shard_id = shard_id
        
        # Buffers
        self._doc_ids: list[str] = []
        self._ords: list[int] = []
        self._embeddings: list[np.ndarray] = []
        self._segment_counter = 0
        
        # Pending files awaiting finalization
        self._pending_paths: list[Path] = []
        
        # Ensure output directory exists
        self.outdir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: SegmentWriterConfig) -> "ChunkSegmentWriter":
        """Create a ChunkSegmentWriter from a config dataclass."""
        return cls(
            outdir=config.segment_dir,
            segment_size=config.segment_size,
            dtype=config.dtype,
            shard_id=config.shard_id,
        )
    
    def add(self, doc_id: str, ord_: int, embedding: np.ndarray) -> None:
        """Add a chunk embedding to the buffer.
        
        Args:
            doc_id: Parent document identifier
            ord_: Chunk ordinal within the document
            embedding: Embedding vector
        """
        self._doc_ids.append(doc_id)
        self._ords.append(ord_)
        self._embeddings.append(embedding)
        
        if len(self._doc_ids) >= self.segment_size:
            self.flush()
    
    def write(
        self, *, paper_doc_ids: list[str], ords: list[int], vecs: np.ndarray
    ) -> Path | None:
        """Batch write: add multiple chunk embeddings and flush.
        
        This is the primary interface for cli.py batch flushes.
        Writes to .pending file (call finalize() after commit to make visible).
        
        Args:
            paper_doc_ids: List of parent document IDs
            ords: List of chunk ordinals
            vecs: Embedding matrix (n_chunks x dim)
        
        Returns:
            Path to the written .pending segment file, or None if empty
        """
        if len(paper_doc_ids) == 0:
            return None
        
        # Add all to buffer and flush
        for doc_id, ord_, vec in zip(paper_doc_ids, ords, vecs):
            self._doc_ids.append(doc_id)
            self._ords.append(ord_)
            self._embeddings.append(vec.reshape(1, -1))
        
        return self.flush()
    
    def flush(self) -> Path | None:
        """Flush buffered embeddings to a .pending file.
        
        The file is NOT visible to consumers until finalize() is called.
        
        Returns:
            Path to the written .pending segment file, or None if buffer was empty
        """
        if not self._doc_ids:
            return None
        
        # Stack embeddings and convert dtype
        X = np.vstack(self._embeddings)
        if self.dtype == "fp16":
            X = X.astype("float16")
        else:
            X = X.astype("float32")
        
        # Generate unique filename with _pending suffix BEFORE .npz
        # (np.savez_compressed auto-appends .npz if not present)
        seg_id = f"{self.shard_id}_{self._segment_counter}_{uuid.uuid4().hex[:8]}"
        filename = f"{CHUNK_SEGMENT_PREFIX}_{seg_id}{PENDING_SUFFIX}{SEGMENT_EXTENSION}"
        path = self.outdir / filename
        
        # Save to disk with fsync for durability
        np.savez_compressed(
            path,
            paper_doc_ids=np.array(self._doc_ids, dtype=object),
            ords=np.array(self._ords, dtype="int32"),
            vecs=X,
            kind=np.array("chunks"),
        )
        
        # fsync to ensure durability before we consider it "written"
        try:
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except Exception:
            pass
        
        n = len(self._doc_ids)
        _eprint(f"[segment] Wrote {n} chunk embeddings to {filename} (pending)")
        
        # Track pending path for finalization
        self._pending_paths.append(path)
        
        # Clear buffers
        self._doc_ids.clear()
        self._ords.clear()
        self._embeddings.clear()
        self._segment_counter += 1
        
        return path
    
    def finalize(self) -> int:
        """Atomically rename all _pending.npz files to final .npz names.
        
        Call this AFTER conn.commit() succeeds to make segments visible to consumers.
        
        Renames: chunk_seg_0_0_abc_pending.npz -> chunk_seg_0_0_abc.npz
        
        Returns:
            Number of segments finalized
        """
        finalized = 0
        for pending_path in self._pending_paths:
            if not pending_path.exists():
                continue
            
            # Remove _pending from the stem (before .npz)
            # e.g., chunk_seg_0_0_abc_pending.npz -> chunk_seg_0_0_abc.npz
            final_path = pending_path.with_name(
                pending_path.name.replace(f"{PENDING_SUFFIX}{SEGMENT_EXTENSION}", SEGMENT_EXTENSION)
            )
            
            try:
                os.replace(pending_path, final_path)
                finalized += 1
            except Exception as e:
                _eprint(f"[segment] WARNING: failed to finalize {pending_path.name}: {e}")
        
        if finalized:
            _eprint(f"[segment] Finalized {finalized} chunk segment(s)")
        
        self._pending_paths.clear()
        return finalized
    
    def cleanup_pending(self) -> int:
        """Remove all pending files (for rollback scenarios).
        
        Call this if write fails, to avoid orphan segments.
        
        Returns:
            Number of .pending files removed
        """
        removed = 0
        for pending_path in self._pending_paths:
            try:
                if pending_path.exists():
                    pending_path.unlink()
                    removed += 1
            except Exception:
                pass
        
        if removed:
            _eprint(f"[segment] Cleaned up {removed} chunk pending file(s)")
        
        self._pending_paths.clear()
        return removed
    
    @property
    def pending_count(self) -> int:
        """Number of pending (not yet finalized) segment files."""
        return len(self._pending_paths)
    
    def close(self) -> None:
        """Flush any remaining data and close the writer.
        
        NOTE: Does NOT call finalize(). Caller must explicitly finalize()
        after DB commit to make segments visible.
        """
        self.flush()
    
    def __enter__(self) -> "ChunkSegmentWriter":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
