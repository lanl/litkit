# litkit/segments/writer.py
"""Segment writers for embedding data in litkit."""

from __future__ import annotations

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


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


@dataclass
class SegmentWriterConfig:
    """Configuration for segment writers."""
    
    segment_dir: Path
    segment_size: int = DEFAULT_EMBED_SEGMENT_SIZE
    dtype: str = DEFAULT_EMBED_SEGMENT_DTYPE
    shard_id: int = 0


class SegmentWriter:
    """Writes paper embedding segments to disk.
    
    Uses doc_id (globally unique file path) for identification.
    Buffers embeddings and flushes to disk when segment_size is reached.
    
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
    ):
        """Initialize the segment writer.
        
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
        self._embeddings: list[np.ndarray] = []
        self._segment_counter = 0
        
        # Ensure output directory exists
        self.outdir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: SegmentWriterConfig) -> "SegmentWriter":
        """Create a SegmentWriter from a config dataclass."""
        return cls(
            outdir=config.segment_dir,
            segment_size=config.segment_size,
            dtype=config.dtype,
            shard_id=config.shard_id,
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
    
    def flush(self) -> Path | None:
        """Flush buffered embeddings to disk.
        
        Returns:
            Path to the written segment file, or None if buffer was empty
        """
        if not self._doc_ids:
            return None
        
        # Stack embeddings and convert dtype
        X = np.vstack(self._embeddings)
        if self.dtype == "fp16":
            X = X.astype("float16")
        else:
            X = X.astype("float32")
        
        # Generate unique filename
        seg_id = f"{self.shard_id}_{self._segment_counter}_{uuid.uuid4().hex[:8]}"
        filename = f"{PAPER_SEGMENT_PREFIX}_{seg_id}{SEGMENT_EXTENSION}"
        path = self.outdir / filename
        
        # Save to disk
        np.savez_compressed(
            path,
            doc_ids=np.array(self._doc_ids, dtype=object),
            embeddings=X,
        )
        
        n = len(self._doc_ids)
        _eprint(f"[segment] Wrote {n} paper embeddings to {filename}")
        
        # Clear buffers
        self._doc_ids.clear()
        self._embeddings.clear()
        self._segment_counter += 1
        
        return path
    
    def close(self) -> None:
        """Flush any remaining data and close the writer."""
        self.flush()
    
    def __enter__(self) -> "SegmentWriter":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ChunkSegmentWriter:
    """Writes chunk embedding segments to disk.
    
    Uses (paper_doc_id, ord) for identification.
    Buffers embeddings and flushes to disk when segment_size is reached.
    
    Segment file format (.npz):
    - doc_ids: array of document IDs (strings)
    - ords: array of chunk ordinals (int32)
    - embeddings: array of embedding vectors (float16 or float32)
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
    
    def flush(self) -> Path | None:
        """Flush buffered embeddings to disk.
        
        Returns:
            Path to the written segment file, or None if buffer was empty
        """
        if not self._doc_ids:
            return None
        
        # Stack embeddings and convert dtype
        X = np.vstack(self._embeddings)
        if self.dtype == "fp16":
            X = X.astype("float16")
        else:
            X = X.astype("float32")
        
        # Generate unique filename
        seg_id = f"{self.shard_id}_{self._segment_counter}_{uuid.uuid4().hex[:8]}"
        filename = f"{CHUNK_SEGMENT_PREFIX}_{seg_id}{SEGMENT_EXTENSION}"
        path = self.outdir / filename
        
        # Save to disk
        np.savez_compressed(
            path,
            doc_ids=np.array(self._doc_ids, dtype=object),
            ords=np.array(self._ords, dtype="int32"),
            embeddings=X,
        )
        
        n = len(self._doc_ids)
        _eprint(f"[segment] Wrote {n} chunk embeddings to {filename}")
        
        # Clear buffers
        self._doc_ids.clear()
        self._ords.clear()
        self._embeddings.clear()
        self._segment_counter += 1
        
        return path
    
    def close(self) -> None:
        """Flush any remaining data and close the writer."""
        self.flush()
    
    def __enter__(self) -> "ChunkSegmentWriter":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
