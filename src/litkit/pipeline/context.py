# litkit/pipeline/context.py
"""Processing context for document ingestion pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from litkit.pipeline.buffers import PaperBuffer, ChunkBuffer

if TYPE_CHECKING:
    import faiss
    from litkit.segments.writer import SegmentWriter, ChunkSegmentWriter


class Embedder(Protocol):
    """Protocol for embedder objects."""
    
    def encode(
        self,
        texts: list[str],
        progress_label: str = "",
    ) -> Any:
        """Encode texts to embeddings."""
        ...


@dataclass
class ProcessingContext:
    """Shared state for document processing pipeline.
    
    This context holds all resources needed for article ingestion:
    - Database connection
    - Embedder instances
    - FAISS indices (for single-node or consumer mode)
    - Segment writers (for producer mode)
    - Lock paths for coordination
    - Embedding buffers for batching
    """
    
    # Database
    conn: sqlite3.Connection
    
    # Embedders
    paper_embedder: Embedder
    chunk_embedder: Embedder
    
    # FAISS indices (None for producer-only mode)
    paper_index: Any = None  # faiss.Index
    chunk_index: Any = None  # faiss.Index
    
    # Segment writers (None for single-node/consumer mode)
    paper_seg_writer: Any = None  # SegmentWriter
    chunk_seg_writer: Any = None  # ChunkSegmentWriter
    
    # Lock paths for coordination
    db_lock_path: Path | None = None
    faiss_lock_path: Path | None = None
    
    # Index paths for saving
    paper_index_path: Path | None = None
    chunk_index_path: Path | None = None
    
    # Embedding buffers
    paper_buffer: PaperBuffer = field(default_factory=PaperBuffer)
    chunk_buffer: ChunkBuffer = field(default_factory=ChunkBuffer)
    
    # Mode flags
    is_faiss_writer: bool = False
    """True if this process writes to FAISS indices."""
    
    is_producer: bool = False
    """True if this process writes segment files (not FAISS)."""
    
    # Statistics
    papers_added: int = 0
    chunks_added: int = 0
    files_processed: int = 0
    
    def cursor(self) -> sqlite3.Cursor:
        """Get a database cursor."""
        return self.conn.cursor()
