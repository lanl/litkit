# litkit/pipeline/buffers.py
"""Embedding buffers for batched document processing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingBuffer:
    """Base buffer for batching embeddings.
    
    Accumulates IDs and texts until batch_size is reached,
    then signals for flushing.
    """
    
    ids: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    batch_size: int = 10000
    
    def add(self, id_: int, text: str) -> None:
        """Add an item to the buffer."""
        self.ids.append(id_)
        self.texts.append(text)
    
    def should_flush(self) -> bool:
        """Check if buffer has reached batch size."""
        return len(self.ids) >= self.batch_size
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ids.clear()
        self.texts.clear()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.ids) == 0


@dataclass
class PaperBuffer(EmbeddingBuffer):
    """Buffer for paper embeddings with doc_id tracking.
    
    Tracks doc_ids for content-addressed segment storage.
    """
    
    doc_ids: list[str] = field(default_factory=list)
    
    def add_paper(
        self,
        paper_id: int,
        text: str,
        doc_id: str,
    ) -> None:
        """Add a paper to the buffer.
        
        Args:
            paper_id: Database paper ID
            text: Paper text (title + abstract)
            doc_id: Document ID for segment storage
        """
        self.ids.append(paper_id)
        self.texts.append(text)
        self.doc_ids.append(doc_id)
    
    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()
        self.doc_ids.clear()
    
    def pop_all(self) -> tuple[list[int], list[str], list[str]]:
        """Pop all items and return (ids, texts, doc_ids)."""
        ids = list(self.ids)
        texts = list(self.texts)
        doc_ids = list(self.doc_ids)
        self.clear()
        return ids, texts, doc_ids


@dataclass
class ChunkBuffer(EmbeddingBuffer):
    """Buffer for chunk embeddings with doc_id and ordinal tracking.
    
    Tracks parent paper doc_ids and chunk ordinals for segment storage.
    """
    
    doc_ids: list[str] = field(default_factory=list)
    ords: list[int] = field(default_factory=list)
    
    def add_chunk(
        self,
        chunk_id: int,
        text: str,
        doc_id: str,
        ord_: int,
    ) -> None:
        """Add a chunk to the buffer.
        
        Args:
            chunk_id: Database chunk ID
            text: Chunk text
            doc_id: Parent paper's document ID
            ord_: Chunk ordinal within the paper
        """
        self.ids.append(chunk_id)
        self.texts.append(text)
        self.doc_ids.append(doc_id)
        self.ords.append(ord_)
    
    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()
        self.doc_ids.clear()
        self.ords.clear()
    
    def pop_all(self) -> tuple[list[int], list[str], list[str], list[int]]:
        """Pop all items and return (ids, texts, doc_ids, ords)."""
        ids = list(self.ids)
        texts = list(self.texts)
        doc_ids = list(self.doc_ids)
        ords = list(self.ords)
        self.clear()
        return ids, texts, doc_ids, ords
