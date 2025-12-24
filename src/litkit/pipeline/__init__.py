# litkit/pipeline/__init__.py
"""Document processing pipeline for litkit.

This module provides the article processing logic extracted from cli.py:
- Processing context with shared state
- Embedding buffers with batching
- Article processing function
- Buffer flush utilities
"""

from litkit.pipeline.helpers import (
    dedupe_ids_and_texts,
    check_file_processed,
    canonical_tar_path,
)
from litkit.pipeline.buffers import (
    EmbeddingBuffer,
    PaperBuffer,
    ChunkBuffer,
)
from litkit.pipeline.context import (
    ProcessingContext,
)
from litkit.pipeline.article import (
    process_article,
    flush_paper_buffer,
    flush_chunk_buffer,
)

__all__ = [
    # Helpers
    "dedupe_ids_and_texts",
    "check_file_processed",
    "canonical_tar_path",
    # Buffers
    "EmbeddingBuffer",
    "PaperBuffer",
    "ChunkBuffer",
    # Context
    "ProcessingContext",
    # Article processing
    "process_article",
    "flush_paper_buffer",
    "flush_chunk_buffer",
]
