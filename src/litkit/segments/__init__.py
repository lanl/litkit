# litkit/segments/__init__.py
"""Segment file management for distributed litkit builds.

This module handles embedding segment I/O for producer/consumer workflows:
- Segment writers for papers and chunks
- Producer/Consumer coordination for multi-node builds
- Build metadata for shard consistency
- Segment ingestion into FAISS indices
"""

from litkit.segments.constants import (
    DEFAULT_EMBED_SEGMENT_SIZE,
    DEFAULT_EMBED_SEGMENT_DTYPE,
)
from litkit.segments.writer import (
    SegmentWriter,
    ChunkSegmentWriter,
    SegmentWriterConfig,
    cleanup_orphan_pending_files,
    PENDING_SUFFIX,
)
from litkit.segments.metadata import (
    write_build_meta,
    read_build_meta,
    validate_shard_consistency,
    has_segment_files,
)
from litkit.segments.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    shard_ckpt_path,
    clear_shard_checkpoints,
)
from litkit.segments.coordination import (
    ProducerCoordinator,
    ConsumerCoordinator,
)
from litkit.segments.ingest import (
    ingest_paper_segments,
    ingest_chunk_segments,
    SegmentIngestConfig,
)

__all__ = [
    # Constants
    "DEFAULT_EMBED_SEGMENT_SIZE",
    "DEFAULT_EMBED_SEGMENT_DTYPE",
    # Writers
    "SegmentWriter",
    "ChunkSegmentWriter",
    "SegmentWriterConfig",
    "cleanup_orphan_pending_files",
    "PENDING_SUFFIX",
    # Metadata
    "write_build_meta",
    "read_build_meta",
    "validate_shard_consistency",
    "has_segment_files",
    # Checkpoints
    "load_checkpoint",
    "save_checkpoint",
    "shard_ckpt_path",
    "clear_shard_checkpoints",
    # Coordination
    "ProducerCoordinator",
    "ConsumerCoordinator",
    # Ingestion
    "ingest_paper_segments",
    "ingest_chunk_segments",
    "SegmentIngestConfig",
]
