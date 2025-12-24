# litkit/ingest/__init__.py
"""TAR-based document ingestion for litkit.

This module provides utilities for reading article metadata and body text
from JATS/NXML files inside tar archives, plus helpers for:
- Shard assignment for multi-node builds
- Tar file type detection
- XML parsing and chunking
"""

from litkit.ingest.ingest import (
    # Type definitions
    ArticleMeta,
    TarMemberMeta,
    # Constants
    CHUNK_TARGET_CHARS,
    BODY_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    # XML parsing
    parse_jats,
    parse_pubmed,
    parse_xml_fileobj,
    # Chunking
    pack_paragraphs,
    # TAR helpers
    iter_tar_paths,
    iter_tar_xml_member_names,
    count_tar_xml_members,
    iter_tar_xml_streams,
    parallel_iter_tar_articles,
)
from litkit.ingest.detection import (
    is_uncompressed_tar,
)
from litkit.ingest.sharding import (
    shard_filter,
)

__all__ = [
    # Type definitions
    "ArticleMeta",
    "TarMemberMeta",
    # Constants
    "CHUNK_TARGET_CHARS",
    "BODY_MIN_CHARS",
    "CHUNK_OVERLAP_CHARS",
    # XML parsing
    "parse_jats",
    "parse_pubmed",
    "parse_xml_fileobj",
    # Chunking
    "pack_paragraphs",
    # TAR helpers
    "iter_tar_paths",
    "iter_tar_xml_member_names",
    "count_tar_xml_members",
    "iter_tar_xml_streams",
    "parallel_iter_tar_articles",
    # Detection
    "is_uncompressed_tar",
    # Sharding
    "shard_filter",
]
