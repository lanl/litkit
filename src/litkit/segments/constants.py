# litkit/segments/constants.py
"""Constants for segment file configuration in litkit."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Segment File Configuration
# ---------------------------------------------------------------------------

# ~131,072 vectors/segment ≈ 0.4 GB per file (fp32, dim=768)
# Half that if stored as fp16.
DEFAULT_EMBED_SEGMENT_SIZE: int = int(
    os.environ.get("LITKIT_EMBED_SEGMENT_SIZE", "131072")
)

# Embedding storage format: fp16 (half precision) or fp32 (single precision)
# fp16 saves ~50% disk space with minimal quality loss for normalized vectors
DEFAULT_EMBED_SEGMENT_DTYPE: str = os.environ.get(
    "LITKIT_EMBED_SEGMENT_DTYPE", "fp16"
)

# ---------------------------------------------------------------------------
# Segment File Naming
# ---------------------------------------------------------------------------

# Segment file patterns
PAPER_SEGMENT_PREFIX = "paper_seg"
CHUNK_SEGMENT_PREFIX = "chunk_seg"
SEGMENT_EXTENSION = ".npz"

# Build metadata file
BUILD_META_FILENAME = "build_meta.json"

# Producer completion signal file pattern
PRODUCER_DONE_PATTERN = "producer_{shard_id}.done"
