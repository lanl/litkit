# litkit/index/constants.py
"""Constants for FAISS index configuration in litkit."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Product Quantizer Configuration
# ---------------------------------------------------------------------------

# Single source of truth for product-quantizer bits.
# Keep this in sync across training & ivfpq_index.
PQ_BITS: int = 8

# ---------------------------------------------------------------------------
# Introspection Fallback Configuration
# ---------------------------------------------------------------------------

# Whether to use faiss.downcast_index as a fallback for index introspection.
# This is useful for read-only introspection of SWIG base-class objects.
# Can be controlled via environment variable for debugging.
USE_DOWNCAST_FALLBACK: bool = os.environ.get(
    "LITKIT_USE_DOWNCAST_FALLBACK", "1"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Index Type Constants
# ---------------------------------------------------------------------------

# Index kind identifiers returned by unwrap_core_and_kind
INDEX_KIND_FLAT = "flat"
INDEX_KIND_HNSW = "hnsw"
INDEX_KIND_IVF = "ivf"
INDEX_KIND_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Default Parameters
# ---------------------------------------------------------------------------

# HNSW defaults
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 128

# IVF-PQ defaults
DEFAULT_IVF_NLIST = 16384
DEFAULT_PQ_M = 64
