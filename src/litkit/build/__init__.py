# src/litkit/build/__init__.py
"""Build orchestration module for litkit.

This module contains:
- Helper functions for text chunking and deduplication
- Backfill and reconciliation logic for FAISS/SQLite consistency
- (Future) BuildConfig and main orchestrator
"""

from litkit.build.helpers import (
    pack_paragraphs,
    dedupe_papers_with_doc_ids,
    dedupe_chunks_with_doc_ids,
    ensure_parent,
    maybe_fsync_dir,
)
from litkit.build.backfill import (
    backfill_unindexed_vectors,
    reconcile_sqlite_flags_with_faiss,
    post_build_sanity_check,
)
from litkit.build.config import (
    BuildConfig,
    build_config_from_args,
)
from litkit.build.indices import (
    init_empty_indices,
)
from litkit.build.consume import (
    run_consume_only_mode,
)

__all__ = [
    # helpers
    "pack_paragraphs",
    "dedupe_papers_with_doc_ids",
    "dedupe_chunks_with_doc_ids",
    "ensure_parent",
    "maybe_fsync_dir",
    # backfill
    "backfill_unindexed_vectors",
    "reconcile_sqlite_flags_with_faiss",
    "post_build_sanity_check",
    # config
    "BuildConfig",
    "build_config_from_args",
    # indices
    "init_empty_indices",
    # consume
    "run_consume_only_mode",
]
