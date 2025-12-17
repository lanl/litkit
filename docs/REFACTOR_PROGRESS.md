# LitKit Refactor Progress

**Last Updated:** 2025-12-17 (Phase 1 Complete)  
**Branch:** `feature/refactor-scope`

## Executive Summary

This document tracks the progress of extracting reusable modules from the monolithic `cli.py` (~5,000 lines) into focused, testable modules. The goal is to transform `cli.py` into a thin CLI layer that dispatches to well-structured library code.

## Completed Work (Jobs 1-8)

### Module Extraction Summary

| Job | Module | Files | Lines | Commit | Status |
|-----|--------|-------|-------|--------|--------|
| 1 | Lock Scope | - | ~50 | `2692d2c` | ✅ Complete |
| 2 | `litkit.config.paths` | 1 | ~200 | `80e4d97` | ✅ Complete |
| 5 | `litkit.concurrent` | 2 | ~400 | `dd0705b`, `586647a` | ✅ Complete |
| 3 | `litkit.db` | 7 | ~815 | `56fdb5c` | ✅ Complete |
| 4 | `litkit.index` | 9 | ~933 | `cb65f99` | ✅ Complete |
| 6 | `litkit.segments` | 6 | ~949 | `396d30e` | ✅ Complete |
| 7 | `litkit.ingest` | 4 | ~464 | `b816d88` | ✅ Complete |
| 8 | `litkit.pipeline` | 5 | ~561 | `561ed5d` | ✅ Complete |

**Total extracted:** ~4,370 lines across 8 modules  
**Dead code removed (Phase 1):** ~93 lines

### Module Structure

```
src/litkit/
├── concurrent/         # Locking primitives
│   ├── __init__.py
│   └── locking.py      # FileLock, flock_guard
│
├── config/             # Configuration and paths
│   ├── __init__.py
│   └── paths.py        # WorkspacePaths, get_default_paths
│
├── db/                 # SQLite operations
│   ├── __init__.py
│   ├── connection.py   # connect_db, init_db
│   ├── schema.py       # SCHEMA constant
│   ├── queries.py      # register_file, already_processed
│   ├── sharding.py     # init_shard_db, merge_shard_databases
│   ├── indexing.py     # mark_in_index, reconcile_sqlite_flags
│   └── temp_tables.py  # Temporary table utilities
│
├── index/              # FAISS index operations
│   ├── __init__.py
│   ├── constants.py    # PQ_BITS, paths
│   ├── factory.py      # ivfpq_index, hnsw_index, flat_ip_index
│   ├── io.py           # faiss_save, faiss_load, faiss_save_force
│   ├── introspection.py # unwrap_core_and_kind, kind_and_core
│   ├── ids.py          # make_id_selector, safe_remove_ids
│   ├── search.py       # faiss_search, auto_set_nprobe
│   ├── training.py     # effective_nlist, safe_pq_m
│   └── dedup.py        # add_with_ids_dedup
│
├── segments/           # Embedding segment I/O
│   ├── __init__.py
│   ├── constants.py    # Segment size defaults
│   ├── writer.py       # SegmentWriter, ChunkSegmentWriter
│   ├── metadata.py     # write_build_meta, read_build_meta
│   ├── coordination.py # ProducerCoordinator, ConsumerCoordinator
│   └── ingest.py       # ingest_paper_segments, ingest_chunk_segments
│
├── ingest/             # Document ingestion
│   ├── __init__.py
│   ├── detection.py    # is_uncompressed_tar
│   ├── sharding.py     # shard_filter (load-balanced)
│   └── ingest.py       # (existing XML parsing, unchanged)
│
└── pipeline/           # Document processing pipeline
    ├── __init__.py
    ├── helpers.py      # dedupe_ids_and_texts, check_file_processed
    ├── buffers.py      # EmbeddingBuffer, PaperBuffer, ChunkBuffer
    ├── context.py      # ProcessingContext dataclass
    └── article.py      # process_article, flush_paper_buffer, flush_chunk_buffer
```

## Completed Work (Job 9)

### Phase 1: Simple Function Migration ✅ Complete

| Function | Status | Notes |
|----------|--------|-------|
| `_safe_pq_m` → `safe_pq_m` | ✅ Complete | All 4 calls replaced |
| `_is_uncompressed_tar` → `is_uncompressed_tar` | ✅ Complete | All calls replaced |
| `_shard_filter` → `shard_filter` | ✅ Complete | ~65-line nested function deleted |
| `_dedupe_ids_and_texts` → `dedupe_ids_and_texts` | ✅ Complete | All 4 calls replaced |

**Commits:**
- `fb9e635` - Start migration: add imports
- `7cd61b8` - Complete Phase 1: replace all calls, delete dead code (~93 lines removed)

### Dead Code Removed

| Function | Lines | Location |
|----------|-------|----------|
| `_dedupe_ids_and_texts` | ~10 | line 888 |
| `_safe_pq_m` | ~12 | line 1779 |
| `_is_uncompressed_tar` | ~6 | line 2606 |
| `_shard_filter` | ~65 | line 3467 (nested) |
| **Total** | **~93** | |

## Future Work

### Phase 2: Import-Time Side Effect Isolation

**Goal:** Make `cli.py` importable without side effects.

**Current issues:**
- `ROOT`, `WORKSPACE` set at import time
- Directories created at import time
- Environment variables set at import time

**Required changes:**
1. Move all path resolution into `prepare_environment()` function
2. Call `prepare_environment()` from `main()` only
3. Remove global variable assignments at module scope

### Phase 3: Remaining Domain Extraction

| Domain | Target Module | Size Estimate |
|--------|---------------|---------------|
| LLM/Prompts | `litkit.llm` | ~200 lines |
| Retrieval | `litkit.retrieval` | ~300 lines |
| Progress | `litkit.progress` | ~150 lines |

### Phase 4: Orchestration Refactor

**Goal:** Split `build_or_update_indices()` (~800 lines) into focused functions.

**Proposed structure:**
```python
# litkit/build/orchestration.py
def run_single_node(config: BuildConfig, services: Services) -> BuildResult
def run_producer(config: BuildConfig, services: Services) -> BuildResult
def run_consumer(config: BuildConfig, services: Services) -> BuildResult
def run_init_indices(config: BuildConfig, services: Services) -> BuildResult

# litkit/build/config.py
@dataclass
class BuildConfig:
    mode: Literal["single", "producer", "consumer", "init"]
    shard_id: int
    num_shards: int
    # ... other config fields
```

## Critique Scorecard

Based on the detailed critique of the monolithic `cli.py`:

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | Scope creep | 🟡 Partial | Modules created, not integrated |
| 2 | Massive file size | 🟡 Partial | ~4,400 lines extracted |
| 3 | Import-time side effects | ❌ Not addressed | Phase 2 |
| 4 | Global state | 🟡 Partial | `ProcessingContext` created |
| 5 | Monolithic build function | ❌ Not addressed | Phase 4 |
| 6 | Locking complexity | ✅ Addressed | `FileLock` centralized |
| 7 | Duplication | 🟡 Partial | New modules coexist with old |
| 8 | Error handling | ❌ Not addressed | Future work |
| 9 | LLM leakage | ❌ Not addressed | Phase 3 |
| 10 | Config sprawl | 🟡 Partial | `WorkspacePaths` created |
| 11 | Testing | 🟡 Partial | Modules testable |
| 12 | Style nits | ❌ Not addressed | Future work |

## Testing

### Build Validation

After each change, run:
```bash
./test_build.sh --clean
```

Expected output:
```
✅ BUILD VALIDATION PASSED
```

### Import Test

Quick smoke test:
```bash
python -c "from litkit.cli import main; print('OK')"
```

### Module Import Test

```bash
python -c "
from litkit.db import init_db
from litkit.index import safe_pq_m, faiss_save
from litkit.segments import SegmentWriter
from litkit.ingest import is_uncompressed_tar, shard_filter
from litkit.pipeline import ProcessingContext
print('All modules import successfully')
"
```

## Decisions Log

### 2025-12-17: Option B Selected for Job 8

**Decision:** Extract article processing as a new `litkit.pipeline` module rather than integrating into existing modules.

**Rationale:**
- Cleaner separation of concerns
- Allows cli.py to continue using its own implementations during transition
- Lower risk of regressions

### 2025-12-17: Gradual Migration Strategy

**Decision:** Replace simple functions first, defer context/flush migration.

**Rationale:**
- `ProcessingContext` has different API than `_IngestContext`
- Buffer management patterns differ
- Need adapter layer for full migration

## Related Files

- `docs/REFACTOR_ROADMAP.md` - Original planning document
- `CODE_REVIEW_SUMMARY.md` - Pre-refactor code review
- `.git/` - Full commit history on `feature/refactor-scope`
