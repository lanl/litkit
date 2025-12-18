# Litkit Refactor Progress

## Overview

This document tracks the progress of the litkit refactoring effort, which extracts functionality from the monolithic `cli.py` into well-organized modules.

## Completed Work

### Phase 1: Module Extraction (COMPLETE)

The following modules have been extracted from `cli.py`:

#### 1. `litkit.concurrent` - Locking utilities
- `locking.py` - FileLock class, lock depth tracking
- Re-exported via `__init__.py`

#### 2. `litkit.db` - Database operations  
- `connection.py` - Connection factory, busy timeout handling
- `schema.py` - SCHEMA constant, DDL for papers/chunks/files tables
- `sharding.py` - Shard-specific DB initialization, merge_shard_databases()
- `queries.py` - Common SQL helpers (mark_in_index, already_processed, etc.)
- `indexing.py` - ensure_in_index_columns migration
- `temp_tables.py` - Temporary table helpers for candidate papers
- Re-exported via `__init__.py`

#### 3. `litkit.index` - FAISS index operations
- `constants.py` - PQ_BITS, USE_DOWNCAST_FALLBACK constants
- `factory.py` - Index creation (hnsw_index, flat_ip_index, ivfpq_index)
- `io.py` - Index save/load with throttling and fsync
- `introspection.py` - unwrap_core_and_kind, kind_and_core, report_faiss_index
- `ids.py` - ID selector creation, safe_remove_ids
- `search.py` - FAISS search with temporary param adjustment
- `dedup.py` - add_with_ids_dedup, faiss_present_ids
- `training.py` - effective_nlist, auto_set_nprobe, pick_nprobe
- Re-exported via `__init__.py`

#### 4. `litkit.segments` - Embedding segment I/O
- `constants.py` - DEFAULT_EMBED_SEGMENT_SIZE, DEFAULT_EMBED_SEGMENT_DTYPE
- `writer.py` - _SegmentWriter, _ChunkSegmentWriter classes
- `metadata.py` - Build metadata for shard consistency
- `coordination.py` - ProducerCoordinator, ConsumerCoordinator classes
- `ingest.py` - Segment ingestion functions
- `checkpoint.py` - Per-shard checkpoint management (NEW - 2024-12-17)
- Re-exported via `__init__.py`

#### 5. `litkit.ingest` - Tar/XML ingestion
- `detection.py` - is_uncompressed_tar helper
- `sharding.py` - shard_filter for load-balanced tar distribution
- Extended `__init__.py` to re-export new helpers

#### 6. `litkit.pipeline` - Build pipeline helpers
- `helpers.py` - dedupe_ids_and_texts function
- `buffers.py` - Batch buffer management (planned)
- `context.py` - _IngestContext class
- `article.py` - Article ingestion functions
- Re-exported via `__init__.py`

### Phase 2: Import-Time Purity (COMPLETE)

All import-time side effects have been eliminated from `cli.py`:

1. **Lazy Runtime Initialization**
   - Created `_Runtime` dataclass for path constants
   - Implemented `get_runtime()` with thread-safe lazy init
   - Module-level `__getattr__` for backward-compatible path access
   - All mkdir/env var setting deferred until first use

2. **Import-Time Side Effects Removed**
   - Python version check moved to `main()`
   - `faiss.cvar.seed` moved to `_init_runtime()`
   - `import litkit.cli` now has zero I/O or side effects

### Phase 3: CLI Wire-up and Schema Fixes (COMPLETE - 2024-12-17)

The extracted modules are now the canonical source, imported and used by `cli.py`:

#### DB Module Integration
- ✅ All 16 DB functions imported from `litkit.db` and used in `cli.py`
- ✅ Schema column mismatch fixed: `mtime_ns` → `mtime` in `register_file()` and queries
- ✅ `merge_shard_databases()` rewritten with correct column names matching schema
- ✅ Preload functions (`preload_paper_id_map`, `preload_chunk_id_map`) verified

#### Other Module Integration
- ✅ `from litkit.index import safe_pq_m`
- ✅ `from litkit.pipeline import dedupe_ids_and_texts`
- ✅ `from litkit.ingest import is_uncompressed_tar, shard_filter`

#### Build Validation
- ✅ Import smoke test passes
- ✅ Multi-node architecture verified (producer → segment → consumer flow)

### Phase 3.5: doc_id Canonicalization Fix (COMPLETE - 2024-12-17)

Fixed a critical correctness bug in doc_id handling for multi-node builds:

#### The Bug
When a paper already exists in the DB (matched by pmcid/pmid), the code used the
*current tar path* as `doc_id` for segment files, but the DB retained the *original*
`doc_id` from first insertion. This caused segment ingestion to fail silently when
the consumer couldn't resolve the new doc_id against the merged DB.

#### The Fix
1. **Canonical doc_id Resolution**: When reusing an existing paper row, fetch both
   `id` and `doc_id` from the DB. Use the DB's canonical `doc_id` (if present) for
   segment files instead of the current tar path.

2. **Resolution Failure Logging**: Added WARNING logs in `_ingest_paper_segments()`
   and `_ingest_chunk_segments()` when doc_id → paper_id or (doc_id, ord) → chunk_id
   resolution fails. This surfaces canonicalization issues early.

#### Changes Made
- Modified paper lookup to `SELECT id, doc_id FROM papers WHERE pmcid/pmid=?`
- Added `canon_doc_id` variable that uses existing doc_id or falls back to current path
- Updated all buffer appends to use `canon_doc_id` instead of raw path
- Added `missing` counter and WARNING log in segment ingestion functions

### Phase 3.6: Multi-Process Checkpoint Safety (COMPLETE - 2024-12-17)

Fixed the "last writer wins" checkpoint bug in multi-node builds:

#### The Bug
Multiple producer processes shared a single checkpoint file. Each process:
1. Loaded the checkpoint once at start
2. Mutated its local copy during processing
3. Wrote the entire dict back under a lock

This caused **checkpoint data loss**: Shard 0 writes `{"/tar0": 500}`, then Shard 1
(which loaded `{}`) writes `{"/tar1": 500}` → overwrites `/tar0` entry. On restart,
those tars would be re-scanned from zero.

#### The Fix
1. **Per-shard checkpoints**: Each producer writes to `build_checkpoint_shard_XX.json`
   - No conflicts between producers
   - Each shard owns its own checkpoint file completely

2. **New checkpoint module**: `litkit/segments/checkpoint.py`
   - `shard_ckpt_path(sqlite_dir, shard_id)` - per-shard path helper
   - `load_checkpoint(ckpt_path, shard_id=None)` - load shared or per-shard
   - `save_checkpoint(..., shard_id=None)` - save to shared or per-shard
   - `clear_shard_checkpoints(sqlite_dir, num_shards)` - cleanup on resharding

3. **Resharding support**: When shard count changes (M→N), the code now:
   - WARNs instead of ERRORs
   - Clears invalidated per-shard checkpoint files
   - Falls back to DB-based resume (`already_processed()` checks)
   - Updates build metadata to reflect new shard count

#### Commits
- `fd41c14`: Add per-shard checkpoint module (WIP)
- `f709070`: Wire up per-shard checkpoints in build_or_update_indices

### Phase 4.1: SQLite Configuration Fixes (COMPLETE - 2024-12-17)

Fixed busy_timeout CLI flag propagation bug:

**The Bug**
- `--sqlite-busy-timeout-ms` CLI flag wasn't being honored by `db_connect_db()` calls
- Root cause: `DEFAULT_BUSY_TIMEOUT_MS` was evaluated at import time, not call time
- Result: query-time connections used 30s default while build-time used user's value

**The Fix**
1. Added `_get_busy_timeout()` helper in `litkit/db/connection.py` that reads from env at call time
2. CLI sets `os.environ["LITKIT_SQLITE_BUSY_TIMEOUT_MS"]` in `main()` after argparse
3. Added default to `litkit/config/paths.py` `setup_environment()` for consistency

**Cleaned up unused imports:**
- Removed `db_ensure_temp_candidates_table` (already called internally by `load_temp_candidates`)
- Removed `db_DEFAULT_BUSY_TIMEOUT_MS` (no longer needed)

### Dead Code Removal (COMPLETE - 2024-12-17)

Removed unused local implementations that were shadowed by imported versions:
- ✅ `_mark_in_index()` - removed (all calls use `db_mark_in_index`)
- ✅ `_flush_pending_marks()` - removed (all calls use `db_flush_pending_marks`)
- ✅ `_SegmentWriter` class - removed (using `SegmentWriter` from litkit.segments)
- ✅ `_ChunkSegmentWriter` class - removed (using `ChunkSegmentWriter` from litkit.segments)
- ✅ `ProducerCoordinator` class - removed (using `SegProducerCoordinator` from litkit.segments)
- ✅ `ConsumerCoordinator` class - removed (using `SegConsumerCoordinator` from litkit.segments)
- ✅ `_write_build_meta()` - removed (using `seg_write_build_meta`)
- ✅ `_read_build_meta()` - removed (using `seg_read_build_meta`)
- ✅ `_has_segment_files()` - removed (using `seg_has_segment_files`)
- ✅ `_validate_shard_consistency()` - removed (using `seg_validate_shard_consistency`)
- ✅ `load_checkpoint()` / `save_checkpoint()` - now using `seg_load_checkpoint()` / `seg_save_checkpoint()`

### Multi-Node Correctness (VERIFIED - 2024-12-17)

The multi-node build architecture works correctly:

1. **Producers** (N nodes): 
   - Each writes to shard-specific SQLite DB (`shard_XX.sqlite3`)
   - Each writes to shard-specific checkpoint (`build_checkpoint_shard_XX.json`)
   - Outputs embedding segments with content-addressed `doc_id`
   - Uses DB's canonical `doc_id` when reusing existing paper rows
   - Marks completion via `.shard_XX_complete` marker files

2. **Consumer** (1 node):
   - Polls for producer completion markers
   - Merges all shard DBs into main `litkit.sqlite3` using `merge_shard_databases()`
   - Ingests segment files, resolving `doc_id → paper_id` against merged DB
   - Logs warnings for any doc_id resolution failures
   - Updates FAISS indices with resolved IDs

## Remaining Work

### Phase 4: Dead Code Removal (COMPLETE - 2024-12-17)

All segment-related local implementations have been removed from cli.py:

**Checkpoint functions (REMOVED):**
- ✅ `load_checkpoint()` - removed, using `seg_load_checkpoint()`
- ✅ `save_checkpoint()` - removed, using `seg_save_checkpoint()`

**Segment classes (REMOVED):**
- ✅ `_SegmentWriter` - removed, using `SegmentWriter` from litkit.segments
- ✅ `_ChunkSegmentWriter` - removed, using `ChunkSegmentWriter` from litkit.segments
- ✅ `ProducerCoordinator` - removed, using `SegProducerCoordinator` from litkit.segments
- ✅ `ConsumerCoordinator` - removed, using `SegConsumerCoordinator` from litkit.segments

**Metadata functions (REMOVED):**
- ✅ `_write_build_meta()` - removed, using `seg_write_build_meta`
- ✅ `_read_build_meta()` - removed, using `seg_read_build_meta`
- ✅ `_validate_shard_consistency()` - removed, using `seg_validate_shard_consistency`
- ✅ `_has_segment_files()` - removed, using `seg_has_segment_files`

**DB helpers (already using imported versions):**
- All DB functions are imported from `litkit.db` and prefixed with `db_`

### Phase 5: Final Cleanup (NOT STARTED)

After dead code removal:
- Remove the SCOPE CONTRACT comment block
- Final lint and test pass
- Update module docstrings

### Phase 6: doc_id Path Stability (DEFERRED)

Current `doc_id` uses absolute tar paths, which breaks if corpus moves to a
different mount point. Options for future work:
- Use `pmcid` as doc_id when available, else `pmid`, else hash
- Use tar-relative paths (filename + member) instead of absolute paths
- Document the limitation explicitly

## Architecture Goals

The final architecture will have:

```
litkit/
├── cli.py              # Thin CLI: argparse, orchestration, exit codes
├── concurrent/         # Locking primitives
├── db/                 # All SQLite operations
├── index/              # All FAISS operations
├── segments/           # Embedding segment I/O + checkpoints
├── ingest/             # Tar/XML parsing
├── pipeline/           # Build pipeline logic
├── embeddings/         # (existing) Embedding models
├── formatting/         # (existing) Answer formatting
└── frontload/          # (existing) Chunk capping
```

## Validation Checklist

After each phase:
- [x] `python -c "import litkit.cli"` succeeds without I/O
- [x] `python -m litkit --version` works
- [ ] Build workflow with sample data passes
- [ ] Query workflow with existing indices works

## Commit History

See git log for detailed commit history. Key commits:
- Phase 1: Module extraction
- Phase 2: Import-time purity  
- Phase 3: Schema fixes and DB module integration (2024-12-17)
- Phase 3.5: doc_id canonicalization fix (2024-12-17)
- Phase 3.6: Per-shard checkpoint module (`fd41c14`, `f709070`) (2024-12-17)

## Current Status

**cli.py is still ~4600 lines.** The modules have been created and imports wired up, but
most function DEFINITIONS still live in cli.py. The next phase is to remove those local
definitions and use the imported versions.

## Suggested Next Task

**Remove FAISS function definitions from cli.py**

cli.py still contains ~500+ lines of FAISS-related functions that already have
equivalents in `litkit.index`:

| cli.py function | Use from litkit.index |
|-----------------|----------------------|
| `_unwrap_core_and_kind()` | `index.introspection.unwrap_core_and_kind` |
| `_kind_and_core()` | `index.introspection.kind_and_core` |
| `_report_faiss_index()` | `index.introspection.report_faiss_index` |
| `_flat_ip_index()` | `index.factory.flat_ip_index` |
| `_hnsw_index()` | `index.factory.hnsw_index` |
| `_ivfpq_index()` | `index.factory.ivfpq_index` |
| `_effective_nlist()` | `index.training.effective_nlist` |
| `_pick_nprobe()` | `index.training.pick_nprobe` |
| `_auto_set_nprobe()` | `index.training.auto_set_nprobe` |
| `_faiss_save()`, `_faiss_save_force()` | `index.io` |
| `_faiss_load()`, `_faiss_load_cached()` | `index.io` |
| `_add_with_ids_dedup()` | `index.dedup.add_with_ids_dedup` |
| `_faiss_present_ids()` | `index.dedup.faiss_present_ids` |
| `_make_id_selector()`, `_safe_remove_ids()` | `index.ids` |
| `_faiss_search()` | `index.search.faiss_search` |
| `_temporary_search_params()` | `index.search` |
| `_extract_ivf()` | `index.introspection.extract_ivf` |

**Estimated reduction:** ~500-600 lines from cli.py

This advances **Job 4 (FAISS/Index)** from REFACTOR_ROADMAP.md.
