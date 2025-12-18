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

### Dead Code Removal (PARTIAL - 2024-12-17)

Removed unused local implementations that were shadowed by imported versions:
- ✅ `_mark_in_index()` - removed (all calls use `db_mark_in_index`)
- ✅ `_flush_pending_marks()` - removed (all calls use `db_flush_pending_marks`)

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

### Phase 4: Dead Code Removal (IN PROGRESS)

cli.py still contains local implementations that duplicate the extracted modules.
These should be removed to shrink cli.py toward its ~500 line target:

**Checkpoint functions (ready to remove):**
- ✅ `load_checkpoint()` - now using `seg_load_checkpoint()`
- ✅ `save_checkpoint()` - now using `seg_save_checkpoint()`
- Note: Local versions still exist but only `seg_*` versions are called

**Segment classes (ready to remove):**
- `_SegmentWriter` - duplicates `litkit.segments.writer.SegmentWriter`
- `_ChunkSegmentWriter` - duplicates `litkit.segments.writer.ChunkSegmentWriter`
- `ProducerCoordinator` - duplicates `litkit.segments.coordination`
- `ConsumerCoordinator` - duplicates `litkit.segments.coordination`

**Metadata functions (ready to remove):**
- `_write_build_meta()` - duplicates `litkit.segments.metadata.write_build_meta`
- `_read_build_meta()` - duplicates `litkit.segments.metadata.read_build_meta`
- `_validate_shard_consistency()` - duplicates `litkit.segments.metadata.validate_shard_consistency`
- `_has_segment_files()` - duplicates `litkit.segments.metadata.has_segment_files`

**DB helpers (still used - need careful removal):**
- `SCHEMA`, `init_db()`, `init_shard_db()`
- `_ensure_in_index_columns()`, `_shard_db_path()`, `_list_shard_dbs()`
- `merge_shard_databases()`, `_connect_db()`
- `_ensure_temp_candidates_table()`, `_load_temp_candidates()`
- `already_processed()`, `register_file()`
- `preload_paper_id_map()`, `preload_chunk_id_map()`

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

## Suggested Next Task

**Dead Code Removal: Segment Classes**

Remove local duplicates from cli.py that are now imported from extracted modules:

1. Remove local `load_checkpoint()` and `save_checkpoint()` (only `seg_*` versions are called)
2. Remove local `_SegmentWriter` and `_ChunkSegmentWriter` classes
3. Remove local `ProducerCoordinator` and `ConsumerCoordinator` classes
4. Remove local `_write_build_meta()`, `_read_build_meta()`, `_validate_shard_consistency()`, `_has_segment_files()`
5. Import and use versions from `litkit.segments` throughout

**Estimated reduction:** ~400-600 lines from cli.py

This directly advances Jobs 5 (Locking) and 6 (Segments) in REFACTOR_ROADMAP.md.
