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

**cli.py is now ~4066 lines** (down from ~4723 at start of 2024-12-18 session).
The major module imports are wired up and local definitions have been removed.

## Work Completed 2024-12-18

### Session: Wire Up Extracted Modules + Remove Local Defs (-657 lines)

#### 1. FAISS Index Functions (-319 lines)
**Commit:** `refactor(cli): replace local FAISS functions with litkit.index imports`

Replaced 17 local FAISS function definitions with imports from `litkit.index`:
- Factory: `flat_ip_index`, `hnsw_index`, `ivfpq_index`, `safe_pq_m`
- I/O: `faiss_save`, `faiss_save_force`, `faiss_load`, `faiss_load_cached`
- Introspection: `unwrap_core_and_kind`, `kind_and_core`, `extract_ivf`, `report_faiss_index`
- IDs: `make_id_selector`, `safe_remove_ids`, `faiss_present_ids`
- Search: `pick_nprobe`, `auto_set_nprobe`, `faiss_search`
- Dedup: `add_with_ids_dedup`

#### 2. Progress Classes (-111 lines)
**Commit:** `refactor(cli): replace local progress classes with litkit.progress imports`

Replaced local `eprint`, `Progress`, `Pulse`, `phase` with imports from `litkit.progress`.

#### 3. FileLock Class (-71 lines)
**Commit:** `refactor(cli): replace local FileLock with litkit.concurrent import`

- Imported `FileLock`, `FLOCK_AVAILABLE`, `in_faiss_lock`, `in_db_lock` from `litkit.concurrent`
- Created thin wrapper that binds `DB_LOCK`/`FAISS_LOCK` paths at runtime
- Deleted local lock depth tracking, fcntl imports, FileLock class

#### 4. Runtime Paths (-92 lines)
**Commit:** `refactor(cli): replace _Runtime with litkit.config.paths.WorkspacePaths`

- Imported `WorkspacePaths` from `litkit.config.paths`
- Deleted `_find_root()`, `_resolve_workspace()`, `_Runtime` dataclass
- Simplified `_init_runtime()` to use `WorkspacePaths.from_env_or_default()`
- Fixed regression risk: updated `WorkspacePaths.setup_environment()` busy timeout 30000→120000ms
- Kept `faiss.cvar.seed` init in cli.py (FAISS-specific)

### Remaining Local Functions (~35 still in cli.py)

**CLI-specific (should stay):**
- `_version_banner()`, `_report_paths()`, `_confirm_rebuild()`, `_resolve_question()`
- Writer guard functions (`_create_writer_guard_or_exit()`, `_maybe_cleanup_own_stale_guard()`)
- LLM config helpers (`_default_base_url_for()`, `_default_api_key_for()`, `_is_openai_cloud()`)
- `_post_build_sanity_check()`, `_init_runtime()`, `__getattr__()`

**Extraction candidates (diminishing returns):**
- `_ingest_paper_segments()`, `_ingest_chunk_segments()` (~190 lines) - module version is simpler, needs enhancement to match cli.py's robust implementation
- `_normalize_and_strip_citations()` (~40 lines) - preprocessing step before existing module
- `_faiss_search()`, `_temporary_search_params()` (~40 lines) - thin wrappers

### Bug Fixes (Post-Wiring)

After the main refactoring, three runtime bugs were discovered during build testing:

#### 5. Dead Code Reference (-7 lines)
**Commit:** `fix(cli): remove dead _ADVISORY_LOCK_DISABLED code after FileLock refactor`

The `_ADVISORY_LOCK_DISABLED` and `_ADVISORY_LOCK_NOTICE_PRINTED` variables were
referenced in `main()` but never defined after the FileLock code was moved to
`litkit.concurrent`. The locking fallback logic is now handled internally by the
`FileLock` class.

#### 6. Underscore-Prefixed Function Calls (59 replacements)
**Commit:** `fix(cli): remove underscore prefixes from FAISS function calls`

When the FAISS functions were imported from `litkit.index`, the imports used
non-underscore names (e.g., `hnsw_index`), but 58+ call sites still used the old
underscore-prefixed names (e.g., `_hnsw_index`). Fixed via sed replacement.

#### 7. Missed extract_ivf Call
**Commit:** `fix(cli): _extract_ivf -> extract_ivf`

One additional underscore-prefixed call (`_extract_ivf`) was missed in the initial
sed replacement.

### Final Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| cli.py lines | 4723 | ~4059 | **-664 (14%)** |
| FAISS local defs | 17 | 0 | -17 |
| Progress local defs | 4 | 0 | -4 |
| FileLock local | 1 | 0 (wrapper) | -1 |
| Runtime/paths local | 3 | 0 | -3 |

### Build Validation
- ✅ Import smoke test passes (`python -c "import litkit.cli"`)
- ✅ Mac local build completes successfully (`./test_build.sh --clean --verbose`)
- Note: `[faiss] remove_ids not supported` warning is expected and harmless (HNSW indices don't support ID removal)

## Work Completed 2024-12-18 (Session 2)

### Enhanced Segment Ingestion Module

**Commit `d22daa4`:** `refactor(segments): enhance ingest module with cli.py features`

Enhanced `litkit/segments/ingest.py` with robust features from cli.py's implementations:
- Atomic file claiming (`.ingesting` rename pattern)
- Content-addressed doc_id resolution with preloaded maps (O(1) lookups)
- Warning logging for resolution failures
- Support for both new format (`doc_ids/vecs`) and legacy format (`ids/vecs`)

Wired `seg_ingest_paper_segments` and `seg_ingest_chunk_segments` into cli.py.

### Fixed HNSW remove_ids Warning

**Commit `3a48a2e`:** `fix(index): skip remove_ids silently for HNSW indices`

Modified `safe_remove_ids()` in `litkit/index/ids.py` to check index type upfront
and return silently for HNSW (which doesn't support ID removal). Eliminates the
noisy `[faiss] remove_ids not supported` warning during paper embedding batches.

### Investigation: `_add_ids_union_compat` Fallback

Investigated whether cli.py's `_add_ids_union_compat` fallback is still needed:

1. **pyproject.toml requires `faiss-cpu>=1.8,<1.9`** - FAISS 1.8+ fully supports
   `add_with_ids` on IndexIDMap2.

2. **`add_with_ids_dedup` already has internal fallback** - The module function
   in `litkit/index/dedup.py` catches `RuntimeError` and falls back to skipping
   present IDs.

3. **Conclusion:** The cli.py `_add_ids_union_compat` is effectively dead code.
   Safe to remove along with the local `_ingest_*` functions.

## Completed 2024-12-18 (Session 3)

### Removed Dead Code

**Commit `06bea4d`:** `refactor(cli): remove dead _ingest_* and _add_ids_union_compat code (-374 lines)`

Removed dead code from cli.py now that segment ingestion uses the module:
- `_ingest_paper_segments()` (~150 lines) - REMOVED
- `_ingest_chunk_segments()` (~160 lines) - REMOVED
- `_add_ids_union_compat()` (~18 lines) - REMOVED
- All `except RuntimeError` fallback blocks referencing `_add_ids_union_compat`

cli.py: 4054 → 3680 lines (-374 lines, 9.2% reduction)

---

## Phase 6: Extract `build_or_update_indices` to `litkit.build` Module

### Overview

The `build_or_update_indices()` function (~750 lines) is the main orchestrator for:
1. Index bootstrap (`--init-indices-only`)
2. Consumer mode (`--consume-only`)
3. IVF-PQ training (when chunks index is new)
4. Tar scanning & article ingestion
5. Final buffer flush
6. SQLite/FAISS reconciliation

This section documents the extraction plan.

### Target Module Structure

```
litkit/build/
├── __init__.py        # Re-exports: build_or_update_indices, BuildConfig
├── config.py          # BuildConfig dataclass (from args namespace)
├── helpers.py         # pack_paragraphs, dedupe, ensure_parent, etc.
├── backfill.py        # backfill_unindexed_vectors, reconcile, sanity_check
├── training.py        # IVF-PQ training logic
└── orchestrator.py    # build_or_update_indices (main entry point)
```

### Function Signatures (Target)

```python
# litkit/build/orchestrator.py
def build_or_update_indices(
    cfg: BuildConfig,
    *,
    paper_seg_writer: SegmentWriter | None = None,
    chunk_seg_writer: ChunkSegmentWriter | None = None,
) -> None:
    ...

# litkit/build/training.py
def train_chunks_ivfpq(
    *,
    chunk_index: faiss.Index,
    chunk_embedder: Embedder,
    cfg: BuildConfig,
    tar_paths: Sequence[Path],
) -> faiss.Index:
    """Returns new trained index (or FLAT fallback). Does NOT save to disk."""
    ...
```

### Phased Extraction Plan

#### Phase 6.1: Extract Helpers + Backfill

**Functions to move to `litkit.build.helpers`:**
- `pack_paragraphs()` (~30 lines) - text chunking
- `_dedupe_papers_with_doc_ids()` (~15 lines)
- `_dedupe_chunks_with_doc_ids()` (~15 lines)
- `_effective_nlist()` (~15 lines) - already in litkit.index.training, verify usage
- `_ensure_parent()` (~3 lines)
- `_clear_chunk_trained_flag()` (~8 lines)
- `_maybe_fsync_dir()` (~10 lines)

**Functions to move to `litkit.build.backfill`:**
- `backfill_unindexed_vectors()` (~50 lines)
- `reconcile_sqlite_flags_with_faiss()` (~40 lines)
- `_post_build_sanity_check()` (~25 lines)

**Technical debt to fix during this phase:**
- **Kill `_PENDING_MARKS`**: Either wire it properly into backfill/reconciliation,
  or drop it and rely on `reconcile_sqlite_flags_with_faiss()` + backfill to repair.
  *Recommendation: Remove it.*
- **De-globalize locks**: `backfill_unindexed_vectors()` and `reconcile_sqlite_flags_with_faiss()`
  should accept lock paths or a lock factory as parameters, not reach into cli.py.

**Restart point:** After this phase, helpers and backfill are modular; orchestrator
still in cli.py. Safe to commit and resume later.

#### Phase 6.2: Create BuildConfig Dataclass

**Create `litkit/build/config.py`:**
```python
@dataclass
class BuildConfig:
    # Paths (from WorkspacePaths)
    db_path: Path
    paper_index_path: Path
    chunk_index_path: Path
    sqlite_dir: Path
    embed_segments_dir: Path
    ckpt_path: Path
    chunk_trained_flag: Path
    db_lock: Path
    faiss_lock: Path
    ckpt_lock: Path
    
    # Build mode flags
    rebuild: bool
    update: bool
    build_only: bool
    consume_only: bool
    init_indices_only: bool
    faiss_writer: bool
    embed_producer: bool
    consume_segments: bool
    
    # Sharding
    shard_id: int
    num_shards: int
    
    # Index params
    papers_index: str  # "hnsw" | "flat"
    chunks_index: str  # "ivfpq" | "flat"
    hnsw_m: int
    efconstruction: int
    efsearch: int
    ivf_nlist: int
    pq_m: int
    nprobe: int | None
    
    # Embedding
    embed_devices: str
    embed_workers: int
    paper_embed_bs: int
    chunk_embed_bs: int
    
    # Chunking
    chunk_target_chars: int
    chunk_min_chars: int
    chunk_overlap: int
    
    # SQLite
    sqlite_journal_mode: str
    sqlite_busy_timeout_ms: int
    
    # Corpus
    tar_dir: Path | None
    tar_manifest: Path | None
    embed_outdir: Path | None
    
    # Misc
    quiet: bool
    
    @classmethod
    def from_args(cls, args, runtime: WorkspacePaths) -> "BuildConfig":
        ...
```

**Technical debt to fix during this phase:**
- **Lazy path globals vs library use**: `BuildConfig.from_args()` takes `WorkspacePaths`
  explicitly. The build module no longer relies on cli's lazy globals.

**Restart point:** Config dataclass ready; orchestrator still in cli.py.

#### Phase 6.3: Extract IVF-PQ Training

**Create `litkit/build/training.py`:**
- Move the ~200-line training block into `train_chunks_ivfpq()`
- Return trained index; let orchestrator handle persistence
- Clear interface: chunk_index, embedder, config, tar_paths → trained index

**Restart point:** Training logic modular; main orchestrator still in cli.py.

#### Phase 6.4: Extract Main Orchestrator

**Create `litkit/build/orchestrator.py`:**
- Move `build_or_update_indices()` with explicit parameters
- Move `iter_tar_articles()` helper (or keep in cli.py as it's also used elsewhere)
- Inject segment writers as parameters (no globals)

**Technical debt to fix during this phase:**
- **CLI flag compatibility checking**: Explicitly reject invalid combinations
  (e.g., `--embed-producer + --faiss-writer` if not well-defined)
- **Centralize role logic**: Single node writer / multi-node producers / consumer-only
  should be determined in one place

**Final signature:**
```python
# cli.py main()
from litkit.build import build_or_update_indices, BuildConfig

cfg = BuildConfig.from_args(args, get_runtime())
build_or_update_indices(
    cfg,
    paper_seg_writer=paper_seg_writer,
    chunk_seg_writer=chunk_seg_writer,
)
```

### Known Technical Debt

**To address during extraction:**

| Issue | Description | Phase |
|-------|-------------|-------|
| `_PENDING_MARKS` semantics | `defaultdict(list)` populated but marks never flushed reliably | 6.1 |
| `_auto_top_papers` leak | Opens DB connection, doesn't close on all paths | Fix immediately in cli.py |
| Lazy path globals | `DB_PATH`, `PAPER_INDEX_PATH` etc. as magical globals | 6.2 |
| FileLock coupling | cli.py subclass binds paths; should be in `litkit.concurrent` | 6.1 |
| Flag incompatibility | `--embed-producer + --faiss-writer` semantics unclear | 6.4 |

### FileLock Migration Plan

**Current state (cli.py):**
```python
class FileLock(_FileLockBase):
    def __init__(self, path: Path):
        get_runtime()  # triggers lazy globals
        super().__init__(path, db_lock_path=DB_LOCK, faiss_lock_path=FAISS_LOCK)
```

**Target state:**
1. `litkit.concurrent.FileLock` accepts optional `db_lock_path`, `faiss_lock_path`
2. cli.py provides factory helpers after `get_runtime()`:
   ```python
   def make_file_lock(path: Path) -> FileLock:
       return FileLock(path, db_lock_path=DB_LOCK, faiss_lock_path=FAISS_LOCK)
   ```
3. `litkit.build` accepts paths or lock factory; constructs `FileLock` locally

### Restart Points Summary

| After Phase | State | Can Resume? |
|-------------|-------|-------------|
| 6.1 | Helpers + backfill in module; orchestrator in cli.py | ✅ |
| 6.2 | BuildConfig ready; orchestrator in cli.py | ✅ |
| 6.3 | Training extracted; orchestrator in cli.py | ✅ |
| 6.4 | Fully extracted; cli.py is thin | ✅ |

---

## Work Completed 2024-12-19 (Bug Fix Session)

### Pre-Extraction Bug Fixes (8 commits)

Before beginning Phase 6.1 extraction, we addressed bugs identified in code review:

| Commit | Fix | Lines |
|--------|-----|-------|
| `c0a788c` | Add `get_runtime()` to 5 helpers for library use safety | +5 |
| `eb27902` | Fix connection leak in `_auto_top_papers` (try/finally) | +8 |
| `e380547` | Remove dead `_PENDING_MARKS` tracking | -23 |
| `9500990` | Reject incompatible flag combinations early in main() | +21 |
| `bbbc8f1` | Handle invalid `LITKIT_WRITER_GUARD_TTL` gracefully | +4 |
| `93298d3` | Move `get_runtime()` after `--offline` handling | +3 |
| `096cccd` | Document `_vector_store_exists` and signal handler semantics | +12 |
| `394c811` | Fix regression: move `get_runtime()` right after `--offline` | +3 |

**Key fixes:**

1. **Lazy runtime safety (`c0a788c`)**: Added `get_runtime()` calls to 5 helper functions
   that access path globals (`_maybe_cleanup_own_stale_guard`, `_create_writer_guard_or_exit`,
   `backfill_unindexed_vectors`, `_post_build_sanity_check`, `_auto_top_papers`). This ensures
   library use (importing and calling these functions outside `main()`) works correctly.

2. **Connection leak (`eb27902`)**: `_auto_top_papers()` now uses try/finally to ensure
   the DB connection closes on all exit paths.

3. **Dead code removal (`e380547`)**: Removed `_PENDING_MARKS` mechanism that was:
   - Populated when FAISS save failed (to defer marking)
   - Never actually consumed/flushed
   - Causing marks to be lost, requiring manual reconcile runs
   
   Now relies entirely on `reconcile_sqlite_flags_with_faiss()` which runs at end of writer builds.

4. **Flag validation (`9500990`)**: Added early rejection of incompatible flag combinations:
   - `--embed-producer + --faiss-writer` (mutually exclusive)
   - `--consume-only + --embed-producer` (mutually exclusive)
   - `--init-indices-only + --embed-producer` (mutually exclusive)

5. **TTL parsing (`bbbc8f1`)**: `LITKIT_WRITER_GUARD_TTL` env var parsing now catches
   `ValueError` and falls back to 86400s with a warning instead of crashing.

6. **--offline ordering (`93298d3`, `394c811`)**: Fixed ordering so `--offline` sets
   `HF_HUB_OFFLINE` before `get_runtime()` calls `setup_environment()`. Then fixed
   regression where `get_runtime()` was moved too far down, causing `NameError` for
   path globals used before initialization.

7. **Documentation (`096cccd`)**: Added docstring to `_vector_store_exists()` explaining
   it requires ALL three paths (both FAISS indices + DB). Added comment explaining
   the signal handler's hard kill via `os._exit(1)` is intentional.

---

## Work Completed 2024-12-19 (Phase 6 Extraction)

### Phase 6.1: Extract helpers and backfill to litkit.build (COMPLETE)

Created the `litkit/build/` module skeleton and extracted helper functions.

**Commit `5e2eec9`:** `refactor(cli): extract helpers to litkit.build module (Phase 6.1)`

Created `litkit/build/helpers.py` with:
- `pack_paragraphs()` - Greedy text chunking with overlap
- `dedupe_papers_with_doc_ids()` - Deduplicate paper buffers
- `dedupe_chunks_with_doc_ids()` - Deduplicate chunk buffers
- `ensure_parent()` - Create parent directory if needed
- `maybe_fsync_dir()` - Optional directory fsync for NFS safety

cli.py: ~3610 lines (down from ~3700, removed ~90 lines of local definitions)

**Commit `b4fcc51`:** `refactor(build): add backfill module to litkit.build (Phase 6.1b)`

Created `litkit/build/backfill.py` with parameterized module versions:
- `backfill_unindexed_vectors()` - Embed and add rows missing from FAISS
- `reconcile_sqlite_flags_with_faiss()` - Sync in_index flags with FAISS state
- `post_build_sanity_check()` - Summary report after build

Module functions accept explicit path/lock parameters for decoupled use.
cli.py imports module versions but still had local definitions shadowing them.

**Commit `8d55639`:** `refactor(cli): convert backfill functions to thin wrappers (Phase 6.1c)`

Replaced local implementations with thin wrappers that delegate to module:
- `backfill_unindexed_vectors()`: ~90 lines → 15 lines
- `reconcile_sqlite_flags_with_faiss()`: ~45 lines → 3 lines
- `_post_build_sanity_check()`: ~25 lines → 7 lines

cli.py: 3506 lines (down from 3633, -127 lines)

### Phase 6.2: Create BuildConfig and extract index bootstrap (IN PROGRESS)

**Commit `602498b`:** `refactor(build): add BuildConfig dataclass (Phase 6.2a)`

Created `litkit/build/config.py` with:
- `BuildConfig` dataclass encapsulating all build parameters
- Properties: `is_multi_node`, `build_mode`, `effective_embed_outdir`
- Methods: `needs_corpus()`, `validate()`
- Factory: `build_config_from_args(args, paths={})`

Design principles:
- All paths explicit (no globals accessed by module functions)
- Sensible defaults matching cli.py constants
- Validation methods for flag consistency

**Commit `71c4556`:** `refactor(build): extract init_empty_indices to litkit.build.indices (Phase 6.2b)`

Created `litkit/build/indices.py` with:
- `init_empty_indices()` - Create empty FAISS indices for bootstrap mode (`--init-indices-only`)

cli.py now uses `BuildConfig` to delegate to module function:
```python
cfg = BuildConfig(
    faiss_writer=args.faiss_writer,
    papers_index=args.papers_index,
    # ... other params
    paper_index_path=PAPER_INDEX_PATH,
    chunk_index_path=CHUNK_INDEX_PATH,
    faiss_lock=FAISS_LOCK,
)
build_init_empty_indices(cfg, FileLock=FileLock)
```

cli.py: 3492 lines (down from 3506, -14 lines)

### Current Module Structure

```
src/litkit/build/
├── __init__.py       # Package exports (12 items)
├── helpers.py        # Text chunking & deduplication (~148 lines)
├── backfill.py       # FAISS/SQLite reconciliation (~268 lines)
├── config.py         # BuildConfig dataclass (~270 lines)
└── indices.py        # Index creation utilities (~85 lines)
```

### Remaining Phase 6.2 Steps

| Step | Description | Est. Lines |
|------|-------------|------------|
| 6.2c | Extract `run_consume_only_mode()` | ~55 |
| 6.2d | Extract index load/create functions | ~100 |
| 6.2e | Extract IVF-PQ training | ~250 |
| 6.2f | Extract tar processing loop | ~500 |
| 6.2g | Final cleanup extraction | ~80 |

---

## Current Status

**cli.py is now ~3492 lines** (down from ~4723 at start of 2024-12-18 session, **-1231 lines / 26% reduction**).

### Summary of Reductions

| Phase | Description | Lines Removed |
|-------|-------------|---------------|
| 2024-12-18 | FAISS, Progress, FileLock, Runtime extraction | ~664 |
| 2024-12-18 | Dead code removal (_ingest_*, _add_ids_union_compat) | ~374 |
| 2024-12-19 | Bug fixes + minor cleanup | ~23 |
| 2024-12-19 | Phase 6.1 helpers + backfill extraction | ~127 |
| 2024-12-19 | Phase 6.2a-b (BuildConfig, init_empty_indices) | ~43 |

### Next Immediate Steps

1. ~~**Pause extraction** - User wants to discuss potential bugs~~ ✅ Done
2. **Resume refactor** - Continue with Phase 6.2c (run_consume_only_mode)
3. **Test after each sub-phase** with `./test_build.sh --clean`

---

## Work Completed 2024-12-19 (Bug Fix Aside Session)

### Code Review Bug Fixes (14 commits)

Comprehensive bug fix session addressing issues found in code review. This work
paused the Phase 6 extraction to fix correctness issues before continuing.

#### Critical Fixes

| Commit | Issue | Description |
|--------|-------|-------------|
| `4bb8f6e` | **DATA LOSS** | Checkpoint/rollback: `conn.commit()` was releasing savepoint, causing partial member data to be committed even on error. Fixed with proper SAVEPOINT/RELEASE/ROLLBACK pattern. |
| `6a3214e` | **DATA LOSS** | Buffer/rollback desync: In-memory buffers could contain IDs rolled back in SQLite. Now truncates buffers to pre-member snapshot on error. |
| `759976d` | **RESOURCE LEAK** | DB connection leak in `_auto_top_papers()`: Early returns bypassed `conn.close()`. Fixed with try/finally. |
| `7f97812` | **BUG** | `LITKIT_NO_LEXICAL` env var checked `!= "0"` instead of `== "1"`, meaning any value (even "false") disabled lexical. |
| `d6b7196` | **DEAD CODE** | `--parse-workers` flag was parsed but never passed to `iter_tar_articles()`. Now wired through. |
| `a6ab735` | **REGRESSION** | `_resolve_question()` lost bare-path logic: `litkit ./question.txt` treated path as literal question instead of reading file. |

#### Edge Case Fixes

| Commit | Issue | Description |
|--------|-------|-------------|
| `277e127` | **SEMANTICS** | `LITKIT_WRITER_GUARD_TTL=0` previously meant "always steal guard" (any guard is >0s old). Now means "never auto-cleanup" (manual deletion required). |
| `37a25b6` | **ERROR HANDLING** | `_vector_store_exists()` silently returned False on permission errors. Now exits with clear error message for OSError (permissions, path issues). |
| `32f1de4` | **UX** | `--reconcile-only` with no existing store hit misleading "No tar shards" error. Now falls through to proper "run a build first" message. |

#### Documentation & Hygiene

| Commit | Issue | Description |
|--------|-------|-------------|
| `e75f064` | **DOCS** | Added SQLite concurrency model documentation to README.md and contract comment in cli.py. Prevents future "optimization" attempts that cause SQLITE_BUSY. |
| `f0eae03` | **HYGIENE** | Removed dead `db_SCHEMA` import, `_idmap_bloom` function (~45 lines). Improved `_vector_store_exists()` exception handling. |
| `eabeea5` | **HYGIENE** | Final flush used different dedupe function than mid-batch flush. Now uses same `dedupe_*_with_doc_ids()` functions throughout. |
| `69d78b0` | **DOCS** | Fixed copy-paste error in `_post_build_sanity_check` docstring (said "backfill" instead of "sanity_check"). |
| `763f9ea` | **DOCS** | Added docstring to `_maybe_cleanup_own_stale_guard` explaining it only cleans THIS process's guards (PID match), not general stale guards. |

#### Reviewed but No Action Required

| Item | Assessment |
|------|------------|
| Writer guard TTL semantics | Correct as-is: TTL≤0 → never auto-cleanup |
| Stale guard cleanup exit behavior | Correct: fail-fast on FS errors is appropriate |
| `_init_runtime()` FAISS seed exception | Silent catch is fine; failure would break more than seed |
| `use_tar` assigned twice | Intentional: different scopes (training vs scanning) |
| `FileLock.__init__` calls `get_runtime()` | Documented design; lock paths require runtime |

### Summary

- **13 commits** for bug fixes and documentation
- **3 items** reviewed and confirmed correct (no changes)
- **Most critical**: SAVEPOINT fix (1.1) and buffer/rollback desync (1.2) prevented potential data loss during tar member processing errors
