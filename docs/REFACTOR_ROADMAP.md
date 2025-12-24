# Refactor Roadmap for cli.py

This document tracks the planned extraction of functionality from the monolithic `cli.py` into purpose-specific modules across 9 incremental jobs.

## Job Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │            JOB 1 (DONE)                 │
                    │         Lock Scope                      │
                    │   (Define what cli.py should contain)   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │             JOB 2 [M]                   │
                    │     Environment & Paths                 │
                    │   (WorkspacePaths + progress utils)     │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────┐
          │                           │                       │
          ▼                           ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│     JOB 3 [M]       │   │     JOB 4 [L]       │   │     JOB 5 [S]       │
│    SQLite / DB      │   │   FAISS / Index     │   │     Locking         │
│   (litkit.db)       │   │  (litkit.index)     │   │ (litkit.concurrent) │
└─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          JOB 6 [M]            │
                    │    Segments (writers/ingest)  │
                    │     (litkit.segments)         │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          JOB 7 [L]            │
                    │   Tar/Producer Pipeline       │
                    │  (litkit.ingest.pipeline)     │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                                                   │
          ▼                                                   ▼
┌─────────────────────────┐                       ┌─────────────────────────┐
│       JOB 8 [M]         │                       │       JOB 9 [S]         │
│ Robustness/Performance  │                       │   UX/Docs Polish        │
└─────────────────────────┘                       └─────────────────────────┘

Legend: [S] = Small (~1 day), [M] = Medium (2-3 days), [L] = Large (4-5 days)
```

## Complexity Estimates

| Job | Name | Complexity | Effort | Risk |
|-----|------|------------|--------|------|
| 1 | Lock Scope | S | ✅ Done | None |
| 2 | Env/Paths | M | 2-3 days | Low |
| 3 | SQLite/DB | M | 2-3 days | Low |
| 4 | FAISS/Index | L | 4-5 days | Medium (training code) |
| 5 | Locking | S | 1 day | Low |
| 6 | Segments | M | 2-3 days | Low |
| 7 | Producer Pipeline | L | 4-5 days | Medium (main loop) |
| 8 | Robustness | M | 2-3 days | Low |
| 9 | UX/Docs | S | 1-2 days | None |

**Total estimated effort: 18-25 days**

---

## JOB 1 — Lock Scope ✅ COMPLETE

**Status:** Done (commit `2692d2c`)

**Deliverables:**
- SCOPE CONTRACT comment at top of cli.py
- This roadmap document
- TODO annotations in cli.py for extraction candidates

---

## JOB 2 — Environment & Paths

**Goal:** Remove all path/env discovery logic from cli.py. Replace ad-hoc globals with `WorkspacePaths` dataclass.

**Target module:** `litkit/config/paths.py` (or `litkit/env.py`)

### Move these responsibilities:
- `_find_root()`, `_resolve_workspace()`
- Directory derivation: `SQLITE_DIR`, `INDICES_DIR`, `EMBED_SEGMENTS_DIR`
- Environment bootstrapping: all `os.environ.setdefault(...)` calls
- Path constants: `DB_PATH`, `PAPER_INDEX_PATH`, `CHUNK_INDEX_PATH`, `CKPT_PATH`

### Also extract progress utilities (Job 2b):
**Target module:** `litkit/progress.py`
- `_eprint()`, `_Progress`, `_Pulse`, `_phase()`
- Progress lock from `litkit.embeddings.base`

### Define dataclass:
```python
@dataclass
class WorkspacePaths:
    root: Path
    workspace: Path
    sqlite_dir: Path
    indices_dir: Path
    embed_segments_dir: Path
    db_path: Path
    paper_index_path: Path
    chunk_index_path: Path
    ckpt_path: Path
    
    @classmethod
    def from_env_or_default(cls, root_override: Path | None = None) -> "WorkspacePaths":
        """Construct from environment variables with sensible defaults."""
        ...
```

### Tests:
- Test env var overrides
- Test directory creation
- Test error handling for invalid configs

### Acceptance criteria:
- No direct references to `ROOT` / `WORKSPACE` / `DB_PATH` primitives in cli.py
- cli.py constructs exactly one `WorkspacePaths` per invocation
- No behavior change for users

---

## JOB 3 — SQLite / DB

**Goal:** Move all SQLite-related code out of cli.py.

**Target modules:**
- `litkit/db/__init__.py`
- `litkit/db/schema.py` — SCHEMA constant, init_db, _ensure_in_index_columns
- `litkit/db/merge.py` — merge_shard_databases, init_shard_db
- `litkit/db/queries.py` — Helpers and query functions

### Move these:
- `SCHEMA` definition
- `init_db()`, `init_shard_db()`
- `_connect_db()`, `_shard_db_path()`, `_list_shard_dbs()`
- `merge_shard_databases()`
- `_ensure_in_index_columns()`, `_ensure_temp_candidates_table()`, `_load_temp_candidates()`
- `already_processed()`, `register_file()`
- `preload_paper_id_map()`, `preload_chunk_id_map()`
- `_mark_in_index()`, `_flush_pending_marks()`, `_PENDING_MARKS`
- `chunk_ids_to_paper_ids()`

### Clean API:
```python
def open_main_db(paths: WorkspacePaths, journal_mode: str, busy_timeout_ms: int) -> Connection:
    ...

def open_shard_db(shard_path: Path, journal_mode: str, busy_timeout_ms: int) -> Connection:
    ...
```

### Tests:
- `init_db()` creates expected tables
- `merge_shard_databases()` on small test DBs
- `already_processed()` / `register_file()` work correctly

### Acceptance criteria:
- No `SCHEMA` or direct SQL DDL in cli.py
- All DB connections created through `litkit.db` functions
- Merge logic callable from tests without importing cli.py

---

## JOB 4 — FAISS / Index

**Goal:** Remove all FAISS-specific utilities from cli.py.

**Target modules:**
- `litkit/index/__init__.py`
- `litkit/index/core.py` — Index creation, loading, saving
- `litkit/index/search.py` — Search operations
- `litkit/index/training.py` — IVF-PQ training logic

### Move these:
- `_unwrap_core_and_kind()`, `_kind_and_core()`, `_extract_ivf()`
- `_flat_ip_index()`, `_hnsw_index()`, `_ivfpq_index()`, `_safe_pq_m()`
- `_faiss_save()`, `_faiss_save_force()`, `_faiss_load()`, `_faiss_load_cached()`
- `_add_with_ids_dedup()`, `_add_ids_union_compat()`, `_safe_remove_ids()`
- `_faiss_search()`, `_auto_set_nprobe()`, `_pick_nprobe()`
- `_make_id_selector()`, `_faiss_present_ids()`
- `PQ_BITS` and related constants
- IVF-PQ training loop (from `build_or_update_indices`)

### Also extract retrieval (Job 4b):
**Target module:** `litkit/retrieval/`
- `shortlist_papers()`
- `search_chunks_constrained()`
- `get_chunks()`
- Lexical prefilter: `_query_terms()`, `_normalize_for_search_py()`, etc.

### Define index manager classes:
```python
@dataclass
class IndexConfig:
    kind: Literal["flat", "ivfpq", "hnsw"]
    dim: int
    nlist: int | None = None
    m: int | None = None
    nbits: int = 8

class PaperIndexManager:
    def __init__(self, path: Path, config: IndexConfig, lock: FileLock | None = None): ...
    def load_or_init(self) -> None: ...
    def add_vectors(self, ids: np.ndarray, X: np.ndarray) -> int: ...
    def remove_ids(self, ids: np.ndarray) -> int: ...
    def search(self, q: np.ndarray, top_k: int, **kwargs) -> tuple[list[int], list[float]]: ...
    def save(self, force: bool = False) -> bool: ...

class ChunkIndexManager:  # Same pattern
    ...
```

### Tests:
- Round-trip: create index, add vectors, save, load, search
- Dedup behavior preserved
- IVF-PQ training works

### Acceptance criteria:
- No `faiss` import or `faiss.*` calls in cli.py
- Index paths and types configured via `IndexConfig` + `WorkspacePaths`
- Paper/Chunk indexing reachable from tests without importing cli.py

---

## JOB 5 — Locking

**Goal:** Centralize locking behavior instead of sprinkling `FileLock`, `DB_LOCK`, `FAISS_LOCK` across cli.py.

**Target module:** `litkit/concurrent/locking.py`

### Move these:
- `FileLock` class
- `DB_LOCK`, `FAISS_LOCK`, `CKPT_LOCK` definitions
- `_faiss_lock_enter()/_exit()`, `_db_lock_enter()/_exit()`, `_in_faiss_lock()`, `_in_db_lock()`
- `WRITER_GUARD`, `_create_writer_guard_or_exit()`, `_maybe_cleanup_own_stale_guard()`

### Define clean API:
```python
class LockManager:
    def __init__(self, paths: WorkspacePaths): ...
    
    @contextmanager
    def db_lock(self): ...
    
    @contextmanager
    def faiss_lock(self): ...
    
    def create_writer_guard_or_exit(self, ttl_sec: int = 86400) -> None: ...
    def cleanup_writer_guard(self) -> None: ...
```

### Document lock ordering:
> **Rule:** Always acquire DB_LOCK before FAISS_LOCK. Never reverse.

### Tests:
- Writer guard creation and cleanup
- DB and FAISS locking are mutually consistent

### Acceptance criteria:
- No raw `threading.Lock` or `FileLock` globals in cli.py
- All lock acquisition goes through `litkit.concurrent.locking`
- Lock ordering policy documented and enforced in one place

---

## JOB 6 — Segments

**Goal:** Separate segment file I/O from cli.py.

**Target modules:**
- `litkit/segments/__init__.py`
- `litkit/segments/writer.py` — Segment writers
- `litkit/segments/ingest.py` — Segment readers/ingesters
- `litkit/segments/coordination.py` — Producer/consumer coordination

### Move these:
- `_SegmentWriter`, `_ChunkSegmentWriter`
- `ProducerCoordinator`, `ConsumerCoordinator`
- `_write_build_meta()`, `_read_build_meta()`, `_validate_shard_consistency()`
- `_ingest_paper_segments()`, `_ingest_chunk_segments()`
- `DEFAULT_EMBED_SEGMENT_SIZE`, `DEFAULT_EMBED_SEGMENT_DTYPE`

### Define configs:
```python
@dataclass
class SegmentWriterConfig:
    segment_dir: Path
    segment_size: int = 131072
    dtype: str = "fp16"
    shard_id: int = 0

@dataclass  
class SegmentIngestConfig:
    segment_dir: Path
    save_every: int = 2
```

### Tests:
- Write segments, then ingest into test DB + index
- Shard consistency validation works

### Acceptance criteria:
- No segment file format knowledge in cli.py
- cli.py calls high-level functions like `ingest_paper_segments()`
- Segment format changeable without touching CLI

---

## JOB 7 — Tar / Producer Pipeline

**Goal:** Remove tar scanning, sharding, and producer loops from cli.py.

**Target modules:**
- `litkit/ingest/pipeline.py` — Main producer pipeline
- `litkit/ingest/sharding.py` — Shard assignment and filtering

### Move these:
- `iter_tar_articles()`, `_is_uncompressed_tar()`
- `_IngestContext`, `_ingest_article()`, `_flush_paper_batch()`, `_flush_chunk_batch()`
- `_shard_filter()` and tar-to-shard assignment logic
- Main producer loop from `build_or_update_indices()`
- Checkpoint logic: `load_checkpoint()`, `save_checkpoint()`

### Define producer config:
```python
@dataclass
class ProducerConfig:
    shard_id: int
    num_shards: int
    tar_dir: Path | None
    tar_manifest: Path | None
    paper_embed_bs: int = 16
    chunk_embed_bs: int = 64
    parse_workers: int = 8
    chunk_target_chars: int = 1200
    chunk_min_chars: int = 300
    chunk_overlap: int = 200
```

### Clean API:
```python
def run_producer(
    config: ProducerConfig,
    paths: WorkspacePaths,
    db_conn: Connection,
    paper_embedder: Embedder,
    chunk_embedder: Embedder,
    segment_writers: tuple[SegmentWriter, ChunkSegmentWriter],
) -> ProducerResult:
    ...
```

### Tests:
- Small synthetic tar with 2-3 documents
- Run producer against temp DB + segment dir
- Verify DB rows and segments produced

### Acceptance criteria:
- No tarfile or manifest parsing in cli.py
- Entire producer path callable via `run_producer()` from tests
- Sharding in `ProducerConfig`, not CLI-only branches

---

## JOB 8 — Robustness, Performance, Resource Management

**Goal:** Ensure robustness under failure, efficiency for large workloads, predictable resource behavior.

### Work areas:

1. **Resource inventory** — Enumerate all files, subprocesses, network I/O, large allocations, GPU touches
2. **Structured resource management** — Context managers everywhere, `safe_open()` utilities
3. **Timeouts and limits** — Define defaults for network ops, max retries, max parallelism
4. **Error classification** — Domain-specific exceptions (`ConfigError`, `InputError`, etc.)
5. **Performance profiling** — Instrument hot paths, batch operations, cache derived values
6. **Concurrency sanity** — Worker pool lifetime, signal handling for graceful shutdown
7. **Memory controls** — Streaming/iterator processing, configurable cache sizes
8. **Hard-failure protection** — Atomic writes (temp + rename), output validation

### Tests:
- Resource cleanup (file handles closed)
- Exit codes on common failures
- Timeouts trigger as expected

### Acceptance criteria:
- Context-managed resources throughout
- Sensible timeouts, retries, limits
- Clear, structured error handling

---

## JOB 9 — UX, Ergonomics, Documentation Polish

**Goal:** Make CLI pleasant and predictable to use.

### Work areas:

1. **Command structure** — Rationalize commands/subcommands/flags
2. **Flag behavior** — Standardize patterns, align env vars with flags
3. **Help output** — One-line summaries, structured help, examples
4. **Output formatting** — Normal/quiet/verbose/debug modes
5. **Error messages** — Clear, actionable, context-rich
6. **Examples** — Task-oriented docs, runnable examples
7. **Migration notes** — Document breaking changes, provide shims
8. **Entrypoint polish** — `--version`, zero-arg behavior, exit codes

### Tests:
- Help output smoke tests
- Key error message tests

### Acceptance criteria:
- Clean, coherent command/flag layout
- High-quality `--help` for all commands
- Updated docs in sync with refactored CLI

---

## Testing Strategy

| Phase | Test Type | Location | Purpose |
|-------|-----------|----------|---------|
| Before extraction | Smoke test | `validate_build.sh` | Catch catastrophic regressions |
| During each job | Unit tests | `tests/test_<module>.py` | Prove extracted code works |
| After each job | Re-run smoke | `validate_build.sh` | Verify nothing broke |

### Smoke Test (validate_build.sh)

Uses `tiny_test.manifest` to:
1. Run a minimal multi-node build
2. Verify segment files consumed
3. Check FAISS index sizes
4. Validate SQLite DB counts
5. Detect duplicate handling

**Run after every refactor commit:**
```bash
./validate_build.sh /path/to/workspace
```

---

## Target Module Structure (Final)

```
src/litkit/
├── __init__.py
├── __main__.py
├── cli.py                      # THIN: argparse + dispatch only (~500 lines)
├── progress.py                 # _Progress, _Pulse, _eprint utilities
├── config/
│   ├── __init__.py
│   └── paths.py                # WorkspacePaths dataclass
├── db/
│   ├── __init__.py
│   ├── schema.py               # SCHEMA, init_db
│   ├── merge.py                # merge_shard_databases
│   └── queries.py              # Helpers, preload maps
├── index/
│   ├── __init__.py
│   ├── core.py                 # Index creation, loading, saving
│   ├── search.py               # _faiss_search, shortlist
│   └── training.py             # IVF-PQ training
├── retrieval/
│   ├── __init__.py
│   ├── papers.py               # shortlist_papers
│   ├── chunks.py               # search_chunks_constrained, get_chunks
│   └── lexical.py              # _query_terms, lexical prefilter
├── concurrent/
│   ├── __init__.py
│   └── locking.py              # FileLock, LockManager
├── segments/
│   ├── __init__.py
│   ├── writer.py               # SegmentWriter, ChunkSegmentWriter
│   ├── ingest.py               # _ingest_paper_segments, _ingest_chunk_segments
│   └── coordination.py         # ProducerCoordinator, ConsumerCoordinator
├── ingest/
│   ├── __init__.py
│   ├── ingest.py               # (existing) XML parsing
│   ├── pipeline.py             # run_producer, main loop
│   └── sharding.py             # _shard_filter, tar assignment
├── llm/
│   ├── __init__.py
│   ├── client.py               # answer_with_llm
│   └── context.py              # pack_context, approx_tokens
├── embeddings/                 # (existing, already modular)
├── formatting/                 # (existing, already modular)
└── frontload/                  # (existing)
```

---

## Global State Elimination

| Current Global | Target |
|----------------|--------|
| `ROOT`, `WORKSPACE` | `WorkspacePaths.root`, `.workspace` |
| `DB_PATH`, `CKPT_PATH` | `WorkspacePaths.db_path`, `.ckpt_path` |
| `PAPER_INDEX_PATH`, `CHUNK_INDEX_PATH` | `WorkspacePaths.paper_index_path`, `.chunk_index_path` |
| `DB_LOCK`, `FAISS_LOCK`, `CKPT_LOCK` | `LockManager` instance |
| `paper_seg_writer`, `chunk_seg_writer` | Constructor arguments |
| `_PENDING_MARKS` | Return value or context object |
| `QUIET` | `OutputConfig.quiet` |
