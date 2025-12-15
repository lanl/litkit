# Refactor Roadmap for cli.py

This document tracks the planned extraction of functionality from the monolithic `cli.py` into purpose-specific modules.

## Current State

`cli.py` is ~5500 lines and handles:
- CLI parsing and argument handling
- Environment discovery and workspace path resolution
- SQLite schema, initialization, migrations, and merge logic
- FAISS index construction, training, saving, loading, and dedup logic
- Concurrency and locking (FileLock, DB/FAISS locks, writer guard)
- Producer pipeline (tar scanning, sharding, embedding production, segment writing)
- Consumer pipeline (segment ingestion, DB + FAISS updates, reconciliation)
- Retrieval logic (shortlisting, chunk search, lexical prefilter, ID mappings)
- LLM orchestration and prompt/context packing
- Progress / logging utilities and user-facing UI

## Target Module Structure

```
src/litkit/
├── __init__.py
├── __main__.py
├── cli.py                    # THIN: argparse + dispatch only
├── config.py                 # Paths, env vars, defaults (no globals)
├── progress.py               # _Progress, _Pulse, _eprint utilities
├── build/
│   ├── __init__.py
│   ├── sharding.py           # Tar-to-shard assignment, build metadata
│   ├── producer.py           # Producer pipeline orchestration
│   ├── consumer.py           # Consumer pipeline orchestration
│   ├── segments.py           # _SegmentWriter, _ChunkSegmentWriter
│   ├── checkpoints.py        # load_checkpoint, save_checkpoint
│   └── coordination.py       # ProducerCoordinator, ConsumerCoordinator
├── db/
│   ├── __init__.py
│   ├── schema.py             # SCHEMA constant, init_db, _ensure_in_index_columns
│   ├── queries.py            # _mark_in_index, chunk_ids_to_paper_ids, etc.
│   └── merge.py              # merge_shard_databases, init_shard_db
├── faiss_ops/
│   ├── __init__.py
│   ├── indices.py            # _faiss_load, _faiss_save, _hnsw_index, _flat_ip_index
│   ├── training.py           # IVF-PQ training logic
│   └── search.py             # _faiss_search, shortlist_papers
├── retrieval/
│   ├── __init__.py
│   ├── papers.py             # shortlist_papers
│   ├── chunks.py             # search_chunks_constrained, get_chunks
│   └── lexical.py            # _query_terms, lexical prefilter SQL
├── llm/
│   ├── __init__.py
│   ├── client.py             # answer_with_llm
│   └── context.py            # pack_context, approx_tokens
├── embeddings/               # Already modular
├── ingest/                   # Already modular
└── formatting/               # Already modular
```

## Domain-to-Module Mapping

| Domain | Target Module | Key Functions/Classes |
|--------|---------------|----------------------|
| **Config/Paths** | `litkit.config` | `_find_root`, `_resolve_workspace`, `ROOT`, `WORKSPACE`, path constants |
| **Progress/Logging** | `litkit.progress` | `_Progress`, `_Pulse`, `_eprint`, `_phase` |
| **Database Schema** | `litkit.db.schema` | `SCHEMA`, `init_db`, `_ensure_in_index_columns` |
| **Database Merge** | `litkit.db.merge` | `merge_shard_databases`, `init_shard_db`, `_shard_db_path` |
| **Database Queries** | `litkit.db.queries` | `_mark_in_index`, `chunk_ids_to_paper_ids`, `already_processed`, `register_file` |
| **FAISS Indices** | `litkit.faiss_ops.indices` | `_faiss_load`, `_faiss_save`, `_hnsw_index`, `_flat_ip_index`, `_ivfpq_index` |
| **FAISS Training** | `litkit.faiss_ops.training` | IVF-PQ training loop (in `build_or_update_indices`) |
| **FAISS Search** | `litkit.faiss_ops.search` | `_faiss_search`, `_auto_set_nprobe`, `_faiss_present_ids` |
| **Build Sharding** | `litkit.build.sharding` | `_shard_filter`, `_write_build_meta`, `_read_build_meta`, `_validate_shard_consistency` |
| **Build Segments** | `litkit.build.segments` | `_SegmentWriter`, `_ChunkSegmentWriter`, segment I/O |
| **Build Coordination** | `litkit.build.coordination` | `ProducerCoordinator`, `ConsumerCoordinator` |
| **Build Checkpoints** | `litkit.build.checkpoints` | `load_checkpoint`, `save_checkpoint`, `CKPT_PATH` |
| **Retrieval Papers** | `litkit.retrieval.papers` | `shortlist_papers` |
| **Retrieval Chunks** | `litkit.retrieval.chunks` | `search_chunks_constrained`, `get_chunks` |
| **Retrieval Lexical** | `litkit.retrieval.lexical` | `_query_terms`, `_normalize_for_search_py`, lexical SQL |
| **LLM Client** | `litkit.llm.client` | `answer_with_llm`, `_clarify_llm_error` |
| **LLM Context** | `litkit.llm.context` | `pack_context`, `approx_tokens`, `SYS_PROMPT` |
| **Citations** | `litkit.formatting.citations` | `_normalize_and_strip_citations` |
| **CLI** | `litkit.cli` | `main()`, argparse setup, dispatch logic |

## Extraction Priority

### Phase 1: Immediate (enables flexible shard count)
1. **`litkit.build.sharding`** — Extract shard assignment and metadata management
   - `_shard_filter()`
   - `_write_build_meta()`, `_read_build_meta()`
   - `_validate_shard_consistency()`
   - `BuildMeta` dataclass

### Phase 2: High Value
2. **`litkit.db`** — Extract database operations
   - Schema + init functions
   - Merge logic (already bulk-optimized)
   - Query helpers

3. **`litkit.faiss_ops`** — Extract FAISS operations
   - Index creation/loading/saving
   - Search operations
   - Training logic

### Phase 3: Cleanup
4. **`litkit.build.segments`** — Segment writers and readers
5. **`litkit.retrieval`** — Search pipeline
6. **`litkit.llm`** — LLM client and context packing
7. **`litkit.progress`** — Progress utilities
8. **`litkit.config`** — Configuration and paths

## Global State to Eliminate

The following globals in cli.py should become explicit config/dependency injection:

| Global | Replacement |
|--------|-------------|
| `ROOT`, `WORKSPACE` | `Config.root`, `Config.workspace` |
| `DB_PATH`, `CKPT_PATH` | `Config.db_path`, `Config.ckpt_path` |
| `PAPER_INDEX_PATH`, `CHUNK_INDEX_PATH` | `Config.paper_index_path`, `Config.chunk_index_path` |
| `DB_LOCK`, `FAISS_LOCK`, `CKPT_LOCK` | `LockManager` instance |
| `paper_seg_writer`, `chunk_seg_writer` | Constructor arguments |
| `_PENDING_MARKS` | Return value from functions |

## Extraction Checklist

- [ ] Create `litkit.build.sharding` module
- [ ] Create `litkit.db` package with schema/merge/queries
- [ ] Create `litkit.faiss_ops` package
- [ ] Create `litkit.build.segments` module
- [ ] Create `litkit.retrieval` package
- [ ] Create `litkit.llm` package
- [ ] Create `litkit.progress` module
- [ ] Create `litkit.config` module
- [ ] Remove global state from cli.py
- [ ] Make cli.py a thin dispatcher only

## Testing Strategy

1. Each extracted module gets unit tests
2. Integration tests verify end-to-end workflows still work
3. Use the existing `validate_build.sh` as a smoke test after each extraction
