# LitKit Code Review Summary

> **Last Updated:** December 12, 2025

This document provides a comprehensive analysis of the litkit codebase as it exists today.

## Architecture Overview

LitKit is a two-stage RAG (Retrieval-Augmented Generation) pipeline designed for HPC environments. It processes the PMC-OA corpus (PubMed Central Open Access) and provides semantic search over scientific literature.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MULTI-NODE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│   │  Producer 0  │   │  Producer 1  │   │  Producer 2  │  (N-1 nodes)  │
│   │  (GPU Node)  │   │  (GPU Node)  │   │  (GPU Node)  │               │
│   │              │   │              │   │              │               │
│   │ • Parse XML  │   │ • Parse XML  │   │ • Parse XML  │               │
│   │ • Embed docs │   │ • Embed docs │   │ • Embed docs │               │
│   │ • Write .npz │   │ • Write .npz │   │ • Write .npz │               │
│   │ • Shard DB   │   │ • Shard DB   │   │ • Shard DB   │               │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘               │
│          │                  │                  │                        │
│          └──────────────────┼──────────────────┘                        │
│                             │                                           │
│                     ┌───────▼────────┐                                  │
│                     │ Shared Storage │  (Lustre/NFS)                    │
│                     │                │                                  │
│                     │ • emb_segments/│  (.npz vector files)             │
│                     │ • sqlite/      │  (shard DBs + main DB)           │
│                     │ • indices/     │  (FAISS files)                   │
│                     └───────┬────────┘                                  │
│                             │                                           │
│                     ┌───────▼────────┐                                  │
│                     │   Consumer     │  (1 dedicated node)              │
│                     │                │                                  │
│                     │ • Merge shards │                                  │
│                     │ • Ingest .npz  │                                  │
│                     │ • Build FAISS  │                                  │
│                     │ • Backfill     │                                  │
│                     └────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Modules

### `src/litkit/cli.py` (~3500 lines)

The main module containing:

| Component | Description |
|-----------|-------------|
| `build_or_update_indices()` | Core build logic for single/multi-node |
| `ProducerCoordinator` | Writes `.shard_XX_complete` markers |
| `ConsumerCoordinator` | Polls for producer completion (10h timeout) |
| `_SegmentWriter` | Writes paper embeddings to `.npz` files |
| `_ChunkSegmentWriter` | Writes chunk embeddings to `.npz` files |
| `merge_shard_databases()` | Merges per-shard SQLite DBs into main |
| `_ingest_paper_segments()` | Consumer ingests paper vectors into FAISS |
| `_ingest_chunk_segments()` | Consumer ingests chunk vectors into FAISS |
| `reconcile_sqlite_flags_with_faiss()` | Ensures `in_index` flags match FAISS |
| `backfill_unindexed_vectors()` | Re-embeds any rows with `in_index=0` |

### `src/litkit/embeddings/`

| Module | Description |
|--------|-------------|
| `base.py` | `Embedder` protocol and progress utilities |
| `factory.py` | Creates paper/chunk embedders |
| `pool.py` | Multi-GPU worker pool for SBERT |
| `specter2.py` | SPECTER2 paper embeddings |
| `sbert_mpnet.py` | SBERT chunk embeddings |
| `devices.py` | GPU detection and thread configuration |

### `src/litkit/ingest/`

| Module | Description |
|--------|-------------|
| `ingest.py` | Tar streaming, XML parsing, parallel parsing |

### `src/litkit/formatting/`

| Module | Description |
|--------|-------------|
| `answers.py` | Citation normalization and reference rendering |

## Data Flow

### 1. Ingestion (Producer Mode)

```
tar files → XML parsing → SQLite (shard DB) → Embedding → .npz segments
```

- Each producer processes a shard of tar files (load-balanced by size)
- Uses `--embed-producer --shard-id X --num-shards N`
- Writes to `litkit_shard_XX.sqlite3` (no lock contention)
- Outputs `papers_shXX_*.npz` and `chunks_shXX_*.npz`

### 2. Consumption (Consumer Mode)

```
.npz segments → FAISS indices + shard DBs → main DB merge → backfill
```

- Single consumer runs with `--faiss-writer --consume-only`
- Polls `emb_segments/` for new `.npz` files
- Waits for `.shard_XX_complete` markers (up to 10 hours)
- Merges shard DBs using `doc_id` for deduplication
- Runs `backfill_unindexed_vectors()` to catch any gaps

### 3. Retrieval (Query Mode)

```
Question → SPECTER2 (Stage 1) → SBERT (Stage 2) → LLM → Answer
```

- Stage 1: HNSW search over paper embeddings
- Stage 2: ANN search over chunks, filtered to candidate papers
- Lexical front-loading for rare terms (e.g., "rulemonkey")
- Per-paper cap to ensure diversity in context

## Coordination Protocol

### Producer Completion Signaling

1. Producer finishes processing its tar shard
2. Writes `.shard_XX_complete` marker file (JSON with timestamp, hostname, pid)
3. Uses atomic write pattern: `.tmp` → `os.replace()` → final

### Consumer Completion Detection

1. Consumer polls `emb_segments/` for markers every 30 seconds
2. Logs progress: "Waiting for producers: X/N complete"
3. Timeout: 36000 seconds (10 hours) to match the cluster's max job time
4. After all producers complete, runs final ingestion pass

### Shard Database Merging

1. Consumer reads all `litkit_shard_*.sqlite3` files
2. Inserts papers using `doc_id` for deduplication (prevents duplicates)
3. Remaps chunk/file `paper_id` references to main DB IDs
4. Deletes shard DBs after successful merge

## Deduplication

Papers are deduplicated across shards using SQLite unique indexes:
- `papers_pmcid_uq` — unique on `pmcid` (preferred)
- `papers_pmid_uq` — unique on `pmid` (fallback)

When merging shard databases, the consumer uses these indexes to prevent duplicates. If a paper has neither pmcid nor pmid, no cross-shard deduplication occurs (rare edge case).

## FAISS Index Types

| Index | Type | Use Case |
|-------|------|----------|
| Papers | HNSW (M=32, efConstruction=200) | Fast ANN for ~millions of papers |
| Chunks | IVF-PQ or FLAT | Depends on corpus size |

- FLAT: Used for small corpora or when IVF-PQ training fails
- IVF-PQ: Used for large corpora (requires training pass)

## Segment File Lifecycle

```
Producer:
  .tmp → os.replace() → final .npz → Producer completes

Consumer:
  .npz → os.replace() → .npz.ingesting → ingest → delete
        (on error)    → os.replace() → .npz (retry next run)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LITKIT_WORKSPACE` | `./workspace` | Writable output directory |
| `LITKIT_SQLITE_BUSY_TIMEOUT_MS` | `120000` | SQLite busy timeout (ms) |
| `LITKIT_SAVE_EVERY_SEC` | `120` | FAISS save throttle (seconds) |
| `LITKIT_EMBED_SEGMENT_SIZE` | `131072` | Vectors per segment file |
| `LITKIT_EMBED_SEGMENT_DTYPE` | `fp16` | Segment storage dtype |

### Key CLI Flags

| Flag | Description |
|------|-------------|
| `--faiss-writer` | This process mutates FAISS indices |
| `--embed-producer` | Producer mode (embed + write segments) |
| `--consume-only` | Consumer mode (ingest segments only) |
| `--init-indices-only` | Bootstrap empty FAISS indices |
| `--shard-id` / `--num-shards` | Producer shard assignment |
| `--tar-manifest` | Path to manifest file listing tar paths |
| `--chunks-index flat` | Use FLAT index (skip IVF-PQ training) |

## Locking Strategy

Lock order (always acquire in this sequence to prevent deadlock):
1. `DB_LOCK` (SQLite writes)
2. `FAISS_LOCK` (FAISS mutations)

Additional guards:
- `WRITER_GUARD`: Prevents multiple `--faiss-writer` processes
- Advisory file locks via `fcntl.flock()` (best-effort on NFS)

## Known Limitations

### No Dead-Letter Queue
- Failed segments are renamed back to `.npz` for retry
- Persistently failing segments will retry indefinitely
- **Mitigation:** Manual inspection if a segment keeps failing

### No Real-Time Monitoring
- Progress is logged to stderr
- No Prometheus/metrics endpoint
- **Mitigation:** Parse logs or use `validate_build.sh`

### Single Consumer Bottleneck
- Only one consumer can ingest segments
- Could be parallelized with sharded FAISS indices
- **Current design:** Acceptable for ~1M paper corpora

## Validation

After a build completes, run:

```bash
./validate_build.sh /path/to/workspace
```

This checks:
1. All segment files consumed (directory empty)
2. FAISS indices exist with non-trivial size
3. SQLite `in_index` flags match FAISS `ntotal`

## Future Work

1. **Dead-letter queue** for permanently failing segments
2. **Prometheus metrics** for monitoring
3. **Sharded FAISS** for parallel consumer ingestion
4. **Incremental updates** without full rebuild
5. **Better error recovery** with automatic backoff

---

**Codebase:** `src/litkit/`  
**Branch:** `feature/multi-producer-sqlite`  
**Status:** Production-ready for multi-node builds
