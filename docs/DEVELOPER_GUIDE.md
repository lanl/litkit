# LitKit Developer Guide

This document contains technical implementation details, optimization strategies, and architectural notes for developers working on LitKit internals.

## Concurrency Model

LitKit supports multi-node builds with a **producer/consumer architecture**:

| Role | Database | Description |
|------|----------|-------------|
| **Producers** (`--embed-producer`) | Shard-specific DB | Each producer writes to its own SQLite file |
| **Consumer** (`--consume-only --faiss-writer`) | Main DB | Merges shard DBs and ingests embedding segments |
| **Single-node** (`--faiss-writer`) | Main DB | All operations in one process |

### ⚠️ Important: Do NOT Share a Main DB Across Writers

SQLite handles concurrent *readers* well, but concurrent *writers* to the same database file will cause `SQLITE_BUSY` errors, especially on network filesystems (NFS/Lustre).

**Correct multi-node setup:**
```bash
# Producers (one per node, each gets own shard DB)
srun --ntasks=N litkit --embed-producer --shard-id $SLURM_PROCID --num-shards N

# Consumer (single node, merges DBs at end)
litkit --consume-only --faiss-writer --num-shards N
```

**Incorrect (will fail):**
```bash
# WRONG: Multiple processes writing to same main DB
srun --ntasks=N litkit --faiss-writer ...  # Race conditions!
```

## Improving Lexical Search

The lexical front-loading module (`litkit/retrieval/lexical.py`) is designed for easy improvement:

### Architecture

```
litkit/retrieval/lexical.py
├── LexicalConfig      # Configuration dataclass (enable, cap, limit, etc.)
├── LexicalResult      # Results with scores dict for future BM25
├── LexicalBackend     # Protocol for swappable backend implementations
├── SqliteLexicalBackend # Default LIKE-based SQLite backend
├── find_rare_terms()  # Pure function for term extraction (no DB deps)
└── merge_lexical_and_ann() # Interleave lexical + ANN results
```

### Improvement Areas

**1. Better Term Extraction** (`find_rare_terms`)
- Current: triggers on digits, hyphens, long terms (9+ chars)
- Ideas: domain-specific vocabularies, stemming, abbreviation expansion, UMLS lookup

**2. Scoring** (`LexicalResult.scores`)
- Current: flat 1.0 for all matches (placeholder)
- Ideas: BM25, TF-IDF, term frequency weighting, title vs body position

**3. Backend Swap** (`LexicalBackend` protocol)
- Default: SQLite LIKE patterns (works everywhere)
- Future: FTS5, Tantivy, Vespa, or external search indices

**4. Configuration** (`LexicalConfig`)
```python
@dataclass
class LexicalConfig:
    enabled: bool = True
    cap: int | None = None      # None => caller computes from k
    limit: int = 200            # SQL LIMIT for lexical scan
    allow_global: bool = False  # Force global scope
    min_term_length: int = 9    # Long-term fallback threshold
    max_terms: int = 8          # Cap # of rare terms
    require_rare_terms: bool = True
```

### Testing Lexical in Isolation

```python
from litkit.retrieval.lexical import find_rare_terms, LexicalConfig

config = LexicalConfig(min_term_length=7, max_terms=5)
terms = find_rare_terms("What causes SARS-CoV-2 infection?", config)
# ['sars-cov-2', 'infection']
```

### Adding a New Backend

Implement the `LexicalBackend` protocol:

```python
from litkit.retrieval.lexical import LexicalBackend, LexicalConfig, LexicalResult

class MyFTS5Backend:
    def __init__(self, db_conn):
        self.db_conn = db_conn
    
    def search(
        self,
        terms: list[str],
        candidate_papers: set[int] | None,
        config: LexicalConfig,
    ) -> LexicalResult:
        # Implement FTS5 search
        chunk_ids = [...]  # Your FTS5 query
        scores = {cid: fts5_score for cid in chunk_ids}
        return LexicalResult(
            chunk_ids=chunk_ids,
            scores=scores,
            terms_used=terms,
            scope="candidates" if candidate_papers else "global",
        )
```

Then use it in `search_chunks_constrained`:
```python
backend = MyFTS5Backend(db_conn)
lexical_result = backend.search(rare_terms, candidate_scope, lexical_config)
```

## High-ROI Efficiency Improvements

These are the highest-impact optimizations identified during code review, ordered by expected ROI for large corpus builds.

### 1. Time-Gated FAISS Saves (High Impact)

**Problem:** `faiss_save()` rewrites the entire index file. Doing this per 20K vectors is expensive on Lustre/NFS, and the I/O can dominate GPU embedding time.

**Current:** Saves after every batch flush (every ~20K vectors).

**Recommendation:** Save at most once every 2-5 minutes, plus a final save at end:
```python
_SAVE_MIN_SEC = 120  # Already defined but unused
_last_save_ts = {"papers": 0.0, "chunks": 0.0}

def throttled_faiss_save(index, path, key: str) -> bool:
    now = time.time()
    if now - _last_save_ts[key] < _SAVE_MIN_SEC:
        return False  # Skip save, reconcile+backfill will repair
    _last_save_ts[key] = now
    return faiss_save(index, path)
```

Since reconcile+backfill repairs any missing vectors on restart, frequent saves are unnecessary for correctness—only for reducing rework after crashes.

### 2. Embedding Cache by Content Hash (Medium-High Impact)

**Problem:** Titles/abstracts repeat across corpora. Chunk boilerplate (references, acknowledgments) repeats. Rebuilds recompute identical embeddings.

**Recommendation:** On-disk cache with `sha1(text) → vector`:
```python
import hashlib
import numpy as np

class EmbeddingCache:
    def __init__(self, cache_dir: Path, dtype="fp16"):
        self.cache_dir = cache_dir
        self.dtype = np.float16 if dtype == "fp16" else np.float32
    
    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()
    
    def get(self, text: str) -> np.ndarray | None:
        path = self.cache_dir / self._key(text)
        if path.exists():
            return np.fromfile(path, dtype=self.dtype)
        return None
    
    def put(self, text: str, vec: np.ndarray):
        path = self.cache_dir / self._key(text)
        vec.astype(self.dtype).tofile(path)
```

**Usage:** Before embedding a batch, partition into cache hits and misses. Only embed misses, then merge results. Expected impact: 30-50% reduction in GPU time on rebuilds.

### 3. Deterministic Vector IDs (Architectural Change)

**Problem:** Current design requires SQLite `lastrowid` for each chunk before writing vectors. This forces row-by-row inserts and makes parallel ID assignment impossible.

**Current:** `INSERT → lastrowid → add_with_ids()`

**Recommendation:** Derive stable IDs from content:
```python
def chunk_vector_id(paper_id: int, ord: int) -> int:
    """Pack (paper_id, ord) into a 64-bit ID for FAISS."""
    return (paper_id << 16) | (ord & 0xFFFF)

def paper_vector_id(doc_id: str) -> int:
    """Hash doc_id to 64-bit ID for FAISS."""
    return int(hashlib.sha1(doc_id.encode()).hexdigest()[:16], 16)
```

**Schema change:** Add `chunks.vector_id INTEGER UNIQUE` column (or use as PRIMARY KEY). Then `executemany()` all chunk inserts without per-row `lastrowid`.

**Impact:** This is the single biggest structural win for multi-node scaling. Producers can generate IDs independently without coordination.

### 4. Memmap-Friendly Segment Ingestion (Medium Impact)

**Current:** Segment consumer iterates over files, calling `add_with_ids()` per segment.

**Recommendation:** Memory-map vectors and batch adds in large contiguous slabs:
```python
def ingest_segments_batched(index, seg_dir: Path, batch_size: int = 100_000):
    """Ingest all segments in one pass with large batch adds."""
    all_ids, all_vecs = [], []
    for seg_path in seg_dir.glob("*.seg"):
        seg = load_segment(seg_path)  # Memory-mapped
        all_ids.extend(seg.ids)
        all_vecs.append(seg.vecs)
        
        if len(all_ids) >= batch_size:
            stacked = np.vstack(all_vecs)
            index.add_with_ids(stacked, np.array(all_ids))
            all_ids, all_vecs = [], []
    
    # Final flush
    if all_ids:
        index.add_with_ids(np.vstack(all_vecs), np.array(all_ids))
```

Expected impact: 2-3x faster segment ingestion vs. many small adds.

## Improving Tar Processing Efficiency

LitKit is designed to process the full PMC-OA corpus (~5M articles) in under 10 hours using multi-node GPU parallelism.

### Current Architecture

| Stage | Parallelism | Bottleneck |
|-------|-------------|------------|
| Tar I/O | Per-shard (N producers) | FS bandwidth, stripe width |
| XML Parsing | Thread pool (8 workers/shard) | CPU-bound |
| Embedding | GPU (multi-device pooling) | GPU memory/bandwidth |
| DB Writes | Per-shard SQLite (lock-free) | Disk IOPS |
| FAISS Updates | Single consumer | Index structure |

### Improvement Areas

#### 1. SQLite Batching

**Current:** Row-by-row `INSERT` statements via `execute()`.

**Improvement:** Use `executemany()` with batched inserts:
```python
# Before
for row in rows:
    cur.execute("INSERT INTO chunks (...) VALUES (...)", row)

# After
cur.executemany("INSERT INTO chunks (...) VALUES (?,?,?)", rows)
```

Expected impact: 2-5x faster DB writes for large batches.

**Implementation notes:**
- Batch papers/chunks buffers before INSERT
- Consider staging table without indices during ingest, then CREATE INDEX at end
- Test with SQLite PRAGMA optimizations (`synchronous=OFF` during bulk, larger `cache_size`)

#### 2. Uncompressed Tar Shards

**Strongly recommended:** Use uncompressed `.tar` files, not `.tar.gz`.

Compressed tars force sequential decompression, serializing the entire I/O path. Uncompressed tars allow:
- Parallel I/O across shard files
- Multi-threaded XML parsing (lxml releases GIL)

Pre-decompress shards: `gunzip -k shard_*.tar.gz`

#### 3. Pipeline Overlap

**Current:** Parse → Embed → Write are sequential per batch.

**Future:** True async pipeline:
- Parser fills embedding queue while GPU processes previous batch
- DB writes happen while next batch embeds

This requires restructuring the main loop with asyncio or thread-based producers/consumers.

#### 4. FAISS Training Optimization

IVF-PQ training samples 150K chunks from tar files. For very large corpora:
- Consider pre-computed training set (sample once, reuse)
- Or train on first N tars only, not random sample from all

## Lustre Filesystem Optimization

On Lustre filesystems, file striping dramatically affects I/O throughput. Under-striped tar shards will bottleneck all downstream GPU processing.

### Recommended Stripe Settings

For tar shards on Lustre:
- `stripe_count`: -1 (all OSTs) or at least 8
- `stripe_size`: 4M or larger for sequential reads

### Checking Stripe Width

```bash
# Check a specific file
lfs getstripe /path/to/shard.tar

# Check directory default
lfs getstripe -d /path/to/tar_shards/
```

### Future Work: Automatic Stripe Detection

LitKit could detect poor striping and warn users at startup:

```python
import subprocess

def check_lustre_stripe(path: Path, min_stripe_count: int = 4) -> tuple[bool, str]:
    """Check if path has adequate Lustre striping.
    
    Returns (ok, message) where ok=True if striping is adequate or non-Lustre.
    """
    # Detect Lustre via statfs magic number (0x0BD00BD0)
    # ... statfs check ...
    
    # Run lfs getstripe
    try:
        result = subprocess.run(
            ["lfs", "getstripe", "-c", str(path)],
            capture_output=True, text=True, timeout=10
        )
        stripe_count = int(result.stdout.strip())
        if stripe_count < min_stripe_count:
            return False, f"Low stripe count ({stripe_count}); recommend >= {min_stripe_count}"
        return True, f"Stripe count: {stripe_count}"
    except FileNotFoundError:
        return True, "lfs not available"  # Degrade gracefully
    except Exception as e:
        return True, f"Stripe check failed: {e}"
```

### User Remediation

If LitKit warns about poor striping:

```bash
# Option 1: Re-stripe existing files (requires copy)
mkdir /new/tar_shards
lfs setstripe -c -1 -S 4M /new/tar_shards
cp /old/tar_shards/*.tar /new/tar_shards/

# Option 2: Set directory default for new files
lfs setstripe -c -1 -S 4M /path/to/tar_shards/
# Then copy/download tars fresh
```

## Known Limitations & Deferred Issues

These issues were identified during code review but deferred to future work. They are documented here for transparency and to guide future improvements.

### 1. Unparsable XML Permanently Skipped

**Current behavior:** When XML parsing fails (`meta is None`), the member is marked as `handled_ok = True` and `processed_count` advances. The checkpoint permanently skips that member on future runs.

**Risk:** If parsing was transiently flaky (I/O error, lxml edge case fixed in a later version), you've encoded data loss into the checkpoint.

**Future improvement:** Distinguish "definitively bad" vs "transient failure":
- For definitively bad: record a skip marker in DB (or skiplist file) with reason/hash/mtime
- For transient: either don't advance checkpoint, or record "bad parse" with a version stamp so future versions can optionally reattempt

### 2. Writer Guard Heartbeat for Long Builds (TODO)

**Current behavior:** The writer guard file is written once at startup with `PID hostname timestamp`. Cross-host TTL eviction (when `LITKIT_WRITER_GUARD_TTL > 0`) cannot verify if the remote process is still alive, so it relies solely on timestamp staleness.

**Risk:** A legitimate long-running build (>24h on HPC) will have its guard evicted when TTL expires, allowing a second writer to start and corrupt the index.

**Future improvement:** Periodic heartbeat updates during long builds.

**Workaround (current):**
- Default TTL is 0 (disabled) - no auto-eviction
- Same-host PID liveness check via `os.kill(pid, 0)` still works
- Set `LITKIT_WRITER_GUARD_TTL=172800` (48h) for very long builds
- Manually remove stale guards: `rm workspace/.writer_guard`

### 3. Writer Guard Identity Verification

**Current behavior:** The writer guard file contains `PID hostname timestamp`. On stale detection, we check if the PID is alive on the same host via `os.kill(pid, 0)`.

**Risk:** PID reuse is rare but real on HPC nodes.

**Future improvement:** Store stronger identity in the guard file (e.g., a nonce).

### 4. Segment File Claiming (`.ingesting` Overwrite Risk)

**Current behavior:** Segment ingestion claims files by renaming `file.npz` to `file.npz.ingesting`. If `.ingesting` already exists (e.g., from a crashed run), `os.replace()` will overwrite it.

**Risk:** Could drop an un-ingested segment if two processes race or a previous crash left a file behind.

**Future improvement:** Use unique claim names with PID and hostname.

### 5. Signal Handlers Use `os._exit(1)`

**Current behavior:** SIGINT/SIGTERM handlers call `os._exit(1)` after cleaning up the writer guard file. This bypasses Python's normal shutdown sequence.

**Implications:**
- SQLite cleanup and in-flight filesystem buffering may be skipped
- FAISS indices may have unflushed data
- Recovery relies on reconcile+backfill on restart

**Design rationale:** This is intentional to avoid deadlocks and ensure the guard file is always cleaned up. The tradeoff is accepted because reconcile+backfill repairs any partial state.

### 6. Preloaded ID Maps Scalability

**Current behavior:** Segment ingestion preloads `paper_id_map` and `chunk_id_map` into memory for O(1) lookups during doc_id resolution.

**Risk:** On very large corpora (tens of millions of chunks), the chunk_id_map could consume significant RAM and cause slow startup.

**Future improvement:** Use batched lookups or on-disk index.

### 7. Producer Mode: Duplicate Segments on Crash

**Current behavior:** Producer nodes write embedding segments, then commit DB rows, then update the checkpoint. A crash between segment write and checkpoint update causes the producer to reprocess the same batch on restart, generating duplicate segment files.

**Impact:** NOT data loss: consumer ingestion is idempotent. Only cost is storage bloat from orphan segment files.

## Version Bump Checklist

When releasing a new version, update the version string in these **5 files**:

| File | Location | Format |
|------|----------|--------|
| `pyproject.toml` | Line ~7 | `version = "X.Y.Z"` |
| `justfile` | Line ~54 | `tag := "vX.Y.Z-" + arch + "-" + flavor` |
| `vector_build_single.sbatch` | Line ~124 | `IMG="...litkit-vX.Y.Z-aarch64-lean.sqfs"` |
| `vector_build_multi.sbatch` | Line ~69 | `IMG="...litkit-vX.Y.Z-aarch64-lean.sqfs"` |
| `vector_resume_consumer.sbatch` | Line ~53 | `IMG="...litkit-vX.Y.Z-aarch64-lean.sqfs"` |

**Quick check:**
```bash
grep -rn "0\.3\." --include="*.py" --include="*.toml" --include="justfile" --include="*.sbatch" .
```

**After version bump:**
1. `just reset && just build` — rebuild container with new tag
2. Copy `.sqfs` to cluster: `/path/to/litkit/sqfs/`
3. Commit and push all changed files
