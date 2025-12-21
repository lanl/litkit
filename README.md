# LitKit

A scalable two-stage RAG (Retrieval-Augmented Generation) pipeline for scientific literature. Build searchable vector indices from JATS/NXML XML corpora and query with natural language questions.

## ⚠️ Important: Data Preparation

**LitKit does not deduplicate papers.** Each XML file in your tar archives is treated as a unique document.

- **Your responsibility:** Ensure tar/tar.gz files do not contain duplicate XML files
- **If duplicates exist:** The same paper will be indexed multiple times, appearing multiple times in search results
- **Recommended:** Deduplicate your corpus before ingestion

## Quick Start

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Build Index

```bash
# Place tar archives in workspace/tar_shards/
litkit --build-only --faiss-writer --tar-dir workspace/tar_shards
```

### Query

```bash
litkit "What mechanisms are described in these papers?"
```

## Platform Guides

| Platform | Guide | Description |
|----------|-------|-------------|
| **HPC Cluster** | [LITKIT_CLUSTER_GUIDE.md](LITKIT_CLUSTER_GUIDE.md) | HPC deployment with Charliecloud, multi-node builds, GPU passthrough |
| **macOS** | [LITKIT_MAC_GUIDE.md](LITKIT_MAC_GUIDE.md) | Local development on MacBook (Apple Silicon or Intel) |

## How It Works

LitKit uses a **two-stage retrieval** approach:

1. **Stage 1 (Papers):** SPECTER2 embeddings (title+abstract) → HNSW index for fast paper-level search
2. **Stage 2 (Chunks):** SBERT embeddings (body text chunks) → IVF-PQ index for passage-level search
3. **LLM Synthesis:** Top chunks are passed to an LLM which generates an answer with citations

## Key Features

- **Streaming tar ingestion** — process tar.gz archives without extraction
- **Resumable builds** — checkpoint-based recovery from interruption
- **Multi-node support** — producer/consumer architecture for parallel processing
- **Air-gapped operation** — works offline with cached HuggingFace models
- **Token-budgeted prompts** — automatic context trimming for LLM limits

## Common Options

```bash
# Build options
--rebuild              # Wipe and rebuild from scratch
--update               # Append new files only
--papers-index {hnsw,flat}
--chunks-index {ivfpq,flat}

# Query options  
--llm-model MODEL      # e.g., gpt-oss:20b, o3-mini
--top-papers N         # Stage 1 shortlist size (default: 500)
--top-chunks N         # Chunks for LLM context (default: 30)
--no-llm               # Retrieval only, no LLM call

# Performance
--embed-devices auto   # Device selection (auto, cpu, mps, cuda:0,cuda:1)
--paper-embed-bs 16    # Paper embedding batch size
--chunk-embed-bs 64    # Chunk embedding batch size
```

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LITKIT_WORKSPACE` | Output directory for indices/database |
| `LITKIT_TAR_DIR` | Default tar shard directory |
| `HF_HOME` | HuggingFace model cache location |
| `HF_HUB_OFFLINE=1` | Force offline mode |

## Full Help

```bash
litkit --help
litkit --version
```

## Generating PDF Documentation

To create a combined PDF of all documentation:

```bash
# Install prerequisites (macOS)
brew install pandoc
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install xetex collection-fontsrecommended

# Generate combined PDF
pandoc README.md LITKIT_MAC_GUIDE.md LITKIT_CLUSTER_GUIDE.md \
  --pdf-engine=xelatex \
  --toc --number-sections \
  -V geometry:margin=1in \
  -o litkit_documentation.pdf
```

## Developer Guide: Improving Lexical Search

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

## Developer Guide: High-ROI Efficiency Improvements

These are the highest-impact optimizations identified during code review, ordered by
expected ROI for large corpus builds.

### 1. Time-Gated FAISS Saves (High Impact)

**Problem:** `faiss_save()` rewrites the entire index file. Doing this per 20K vectors
is expensive on Lustre/NFS, and the I/O can dominate GPU embedding time.

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

Since reconcile+backfill repairs any missing vectors on restart, frequent saves are
unnecessary for correctness—only for reducing rework after crashes.

### 2. Embedding Cache by Content Hash (Medium-High Impact)

**Problem:** Titles/abstracts repeat across corpora. Chunk boilerplate (references,
acknowledgments) repeats. Rebuilds recompute identical embeddings.

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

**Usage:** Before embedding a batch, partition into cache hits and misses. Only embed
misses, then merge results. Expected impact: 30-50% reduction in GPU time on rebuilds.

### 3. Deterministic Vector IDs (Architectural Change)

**Problem:** Current design requires SQLite `lastrowid` for each chunk before writing
vectors. This forces row-by-row inserts and makes parallel ID assignment impossible.

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

**Schema change:** Add `chunks.vector_id INTEGER UNIQUE` column (or use as PRIMARY KEY).
Then `executemany()` all chunk inserts without per-row `lastrowid`.

**Impact:** This is the single biggest structural win for multi-node scaling. Producers
can generate IDs independently without coordination.

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

---

## Developer Guide: Improving Tar Processing Efficiency

LitKit is designed to process the full PMC-OA corpus (~5M articles) in under 10 hours
using multi-node GPU parallelism.

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

Compressed tars force sequential decompression, serializing the entire I/O path.
Uncompressed tars allow:
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

## Developer Guide: Lustre Filesystem Optimization

On Lustre filesystems, file striping dramatically affects I/O throughput. Under-striped
tar shards will bottleneck all downstream GPU processing.

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

**Usage in CLI startup:**
```python
if args.tar_dir:
    ok, msg = check_lustre_stripe(args.tar_dir)
    if not ok:
        _eprint(f"[warn] Tar directory has suboptimal Lustre striping: {msg}")
        _eprint(f"[warn] Consider: lfs setstripe -c -1 {args.tar_dir}")
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

## License

Proprietary — LANL
