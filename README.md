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

## License

Proprietary — LANL
