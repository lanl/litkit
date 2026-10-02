# LitKit

A scalable two-stage RAG (Retrieval-Augmented Generation) pipeline for scientific literature. Build searchable vector indices from JATS/NXML XML corpora and query with natural language questions.

## Key Features

- **Two-stage retrieval** — SPECTER2 for paper-level search, SBERT for passage-level search
- **Streaming tar ingestion** — process tar archives without extraction
- **Resumable builds** — checkpoint-based recovery from interruption
- **Multi-node support** — producer/consumer architecture for HPC clusters
- **Air-gapped operation** — works offline with cached HuggingFace models
- **Token-budgeted prompts** — automatic context trimming for LLM limits

## How It Works

LitKit uses a **two-stage retrieval** approach:

1. **Stage 1 (Papers):** SPECTER2 embeddings index titles and abstracts for fast paper-level similarity search
2. **Stage 2 (Chunks):** SBERT embeddings index body text chunks for passage-level retrieval
3. **LLM Synthesis:** Top chunks are passed to an LLM which generates an answer with citations

## Platform Guides

| Platform | Guide | Description |
|----------|-------|-------------|
| **HPC Cluster** | [LITKIT_CLUSTER_GUIDE.md](LITKIT_CLUSTER_GUIDE.md) | HPC deployment with Charliecloud, multi-node builds, GPU passthrough |
| **macOS** | [LITKIT_MAC_GUIDE.md](LITKIT_MAC_GUIDE.md) | Local development on MacBook (Apple Silicon or Intel) |

## Quick Start

### Prerequisites

- **Python 3.12** (exact version required)
- **uv** package manager

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone https://github.com/lanl/litkit.git
cd litkit
uv sync
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

## ⚠️ Important: Data Preparation

**LitKit does not deduplicate papers.** Each XML file in your tar archives is treated as a unique document.

- **Your responsibility:** Ensure tar files do not contain duplicate XML files
- **If duplicates exist:** The same paper will be indexed multiple times
- **Recommended:** Use uncompressed `.tar` files (not `.tar.gz`) for parallel processing

## CLI Reference

### Build Options

| Option | Description |
|--------|-------------|
| `--build-only` | Build indices without running a query |
| `--faiss-writer` | Enable FAISS index writes (required for builds) |
| `--rebuild` | Wipe and rebuild from scratch |
| `--update` | Append new files only |
| `--tar-dir DIR` | Directory containing tar shards |
| `--tar-manifest FILE` | File listing tar paths (one per line) |

### Query Options

| Option | Description |
|--------|-------------|
| `--llm-model MODEL` | LLM model name (default: `gpt-oss:20b`) |
| `--top-papers N` | Stage 1 shortlist size (default: 500) |
| `--top-chunks N` | Chunks for LLM context (default: 30) |
| `--no-llm` | Retrieval only, skip LLM call |
| `--per-paper-cap N` | Max chunks per paper in context (default: 3) |

### Performance Options

| Option | Description |
|--------|-------------|
| `--embed-devices SPEC` | Device selection: `auto`, `cpu`, `mps`, `cuda:0,cuda:1` |
| `--paper-embed-bs N` | Paper embedding batch size (default: 16) |
| `--chunk-embed-bs N` | Chunk embedding batch size (default: 64) |
| `--parse-workers N` | Parallel XML parsing threads (default: 8) |

### Other Options

| Option | Description |
|--------|-------------|
| `--quiet` | Suppress progress output |
| `-y, --yes` | Skip confirmation prompts |
| `--offline` | Force HuggingFace offline mode |
| `--version` | Print version and exit |

For the complete list, run `litkit --help`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LITKIT_WORKSPACE` | Output directory for indices and database |
| `LITKIT_TAR_DIR` | Default tar shard directory |
| `HF_HOME` | HuggingFace model cache location |
| `HF_HUB_OFFLINE=1` | Force offline mode |
| `LITKIT_QUIET=1` | Suppress output (like `--quiet`) |
| `LITKIT_DEBUG=1` | Enable verbose debugging |

## Documentation

- **Platform Guides:** See [LITKIT_CLUSTER_GUIDE.md](LITKIT_CLUSTER_GUIDE.md) and [LITKIT_MAC_GUIDE.md](LITKIT_MAC_GUIDE.md)
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md); release notes are in [CHANGELOG.md](CHANGELOG.md)

### Generating PDF Documentation

```bash
# Install prerequisites (macOS)
brew install pandoc
brew install --cask basictex

# Generate combined PDF
pandoc README.md LITKIT_MAC_GUIDE.md LITKIT_CLUSTER_GUIDE.md \
  --pdf-engine=xelatex \
  --toc --number-sections \
  -V geometry:margin=1in \
  -o litkit_documentation.pdf
```

## License

LitKit is released under the MIT License; see [LICENSE](LICENSE).

© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

LANL software release O5068.
