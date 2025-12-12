# LitKit: MacBook Pro Installation & Usage Guide

A step-by-step guide for installing and using LitKit on macOS to query scientific literature using the hosted LLM API.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Setting Up the AI Portal](#setting-up-the-ai-portal)
5. [Preparing Your Papers](#preparing-your-papers)
6. [Building the Index](#building-the-index)
7. [Querying Your Literature](#querying-your-literature)
8. [Command Reference](#command-reference)
9. [Troubleshooting](#troubleshooting)
10. [Tips & Best Practices](#tips--best-practices)

---

## Overview

**LitKit** is an air-gapped RAG (Retrieval-Augmented Generation) pipeline designed for scientific literature search. It enables you to:

- **Ingest** scientific papers from JATS/NXML format (commonly used by PubMed Central)
- **Embed** paper titles, abstracts, and body text using state-of-the-art models
- **Index** embeddings for fast similarity search using FAISS
- **Query** your literature collection with natural language questions
- **Answer** using an LLM with citations to the source material

### How It Works

LitKit uses a **two-stage retrieval** approach:

1. **Stage 1 (Papers)**: Your question is embedded using SPECTER2 (a scientific paper embedding model) and matched against paper-level embeddings to find the most relevant papers.

2. **Stage 2 (Chunks)**: Text chunks from the shortlisted papers are searched using SBERT embeddings to find the most relevant passages.

3. **LLM Synthesis**: The top chunks are passed to an LLM (via the hosted LLM API) which synthesizes an answer with citations.

---

## System Requirements

### Hardware
- **macOS**: Sonoma, Ventura, or Monterey (Apple Silicon or Intel)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: ~5 GB for models + index storage (varies with corpus size)
- **Network**: Access to `llm.example.com` for LLM queries

### Software
- **Python 3.12** (exact version required)
- **pip** or **uv** (Python package installer)

### Check Your Python Version
```bash
python3 --version
# Should output: Python 3.12.x
```

If you don't have Python 3.12, install it via Homebrew:
```bash
brew install python@3.12
```

---

## Installation

You can install LitKit from either a ZIP archive or a Git clone.

### Option A: Install from ZIP Archive (Recommended)

If you received LitKit as a ZIP file:

```bash
# 1. Extract the archive
unzip litkit-v0.3.33.zip
cd litkit

# 2. Create a virtual environment with Python 3.12
python3.12 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install LitKit and dependencies
pip install -e .
```

### Option B: Install from Git (If You Have Access)

```bash
# 1. Clone the repository
git clone https://github.com/lanl/litkit.git
cd litkit

# 2. Create a virtual environment
python3.12 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install
pip install -e .
```

### Verify Installation

```bash
litkit --version
# Should output: litkit 0.3.33
```

---

## Setting Up the AI Portal

LitKit uses the hosted LLM API to generate answers from retrieved context. You need an API key.

### Step 1: Save Your API Key

Create a file containing your AI Portal API key:

```bash
# Create the key file (replace YOUR_API_KEY with your actual key)
echo "YOUR_API_KEY_HERE" > ~/.llm_api_key

# Secure the file permissions
chmod 600 ~/.llm_api_key
```

### Step 2: Test the Connection (Optional)

You can test your API key with curl:

```bash
curl -H "Authorization: Bearer $(cat ~/.llm_api_key)" \
     https://llm.example.com/v1/models
```

If successful, you'll see a list of available models.

---

## Preparing Your Papers

LitKit ingests papers from `.tar.gz` archives containing JATS/NXML XML files. If your papers were converted from PDFs using the `text-fetch` utility, they should already be in the correct format.

### Expected Structure

Your tar archive should contain XML files at the top level or in subdirectories:

```
papers.tar.gz
├── paper1.xml
├── paper2.xml
├── paper3.xml
└── ...
```

### Step 1: Create the Workspace Directory

```bash
# From the litkit directory
mkdir -p workspace/tar_shards
```

### Step 2: Copy Your Papers Archive

```bash
cp /path/to/your/papers.tar.gz workspace/tar_shards/
```

### Step 3: Create a Manifest File

The manifest tells LitKit which archives to process:

```bash
echo "papers.tar.gz" > workspace/papers.manifest
```

Or if you have multiple archives:

```bash
cat > workspace/papers.manifest << 'EOF'
papers_batch1.tar.gz
papers_batch2.tar.gz
papers_batch3.tar.gz
EOF
```

---

## Building the Index

Before you can query your papers, LitKit needs to build a searchable index.

### First-Time Model Download

The first time you run LitKit, it will download the required embedding models from HuggingFace (~2 GB total):

- **SPECTER2**: For paper-level embeddings
- **all-mpnet-base-v2**: For chunk-level embeddings

This download happens automatically and is cached in `workspace/hf_cache/`.

### Build Command (Small Corpus)

For a small collection of papers (< 10,000), use FLAT indices for simplicity:

```bash
cd litkit
source .venv/bin/activate

litkit --build-only \
       --faiss-writer \
       --tar-dir workspace/tar_shards \
       --papers-index flat \
       --chunks-index flat
```

Or using a manifest file:

```bash
litkit --build-only \
       --faiss-writer \
       --tar-manifest workspace/papers.manifest \
       --papers-index flat \
       --chunks-index flat
```

### Build Command (Larger Corpus)

For larger collections (> 10,000 papers), use HNSW for papers and IVF-PQ for chunks:

```bash
litkit --build-only \
       --faiss-writer \
       --tar-manifest workspace/papers.manifest \
       --papers-index hnsw \
       --chunks-index ivfpq
```

### What Happens During Build

1. **Scanning**: LitKit streams through your tar archives without extracting
2. **Parsing**: XML files are parsed to extract title, abstract, and body paragraphs
3. **Chunking**: Body text is split into ~1200-character chunks with overlap
4. **Embedding**: Text is embedded using the downloaded models
5. **Indexing**: Embeddings are added to FAISS indices
6. **Saving**: SQLite database and FAISS indices are written to `workspace/`

### Expected Output

```
[version] litkit 0.3.33
[device] using mps              # or cpu on Intel Macs
[paths] using workspace/tar_shards as source directory for tar shards
[paths] using /path/to/litkit/workspace as writable directory
[build] using DB at workspace/sqlite/litkit.sqlite3
[scan] found 1 tar shards in shard 0/1
[progress] [scan] papers.tar.gz: 50/50  (100.0%)  12.3/s
[done] indexed 50 papers and 847 chunks (this run)
```

### Resume After Interruption

If the build is interrupted, simply re-run the same command. LitKit maintains checkpoints and will resume from where it left off.

---

## Querying Your Literature

Once the index is built, you can ask questions about your papers.

### Basic Query

```bash
litkit "What is the main mechanism described in these papers?" \
       --llm-model gtp-oss-120b \
       --openai-base-url "https://llm.example.com/v1" \
       --openai-api-key "$(cat ~/.llm_api_key)"
```

### Query with More Context

Retrieve more chunks for a more comprehensive answer:

```bash
litkit "What experimental methods were used?" \
       --llm-model gtp-oss-120b \
       --openai-base-url "https://llm.example.com/v1" \
       --openai-api-key "$(cat ~/.llm_api_key)" \
       --top-papers 100 \
       --top-chunks 30
```

### Query from a File

For longer questions, save them to a file:

```bash
echo "What are the key findings regarding the relationship between X and Y?" > question.txt

litkit --question-file question.txt \
       --llm-model gtp-oss-120b \
       --openai-base-url "https://llm.example.com/v1" \
       --openai-api-key "$(cat ~/.llm_api_key)"
```

### Retrieval Only (No LLM)

To see what chunks would be selected without calling the LLM:

```bash
litkit "What is HIV?" --no-llm
```

This prints the retrieved context, useful for debugging or when you want to manually review sources.

### Expected Output Format

```
Based on the provided context, the main mechanism involves... [1, 2].

The experimental results demonstrate that... [3].

REFERENCES
==========
[1] Smith et al. "Title of Paper One" (PMID:12345678)
[2] Jones et al. "Title of Paper Two" (PMCID:PMC9876543)
[3] Wang et al. "Title of Paper Three" (PMID:11111111)
```

---

## Command Reference

### Build Options

| Flag | Description | Default |
|------|-------------|---------|
| `--build-only` | Build indices without running a query | - |
| `--faiss-writer` | Enable index modifications (required for builds) | - |
| `--tar-dir PATH` | Directory containing tar archives | workspace/tar_shards |
| `--tar-manifest FILE` | Text file listing tar archives (one per line) | - |
| `--papers-index {hnsw,flat}` | Index type for papers | hnsw |
| `--chunks-index {ivfpq,flat}` | Index type for chunks | ivfpq |
| `--rebuild` | Wipe existing index and rebuild from scratch | - |
| `--update` | Add new files without reprocessing existing ones | - |

### Query Options

| Flag | Description | Default |
|------|-------------|---------|
| `--llm-model MODEL` | LLM model name | gpt-oss:20b |
| `--openai-base-url URL` | API endpoint URL | localhost:1234 |
| `--openai-api-key KEY` | API authentication key | - |
| `--top-papers N` | Number of candidate papers (Stage 1) | 500 |
| `--top-chunks N` | Number of chunks for LLM context (Stage 2) | 30 |
| `--no-llm` | Retrieval only; skip LLM call | - |
| `--question-file FILE` | Read question from file (use `-` for stdin) | - |

### Performance Options

| Flag | Description | Default |
|------|-------------|---------|
| `--embed-devices DEVS` | Embedding devices (auto, cpu, mps, cuda:0) | auto |
| `--paper-embed-bs N` | Batch size for paper embeddings | 16 |
| `--chunk-embed-bs N` | Batch size for chunk embeddings | 64 |
| `--per-paper-cap N` | Max chunks per paper in final context | 3 |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LITKIT_WORKSPACE` | Override workspace directory |
| `HF_HOME` | HuggingFace cache location |
| `HF_HUB_OFFLINE=1` | Force offline mode (no model downloads) |

---

## Troubleshooting

### "No module named 'litkit'"

Make sure you activated the virtual environment:
```bash
source .venv/bin/activate
```

### "Model not found" or Download Errors

LitKit downloads models on first use. Ensure you have internet access. If behind a proxy:
```bash
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
```

### "401 Unauthorized" from AI Portal

Check that your API key is correct:
```bash
cat ~/.llm_api_key
```

Verify the key works:
```bash
curl -H "Authorization: Bearer $(cat ~/.llm_api_key)" \
     https://llm.example.com/v1/models
```

### "No tar shards found"

Ensure your tar file is in the correct location and has a valid extension:
```bash
ls workspace/tar_shards/
# Should show: papers.tar.gz
```

### Slow Embedding Performance

On Apple Silicon Macs, LitKit should automatically use MPS (Metal Performance Shaders). Verify:
```bash
litkit --version
# Look for: [device] using mps
```

If it shows `cpu`, check your PyTorch installation:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should output: True
```

### Index Corruption

If you suspect index corruption, rebuild from scratch:
```bash
litkit --rebuild --faiss-writer \
       --tar-manifest workspace/papers.manifest \
       --papers-index flat \
       --chunks-index flat -y
```

The `-y` flag skips the confirmation prompt.

---

## Tips & Best Practices

### 1. Start Small
For testing, create a manifest with just one or two papers first:
```bash
# Single paper test
litkit --build-only --faiss-writer \
       --tar-dir workspace/tar_shards \
       --papers-index flat --chunks-index flat
```

### 2. Use FLAT Indices for Small Corpora
For fewer than ~10,000 papers, FLAT indices are simpler and often faster than HNSW/IVF-PQ:
```bash
--papers-index flat --chunks-index flat
```

### 3. Save Common Options in a Script

Create a helper script for repeated use:
```bash
cat > query.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
litkit "$1" \
    --llm-model gtp-oss-120b \
    --openai-base-url "https://llm.example.com/v1" \
    --openai-api-key "$(cat ~/.llm_api_key)"
EOF
chmod +x query.sh

# Usage:
./query.sh "What is the main finding?"
```

### 4. Backup Your Index
The built index is stored in `workspace/`. Back it up to avoid rebuilding:
```bash
tar czf litkit-workspace-backup.tar.gz workspace/sqlite workspace/indices
```

### 5. Check Retrieved Context First
Before relying on LLM answers, use `--no-llm` to verify the right papers/chunks are being retrieved.

---

## Quick Start Summary

```bash
# 1. Extract and install
unzip litkit-v0.3.33.zip && cd litkit
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Setup API key
echo "YOUR_KEY" > ~/.llm_api_key && chmod 600 ~/.llm_api_key

# 3. Add your papers
mkdir -p workspace/tar_shards
cp /path/to/papers.tar.gz workspace/tar_shards/

# 4. Build index
litkit --build-only --faiss-writer \
       --tar-dir workspace/tar_shards \
       --papers-index flat --chunks-index flat

# 5. Query
litkit "What is the main finding?" \
       --llm-model gtp-oss-120b \
       --openai-base-url "https://llm.example.com/v1" \
       --openai-api-key "$(cat ~/.llm_api_key)"
```

---

## Getting Help

- **Version**: `litkit --version`
- **Full help**: `litkit --help`
- **Author**: William S. Hlavacek (hlavacek@lanl.gov)

---

*LitKit v0.3.33 — Air-gapped RAG for Scientific Literature*
