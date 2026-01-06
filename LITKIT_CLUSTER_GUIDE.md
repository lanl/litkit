# LitKit: HPC Cluster Guide

A guide for running LitKit on HPC clusters using Charliecloud containers with GPU passthrough.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Building the Container](#building-the-container)
4. [Caching HuggingFace Models](#caching-huggingface-models)
5. [Understanding Multi-Node Builds](#understanding-multi-node-builds)
6. [Why Uncompressed Tar Files?](#why-uncompressed-tar-files)
7. [Customizing SLURM Batch Scripts](#customizing-slurm-batch-scripts)
8. [Running Single-Node Builds](#running-single-node-builds)
9. [Running Multi-Node Builds](#running-multi-node-builds)
10. [Querying Your Index](#querying-your-index)
11. [Troubleshooting](#troubleshooting)
12. [Command Reference](#command-reference)
13. [Appendix A: HPC Cluster Reference (Site-Specific)](#appendix-a-hpc-cluster-reference-site-specific)

---

## Overview

**LitKit** processes scientific literature corpora (JATS/NXML XML files in tar archives) into searchable vector indices for RAG (Retrieval-Augmented Generation) pipelines.

On HPC clusters, LitKit runs inside a Charliecloud container with GPU passthrough via NVIDIA CDI (Container Device Interface). This guide covers:

- **Single-node builds**: For small-to-medium corpora (< 50 GB)
- **Multi-node builds**: For large corpora (50+ GB) using producer/consumer architecture
- **Querying**: Natural language questions against your indexed literature

### Architecture

```
[Tar Archives] → [LitKit Build] → [FAISS Indices + SQLite DB] → [Query]
                      │
         ┌───────────┴───────────┐
    Single-node              Multi-node
    (one process)       (producers + consumer)
```

---

## Prerequisites

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× CUDA-capable | 2+ GPUs per node |
| GPU Memory | 16 GB | 32+ GB |
| System Memory | 64 GB | 128+ GB |
| Local SSD | 100 GB | 500+ GB (for staging) |

### Software Requirements

- **Charliecloud** 0.37+ with CDI support
- **CUDA Toolkit** 12.x (for GPU library bindings)
- **NVIDIA CDI specs** generated for your GPU nodes
- **uv** package manager (for lockfile regeneration)

### CUDA Setup Options

You need CUDA libraries available on the host to bind into the container. Two approaches:

**Option A: Use site CUDA module** (recommended on managed clusters)
```bash
module load cuda/12.5.0
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"

# Pick ARM runtime lib directory (SBSA for Grace/Hopper, aarch64 for older)
if [ -d "$CUDA_HOME/targets/sbsa-linux/lib" ]; then
  export CUDA_LIBA="$CUDA_HOME/targets/sbsa-linux/lib"
elif [ -d "$CUDA_HOME/targets/aarch64-linux/lib" ]; then
  export CUDA_LIBA="$CUDA_HOME/targets/aarch64-linux/lib"
fi
export CUDA_LIBB="$CUDA_HOME/lib64"
```

**Option B: Extract from NVIDIA container** (for clusters without CUDA modules)
```bash
module load charliecloud
ch-image pull nvidia/cuda:12.5.0-devel-ubuntu22.04

DEST=/path/to/cuda-12.5-host
mkdir -p "$DEST"
ch-run nvidia/cuda:12.5.0-devel-ubuntu22.04 -- bash -lc \
  'tar -C /usr/local -cf - cuda-12.5' | tar -C "$DEST" -xvf -

export CUDA_BASE="$DEST/cuda-12.5"
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
```

### Container Image

LitKit is distributed as a SquashFS container image:
```
litkit-v0.3.35-aarch64-lean.sqfs    # ARM64 (aarch64)
litkit-v0.3.35-x86_64-lean.sqfs     # x86-64 (if available)
```

---

## Building the Container

LitKit containers are built using the repo's `justfile` and either `Dockerfile.lean` or `Dockerfile.fat`.

### Dockerfile Flavors

| Flavor | File | CUDA in Image | Portability | Use Case |
|--------|------|---------------|-------------|----------|
| **lean** | `Dockerfile.lean` | No | High | Recommended for most clusters |
| **fat** | `Dockerfile.fat` | Yes | Low | Single-cluster deployments |

**lean** (recommended): No CUDA userspace baked in. At runtime, you bind host CUDA libraries via CDI. More portable—works across clusters with different driver versions.

**fat**: CUDA userspace injected at build time. Simpler to run, but baked libs must be ≤ site driver version.

### Building with the Justfile

```bash
# Ensure Charliecloud is loaded
module load charliecloud

# Build lean container (default)
just build

# Output: sqfs/litkit-v0.3.35-<arch>-lean.sqfs
```

### Build Commands

| Command | Description |
|---------|-------------|
| `just build` | Build container and export to sqfs |
| `just release` | Build + save wheel + export (full release) |
| `just reset` | Clear Charliecloud build cache |

### Regenerating the Lockfile

If you modify `pyproject.toml` (add/remove dependencies), regenerate `uv.lock`:

```bash
# Install uv if needed
pip install uv

# Regenerate lockfile
uv lock --python 3.12
```

The lockfile ensures reproducible builds.

---

## Caching HuggingFace Models

LitKit uses two embedding models from HuggingFace:
- **SPECTER2** (`allenai/specter2_base`) for paper-level embeddings
- **SBERT** (`sentence-transformers/all-mpnet-base-v2`) for chunk-level embeddings

### Why Cache Models?

Air-gapped HPC clusters cannot download models at runtime. You must:
1. Pre-download models to a persistent cache directory
2. Bind the cache into the container
3. Set `HF_HUB_OFFLINE=1` to prevent download attempts

### Creating a Persistent Cache

```bash
# Create cache directory (shared filesystem, persistent across jobs)
mkdir -p /path/to/hf_cache
export HF_HOST=/path/to/hf_cache
```

### Converting to SafeTensors

PyTorch's default model format (`.bin`) uses pickle, which has security concerns and slower loading. **SafeTensors** is preferred:

- No arbitrary code execution (safer)
- Memory-mapped loading (faster)
- Required by LitKit when `TRANSFORMERS_USE_SAFE_TENSORS=1`

The `setup_safetensors.sh` script handles conversion:

```bash
# Set required variables
export IMG=/path/to/litkit.sqfs
export HF_HOST=/path/to/hf_cache
export CUDA_BASE=/path/to/cuda-12.x
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"

# Optional: CDI spec dir (if you have GPU access)
export CDI_SPEC_DIR=/path/to/cdi-specs

# Run conversion (idempotent - safe to run multiple times)
./setup_safetensors.sh
```

The script:
1. Seeds the host cache from the container image
2. Converts `.bin` → `.safetensors` for SPECTER2 and SBERT
3. Removes `.bin` artifacts after successful conversion

### Environment Variables for Offline Mode

```bash
--set-env="HF_HOME=/app/hf_cache"
--set-env="HF_HUB_OFFLINE=1"
--set-env="TRANSFORMERS_OFFLINE=1"
--set-env="TRANSFORMERS_USE_SAFE_TENSORS=1"
```

---

## Understanding Multi-Node Builds

### When to Use Multi-Node

| Corpus Size | Recommended Mode |
|-------------|------------------|
| < 10 GB | Single-node |
| 10-50 GB | Single-node (or 2-node) |
| 50-200 GB | 3-4 nodes |
| 200+ GB | 4+ nodes |

### Producer/Consumer Architecture

For large corpora, LitKit uses a producer/consumer model:

```
[Producer 0] ──┐
[Producer 1] ──┼── segments/ ──► [Consumer/Writer] ──► FAISS indices
[Producer N] ──┘
```

- **Producers** (`--embed-producer`): Scan tar files, embed text, write segment files
- **Consumer** (`--consume-only --faiss-writer`): Ingest segments into FAISS indices

> ⚠️ **CRITICAL: Node Count Must Be Consistent**
> 
> Once you start a multi-node build with N producer nodes, you **MUST** restart 
> with exactly N producer nodes. Changing the node count mid-build will corrupt 
> your vector store.
>
> **Why?** Each producer writes to shard-specific files:
> - SQLite: `litkit_shard_00.sqlite3`, `litkit_shard_01.sqlite3`, ...
> - Segments: `paper_seg_shard00_*.npz`, `chunk_seg_shard01_*.npz`, ...
> - Checkpoints: `ckpt_shard_0.json`, `ckpt_shard_1.json`, ...
>
> The consumer expects exactly N shards. Adding or removing nodes breaks this mapping.

### Tar Staging to Local SSD

The multi-node batch script (`vector_build_multi.sbatch`) implements **local staging** to avoid NFS bottlenecks:

**Problem**: Writing many small files to NFS/Lustre causes "metadata storms" that slow down all cluster users.

**Solution**: Producers write to node-local SSD, then rsync to NFS at completion.

```
Producer workflow:
1. Write segments to /local/ssd/litkit_jobid_shard0/
2. Write SQLite to /local/ssd/litkit_jobid_shard0/
3. On success: rsync everything to NFS
4. Cleanup local staging directory
```

**Benefits**:
- 47× faster for small-file creates (vs direct NFS writes)
- No NFS lock contention between producers
- Failed producers leave local artifacts for debugging

---

## Why Uncompressed Tar Files?

**LitKit strongly recommends using uncompressed `.tar` files instead of `.tar.gz`.**

### The Problem with Compressed Tars

```
.tar.gz decompression:  [gz stream] → [single thread] → [tar entries]
                                           │
                              Cannot parallelize!
```

Gzip decompression is inherently **serial**—you cannot seek to random offsets in a compressed stream. This means:
- Only one CPU can decompress at a time
- XML parsing must wait for decompression
- GPUs sit idle during I/O

### Uncompressed Tar Enables Parallelism

```
.tar file:  [tar entries] → [parallel workers] → [parsed XML]
                 │                  │
            Random access    lxml releases GIL
```

With uncompressed tars:
- Multiple workers can seek to different tar members simultaneously
- lxml releases the Python GIL during parsing
- The `--parse-workers` flag controls parallelism (default: 8)

### Converting Your Archives

```bash
# Decompress (keeps original)
gunzip -k corpus.tar.gz

# Or create uncompressed copy
zcat corpus.tar.gz > corpus.tar
```

**Trade-off**: Uncompressed files are 3-5× larger, but parsing is 5-10× faster on multi-core systems.

---

## Customizing SLURM Batch Scripts

The batch scripts (`vector_build_single.sbatch`, `vector_build_multi.sbatch`) require customization for your cluster. Key variables to modify:

### Essential Variables

```bash
# CUSTOMIZE FOR YOUR CLUSTER:

# Where your LitKit installation lives
export LITKIT_REPO="/path/to/litkit"

# Shared filesystem workspace (NFS/Lustre)
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"

# Node-local fast storage (SSD, NVMe, or tmpfs)
# Used for staging to avoid NFS bottlenecks
export LOCAL_SSD="/local/scratch"

# CDI spec directory for GPU passthrough
export CDI_SPEC_DIR="/path/to/cdi-specs"

# CUDA toolkit libraries
export CUDA_BASE="/path/to/cuda-12.x"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"   # ARM64
export CUDA_LIBB="${CUDA_BASE}/lib64"

# HuggingFace model cache (persistent across jobs)
export HF_HOST="/path/to/hf_cache"

# Container image path
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.35-aarch64-lean.sqfs"
```

### Manifest Files

Create a manifest file listing tar archive paths (one per line):

```bash
cat > workspace/corpus.manifest << 'EOF'
/data/corpus/part_001.tar
/data/corpus/part_002.tar
/data/corpus/part_003.tar
EOF
```

Set in batch script:
```bash
export LITKIT_TAR_MANIFEST="${LITKIT_WORKSPACE}/corpus.manifest"
```

### SLURM Resource Requests

Adjust for your cluster's hardware:

```bash
#SBATCH --partition=gpu           # Your GPU partition
#SBATCH --nodes=3                 # Total nodes (N-1 producers + 1 consumer)
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --cpus-per-task=32        # CPUs for XML parsing
#SBATCH --mem=128G                # Memory per node
#SBATCH --gres=gpu:2              # GPUs per node
#SBATCH --time=10:00:00           # Wall time
```

---

## Running Single-Node Builds

### When to Use

- Corpus size < 50 GB
- Testing and development
- Quick validation runs

### Basic Command

```bash
litkit --faiss-writer --build-only --rebuild --yes \
       --tar-manifest /workspace/corpus.manifest
```

### Using the Batch Script

```bash
# Edit vector_build_single.sbatch to set your paths
sbatch -p YOUR_GPU_PARTITION vector_build_single.sbatch
```

### Key Options

| Option | Description |
|--------|-------------|
| `--faiss-writer` | Enable FAISS index writes (required) |
| `--build-only` | Build without running a query |
| `--rebuild` | Wipe and rebuild from scratch |
| `--update` | Append new files only |
| `--tar-manifest FILE` | Path to manifest file |
| `--tar-dir DIR` | Directory containing tar files (alternative to manifest) |

---

## Running Multi-Node Builds

### When to Use

- Corpus size > 50 GB
- Need faster builds via parallelism
- Multiple GPUs across multiple nodes

### Architecture

With N nodes:
- Nodes 0 to N-2: **Producers** (embed text, write segments)
- Node N-1: **Consumer** (ingest segments, write FAISS indices)

### Using the Batch Script

```bash
# Edit vector_build_multi.sbatch to set your paths
# Adjust --nodes=N as needed

sbatch -p YOUR_GPU_PARTITION vector_build_multi.sbatch
```

### Bootstrap Phase

The script automatically creates empty FAISS indices before producers start:

```bash
litkit --faiss-writer --init-indices-only \
       --papers-index hnsw --chunks-index flat
```

### Monitoring Progress

```bash
# Watch job output
tail -f litkit_multi_*.out

# Check GPU utilization across nodes
JOB=$(squeue --me -h -o %i | head -1)
for n in $(scontrol show hostnames $(squeue -j $JOB -h -o %N)); do
  srun --jobid=$JOB --overlap -N1 -n1 -w $n \
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
done
```

---

## Querying Your Index

### Using a Question File

For long or complex questions, use `--question-file`:

```bash
# Create question file
cat > workspace/question.txt << 'EOF'
What molecular mechanisms have been proposed to explain
the interaction between protein X and protein Y in the
context of cellular signaling pathways?
EOF

# Query (retrieval + LLM)
litkit --question-file /workspace/question.txt

# Query (retrieval only, no LLM)
litkit --no-llm --question-file /workspace/question.txt
```

### Interactive Queries

```bash
litkit "What is the role of autophagy in cancer?"
```

### With External LLM API

```bash
litkit --llm-model gpt-4 \
       --openai-base-url "https://api.example.com/v1" \
       --openai-api-key "$API_KEY" \
       --question-file /workspace/question.txt
```

### Key Query Options

| Option | Description |
|--------|-------------|
| `--question-file FILE` | Read question from file |
| `--no-llm` | Retrieval only (print context, skip LLM) |
| `--top-papers N` | Stage 1 shortlist size (default: 500) |
| `--top-chunks N` | Chunks for LLM context (default: 30) |
| `--per-paper-cap N` | Max chunks per paper (default: 3) |
| `--llm-model MODEL` | LLM model name |

---

## Troubleshooting

### Verifying GPU Setup

Before running builds, verify PyTorch can see GPUs:

```bash
ch-run YOUR_IMAGE.sqfs \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  -- /root/.local/share/uv/tools/litkit/bin/python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

**Expected output**:
```
Torch: 2.5.1 CUDA build: 12.5
CUDA available: True
Device: Tesla V100-PCIE-32GB
```

### Monitoring SLURM Jobs

```bash
# Check job status
squeue --me

# Get job ID for completed job
jid=$(sacct -X -n --name=litkit-build --starttime=now-7days -o JobID | tail -1)

# View logs
less "litkit-build.$jid.out"
less "litkit-build.$jid.err"

# Get timing and status
sacct -X -j "$jid" -o JobID,JobName%30,Partition,State,ExitCode,Elapsed
```

### "CUDA not available" inside container

1. Verify CDI spec exists:
   ```bash
   ls $CDI_SPEC_DIR/nvidia.json
   ```

2. Check CDI flags in ch-run:
   ```bash
   --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all
   ```

3. Verify CUDA library bindings:
   ```bash
   ls "$CUDA_LIBA"/libcuda*
   ls "$CUDA_LIBB"/libcudart*
   ```

### "database is locked" errors

These are transient on NFS/Lustre. Increase timeout:
```bash
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000  # 5 minutes
```

### Low GPU utilization (bursty pattern)

Usually caused by compressed `.tar.gz` files blocking parallel parsing.

**Solutions**:
1. Convert to uncompressed `.tar` files
2. Increase `--parse-workers` (e.g., 16 or 32)
3. Use more producer nodes

### Producer failed, consumer waiting forever

Check producer logs for errors. If a producer crashes:
1. Fix the underlying issue
2. Restart the **entire** job (same node count!)
3. Checkpoints allow resumption from last successful point

### Stale writer guard file

If a previous job crashed without cleanup:
```bash
rm workspace/.writer_guard
```

The batch script does this automatically at job start.

### "pam_slurm_adopt" error when SSH to node

You don't have an active allocation on that node:
```bash
# Check your allocations
squeue --me

# Request a new allocation
salloc -N1 -t 1:00:00 -p YOUR_PARTITION --no-shell
```

---

## Command Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LITKIT_WORKSPACE` | Output directory for indices/DB |
| `LITKIT_TAR_DIR` | Default tar directory |
| `LITKIT_TAR_MANIFEST` | Default manifest file |
| `HF_HOME` | HuggingFace model cache |
| `HF_HUB_OFFLINE=1` | Force offline mode (no downloads) |
| `TRANSFORMERS_USE_SAFE_TENSORS=1` | Use SafeTensors format |
| `LITKIT_SQLITE_BUSY_TIMEOUT_MS` | SQLite lock timeout (default: 120000) |
| `LITKIT_SAVE_EVERY_SEC` | FAISS save interval (default: 120) |
| `LITKIT_DEBUG=1` | Enable verbose debugging |
| `LITKIT_QUIET=1` | Suppress output |

### Key CLI Flags

```bash
# Build modes
--faiss-writer          # Enable FAISS writes (required for builds)
--embed-producer        # Producer mode (write segments, not FAISS)
--consume-only          # Consumer mode (ingest segments only)
--init-indices-only     # Create empty indices and exit

# Index types
--papers-index {hnsw,flat}   # Paper index type (default: hnsw)
--chunks-index {ivfpq,flat}  # Chunk index type (default: ivfpq)

# Multi-node
--shard-id N            # This producer's shard ID (0-indexed)
--num-shards N          # Total number of producer shards

# Performance
--parse-workers N       # Parallel XML parsing threads (default: 8)
--embed-devices SPEC    # GPU devices (auto, cpu, cuda:0,cuda:1)
--paper-embed-bs N      # Paper embedding batch size (default: 16)
--chunk-embed-bs N      # Chunk embedding batch size (default: 64)
```

### Utility Scripts

| Script | Description |
|--------|-------------|
| `setup_safetensors.sh` | Convert HF models to SafeTensors format |
| `vector_build_single.sbatch` | SLURM script for single-node builds |
| `vector_build_multi.sbatch` | SLURM script for multi-node builds |
| `vector_resume_consumer.sbatch` | Resume consumer after producer completion |

---

# Appendix A: HPC Cluster Reference (Site-Specific)

> **Note:** This appendix contains site-specific details.
> For public releases, this section can be removed entirely.

## HPC Hardware

### V100 Nodes (gpu-v100 partition)

| Hostname | GPUs | Memory |
|----------|------|--------|
| gpu-node1 | 2× Tesla V100-PCIE-32GB | 32 GB per GPU |
| gpu-node2 | 2× Tesla V100-PCIE-32GB | 32 GB per GPU |
| gpu-node3 | 2× Tesla V100-PCIE-32GB | 32 GB per GPU |
| gpu-node4 | 2× Tesla V100-PCIE-32GB | 32 GB per GPU |

**Submit jobs with:** `sbatch -p gpu-v100`

### GH200 Nodes (gpu-gh200 partition)

| Hostname | GPU | Memory |
|----------|-----|--------|
| gh-node1 | NVIDIA GH200 (Grace Hopper) | 96 GB unified |
| gh-node2 | NVIDIA GH200 (Grace Hopper) | 96 GB unified |

**Submit jobs with:** `sbatch -p gpu-gh200`

## HPC-Specific Setup

### One-Time CDI Generation

**For V100 nodes:**
```bash
salloc -N1 -t 1:00:00 -p gpu-v100 --no-shell
ssh gpu-node1

mkdir -p /path/to/cdi-v100
nvidia-ctk cdi generate \
  --format=json \
  --output=/path/to/cdi-v100/nvidia.json

nvidia-ctk cdi list --spec-dir=/path/to/cdi-v100

exit
scancel <jobid>
```

**For GH200 nodes:**
```bash
salloc -N1 -t 1:00:00 -p gpu-gh200 --no-shell
ssh gh-node1

mkdir -p /path/to/cdi-grace
nvidia-ctk cdi generate \
  --format=json \
  --output=/path/to/cdi-grace/nvidia.json

nvidia-ctk cdi list --spec-dir=/path/to/cdi-grace

exit
scancel <jobid>
```

### CUDA Toolkit Extraction

```bash
cd /path/to
module purge
module load charliecloud/0.42

ch-image pull nvidia/cuda:12.5.0-devel-ubuntu22.04

DEST=/path/to/cuda-12.5-host
mkdir -p "$DEST"
ch-run nvidia/cuda:12.5.0-devel-ubuntu22.04 -- bash -lc \
  'tar -C /usr/local -cf - cuda-12.5' | tar -C "$DEST" -xvf -
```

### SafeTensors Conversion (HPC)

```bash
cd /path/to/litkit

export IMG=$(pwd)/sqfs/litkit-v0.3.35-aarch64-lean.sqfs
export HF_HOST=/path/to/hf_cache_persist
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"

# For V100 nodes:
export CDI_SPEC_DIR=/path/to/cdi-v100

# For GH200 nodes:
# export CDI_SPEC_DIR=/path/to/cdi-grace

./setup_safetensors.sh
```

### HuggingFace Cache

```bash
mkdir -p /path/to/hf_cache_persist
```

## HPC Paths Quick Reference

| Resource | Path |
|----------|------|
| LitKit repo | `/path/to/litkit` |
| Container | `sqfs/litkit-v0.3.35-aarch64-lean.sqfs` |
| V100 CDI spec | `/path/to/cdi-v100` |
| GH200 CDI spec | `/path/to/cdi-grace` |
| CUDA toolkit | `/path/to/cuda-12.5-host/cuda-12.5` |
| HF cache | `/path/to/hf_cache_persist` |
| PMC-OA corpus | `/path/to/PMC-OA` |
| Local SSD | `/local/scratch` |

## HPC SLURM Commands

```bash
# V100 single-node
sbatch -p gpu-v100 vector_build_single.sbatch

# V100 multi-node (3 nodes)
sbatch -p gpu-v100 vector_build_multi.sbatch

# GH200 single-node
sbatch -p gpu-gh200 --export=ALL,TARGET=gh vector_build_single.sbatch

# Check job status
squeue --me

# Get job elapsed time and status
jid=$(sacct -X -n --name=litkit-build --starttime=now-7days -o JobID | tail -1)
sacct -X -j "$jid" -o JobID,JobName%30,Partition,State,ExitCode,Elapsed
```

## HPC: Verifying GPU Setup

**V100 (gpu-v100):**
```bash
cd /path/to/litkit
module purge
module load charliecloud/0.42

export CDI_SPEC_DIR=/path/to/cdi-v100
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
export IMG="$(pwd)/sqfs/litkit-v0.3.35-aarch64-lean.sqfs"

ch-run "$IMG" \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  -- /root/.local/share/uv/tools/litkit/bin/python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## hosted LLM API Integration

```bash
KEY="$(head -n1 ~/.llm_api_key)"
BASE="https://llm.example.com/v1"

litkit --llm-model gpt-oss-120b \
       --openai-base-url "$BASE" \
       --openai-api-key "$KEY" \
       --question-file /workspace/question.txt
```

**Certificate binding for container:**
```bash
--bind "/etc/pki/tls/certs/ca-bundle.crt:/workspace/site-ca.pem"
--set-env="SSL_CERT_FILE=/workspace/site-ca.pem"
--set-env="REQUESTS_CA_BUNDLE=/workspace/site-ca.pem"
```

---

*LitKit v0.3.35 — Air-gapped RAG for Scientific Literature*
