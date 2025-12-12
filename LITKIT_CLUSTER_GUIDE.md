# LitKit: HPC Installation & Usage Guide

A step-by-step guide for running LitKit on an HPC cluster using Charliecloud containers.

## Table of Contents

1. [Overview](#overview)
2. [HPC Hardware](#hpc-hardware)
3. [One-Time Setup](#one-time-setup)
4. [Building the Container](#building-the-container)
5. [Running Single-Node Builds](#running-single-node-builds)
6. [Running Multi-Node Builds](#running-multi-node-builds)
7. [Querying Your Literature](#querying-your-literature)
8. [Monitoring GPU Usage](#monitoring-gpu-usage)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Command Reference](#command-reference)

---

## Overview

**LitKit** runs on HPC inside a Charliecloud container with GPU passthrough via NVIDIA CDI (Container Device Interface). This guide covers:

- **V100 nodes** (`gpu-v100` partition): 4 nodes, 2 GPUs each, 32 CPUs per node
- **GH200 nodes** (`gpu-gh200` partition): 2 nodes (Grace Hopper), 72 ARM cores + H100 GPU

Both node types are ARM64 (aarch64) and share the same container image.

---

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

---

## One-Time Setup

These steps only need to be done once per cluster.

### Step 1: Generate CDI Specs for GPU Passthrough

**For V100 nodes** (run on any V100 node):
```bash
# Get an interactive session
salloc -N1 -t 1:00:00 -p gpu-v100 --no-shell
ssh gpu-node1  # or whichever node you got

# Generate CDI spec
mkdir -p /path/to/cdi-v100
nvidia-ctk cdi generate \
  --format=json \
  --output=/path/to/cdi-v100/nvidia.json

# Verify
nvidia-ctk cdi list --spec-dir=/path/to/cdi-v100

# Exit and release allocation
exit
scancel <jobid>
```

**For GH200 nodes** (run on any Grace node):
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

### Step 2: Extract CUDA Toolkit from NVIDIA Image

This provides a stable, module-free CUDA installation for container bindings:

```bash
cd /path/to
module purge
module load charliecloud/0.42

# Pull CUDA 12.5 image (multi-arch, will get ARM64)
ch-image pull nvidia/cuda:12.5.0-devel-ubuntu22.04

# Extract to host
DEST=/path/to/cuda-12.5-host
mkdir -p "$DEST"
ch-run nvidia/cuda:12.5.0-devel-ubuntu22.04 -- bash -lc \
  'tar -C /usr/local -cf - cuda-12.5' | tar -C "$DEST" -xvf -
```

### Step 3: Create Persistent HuggingFace Cache

```bash
mkdir -p /path/to/hf_cache_persist
```

### Step 4: Convert SPECTER2 to SafeTensors (Required)

From the litkit repo, run once:
```bash
cd /path/to/litkit
./setup_safetensors.sh
```

---

## Building the Container

### Using the Justfile (Recommended)

```bash
cd /path/to/litkit
module purge
module load charliecloud/0.42

# Build container (Dockerfile.lean by default)
just build

# Output: sqfs/litkit-v0.3.33-aarch64-lean.sqfs
```

### Manual Build

```bash
ch-image build -f Dockerfile.lean -t litkit-lean .
ch-convert litkit-lean sqfs/litkit-v0.3.33-aarch64-lean.sqfs
```

---

## Running Single-Node Builds

### Environment Setup (V100)

```bash
cd /path/to/litkit
module purge
module load charliecloud/0.42

export CDI_SPEC_DIR=/path/to/cdi-v100
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
export IMG="$(pwd)/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export HF_HOST=/path/to/hf_cache_persist
```

### Environment Setup (GH200)

```bash
cd /path/to/litkit
module purge
module load charliecloud/0.42

export CDI_SPEC_DIR=/path/to/cdi-grace
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
export IMG="$(pwd)/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export HF_HOST=/path/to/hf_cache_persist
```

### Verify Setup

```bash
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/bin:/usr/bin:/bin" \
  "$IMG" -- \
  litkit --version
```

### Build Vector Store (Small Test)

```bash
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "/path/to/test_tar_shards:/path/to/test_tar_shards" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "$HF_HOST:/app/hf_cache" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/bin:/usr/bin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="HF_HOME=/app/hf_cache" \
  --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
  "$IMG" -- \
  litkit --faiss-writer --build-only --rebuild --yes \
         --tar-manifest /workspace/test.manifest
```

### Using SLURM Batch Scripts

**V100 single-node:**
```bash
sbatch -p gpu-v100 vector_build_single.sbatch
```

**GH200 single-node:**
```bash
sbatch -p gpu-gh200 --export=ALL,TARGET=gh vector_build_single.sbatch
```

---

## Running Multi-Node Builds

For large corpora (126+ GB), use the producer/consumer architecture.

### Architecture

```
[Producer 0] ──┐
[Producer 1] ──┼── segments/ ──► [Consumer/Writer] ──► FAISS indices
[Producer N] ──┘
```

- **Producers**: Scan tar files, embed chunks, write segment files
- **Consumer**: Ingests segments into FAISS, manages SQLite

### Submit Multi-Node Job

```bash
# 4 nodes: 3 producers + 1 consumer
sbatch -p gpu-v100 vector_build_large.sbatch

# Or adjust node count
sed -i 's/--nodes=4/--nodes=3/' vector_build_large.sbatch
sbatch -p gpu-v100 vector_build_large.sbatch
```

### Create a Manifest File

For PMC-OA corpus:
```bash
cat > workspace/pmcoa.manifest << 'EOF'
/path/to/PMC-OA/oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz
/path/to/PMC-OA/oa_comm_xml.PMC008xxxxxx.baseline.2025-06-26.tar.gz
/path/to/PMC-OA/oa_comm_xml.PMC009xxxxxx.baseline.2025-06-26.tar.gz
EOF
```

---

## Querying Your Literature

### Retrieval Only (No LLM)

```bash
cat workspace/question.txt
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "$HF_HOST:/app/hf_cache" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/bin:/usr/bin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="HF_HOME=/app/hf_cache" \
  "$IMG" -- \
  litkit --no-llm --question-file /workspace/question.txt
```

### With hosted LLM API

```bash
KEY="$(head -n1 ~/.llm_api_key)"
BASE="https://llm.example.com/v1"

ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "$HF_HOST:/app/hf_cache" \
  --bind "/etc/pki/tls/certs/ca-bundle.crt:/workspace/site-ca.pem" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="SSL_CERT_FILE=/workspace/site-ca.pem" \
  --set-env="REQUESTS_CA_BUNDLE=/workspace/site-ca.pem" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/bin:/usr/bin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="HF_HOME=/app/hf_cache" \
  --set-env="OPENAI_BASE_URL=$BASE" \
  --set-env="OPENAI_API_KEY=$KEY" \
  "$IMG" -- \
  litkit --llm-model gpt-oss-120b \
         --openai-base-url "$BASE" \
         --openai-api-key "$KEY" \
         --question-file /workspace/question.txt
```

---

## Monitoring GPU Usage

### Check GPU Utilization Across Nodes

```bash
JOB=$(squeue --me -h -o %i | head -1)
for n in $(scontrol show hostnames $(squeue -j $JOB -h -o %N)); do
  srun --jobid=$JOB --overlap -N1 -n1 -w $n -c1 --cpu-bind=none \
    bash -lc 'echo === $(hostname) ===; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; echo'
done
```

### Expected Output (Multi-GPU Working)

```
=== gpu-node1 ===
0, Tesla V100-PCIE-32GB, 85 %, 3500 MiB, 32768 MiB
1, Tesla V100-PCIE-32GB, 78 %, 2700 MiB, 32768 MiB

=== gpu-node3 ===
0, Tesla V100-PCIE-32GB, 82 %, 3500 MiB, 32768 MiB
1, Tesla V100-PCIE-32GB, 90 %, 2700 MiB, 32768 MiB
```

### Continuous Monitoring

```bash
watch -n 5 "for n in \$(scontrol show hostnames \$(squeue -j $JOB -h -o %N) | head -3); do
  srun --jobid=\$JOB --overlap -N1 -n1 -w \$n -c1 --cpu-bind=none \
    bash -lc \"echo '=== '\\\$(hostname)' ==='; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader\" 2>/dev/null
done"
```

---

## Performance Tuning

### Parallel XML Parsing

For **uncompressed tar files** (`.tar` but not `.tar.gz`), LitKit can parse XML files in parallel using multiple CPU cores. This significantly improves ingestion throughput on multi-core systems.

```bash
# Use 16 parallel XML parsing workers (for uncompressed .tar files)
litkit --faiss-writer --build-only --rebuild --yes \
       --parse-workers 16 \
       --tar-manifest /workspace/pmcoa_uncompressed.manifest
```

| Option | Default | Description |
|--------|---------|-------------|
| `--parse-workers` | 8 | Number of parallel XML parsing workers |

**Notes:**
- Parallel parsing **only works with uncompressed `.tar` files**
- Compressed `.tar.gz` files always use sequential parsing (decompression is inherently serial)
- Higher values help on nodes with many CPU cores (e.g., GH200's 72 ARM cores)
- On V100 nodes (32 CPUs), `--parse-workers 8` is usually sufficient

### Converting .tar.gz to .tar for Faster Ingestion

To benefit from parallel parsing, you can decompress your tar archives:

```bash
# Decompress a single archive
gunzip -k /path/to/PMC-OA/oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz

# Or create an uncompressed copy
zcat file.tar.gz > file.tar
```

**Trade-off**: Uncompressed files are ~3-5x larger but can be parsed in parallel.

---

## Troubleshooting

### "pam_slurm_adopt" error when SSH to node

You don't have an active allocation on that node:
```bash
# Check your allocations
squeue --me

# Request a new allocation
salloc -N1 -t 1:00:00 -p gpu-v100 --no-shell
```

### "CUDA not available" inside container

1. Verify CDI spec exists:
   ```bash
   ls /path/to/cdi-v100/nvidia.json
   ```

2. Check CDI flags are present in ch-run:
   ```bash
   --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all
   ```

3. Verify CUDA lib bindings:
   ```bash
   ls "$CUDA_LIBA"/libcuda*
   ls "$CUDA_LIBB"/libcudart*
   ```

### "database is locked" errors

These are **transient and benign** on Lustre. Segment files remain on disk and are retried automatically. To reduce frequency:
```bash
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000  # 5 minutes
```

### Low GPU utilization (bursty pattern)

This pattern occurs when using compressed `.tar.gz` files, which require single-threaded decompression.

**Solutions:**
1. **Decompress archives** to `.tar` format and use `--parse-workers 16` (see [Performance Tuning](#performance-tuning))
2. **Use multiple producer nodes** to parallelize across tar files (see [Running Multi-Node Builds](#running-multi-node-builds))

### Container not found

```bash
# Rebuild the container
cd /path/to/litkit
just build

# Verify
ls -la sqfs/litkit-v0.3.33-aarch64-lean.sqfs
```

---

## Command Reference

### SLURM Submission

| Target | Command |
|--------|---------|
| V100 single-node | `sbatch -p gpu-v100 vector_build_single.sbatch` |
| V100 multi-node | `sbatch -p gpu-v100 vector_build_multi.sbatch` |
| GH200 single-node | `sbatch -p gpu-gh200 --export=ALL,TARGET=gh vector_build_single.sbatch` |

### Key Paths

| Resource | Path |
|----------|------|
| LitKit repo | `/path/to/litkit` |
| Container | `sqfs/litkit-v0.3.33-aarch64-lean.sqfs` |
| V100 CDI spec | `/path/to/cdi-v100` |
| GH200 CDI spec | `/path/to/cdi-grace` |
| CUDA toolkit | `/path/to/cuda-12.5-host/cuda-12.5` |
| HF cache | `/path/to/hf_cache_persist` |
| PMC-OA corpus | `/path/to/PMC-OA` |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `LITKIT_WORKSPACE` | Output directory for indices/DB |
| `HF_HOME` | HuggingFace model cache |
| `LITKIT_SQLITE_BUSY_TIMEOUT_MS` | SQLite lock timeout (default: 120000) |
| `LITKIT_SAVE_EVERY_SEC` | FAISS save interval (default: 120) |

---

## Quick Start Summary

```bash
# 1. SSH to HPC
ssh login-node

# 2. Clone/update repo
cd /path/to/litkit
git pull

# 3. Rebuild container (if needed)
module load charliecloud/0.42
just build

# 4. Submit job
sbatch -p gpu-v100 vector_build_single.sbatch

# 5. Monitor
squeue --me
tail -f litkit-build.*.err
```

---

## Getting Help

- **Version**: `litkit --version`
- **Full help**: `litkit --help`
- **Author**: William S. Hlavacek (hlavacek@lanl.gov)

---

*LitKit v0.3.33 — Air-gapped RAG for Scientific Literature on HPC*
