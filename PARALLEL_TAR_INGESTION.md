# Parallel Tar Ingestion - Status & Testing Guide

## Quick Start: Resume Testing on HPC

When resuming work on this feature, run these tests:

### Step 1: Prepare an Uncompressed Test File

```bash
# SSH to HPC and get a node
ssh login-node
salloc -N1 -t 2:00:00 -p gpu-v100 --no-shell
ssh gpu-node1  # or whichever node you got

# Decompress one tar.gz file for testing
cd /path/to/PMC-OA
gunzip -k oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz
# Creates: oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar (~42 GB)

# Create a manifest for the uncompressed file
echo "/path/to/PMC-OA/oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar" > /path/to/litkit/workspace/uncompressed.manifest
```

### Step 2: Run the Parallel Test

```bash
cd /path/to/litkit
module purge
module load charliecloud/0.42

# Set up environment (V100)
export CDI_SPEC_DIR=/path/to/cdi-v100
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
export IMG="$(pwd)/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export HF_HOST=/path/to/hf_cache_persist

# Run with parallel parsing (16 workers)
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "/path/to/PMC-OA:/path/to/PMC-OA" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "$HF_HOST:/app/hf_cache" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/bin:/usr/bin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="HF_HOME=/app/hf_cache" \
  --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
  "$IMG" -- \
  litkit --faiss-writer --build-only --rebuild --yes \
         --parse-workers 16 \
         --tar-manifest /workspace/uncompressed.manifest 2>&1 | tee parallel_test.log
```

### Step 3: Verify Success

**Look for these indicators:**

1. **Parallel mode confirmation:**
   ```
   [scan] using parallel XML parsing (16 workers) for oa_comm_xml.tar
   ```

2. **Improved scan rate (target: >10 papers/sec):**
   ```
   [progress] [scan] oa_comm_xml.tar: 50000/550000  (9.1%)  25.3/s
   ```
   - Old sequential rate: 2-5 papers/sec
   - New parallel rate: 15-50+ papers/sec (depending on CPU count)

3. **More sustained GPU utilization:**
   ```bash
   # In another terminal
   watch -n 2 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
   ```
   - Old: Bursty (0%...0%...85%...0%...0%)
   - New: More sustained (60%...75%...80%...65%)

---

## Implementation Status: ✅ COMPLETE (2025-12-11)

The parallel tar ingestion feature has been implemented and is ready for testing.

### Code Changes

| File | Change |
|------|--------|
| `src/litkit/ingest/ingest.py` | Added `TarMemberMeta`, `parallel_iter_tar_articles()` |
| `src/litkit/cli.py` | Added `--parse-workers` option, `iter_tar_articles()` helper |
| `LITKIT_HPC_GUIDE.md` | Added "Performance Tuning" section |

### New CLI Option

```
--parse-workers N    Number of parallel XML parsing workers (default: 8)
```

### How It Works

1. **For uncompressed `.tar` files:**
   - Pre-scans the tar to get member offsets
   - Workers read their assigned members via random-access (seeks to file offsets)
   - XML parsing happens in parallel using ProcessPoolExecutor
   - Results are yielded in order to the main thread

2. **For compressed `.tar.gz` files:**
   - Falls back to sequential parsing (decompression is inherently serial)
   - Parallel parsing is not possible without pre-decompression

### Key Functions

```python
# In src/litkit/ingest/ingest.py
parallel_iter_tar_articles(tar_path, workers=8)
  """Parallel iterator for uncompressed .tar files."""

# In src/litkit/cli.py  
iter_tar_articles(tar_path, parse_workers=8)
  """Unified iterator - uses parallel for .tar, sequential for .tar.gz."""
```

---

## Problem Background

### Original Bottleneck (2025-12-11)

During large-scale testing on HPC, we discovered the tar scanning pipeline was single-threaded:

```
[scan] oa_comm_xml.PMC008xxxxxx.baseline.2025-06-26.tar.gz: 12500/550499  (2.3%)  2.0/s
```

- **Scan rate**: 2-5 papers/second (single-threaded)
- **Papers per tar file**: ~550,000
- **Time per tar file**: ~61 hours (unacceptable)
- **GPUs**: Only active in short bursts (~85 sec) with long gaps (~2+ hours)

### Root Cause

The sequential architecture left CPUs idle:
```
[tar.gz on Lustre] → [Single thread: gzip decompress + tar extract + XML parse] → [GPU batch]
```

Each producer used **1 CPU** for the entire pipeline, leaving 31-127 CPUs idle on HPC nodes.

---

## Technical Design

### Architecture After Changes

```
[.tar file (random access)]
       ↓
[Worker pool: N processes read tar members at different offsets]
       ↓
[Worker pool: Each process parses XML locally]
       ↓
[Main thread: receives (metadata, article) tuples in order]
       ↓
[DB write + chunk accumulation]
       ↓
[GPU batch embedding]
```

### Expected Speedup

| Workers | Papers/sec | Time per 550K tar |
|---------|------------|-------------------|
| 1 (sequential) | 2.5 | ~61 hours |
| 8 | ~15-20 | ~8-10 hours |
| 16 | ~25-35 | ~4-6 hours |
| 32 | ~40-60 | ~2.5-4 hours |

### Important Limitation

**Parallel parsing only works with uncompressed `.tar` files.**

Compressed `.tar.gz` files require sequential decompression (gzip is inherently serial). To benefit from parallel parsing:

```bash
# Decompress ahead of time
gunzip -k archive.tar.gz   # Creates archive.tar (keeps original)
# OR
zcat archive.tar.gz > archive.tar
```

**Trade-off**: Uncompressed files are ~3-5x larger but can be parsed in parallel.

---

## Testing Comparison (TODO)

### Baseline Test (Sequential, Compressed)

```bash
# Create manifest for compressed file
echo "/path/to/PMC-OA/oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz" > /workspace/compressed.manifest

# Run with parse-workers=1 (forces sequential)
time litkit --faiss-writer --build-only --rebuild --yes \
       --parse-workers 1 \
       --tar-manifest /workspace/compressed.manifest 2>&1 | tee baseline.log
```

### Parallel Test (Parallel, Uncompressed)

```bash
# Run with parse-workers=16
time litkit --faiss-writer --build-only --rebuild --yes \
       --parse-workers 16 \
       --tar-manifest /workspace/uncompressed.manifest 2>&1 | tee parallel.log
```

### Metrics to Compare

| Metric | Baseline (sequential) | Target (parallel) |
|--------|----------------------|-------------------|
| Scan rate | 2-5 papers/sec | >15 papers/sec |
| GPU utilization pattern | Bursty (0-90-0-90%) | Sustained (60-90%) |
| Time per 550K tar | ~61 hours | <8 hours |

---

## Historical Test Run (2025-12-11, Job 16802995)

### Configuration
- 3 nodes: 2 producers (gpu-node1, gpu-node3), 1 consumer (gpu-node4)
- Manifest: 3 tar files (~30 GB total)
- Runtime: ~2 hours observed

### Observations (Before Parallel Fix)

**Multi-node parallelism confirmed:**
```
[scan] oa_comm_xml.PMC008xxxxxx: 3001/550499  (0.5%)  2.3/s  ← Producer 1 (gpu-node1)
[scan] oa_comm_xml.PMC009xxxxxx: 5221/604299  (0.9%)  4.5/s  ← Producer 0 (gpu-node3)
```

**GPU bursts working correctly:**
```
[progress] Embedding chunks (producer, 20022): completed in 89s — 222.8/s
```

**Bottleneck confirmed:**
- Scan rate: 2-5 papers/sec (single-threaded)
- GPU burst: ~90 seconds
- Inter-burst gap: ~7-10 minutes (waiting for 20k chunks to accumulate)

---

## Other Potential Bottlenecks (After Tar Fix)

### 1. SQLite Writes (Producers) - Risk: Medium

Currently inserting papers/chunks row-by-row. At 30-60 papers/sec, that's ~500-1000 INSERT/sec.

**Mitigation if needed**: Batch INSERTs with `executemany`.

### 2. Consumer Segment Ingestion - Risk: Medium-High

Consumer does: FAISS add → SQLite UPDATE → faiss.write_index() [sometimes]

**Mitigation if needed**: Batch multiple segments before writing indices.

### 3. FAISS Index Scaling - Risk: Low

Using FLAT or IVF-PQ. Both have O(1) add operations.

---

## Files Changed

| File | Purpose |
|------|---------|
| `src/litkit/ingest/ingest.py` | Added `TarMemberMeta`, `parallel_iter_tar_articles()` |
| `src/litkit/cli.py` | Added `--parse-workers`, `iter_tar_articles()`, `Iterator` import |
| `LITKIT_HPC_GUIDE.md` | Added "Performance Tuning" section |
| `PARALLEL_TAR_INGESTION.md` | This file - updated with implementation status |

---

## References

- HPC V100 nodes: 32 CPUs, 2 GPUs each
- HPC GH200 nodes: 72 ARM cores, 1 H100 GPU
- PMC-OA corpus: ~126 GB compressed, ~550K papers per large tar file

---

*Last updated: 2025-12-11 — Implementation complete, testing pending*
