# Parallel Tar Ingestion - Implementation Plan

## Problem Identified (2025-12-11)

During large-scale testing on HPC, we discovered a significant I/O bottleneck in the tar scanning pipeline.

### Observed Behavior

```
[scan] oa_comm_xml.PMC008xxxxxx.baseline.2025-06-26.tar.gz: 12500/550499  (2.3%)  2.0/s
[scan] oa_comm_xml.PMC009xxxxxx.baseline.2025-06-26.tar.gz: 20765/604299  (3.4%)  3.3/s
```

- **Scan rate**: 2-3 papers/second (single-threaded)
- **Papers per tar file**: ~550,000
- **Time per tar file**: ~61 hours (unacceptable)
- **GPUs**: Only active in short bursts (~85 sec) with long gaps (~2+ hours)

### Root Cause

The current architecture is single-threaded:
```
[tar.gz on Lustre] → [Single thread: gzip decompress + tar extract + XML parse + DB write] → [GPU batch]
```

Each producer uses **1 CPU** for the entire pipeline, leaving 31-127 CPUs idle.

---

## Solution: Decompressed .tar with Parallel Reading

### Phase 1: Pre-decompress tar.gz files (One-time)

```bash
# On HPC, decompress all tar.gz files in parallel
cd /path/to/PMC-OA

# Single file example:
pigz -dk oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz
# Creates: oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar

# Parallel decompression of all files:
ls *.tar.gz | parallel -j4 'pigz -dk {}'

# Disk space required: ~126 GB (same as compressed, roughly 1:1 for XML)
```

### Phase 2: Code Changes

Modify `src/litkit/ingest/ingest.py` to support:

1. **Uncompressed .tar files**: Direct random-access reading
2. **Parallel tar member processing**: Multiple threads can read different offsets simultaneously
3. **Thread pool for XML parsing**: Parse XML in parallel using available CPUs

### Architecture After Changes

```
[.tar file (random access)]
       ↓
[Thread pool: N threads read tar members at different offsets]
       ↓
[Thread pool: M threads parse XML in parallel]
       ↓
[Main thread: DB write + chunk accumulation]
       ↓
[GPU batch embedding]
```

### Expected Speedup

| Workers | Papers/sec | Time per 550K tar |
|---------|------------|-------------------|
| 1 (current) | 2.5 | ~61 hours |
| 8 | ~15-20 | ~8-10 hours |
| 16 | ~25-35 | ~4-6 hours |
| 32 | ~40-60 | ~2.5-4 hours |
| 64 | ~60-90 | ~1.5-2.5 hours |

---

## Implementation Steps (Next Session)

### Step 1: Diagnostic (verify bottleneck)

```bash
# Check if gzip decompression is the bottleneck
time zcat /path/to/PMC-OA/oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz > /dev/null

# If fast (< 10 min): XML parsing is the bottleneck → parallelize parsing
# If slow (> 30 min): gzip is the bottleneck → must pre-decompress
```

### Step 2: Modify iter_tar_xml_streams()

In `src/litkit/ingest/ingest.py`, add support for:
- `.tar` files (uncompressed) with random access
- Thread-safe member iteration
- Parallel XML parsing with ThreadPoolExecutor

### Step 3: Add CLI option

```
--tar-workers N    Number of parallel tar reading threads (default: 8)
```

### Step 4: Test on small subset

```bash
# Decompress one tar file for testing
pigz -dk oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz

# Test with new parallel reader
litkit --build-only --tar-dir /path/to/decompressed --tar-workers 16
```

---

## Git Branch Management

### Current Branch Status

The `fix/multi-node-producer-consumer` branch contains:
- ✅ EmbeddingPool race condition fix
- ✅ Numpy array truth value fix
- ✅ Multi-node producer/consumer coordination
- ✅ Large-scale test scripts

These fixes are complete and should be merged.

### Wrap Up Current Branch (run on laptop)

```bash
cd /path/to/litkit

# Ensure all changes are committed
git status
git add -A
git commit -m "Final documentation for multi-node fixes"

# Push to remote
git push origin fix/multi-node-producer-consumer

# Merge to main (or create PR)
git checkout main
git pull origin main
git merge fix/multi-node-producer-consumer
git push origin main

# Delete the feature branch (optional)
git branch -d fix/multi-node-producer-consumer
git push origin --delete fix/multi-node-producer-consumer
```

### Create New Branch for Parallel Ingestion

```bash
# Start fresh from main
git checkout main
git pull origin main

# Create new feature branch
git checkout -b feature/parallel-tar-ingestion

# Push branch to remote
git push -u origin feature/parallel-tar-ingestion
```

### On HPC (after laptop push)

```bash
cd /path/to/litkit

# Switch to main and update
git checkout main
git pull origin main

# Rebuild container with merged fixes
just build-lean

# Later, when parallel ingestion branch is ready:
git fetch origin
git checkout feature/parallel-tar-ingestion
```

---

## Files to Modify

1. `src/litkit/ingest/ingest.py` - Add parallel tar reading
2. `src/litkit/cli.py` - Add `--tar-workers` CLI option
3. New file: `src/litkit/ingest/parallel_tar.py` (optional, for cleaner separation)

---

---

## Why Was This Bottleneck Not Noticed Earlier?

### 1. The Design Was Reasonable for Small-Scale Testing

The original code was developed on:
- Local SSDs (fast I/O)
- Small tar files (few thousand papers)
- Focus on correctness, not throughput

At 50 papers/sec (reasonable for SSD + lxml), a 10,000-paper tar file takes ~3 minutes. Nobody notices.

### 2. The Corpus Scale Changed, The Architecture Didn't

PMC-OA baseline files have **550,000+ papers** per tar. That's 50-100x larger than typical test files. The sequential tar.gz reader that was "good enough" for development became the bottleneck at scale.

### 3. HPC Lustre Is Slow for Sequential Reads

Lustre is optimized for **large parallel I/O**, not sequential streaming of compressed archives. The 2-5 papers/sec observed is probably:
- 50% gzip decompression (CPU-bound)
- 50% lxml parsing (CPU-bound)
- I/O itself might be fine, but we're CPU-bound on a single core

### 4. Focus Was on GPU, Not Pre-GPU Pipeline

Effort was spent on:
- Multi-GPU embedding pool ✓
- Multi-node producer/consumer ✓
- FAISS index building ✓

But the **data preparation pipeline** (tar→XML→chunks) was treated as "simple" and left single-threaded.

### 5. Lesson Learned

When scaling to HPC:
1. Profile the **entire pipeline**, not just the GPU
2. Parallel filesystems (Lustre/GPFS) have different characteristics than local SSDs
3. CPU-bound preprocessing (parsing, decompression) can dominate wall-clock time

---

## Other Potential Bottlenecks (After Tar Fix)

### 1. SQLite Writes (Producers) - Risk: Medium

Currently inserting papers/chunks row-by-row. At 30-60 papers/sec, that's ~500-1000 INSERT/sec. SQLite on Lustre might struggle.

**Mitigation**: Batch INSERTs with `executemany` (e.g., 1000 rows per call).

### 2. Consumer Segment Ingestion - Risk: Medium-High

Consumer currently does:
```
for each segment:
    FAISS add_with_ids()
    SQLite UPDATE in_index=1
    faiss.write_index() [sometimes]
```

At high segment throughput, this could become the bottleneck.

**Mitigation**: Batch multiple segments before writing indices.

### 3. FAISS Index Scaling - Risk: Low

Using `--chunks-index flat` for simplicity. FLAT has O(n) search but **O(1) add**. Building scales linearly.

---

## Database Lock Errors (Observed 2025-12-11)

During large-scale testing, we observed transient SQLite lock errors:
```
[segments] ERROR ingesting chunks_sh00_1765493063_000001.npz: OperationalError: database is locked
```

**These are benign:**
- Only 2 errors over 2 hours of operation
- Segment files remain on disk and are retried on next consumer poll
- Expected behavior on Lustre with concurrent access
- The 300-second busy_timeout handles most contention

---

## Test Run Results (2025-12-11, Job 16802995)

### Configuration
- 3 nodes: 2 producers (gpu-node1, gpu-node3), 1 consumer (gpu-node4)
- Manifest: 3 tar files (~30 GB total)
- Runtime: ~2 hours observed, job still running

### Observations

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

## References

- HPC V100 nodes: 32 CPUs, 2 GPUs each
- Production cluster: potentially 64-128 CPUs per node
- PMC-OA corpus: ~126 GB compressed, ~550K papers per large tar file
