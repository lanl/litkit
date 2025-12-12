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

## References

- HPC V100 nodes: 32 CPUs, 2 GPUs each
- Production cluster: potentially 64-128 CPUs per node
- PMC-OA corpus: ~126 GB compressed, ~550K papers per large tar file
