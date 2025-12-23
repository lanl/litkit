# Litkit Multi-Node Build Performance Bottleneck Analysis

**Date:** 2025-12-23  
**Issue:** Tar scan rate collapsed from ~29/s to ~0.5/s during multi-node build  
**Cluster:** HPC (gpu-v100 partition, V100 nodes)

## Symptoms

During `medium_test.manifest` processing (4 tar files, ~7GB total):

1. **Initial scan rate:** ~29 files/second (healthy)
2. **After ~655 files:** Rate collapsed to ~4/s, then ~2.5/s, eventually ~0.5/s
3. **Both producers slowed simultaneously** despite being on different nodes
4. **Job timed out** at 10 hours with only 28% of the large tar processed
5. **Producer 1 finished** (smaller shard) while Producer 0 was stuck

## Log Evidence

```
# Healthy start:
[progress] [scan] oa_noncomm_xml.PMC012xxxxxx.baseline.2025-06-26.tar: 147/62348  (0.2%)  29.2/s

# Embedding starts, scan pauses:
[progress] Embedding chunks (producer, 20031): 0/20031
[consumer] Progress: 0/2 producers complete  ← 143 seconds of embedding...
[consumer] Progress: 0/2 producers complete
[consumer] Progress: 0/2 producers complete
[consumer] Progress: 0/2 producers complete
[done] Embedding chunks (producer, 20031): completed in 143s

# Scan resumes at much lower rate:
[progress] [scan] oa_noncomm_xml.PMC012xxxxxx.baseline.2025-06-26.tar: 700/11229  (6.2%)  4.0/s
```

The displayed rate (4.0/s, 0.5/s) is **cumulative average** over total elapsed time, not instantaneous. But instantaneous rate also degraded: near the end, ~4 files per 30-second consumer poll = 0.13 files/s.

## Root Causes (Priority Order)

### 1. CPU Starvation (CRITICAL)

**Problem:** The `srun` commands do NOT request CPUs for each step.

```bash
# Current (broken):
srun -N1 -n1 -w "$target_node" "$CHRUN" ...

# The job requests --cpus-per-task=32, but srun steps don't inherit this!
```

On many Slurm configurations, this means each producer effectively gets **1 CPU**. With `--parse-workers 16`, you have 16 threads fighting over 1 core = instant contention + overhead collapse.

**Fix:**
```bash
srun -N1 -n1 -w "$target_node" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    --export=ALL \
    "$CHRUN" ...
```

**Verification:** Inside container, check:
```bash
echo "nproc=$(nproc)"
grep Cpus_allowed_list /proc/self/status
```

### 2. Hidden Threading Oversubscription (HIGH)

Libraries like OpenMP, MKL, OpenBLAS spawn threads behind the scenes. With 16 parse workers already, these cause severe oversubscription.

**Fix:** Set before ch-run:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

Pass to container via `--set-env=OMP_NUM_THREADS=1` etc.

### 3. NFS Bottleneck (MEDIUM)

**Problem:** All I/O goes through NFS-mounted VAST storage:
- Tar file reads (sequential but large: 5.8GB)
- SQLite writes (frequent, small, IOPS-sensitive)
- Embedding segment writes (.npz.pending files)

NFS latency + lots of per-record writes + tar member reads is a punishing mix.

**Fix:** Stage hot I/O to node-local storage:
```bash
# Node-local tmpfs (128GB ramdisk at /tmp or /ram)
LOCAL_WS="/tmp/litkit_${SLURM_JOB_ID}_shard${shard_id}"
mkdir -p "$LOCAL_WS/tars" "$LOCAL_WS/sqlite" "$LOCAL_WS/emb_segments"

# Copy tar once up front
cp /nfs/path/to/tar "$LOCAL_WS/tars/"

# Run litkit with local workspace
LITKIT_WORKSPACE="$LOCAL_WS" litkit ...

# rsync results back to NFS at end
rsync -a "$LOCAL_WS/sqlite/" /path/to/workspace/sqlite/
rsync -a "$LOCAL_WS/emb_segments/" /path/to/workspace/emb_segments/
```

### 4. Scan Blocked During Embedding (EXPECTED)

The scan loop does NOT overlap with embedding—it waits for each embedding batch to complete. This is by design (simplifies memory management), but it means:

- 143 seconds of embedding = 143 seconds of zero scan progress
- Cumulative average rate tanks even if scan is healthy when running

**Not a bug.** But important to understand when interpreting rate numbers.

## Infrastructure Details

| Resource | HPC gpu-v100 |
|----------|----------------|
| CPU per node | 32 cores |
| RAM per node | 255 GB |
| GPU | 2x V100 |
| Ramdisk | 128 GB at `/ram` |
| Local disk | 3.5 TB (on some nodes) |
| Network FS | VAST via NFS (vers=3, 1MB blocks) |

## Fix Implementation Plan

### Phase 1: CPU Fixes (Immediate)

1. Add `--cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores --export=ALL` to every srun
2. Add threading clamps (OMP_NUM_THREADS=1, etc.)
3. Pass threading clamps to container via --set-env
4. Add diagnostic output (nproc, Cpus_allowed_list) at start

**Expected outcome:** If CPU starvation was the main issue, scan rate should stay at 20-30/s throughout.

### Phase 2: Local Staging (If Still Slow)

1. Stage tar files to /tmp before processing
2. Write SQLite and segments to /tmp
3. rsync results back to NFS at producer completion
4. Modify manifest to point to local paths

**Expected outcome:** Eliminates NFS from hot loop entirely.

### Phase 3: Pathological XML Detection (If Still Slow)

1. Add per-document timing instrumentation
2. Log top-N slowest files by parse time
3. Identify if specific XMLs dominate runtime
4. Add `--max-xml-bytes` or `--skip-large-docs` flags

## Testing Checklist

- [ ] Phase 1: CPU fixes applied to vector_build_multi.sbatch
- [ ] Phase 1: Test run shows nproc=32 inside container
- [ ] Phase 1: Scan rate stays above 15/s throughout
- [ ] Phase 2: If needed, implement /tmp staging
- [ ] Phase 2: Verify rsync copies results correctly
- [ ] Phase 3: If needed, add slow-doc logging

## Files Modified

- `vector_build_multi.sbatch` — CPU allocation + threading clamps
- `BOTTLENECK.md` — This document

## References

- Slurm CPU binding: https://slurm.schedmd.com/cpu_management.html
