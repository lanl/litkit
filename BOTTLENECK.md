# Litkit Multi-Node Build Performance Optimization

**Date:** 2025-12-23 (Updated)  
**Cluster:** HPC (gpu-v100 partition, V100 nodes for testing)  
**Production Target:** H100 nodes with 4 GPUs each  

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: CPU Allocation | ✅ COMPLETE | `affinity_cpus=32` confirmed inside container |
| Phase 1: GPU Access | ✅ COMPLETE | 2× V100 per producer confirmed via nvidia-smi |
| Phase 2: Instrumentation | ✅ COMPLETE | Instantaneous rate + per-cycle timing added |
| Phase 3: Output Staging | 🔲 NEXT | Stage SQLite + segments to local SSD |
| Phase 4: Per-Tar Streaming | 🔲 TODO | Stage one tar at a time to local SSD |
| Phase 5: Scale to Production | 🔲 TODO | 16+ producers on H100 nodes |

**Commits:**
- `c959c19` - Instantaneous scan rate display
- `a0f40b1` - Per-cycle timing instrumentation

---

## Storage Performance Evidence (gpu-node1)

### Measured Benchmarks

| Operation | NFS | Local SSD | Ratio | Interpretation |
|-----------|-----|-----------|-------|----------------|
| Sequential read | 124 MB/s | 350 MB/s | 2.8× | Tar streaming |
| Sequential write | - | 309 MB/s | - | Segment output |
| 20k file creates | ~52s | ~1.1s | **47×** | **Metadata storm** |
| Stage 64GB tar | - | ~5m38s | one-time | Amortized over hours |

**The 47× metadata slowdown is the smoking gun.** SQLite journals, segment files, and checkpoint markers create exactly this pattern. This explains the "mysterious decay" observed in sustained runs.

### Storage Map (HPC gpu-v100)

```
# Node-local SSD (RECOMMENDED for staging):
/local/scratch    ~447G xfs     Safe for large tar + workspace staging

# Node-local tmpfs (CAUTION: competes with RAM):
/ram                   ~128G tmpfs   Risky for large files; use only for small temp

# Shared NFS (AVOID for high-churn writes):
/nfs/home             NFS           Corpus input (/path/to/PMC-OA)
/nfs/projects         NFS           Current workspace (slow for metadata)
```

### Why Local Staging is Critical

1. **SQLite operations** - WAL/journal writes, checkpoints → many small fsyncs
2. **Segment files** - `.npz.pending` → `.npz` renames per batch
3. **Marker files** - Producer completion markers, writer guards

Each of these operations hits NFS metadata server. With 16 producers, this becomes a storm.

---

## Instrumentation Output (Phase 2 Complete)

The new instrumentation shows:

**Scan progress with instantaneous rate:**
```
[scan] file.tar: 500/10000  (5.0%)  28.3/s now, 3.5/s avg
```

**Per-cycle timing after each batch flush:**
```
[cycle] docs=600 chunks=20016 | scan=45.2s embed=143.1s seg=0.8s db=0.2s | chunks/doc=33.4
```

**What to watch for:**
- `scan=` should be ~25-45s per 20K chunk cycle (healthy if so)
- `embed=` should be ~140-150s on V100, ~35-40s on H100
- `seg=` and `db=` should be <5s (if higher, NFS is the bottleneck)
- `chunks/doc=` should be stable ~30-35 (growing = chunk explosion)

---

## Phase 3: Output Staging (NEXT)

### Problem

Every segment write and SQLite commit currently goes to NFS. With measured 47× metadata slowdown, this dominates walltime at scale.

### Solution

Stage all high-churn outputs to node-local SSD, rsync back at end.

### Implementation (vector_build_multi.sbatch)

```bash
# --- LOCAL STAGING SETUP ---
LOCAL_SSD="/local/scratch"
LOCAL_STAGE="${LOCAL_SSD}/litkit_${SLURM_JOB_ID}_shard${shard_id}"
mkdir -p "${LOCAL_STAGE}/sqlite" "${LOCAL_STAGE}/emb_segments"

# Producer runs with local workspace:
export LITKIT_WORKSPACE="${LOCAL_STAGE}"
litkit --embed-producer \
    --tar-manifest "$NFS_TAR_MANIFEST" \
    --embed-outdir "${LOCAL_STAGE}/emb_segments" \
    ...

# After producer completes (before marking done):
rsync -a --inplace "${LOCAL_STAGE}/sqlite/" "${NFS_WORKSPACE}/sqlite/"
rsync -a --inplace "${LOCAL_STAGE}/emb_segments/" "${NFS_WORKSPACE}/emb_segments/"
rm -rf "${LOCAL_STAGE}"  # cleanup
```

### Storage Budget

| Item | Size per Producer | Notes |
|------|-------------------|-------|
| SQLite shard DB | 10-50 MB | Grows with corpus |
| Embedding segments | 100-500 MB | ~20K chunks × 3KB each |
| Temp files | <10 MB | Checkpoints, markers |
| **Total** | **< 1 GB** | Easily fits on 447G SSD |

---

## Phase 4: Per-Tar Streaming

### Problem

With 16 producers reading tars from NFS simultaneously, aggregate bandwidth demand may exceed NFS capacity. Measured NFS read: 124 MB/s vs local 350 MB/s.

### Solution

Stage one tar at a time to local SSD, process, delete, repeat.

### Implementation Concept

```bash
# For each tar in this producer's shard:
for tar_path in $(shard_tars); do
    # Stage tar to local SSD (one-time cost ~5-6 min for 64GB)
    local_tar="${LOCAL_SSD}/current_shard.tar"
    cp "$tar_path" "$local_tar"
    
    # Create single-tar manifest
    echo "$local_tar" > "${LOCAL_STAGE}/single.manifest"
    
    # Process from local
    litkit --embed-producer --tar-manifest "${LOCAL_STAGE}/single.manifest" ...
    
    # Delete before staging next
    rm "$local_tar"
done
```

### Storage Budget

| Item | Size | Notes |
|------|------|-------|
| Largest tar | ~64 GB | Uncompressed PMC-OA shard |
| Local SSD | 447 GB | Plenty of headroom |
| RAM tmpfs | 128 GB | **DO NOT USE** for tars (OOM risk) |

---

## Multi-Node Workflow

### Two-Phase Pattern (Recommended)

**Phase 1: Producers (parallel, all local)**
```
Producer 0 on gpu-node1: reads NFS tar → writes local SSD → rsync to NFS at end
Producer 1 on gpu-node2: reads NFS tar → writes local SSD → rsync to NFS at end
...
```

**Phase 2: Consumer (single node)**
```
Consumer on gpu-node3: reads NFS segments → merges into FAISS → writes NFS indices
```

### Critical: Batch Rsync, Not Per-File

Per-file sync reintroduces metadata storm. Rsync in bulk at shard completion:

```bash
# GOOD: batch rsync at end
rsync -a "${LOCAL_STAGE}/" "${NFS_WORKSPACE}/"

# BAD: per-file sync during run
# (Don't do this - defeats the purpose of local staging)
```

---

## Infrastructure Reference

### HPC gpu-v100 (V100 - Testing)

| Resource | Value |
|----------|-------|
| CPU per node | 32 cores |
| RAM per node | 255 GB |
| GPU | 2× V100-PCIE-32GB |
| Local SSD | **447 GB at `/local/scratch`** (xfs) |
| Tmpfs | 128 GB at `/ram` (competes with RAM) |
| Network FS | VAST via NFS |

### Path Reference (Testbed)

```bash
# Corpus input (NFS - read only):
NFS_CORPUS_DIR="/path/to/PMC-OA"

# Shared workspace (NFS - avoid high-churn writes):
NFS_WORKSPACE="/path/to/litkit/workspace"

# Local staging (SSD - use for all outputs):
LOCAL_SSD="/local/scratch"
LOCAL_STAGE="${LOCAL_SSD}/litkit_${SLURM_JOB_ID}_shard${shard_id}"
```

### HPC H100 Partition (Production)

| Resource | Value |
|----------|-------|
| CPU per node | TBD |
| RAM per node | TBD |
| GPU | 4× H100 |
| Local storage | TBD - **VERIFY PATHS** |

**Note:** Production paths will differ. Swap testbed paths for production equivalents.

---

## Scaling Projections (Updated)

### On V100 (Current Test)

| Producers | Est. Time (Medium Test) | Notes |
|-----------|------------------------|-------|
| 2 | ~6-7 hours | No staging |
| 2 + output staging | ~4-5 hours | Expected improvement |
| 4 + staging | ~2-3 hours | Should scale linearly |

### On H100 (Production)

With 4 GPUs/node and ~4× faster embedding, I/O becomes limiting factor sooner.

| Producers | Est. Time (Full Corpus) | Notes |
|-----------|------------------------|-------|
| 4 | ~24 hours | Embedding dominated |
| 8 | ~12 hours | Balanced |
| 16 | ~6-8 hours | Consumer may limit |
| 16 + full staging | **~3-4 hours** | Target |

---

## Testing Checklist

### Phase 1 (Complete)
- [x] CPU fixes applied to vector_build_multi.sbatch
- [x] Test run shows affinity_cpus=32 inside container
- [x] GPU access confirmed (nvidia-smi shows devices)
- [x] Scan rate healthy at start (~25-30/s)

### Phase 2 (Complete)
- [x] Add instantaneous rate to scan progress (c959c19)
- [x] Add per-cycle timing to ingest_loop.py (a0f40b1)
- [x] Measure storage performance (47× metadata slowdown)
- [x] Identify staging target: `/local/scratch` (447G)
- [ ] Rebuild container with instrumentation
- [ ] Collect timing data from test run

### Phase 3 (Next)
- [ ] Implement output staging in sbatch
- [ ] Verify rsync correctly copies results
- [ ] Compare cycle timing with/without staging

### Phase 4
- [ ] Implement per-tar streaming
- [ ] Test with 64GB tar file
- [ ] Compare scan rate with local vs NFS read

### Phase 5
- [ ] Test with 4 producers
- [ ] Test with 8 producers  
- [ ] Test with 16 producers
- [ ] Time end-of-job consumer merge
- [ ] Identify final walltime limiter

---

## References

- Slurm CPU binding: https://slurm.schedmd.com/cpu_management.html
- FAISS threading: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
