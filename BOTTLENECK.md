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
| Phase 2: Instrumentation | 🔲 NEXT | Per-cycle timers needed before further optimization |
| Phase 3: Output Staging | 🔲 TODO | Stage SQLite + segments to node-local storage |
| Phase 4: Per-Tar Streaming | 🔲 TODO | Stage one tar at a time to local, process, delete |
| Phase 5: Scale to Production | 🔲 TODO | 16+ producers on H100 nodes |

**Current Test Job:** 16822214 (medium_test.manifest, 3 nodes, 2 producers)

---

## Confirmed Findings

### CPU Allocation Works ✓

```
[diag:producer] Cpus_allowed=0-15,32-47
[diag:producer] affinity_cpus= 32
[diag:producer] GPUs: GPU 0: Tesla V100-PCIE-32GB
                      GPU 1: Tesla V100-PCIE-32GB
```

The `--cpus-per-task` fix propagates correctly. All roles see 32 CPUs.

### Initial Scan Rate is Healthy ✓

- ~25-30 files/second at start
- 16 parallel XML parse workers active

### Architecture: Scan-Embed-Scan (By Design)

The pipeline does NOT overlap scanning with embedding:
1. Scan XML files, buffer chunks
2. When buffer hits ~20K chunks: **STOP scanning**
3. Embed batch on GPU (~140-155s on V100)
4. Write segment file to NFS
5. Commit SQLite
6. **RESUME scanning**

This explains why cumulative rate drops after embedding phases. The instantaneous rate during active scanning remains healthy.

---

## What We Don't Know Yet (Need Instrumentation)

| Question | Why It Matters |
|----------|----------------|
| Time spent in scan/parse vs embedding? | Identifies true bottleneck |
| Time spent writing segments to NFS? | If high, local staging helps |
| Time spent in SQLite commits/fsync? | If high, local staging helps |
| Chunks per document trend? | Chunk explosion = runaway embedding time |
| Does scan rate degrade over hours? | Would indicate NFS degradation |

**Cannot optimize without measurements.** Current logs show only cumulative progress.

---

## Optimization Plan

### Phase 2: Add Per-Cycle Instrumentation (NEXT)

Add timers to `process_tar_files()` in `src/litkit/build/ingest_loop.py`:

```python
# Per batch cycle:
t_scan_parse   # Time spent iterating tar + parsing XML
t_embed        # Time spent in GPU embedding
t_segment_write # Time spent writing .npz.pending + fsync
t_sqlite_commit # Time spent in DB commit

# Trend tracking:
chunks_per_doc  # Rolling average (detect chunk explosion)
```

**Output format:**
```
[cycle] scan=8.2s embed=142.3s seg_write=1.5s db_commit=0.3s | chunks/doc=32.1 avg
```

### Phase 3: Stage OUTPUTS to Node-Local Storage

**Problem:** Every segment write and SQLite commit goes to NFS. On NFS, small writes + fsync is expensive due to metadata server round-trips.

**Solution:** Write all outputs to node-local storage, rsync back at end.

```bash
# At producer start:
LOCAL="$LOCAL_SCRATCH/litkit_${SLURM_JOB_ID}_shard${shard_id}"
mkdir -p "$LOCAL/sqlite" "$LOCAL/emb_segments"

# litkit writes to local:
export LITKIT_WORKSPACE="$LOCAL"
litkit --embed-producer ...

# At producer completion:
rsync -a "$LOCAL/sqlite/" "$NFS_WORKSPACE/sqlite/"
rsync -a "$LOCAL/emb_segments/" "$NFS_WORKSPACE/emb_segments/"
```

**Storage requirements for outputs:**
- SQLite shard DB: ~10-50 MB per producer
- Embedding segments: ~100-500 MB per producer (depends on chunks)
- Total: < 1 GB per producer (easily fits in /tmp)

### Phase 4: Per-Tar Streaming (Stage Input)

**Problem:** On H100 with faster GPUs, embedding time drops. Scan/parse + tar reads may become the bottleneck. With 16 producers hitting NFS simultaneously, per-node read rate could drop.

**Solution:** Stream one tar at a time to local storage:

```bash
for tar_path in $(cat shard_tars.txt); do
    # Stage one tar to local
    rsync "$tar_path" "$LOCAL/current.tar"
    
    # Process from local
    litkit --tar-manifest "$LOCAL/single.manifest" ...
    
    # Delete before staging next
    rm "$LOCAL/current.tar"
done
```

**Storage requirements:**
- Largest uncompressed tar: ~64 GB
- Need: /tmp or local NVMe with ≥70 GB free

**⚠️ TODO:** Verify node-local storage on target nodes:
- `/tmp` - tmpfs (RAM-backed) or SSD?
- `/ram` - ramdisk available?
- Local NVMe/SSD path and size?
- H100 partition storage layout?

### Phase 5: Scale to Production

**Target:** Process full PMC-OA corpus (~185GB, 5M+ articles) on H100 nodes.

**Prerequisites:**
1. ✅ CPU allocation working
2. 🔲 Instrumentation shows where time goes
3. 🔲 Output staging validated
4. 🔲 Per-tar streaming validated
5. 🔲 End-of-job consumer merge timed

**Expected challenges at 16 producers:**
- Consumer may become bottleneck (single node merging segments)
- NFS metadata ops scale poorly (many concurrent file creates)
- End-of-job merge could dominate walltime

---

## Scaling Projections

### On V100 (Current Test)

| Producers | Est. Time (Medium Test) | Notes |
|-----------|------------------------|-------|
| 2 | ~6-7 hours | Current test |
| 4 | ~3-4 hours | Limited by embedding |
| 8 | ~2-3 hours | NFS contention likely |

### On H100 (Production)

With 4 GPUs/node, embedding is ~4× faster than V100. The pipeline becomes I/O bound sooner.

| Producers | Est. Time (Full Corpus) | Notes |
|-----------|------------------------|-------|
| 4 | ~24 hours | Embedding dominated |
| 8 | ~12 hours | Balanced |
| 16 | ~6-8 hours | I/O + consumer may limit |
| 16 + local staging | ~3-4 hours | Target |

**Estimates are speculative without instrumentation data.**

---

## Infrastructure Reference

### HPC gpu-v100 (V100)

| Resource | Value |
|----------|-------|
| CPU per node | 32 cores |
| RAM per node | 255 GB |
| GPU | 2× V100-PCIE-32GB |
| Ramdisk | 128 GB at `/ram` |
| Local disk | 3.5 TB (some nodes) |
| Network FS | VAST via NFS |

### HPC H100 Partition (Production)

| Resource | Value |
|----------|-------|
| CPU per node | TBD |
| RAM per node | TBD |
| GPU | 4× H100 |
| Local storage | TBD - **VERIFY** |

---

## Files to Modify

| File | Change |
|------|--------|
| `src/litkit/build/ingest_loop.py` | Add per-cycle timers |
| `src/litkit/progress.py` | Already updated with instantaneous rate |
| `vector_build_multi.sbatch` | Add local staging logic |
| `BOTTLENECK.md` | This document |

---

## Testing Checklist

### Phase 1 (Complete)
- [x] CPU fixes applied to vector_build_multi.sbatch
- [x] Test run shows affinity_cpus=32 inside container
- [x] GPU access confirmed (nvidia-smi shows devices)
- [x] Scan rate healthy at start (~25-30/s)

### Phase 2 (Next)
- [ ] Add per-cycle timers to ingest_loop.py
- [ ] Rebuild container with instrumentation
- [ ] Run test job and collect timing data
- [ ] Analyze: what % of time is scan vs embed vs I/O?

### Phase 3
- [ ] Implement output staging in sbatch
- [ ] Verify rsync correctly copies results
- [ ] Compare total time with/without staging

### Phase 4
- [ ] Verify node-local storage availability
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
