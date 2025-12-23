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
| Phase 3: Output Staging | ✅ COMPLETE | Stage SQLite + segments to local SSD |
| Phase 4: Tar Input Staging | 🔲 NEXT | Stage tar files to local SSD for 2.8× read speedup |
| Phase 5: Scale to Production | 🔲 TODO | 16+ producers on H100 nodes |

**Commits:**
- `c959c19` - Instantaneous scan rate display
- `a0f40b1` - Per-cycle timing instrumentation
- `0eb8dba` - Output staging to local SSD (flat rsync to NFS)
- `935ef57` - Fix: Skip FAISS index loading for `--embed-producer` mode
- `878001c` - Baseline config: `USE_LOCAL_STAGING=0`, no step-level `--gres`
- `b77fe27` - Producer PID tracking and failure handling

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

## Phase 3: Output Staging (IMPLEMENTED, TESTING)

### Problem

Every segment write and SQLite commit currently goes to NFS. With measured 47× metadata slowdown, this dominates walltime at scale.

### Solution

Stage all high-churn outputs to node-local SSD, rsync back at end.

### Implementation (vector_build_multi.sbatch) ✅

The sbatch script now:
1. Validates LOCAL_SSD (exists, mountpoint, writable, ≥50GB free)
2. Sets `LITKIT_WORKSPACE=/local_stage` inside container
3. Rsyncs outputs to FLAT NFS directories after litkit completes
4. Falls back to NFS if LOCAL_SSD unavailable

**Key design choices:**
- `--exclusive` omitted for backgrounded producers (avoids step scheduling friction)
- GPUs inherited from job-level allocation (no step-level `--gres` to avoid conflicts)
- FLAT rsync destinations (consumer expects non-recursive globs)
- Conditional rsync on success only (preserves staging dir on failure for debugging)

### Baseline vs Staging Mode

For testing, we run baseline (no staging) first to establish a control:

```bash
# Baseline (default): all writes go to NFS
sbatch vector_build_multi.sbatch

# With local staging: writes to SSD, rsync at end
USE_LOCAL_STAGING=1 sbatch vector_build_multi.sbatch
```

### Bug Fix: Producer FAISS Loading (935ef57)

**Problem:** When `LITKIT_WORKSPACE=/local_stage`, producers looked for FAISS indices at `/local_stage/indices/`, but bootstrap created them on NFS at `/workspace/indices/`.

**Root cause:** `cli.py` called `load_or_create_paper_index()` for all modes, including `--embed-producer`.

**Fix:** Skip index loading when `args.embed_producer=True`. Producers don't need indices (they write segments, not FAISS).

```python
if args.embed_producer:
    paper_index = None
    chunk_index = None
    needs_training = False
else:
    paper_index = build_load_or_create_paper_index(...)
    chunk_index, needs_training = build_load_or_create_chunk_index(...)
```

### Storage Budget

| Item | Size per Producer | Notes |
|------|-------------------|-------|
| SQLite shard DB | 10-50 MB | Grows with corpus |
| Embedding segments | 100-500 MB | ~20K chunks × 3KB each |
| Temp files | <10 MB | Checkpoints, markers |
| **Total** | **< 1 GB** | Easily fits on 447G SSD |

---

## Phase 4: Tar Input Staging (NEXT)

### Problem

Producers still read tar files from NFS. With 16 producers reading simultaneously:
- **NFS read bandwidth:** 124 MB/s per producer
- **Local SSD read:** 350 MB/s (2.8× faster)
- **Aggregate NFS demand:** 16 × 124 MB/s = 2 GB/s (exceeds typical NFS capacity)

Output staging (Phase 3) helps metadata/write pain, but does NOT remove the read bottleneck at scale.

### Solution

Stage each shard's tar files to local SSD, generate a per-shard local manifest, and point litkit at that manifest.

### Implementation Algorithm

**Insert this block inside the srun `bash -lc '...'` portion of run_producer(), AFTER staging validation:**

```bash
# ---- Tar Input Staging ----
HOST_CORPUS_ROOT="/path/to/PMC-OA"
CONTAINER_CORPUS_ROOT="/path/to/PMC-OA"  # same due to bind
HOST_STAGE_TAR_DIR="${PRODUCER_LOCAL_STAGE_HOST}/tars"
CONTAINER_STAGE_TAR_DIR="/local_stage/tars"

# Default: use NFS manifest
MANIFEST_FOR_LITKIT="${CONTAINER_NFS_WS}/${PRODUCER_MANIFEST_FILE}"

if [[ "${ACTUAL_USE_LOCAL_STAGING}" == "1" ]]; then
    echo "[Producer ${PRODUCER_NODE_ID}] [tar-stage] Staging shard tar(s) to local SSD."
    mkdir -p "${HOST_STAGE_TAR_DIR}"
    
    # 1) Read NFS manifest
    HOST_MANIFEST_PATH="${LITKIT_WORKSPACE}/${PRODUCER_MANIFEST_FILE}"
    
    # 2) Select tars for THIS shard using crc32(basename) % num_shards
    # This matches litkit's internal sharding algorithm
    pick_for_shard_py='
import os, sys, zlib
shard_id=int(sys.argv[1]); num=int(sys.argv[2])
for line in sys.stdin:
    p=line.strip()
    if not p or p.startswith("#"): continue
    if ".tar" in p:
        b=os.path.basename(p.split()[0])
        h=zlib.crc32(b.encode("utf-8")) & 0xffffffff
        if (h % num) == shard_id:
            print(p.split()[0])
'
    mapfile -t SHARD_TARS < <(cat "${HOST_MANIFEST_PATH}" | python3 -c "${pick_for_shard_py}" "${PRODUCER_NODE_ID}" "${PRODUCER_NUM_SHARDS}")
    
    # 3) Stage selected tars to local SSD
    for tar_path in "${SHARD_TARS[@]}"; do
        [[ "${tar_path}" != /* ]] && tar_path="${HOST_CORPUS_ROOT}/${tar_path}"
        [[ ! -f "${tar_path}" ]] && continue
        dest="${HOST_STAGE_TAR_DIR}/$(basename "${tar_path}")"
        if [[ ! -f "${dest}" ]]; then
            echo "[tar-stage] Copy -> $(basename "${tar_path}")"
            cp -p "${tar_path}" "${dest}"
        fi
    done
    
    # 4) Write local manifest with rewritten tar paths
    CONTAINER_LOCAL_MANIFEST="/local_stage/manifest_shard_${PRODUCER_NODE_ID}.manifest"
    for f in "${HOST_STAGE_TAR_DIR}"/*.tar; do
        echo "${CONTAINER_STAGE_TAR_DIR}/$(basename "$f")"
    done > "${PRODUCER_LOCAL_STAGE_HOST}/manifest_shard_${PRODUCER_NODE_ID}.manifest"
    
    MANIFEST_FOR_LITKIT="${CONTAINER_LOCAL_MANIFEST}"
    echo "[tar-stage] Using local manifest: ${MANIFEST_FOR_LITKIT}"
fi

# Then in litkit_cmd, use:
#   --tar-manifest "${MANIFEST_FOR_LITKIT}"
```

### Storage Budget

| Item | Size | Notes |
|------|------|-------|
| Largest tar | ~64 GB | Uncompressed PMC-OA shard |
| Tars per producer | ~100-200 GB | Depends on sharding |
| Local SSD | 447 GB | Enough for tars + outputs |
| RAM tmpfs | 128 GB | **DO NOT USE** (OOM risk) |

### Expected Speedup

| Operation | NFS | Local SSD | Ratio |
|-----------|-----|-----------|-------|
| Read 64GB tar | ~8.6 min | ~3.0 min | 2.8× |
| Staging cost | - | ~5-6 min | one-time |

For a multi-hour job, staging cost is amortized. Net gain is significant when multiple tars are processed per producer.

### Deferred

Tar staging adds ~100 lines of complexity. We will validate output staging (Phase 3) first before implementing this.

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

## Red Team Feedback (2025-12-23)

### Issues Raised and Responses

| Issue | Valid? | Response |
|-------|--------|----------|
| `USE_LOCAL_STAGING=1` changes baseline | ✅ Yes | Changed default to 0 (`878001c`) |
| Step-level `--gres=gpu:2` conflicts with job-level | ⚠️ Site-specific | Removed step-level for simplicity (`878001c`) |
| Background producers + `--kill-on-bad-exit` doesn't fail fast | ✅ Yes | Noted; single `wait` at end is too late. Deferred fix. |
| Flat rsync unsafe if filenames not unique | ❌ Already handled | Filenames include shard_id: `litkit_shard_XX.sqlite3`, `paper_seg_shardXX_*.npz` |
| Consumer competing for GPUs | ⚠️ Minor | Consumer on dedicated node; job-level allocation is per-node |
| `mountpoint -q` not portable | ✅ Acknowledged | HPC-specific, acceptable for now |

### Background PID Tracking (IMPLEMENTED - b77fe27)

Previous pattern (problematic):
```bash
for i in ...; do run_producer $i &; done
wait  # waits for ALL, but runs AFTER consumer finishes (too late!)
```

**New pattern (implemented):**
```bash
# Capture PIDs at launch
PRODUCER_PIDS=()
for ((i=0; i<NUM_PRODUCERS; i++)); do
    run_producer $i "${NODELIST[$i]}" &
    PRODUCER_PIDS+=($!)
done

# After consumer completes, check each producer
FAILED_PRODUCERS=()
for ((i=0; i<NUM_PRODUCERS; i++)); do
    if ! wait ${PRODUCER_PIDS[$i]}; then
        FAILED_PRODUCERS+=($i)
    fi
done

if [[ ${#FAILED_PRODUCERS[@]} -gt 0 ]]; then
    echo "FATAL: ${#FAILED_PRODUCERS[@]} producer(s) failed"
    exit 1
fi
```

**Why this matters:** If a producer fails early, the script now reports which producer failed and exits with error code 1, preventing wasted GPU hours.

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
- [x] Rebuild container with instrumentation (in progress)
- [ ] Collect timing data from test run

### Phase 3 (In Progress)
- [x] Implement output staging in sbatch (USE_LOCAL_STAGING=1)
- [x] Fix producer FAISS loading bug (935ef57)
- [x] Baseline config: USE_LOCAL_STAGING=0 (878001c)
- [ ] Run baseline test (no staging)
- [ ] Run staging test (USE_LOCAL_STAGING=1)
- [ ] Compare `seg=` and `db=` timing between runs

### Phase 4 (Deferred)
- [ ] Implement tar input staging
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
