# Litkit Multi-Node Build Performance Optimization

**Date:** 2025-12-24 (Updated)  
**Version:** 0.3.35  
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
| Phase 3.1: Segment Naming | ✅ FIXED | `_pending` now BEFORE `.npz` (critical bug) |
| Phase 3.2: Single-Node Build | ✅ TESTED | vector_build_single.sbatch with local staging |
| Phase 3.3: Periodic Rsync | 🔲 NEXT | Checkpoint-safe local staging for long jobs |
| Phase 4: Tar Input Staging | 🔲 TODO | Stage tar files to local SSD for 2.8× read speedup |
| Phase 5: Scale to Production | 🔲 TODO | 16+ producers on H100 nodes |

**Commits (2025-12-23):**
- `c959c19` - Instantaneous scan rate display
- `a0f40b1` - Per-cycle timing instrumentation
- `0eb8dba` - Output staging to local SSD (flat rsync to NFS)
- `935ef57` - Fix: Skip FAISS index loading for `--embed-producer` mode
- `878001c` - Baseline config: `USE_LOCAL_STAGING=0`, no step-level `--gres`
- `b77fe27` - Producer PID tracking and failure handling
- `33360ba` - Watchdog to scancel job on early producer failure

**Commits (2025-12-24):**
- `7d64c55` - Bash 4.4 safe polling monitor (no double-wait SIGCHLD bug)
- `9931b22` - Handle rc=127 as "already reaped" + kill process groups in abort
- `fdc2bba` - Safe rsync version logging (quote command substitution)
- `9572f71` - Default to 3 nodes + `USE_LOCAL_STAGING=1` in vector_build_multi
- `6273865` - Add local staging support to vector_build_single.sbatch
- `3b439ac` - **CRITICAL:** Fix segment file naming bug (see below)
- `61d55b8` - Add CUDA/GPU bindings to `just ask-file` target
- `03e56bd` - Fix vector store filenames in ask-file diagnostics
- `c200620` - Bump version 0.3.34 → 0.3.35

**Tested (2025-12-24, Job 16822876):**
- **Single-node build** with local SSD staging: ✅ SUCCESS
- **Corpus:** tiny_test.manifest (~40MB uncompressed)
- **Results:** 1,015 papers, 13,724 chunks indexed
- **Throughput:** 107.6 chunks/sec embedding (single V100)
- **Local staging:** 444GB free on `/local/scratch`, rsync back succeeded
- **End-to-end RAG:** `just ask-file` query with LLM response working

---

## CRITICAL BUG FIX: Segment File Naming (3b439ac)

### Problem

After running Job 16822787 with local staging, FAISS indices were empty despite producers successfully generating embeddings:

```
[segment] Wrote 20031 chunk embeddings to chunk_seg_1_0_ca894822.npz.pending (pending)
...
[summary] papers: db=12656 in_index=0 faiss_ntotal=0
[summary] chunks: db=339716 in_index=0 faiss_ntotal=0
```

**Evidence on disk:**
```bash
$ ls workspace/emb_segments/
chunk_seg_1_0_ca894822.npz.pending.npz   # WRONG: double .npz!
producer_1.done
```

### Root Cause

`np.savez_compressed()` auto-appends `.npz` if the filename doesn't already end with it:

```python
# OLD (buggy):
filename = f"{prefix}_{seg_id}{SEGMENT_EXTENSION}{PENDING_SUFFIX}"
# = chunk_seg_1_0_abc.npz.pending
# numpy sees ".pending" → adds ".npz" → "chunk_seg_1_0_abc.npz.pending.npz"
```

Then `finalize()` tried to rename `...npz.pending` (which doesn't exist) → silent failure → consumer never saw segments.

### Fix

Put `_pending` BEFORE `.npz`:

```python
# NEW (fixed):
filename = f"{prefix}_{seg_id}{PENDING_SUFFIX}{SEGMENT_EXTENSION}"
# = chunk_seg_1_0_abc_pending.npz
# numpy sees ".npz" → leaves it alone
```

And `finalize()` now renames `*_pending.npz` → `*.npz`.

### Impact

- **All multi-node builds prior to 3b439ac produced empty FAISS indices**
- Single-node builds worked (no segment protocol)
- Fix requires container rebuild

---

## Bash 4.4 Polling Monitor (7d64c55)

### Problem

The original watchdog pattern used `wait -n` which:
1. Relies on Bash 4.3+ (HPC has 4.4, but fragile)
2. Can race with the main `wait` loop, causing "double-wait" SIGCHLD issues
3. Automatically reaps processes, confusing later `wait $PID` calls

### Solution: Polling Monitor

```bash
REAPED=()  # Track which PIDs have already been reaped

monitor_procs() {
    while true; do
        for ((i=0; i<${#PRODUCER_PIDS[@]}; i++)); do
            pid="${PRODUCER_PIDS[$i]}"
            [[ " ${REAPED[*]} " == *" $pid "* ]] && continue  # Already handled
            
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null; rc=$?
                REAPED+=("$pid")
                
                # rc=127 means already reaped (not an error)
                if [[ "$rc" -ne 0 && "$rc" -ne 127 ]]; then
                    echo "[monitor] Producer $i (PID $pid) FAILED (rc=$rc)" >&2
                    abort_all
                    return 1
                fi
            fi
        done
        sleep 15
    done
}
```

### Key Improvements

1. **No `wait -n`** - Uses `kill -0` to poll, then explicit `wait $pid`
2. **REAPED tracking** - Prevents double-wait race conditions
3. **rc=127 handling** - Treats "already reaped" as success (Bash quirk)
4. **Process group kills** - `abort_all` uses `kill -- -$pid` to kill entire process trees

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
/nfs/home              NFS           Corpus input (/path/to/PMC-OA)
/nfs/projects          NFS           Current workspace (slow for metadata)
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

### Producer Watchdog (IMPLEMENTED - 33360ba)

The PID tracking above only detects failures AFTER the consumer finishes. A producer failing at minute 5 could still let the consumer run for hours. The watchdog fixes this:

```bash
watch_producers() {
    while true; do
        for ((i=0; i<NUM_PRODUCERS; i++)); do
            pid="${PRODUCER_PIDS[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                if ! wait "$pid" 2>/dev/null; then
                    echo "[watchdog] FATAL: Producer $i failed; cancelling job" >&2
                    scancel "${SLURM_JOB_ID}"
                    return 1
                fi
            fi
        done
        sleep 15
    done
}

watch_producers &
WATCHDOG_PID=$!

# ... run consumer ...

# After consumer:
kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
```

**Behavior:** Polls every 15s. If any producer dies with nonzero exit, calls `scancel` to terminate the entire job immediately.

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

### Phase 3 (Complete - Testing)
- [x] Implement output staging in sbatch (USE_LOCAL_STAGING=1)
- [x] Fix producer FAISS loading bug (935ef57)
- [x] Baseline config: USE_LOCAL_STAGING=0 (878001c)
- [x] Add local staging to vector_build_single.sbatch (6273865)
- [x] Test run Job 16822787 (staging working, seg=4.4s)
- [ ] Full test with fixed segment naming

### Phase 3.1: Segment Naming Bug Fix (Complete)
- [x] Identify bug: `np.savez_compressed` appends `.npz` (3b439ac)
- [x] Fix: `_pending` suffix now BEFORE `.npz`
- [x] Update `finalize()` and `cleanup_orphan_pending_files()`
- [ ] Rebuild container with fix
- [ ] Re-run multi-node test

### Bash 4.4 Compatibility (Complete)
- [x] Replace `wait -n` watchdog with polling monitor (7d64c55)
- [x] Add REAPED tracking to prevent double-wait
- [x] Handle rc=127 as "already reaped" (9931b22)
- [x] Kill process groups in abort_all (9931b22)

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

## Known Issues / Future Work

### Partial Failure Recovery (NOT SUPPORTED)

**Problem:** If a job is cancelled or a producer fails mid-run, work is lost and cannot be resumed.

**What happens on cancel/failure:**
1. Producers write to local SSD (`/local/scratch/litkit_JOBID_shardX/`)
2. Rsync to NFS happens ONLY on successful producer completion
3. If producer crashes/cancelled → local staging is deleted → work lost

**Current state after a partial failure:**
| Component | Status |
|-----------|--------|
| Completed producers | ✅ Segments on NFS |
| Failed/cancelled producers | ❌ Work lost (local staging deleted) |
| Consumer | ❌ Never ran or incomplete |
| FAISS indices | ❌ Empty (bootstrap only) |

**Workarounds:**

1. **Clean restart** (simplest)
   ```bash
   rm -rf workspace/emb_segments/* workspace/indices/* workspace/sqlite/*
   sbatch vector_build_multi.sbatch
   ```

2. **Consumer-only** (partial data)
   ```bash
   sbatch vector_resume_consumer.sbatch
   ```
   Ingests whatever segments are on NFS. Only gives data from completed producers.

3. **Re-run single producer** (complex)
   Not currently supported. Would require manual shard configuration.

**Future fix ideas:**
- [ ] Periodic rsync (every N cycles) - trades off I/O for fault tolerance
- [ ] Checkpoint to NFS before local staging (startup overhead)
- [ ] SLURM `--signal=TERM@300` to trigger cleanup rsync on timeout
- [ ] Track tar progress in checkpoint file for per-tar resume

**Priority:** Medium. For production, use shorter manifests or accept occasional full restarts.

---

## Phase 3.3: Periodic Rsync (PROPOSED)

### Problem

Local SSD staging is **essential** for performance (47× metadata speedup), but creates a fault tolerance problem:

| Scenario | Without Periodic Rsync | With Periodic Rsync |
|----------|------------------------|---------------------|
| Job completes | ✅ Rsync at end | ✅ Rsync at end |
| Job times out (10hr limit) | ❌ All work lost | ✅ Lose max 30 min |
| Producer crashes | ❌ All work lost | ✅ Lose max 30 min |
| Restart with `--update` | ❌ Cannot resume | ✅ Resumes from last rsync |

**Disabling local staging is NOT viable** - NFS metadata storm makes tar processing ~47× slower.

### Solution: Background Rsync Loop

Add a background process that rsyncs to NFS every N minutes during the build:

```bash
# Start background rsync loop (runs every 30 min)
rsync_checkpoint_loop() {
    while true; do
        sleep 1800  # 30 minutes
        /usr/bin/rsync -a --delay-updates "$LOCAL_STAGE_DIR/sqlite/" "$WORKSPACE/sqlite/" 2>/dev/null || true
        /usr/bin/rsync -a --delay-updates "$LOCAL_STAGE_DIR/indices/" "$WORKSPACE/indices/" 2>/dev/null || true
        /usr/bin/rsync -a --delay-updates "$LOCAL_STAGE_DIR/emb_segments/" "$WORKSPACE/emb_segments/" 2>/dev/null || true
        echo "[checkpoint] rsync to NFS at $(date)"
    done
}

rsync_checkpoint_loop &
RSYNC_LOOP_PID=$!

# Run litkit as normal...
"${CHR[@]}" "$IMG" -- litkit "${LITKIT_ARGS[@]}"
BUILD_RC=$?

# Kill background rsync loop
kill $RSYNC_LOOP_PID 2>/dev/null || true

# Final rsync (as before)
if [[ "$BUILD_RC" -eq 0 ]]; then
    rsync -a --delay-updates "$LOCAL_STAGE_DIR/" "$WORKSPACE/"
fi
```

### Why This Works

1. **rsync is incremental** - After first sync, subsequent syncs only transfer changed files
2. **SQLite/FAISS grow monotonically** - rsync handles appends efficiently
3. **`--delay-updates`** - Atomic file replacement, safe for concurrent readers
4. **30-min interval** - Balances checkpoint frequency vs I/O overhead

### Overhead Estimate

| Item | First Sync | Subsequent Syncs |
|------|------------|------------------|
| SQLite (~25MB) | ~1s | <1s (delta) |
| Indices (~50MB) | ~2s | <1s (delta) |
| Segments (~100MB) | ~3s | <1s (delta) |
| **Total** | ~6s | ~3s |

For a 10-hour job with 20 checkpoints: **~60s total overhead** (negligible vs 47× metadata storm).

### Restart After Timeout

```bash
# After job timeout, NFS has data up to last checkpoint
# Restart with --update (appends to existing index)
REBUILD=0 sbatch vector_build_single.sbatch
```

Litkit's `--update` mode:
1. Loads existing FAISS indices from NFS
2. Reads checkpoint file to find last processed tar
3. Continues from where it left off

### Implementation Status

- [ ] Add `RSYNC_CHECKPOINT_INTERVAL` knob (default 1800s = 30min)
- [ ] Add rsync loop to `vector_build_single.sbatch`
- [ ] Add rsync loop to `vector_build_multi.sbatch` (per-producer)
- [ ] Test restart after simulated timeout
- [ ] Document in README

---

## Constrained Environment Strategy

### Constraints

Some clusters have strict limits:
- **Storage:** 500GB quota
- **Job time:** 10 hours max
- **Cannot decompress full corpus** at once

### Solution: Batch Processing with Incremental Builds

**Step 1: Split corpus into batches**
```bash
# Create batch manifests (~100GB uncompressed each)
ls /path/to/*.tar | head -20 > batch1.manifest
ls /path/to/*.tar | tail -n +21 | head -20 > batch2.manifest
# etc.
```

**Step 2: Decompress-process-delete cycle**
```bash
# For each batch:
# 1. Decompress this batch's tar.gz → tar (to temp storage)
for f in $(cat batchN.manifest.gz); do gunzip -k "$f"; done

# 2. Run litkit with --update (not --rebuild) to append
sbatch --export=MANIFEST=batchN.manifest,REBUILD=0 vector_build_single.sbatch

# 3. Delete decompressed tars after successful embedding
rm /path/to/temp/*.tar
```

**Step 3: Use periodic rsync for checkpoint safety**

With periodic rsync enabled:
- Each 10-hour job can be interrupted safely
- Restart continues from last checkpoint
- Progress accumulates across jobs

### Storage Budget

| Item | Size | Notes |
|------|------|-------|
| Decompressed batch | ~100GB | Rotated per batch |
| SQLite DB | ~50-100MB | Grows with corpus |
| FAISS indices | ~100-500MB | Grows with corpus |
| Working headroom | ~50GB | For checkpoints, temp files |
| **Total** | ~250GB | Well under 500GB limit |

### Time Budget

| Operation | Time | Notes |
|-----------|------|-------|
| Decompress batch | ~30 min | One-time per batch |
| Process batch (8 GPUs) | ~2-3 hours | With local staging |
| Checkpoint rsync | ~3s | Every 30 min |
| **Total per batch** | ~3 hours | Fits 10hr limit easily |

### Full Corpus Estimate

- 160GB compressed → ~500-800GB uncompressed
- Split into 5-8 batches
- Each batch: ~3 hours
- **Total: ~15-24 hours** across 3-4 job submissions

---

## FAISS Dedup Bottleneck Fix (2026-01-13)

### Root Cause Analysis

**Reported:** Email thread 2026-01-08  
**Confirmed by:** Code review 2026-01-13  
**Status:** ✅ FIX IMPLEMENTED (2026-01-13)

#### Symptom

Processing rate starts high (~100 files/sec) but degrades progressively as the build continues. The slowdown correlates with index size growth.

#### Root Cause

The `safe_remove_ids()` function (in `src/litkit/index/ids.py`) calls FAISS `index.remove_ids()`, which has **O(N) complexity** for IndexFlatIP wrapped in IndexIDMap2. As the index grows:

- 10K vectors → each removal scans 10K vectors
- 100K vectors → each removal scans 100K vectors  
- 1M vectors → each removal scans 1M vectors

This explains the "mysterious decay" in processing rate—it's proportional to index size.

#### Compounding Factor: Double-Call Bug

The code calls `safe_remove_ids()` **TWICE** per batch:

1. Explicitly in caller code before `add_with_ids_dedup()`
2. Again inside `add_with_ids_dedup()` itself

This doubles the O(N) cost per batch.

**Evidence in `ingest_loop.py`:**
```python
sel = make_id_selector(u_ids)
with FileLock(faiss_lock):
    safe_remove_ids(paper_index, sel)    # FIRST CALL - explicit
    added, ids_added = add_with_ids_dedup(paper_index, u_ids, Xp)  # SECOND CALL inside
```

**Inside `add_with_ids_dedup()` (dedup.py):**
```python
def add_with_ids_dedup(index, ids, X):
    ...
    sel = make_id_selector(ids_arr)
    safe_remove_ids(index, sel)   # Redundant second call
    index.add_with_ids(X, ids_arr)
```

#### Why HNSW Users Don't See This

The `safe_remove_ids` function has a special case:
```python
if kind == INDEX_KIND_HNSW:
    return 0  # No-op - HNSW doesn't support ID removal
```

HNSW index users bypass this bottleneck entirely.

### Affected Files

| File | Double-Call Sites | Notes |
|------|-------------------|-------|
| `src/litkit/build/ingest_loop.py` | 6 | Main build loop, faiss_writer mode |
| `src/litkit/build/backfill.py` | 2 | Backfill reconciliation |
| `src/litkit/segments/ingest.py` | 2 | Consumer segment ingestion |

### Fix Implementation

#### Phase 1: Eliminate Double-Calls ✅ COMPLETE

Removed explicit `safe_remove_ids()` calls that immediately preceded `add_with_ids_dedup()`. The dedup function already handles removal internally.

**Files modified:**

1. **`src/litkit/build/ingest_loop.py`** - Removed 6 explicit calls ✅
   - Paper batch flush in faiss_writer mode
   - Chunk batch flush in faiss_writer mode
   - Final buffer flush sections (process_tar_files and flush_final_buffers)
   
2. **`src/litkit/build/backfill.py`** - Removed 2 explicit calls ✅
   - Paper backfill
   - Chunk backfill

3. **`src/litkit/segments/ingest.py`** - Removed 2 explicit calls ✅
   - `ingest_paper_segments()` 
   - `ingest_chunk_segments()`

#### Phase 2: Add skip_dedup for Rebuild Mode ✅ COMPLETE

Added `skip_dedup` parameter to `add_with_ids_dedup()` in `src/litkit/index/dedup.py`:

```python
def add_with_ids_dedup(
    index: faiss.Index,
    ids: list[int],
    X: np.ndarray,
    skip_dedup: bool = False,  # NEW PARAMETER
) -> tuple[int, np.ndarray]:
    """Add vectors with IDs, optionally skipping dedup for rebuild mode."""
    ...
    # skip_dedup=True is safe for --rebuild mode (index starts empty)
    if not skip_dedup:
        sel = make_id_selector(ids_arr)
        safe_remove_ids(index, sel)
    ...
```

**Phase 2.2:** Updated `process_tar_files()` in `ingest_loop.py` to pass `skip_dedup=rebuild`:
- Paper batch flush (faiss_writer mode) ✅
- Chunk batch flush (faiss_writer mode) ✅
- Final buffer flush (inside process_tar_files) ✅

**Remaining:** The standalone `flush_final_buffers()` function doesn't receive the `rebuild` flag and defaults to `skip_dedup=False`. This is a minor gap affecting only the final buffer flush when called externally.

#### Phase 3: Test

| Test Case | Expected Result |
|-----------|-----------------|
| `--rebuild` small corpus | Fast, constant rate |
| `--rebuild` medium corpus | Fast, constant rate |
| `--update` on existing index | Dedup works correctly |
| Segment ingestion | No regression |

### Expected Performance Impact

| Scenario | Before | After Phase 1 | After Phase 2 |
|----------|--------|--------------|---------------|
| `--rebuild` 100K chunks | 2× O(N) per batch | 1× O(N) per batch | O(1) per batch |
| `--update` 100K chunks | 2× O(N) per batch | 1× O(N) per batch | 1× O(N) per batch |
| Processing rate | Degrades with N | 2× improvement | Constant (rebuild) |

---

## Comprehensive Scalability Code Review Plan

### Motivation

The FAISS dedup bottleneck reveals a pattern: code that works at small scale (1K papers) may fail at production scale (1M+ papers). This section outlines a systematic review to find other scalability issues.

### Review Categories

#### 1. Algorithmic Complexity (O(N²) and O(N) in loops)

**What to look for:**
- Nested loops over growing data structures
- List/set membership checks in loops (O(N) per check)
- String concatenation in loops
- Repeated computation that could be cached

**Files to prioritize:**
- `src/litkit/build/ingest_loop.py` - Main processing loop
- `src/litkit/segments/ingest.py` - Segment processing
- `src/litkit/db/queries.py` - Database operations
- `src/litkit/ingest/ingest.py` - XML parsing

**Search patterns:**
```bash
grep -n "for.*in.*:" src/litkit/**/*.py  # All loops
grep -n "\.append\|\.extend" src/litkit/**/*.py  # Growing lists
grep -n "in list\|in set" src/litkit/**/*.py  # Membership checks
```

#### 2. I/O Patterns

**What to look for:**
- Unbatched writes (fsync per item instead of per batch)
- Small reads that could be batched
- Missing file buffering
- NFS-hostile patterns (many small files, frequent metadata ops)

**Files to check:**
- `src/litkit/segments/writer.py` - Segment output
- `src/litkit/index/io.py` - FAISS save/load
- `src/litkit/db/connection.py` - SQLite operations

**Known good patterns to verify:**
- SQLite WAL mode enabled
- FAISS save throttling (save every N batches)
- Segment batching (write every N chunks)

#### 3. Lock Contention

**What to look for:**
- Fine-grained locks held during I/O
- Locks acquired in different orders (deadlock risk)
- Long-held locks that serialize parallel work

**Files to check:**
- `src/litkit/concurrent/locking.py` - Lock implementation
- All files using `FileLock` - Lock usage patterns

**Check for:**
```bash
grep -n "FileLock\|with.*lock" src/litkit/**/*.py
```

#### 4. Memory Growth

**What to look for:**
- Unbounded buffers (lists that grow without limit)
- Cached data that's never evicted
- Large objects held beyond their useful lifetime

**Files to check:**
- `src/litkit/build/ingest_loop.py` - Buffer management
- `src/litkit/embeddings/pool.py` - Embedding cache
- `src/litkit/db/queries.py` - Preloaded maps

**Search patterns:**
```bash
grep -n "= \[\]" src/litkit/**/*.py  # Empty list init
grep -n "\.clear()" src/litkit/**/*.py  # Buffer clearing
```

#### 5. SQLite Patterns

**What to look for:**
- Missing indices on frequently-queried columns
- N+1 query patterns (query per item instead of batch)
- Large transactions that block readers
- VACUUM/ANALYZE not run after bulk inserts

**Files to check:**
- `src/litkit/db/schema.py` - Index definitions
- `src/litkit/db/queries.py` - Query patterns
- `src/litkit/db/indexing.py` - Batch operations

**Verify indices exist for:**
- `papers.doc_id` (used in segment resolution)
- `papers.pmcid`, `papers.pmid` (used in dedup)
- `chunks.paper_id` (used in paper→chunk lookup)
- `files.path` (used in already_processed check)

#### 6. Embedding Pipeline

**What to look for:**
- GPU underutilization (small batches)
- CPU-GPU transfer overhead
- Model loading per batch instead of once

**Files to check:**
- `src/litkit/embeddings/base.py` - Embedder interface
- `src/litkit/embeddings/pool.py` - Batch management
- `src/litkit/embeddings/hf_local.py` - HuggingFace backend

### Review Checklist

| Category | Status | Findings |
|----------|--------|----------|
| Algorithmic complexity | 🔲 TODO | |
| I/O patterns | 🔲 TODO | |
| Lock contention | 🔲 TODO | |
| Memory growth | 🔲 TODO | |
| SQLite patterns | 🔲 TODO | |
| Embedding pipeline | 🔲 TODO | |

### Priority Order

1. **CRITICAL** - Fix FAISS dedup bottleneck (this document)
2. **HIGH** - SQLite patterns (N+1 queries common source of slowdown)
3. **HIGH** - I/O patterns (NFS sensitivity)
4. **MEDIUM** - Memory growth (OOM risk on large corpus)
5. **MEDIUM** - Lock contention (parallelism limiter)
6. **LOW** - Embedding pipeline (usually GPU-bound)

---

## References

- Slurm CPU binding: https://slurm.schedmd.com/cpu_management.html
- FAISS threading: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
- FAISS IndexIDMap2 limitations: https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-remove-vectors-from-an-index
