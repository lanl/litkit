# Deployment Checklist for Producer-Consumer Fix

## Summary of Changes

This update fixes the multi-node producer-consumer architecture by:
1. Adding `--consume-only` flag for dedicated consumer nodes
2. Adding bootstrap phase to create FAISS indices before producers start
3. Simplifying the consumer polling logic

## Files Modified

1. **src/litkit/cli.py**
   - Added `--consume-only` CLI argument
   - Implemented consume-only mode with infinite polling loop
   - Consumer skips all tar scanning and embedding work

2. **vector_build_multi.sbatch**
   - Added bootstrap phase to create FAISS indices
   - Updated consumer to use `--consume-only` flag
   - Removed bash polling loop (now handled in Python)

3. **PRODUCER_CONSUMER_IMPLEMENTATION.md** (new)
   - Complete documentation of the fix
   - Testing plan and monitoring commands

## Deployment Steps

### On Your Laptop

```bash
cd /path/to/litkit

# Check status
git status

# Stage changes
git add src/litkit/cli.py vector_build_multi.sbatch \
    PRODUCER_CONSUMER_IMPLEMENTATION.md DEPLOYMENT_CHECKLIST.md

# Commit
git commit -m "Add --consume-only flag and bootstrap phase for multi-node deployment"

# Push to remote
git push origin fix/multi-node-producer-consumer
```

### On HPC

```bash
# SSH to HPC
ssh cluster.example.com

# Navigate to repo
cd /path/to/litkit

# Pull latest changes
git pull origin fix/multi-node-producer-consumer

# Verify changes
grep -n "consume-only" src/litkit/cli.py
grep -n "Bootstrap" vector_build_multi.sbatch

# Submit job
sbatch vector_build_multi.sbatch

# Monitor
squeue --me
tail -f litkit_multi_<JOBID>.out
```

## What the Job Will Do

1. **Bootstrap Phase** (~1-2 minutes)
   - Consumer node creates empty FAISS indices
   - Trains IVF-PQ index with small sample
   - Exits cleanly

2. **Producer Phase** (parallel)
   - 2-3 nodes embed papers and chunks
   - Write segments to `/workspace/emb_segments/`
   - All GPUs utilized for embedding

3. **Consumer Phase** (continuous)
   - Polls for new segment files every 30 seconds
   - Ingests segments into FAISS indices
   - Saves indices periodically
   - Continues until job times out or is cancelled

## Monitoring Commands

```bash
# Check GPU usage on all nodes
srun --jobid=<JOBID> nvidia-smi

# Watch segment queue
watch -n 5 "ls /path/to/litkit/workspace/emb_segments/*.npz 2>/dev/null | wc -l"

# Check FAISS progress
sqlite3 /path/to/litkit/workspace/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*) FROM chunks WHERE in_index=1"

# Search logs
grep "\[consumer\]" litkit_multi_<JOBID>.out
grep "\[Producer" litkit_multi_<JOBID>.out
```

## Expected Behavior

### Success Indicators
- ✅ Bootstrap completes without errors
- ✅ All producer GPUs show high utilization (nvidia-smi)
- ✅ Consumer GPU memory remains low (<1GB)
- ✅ Segment files appear and disappear from `/emb_segments/`
- ✅ FAISS `ntotal` increases over time
- ✅ No "CHUNK index does not exist" errors

### Logs to Look For
```
=== Bootstrapping FAISS Indices ===
Running bootstrap on consumer node: gpu-node4
[train] chunk training samples: target=150000 collected=...
[train] training IVF-PQ: nlist=... m=64
=== Bootstrap Complete ===

[Producer 0] Starting on gpu-node1
[Producer 1] Starting on gpu-node3

[consumer] Starting consume-only mode
[consumer] Ingested X paper vectors and Y chunk vectors
```

## Performance Expectations

- **Before:** 2-4 GPUs active (consumer wasted GPUs on duplicate work)
- **After:** 4-6 GPUs active (all producer GPUs utilized)
- **Speedup:** ~2x faster embedding (4 vs 2 active embedding GPUs)

## Troubleshooting

### If Bootstrap Fails
```bash
# Check logs
tail -50 litkit_multi_<JOBID>.err

# Common issues:
# - Missing HF cache: check $HF_HOST bind mount
# - Insufficient training samples: normal, will fall back to FLAT index
```

### If Producers Crash
```bash
# Check for index errors
grep "index does not exist" litkit_multi_<JOBID>.err

# Should not happen with bootstrap, but if it does:
# Verify indices exist before retrying
ls -lh /path/to/litkit/workspace/indices/
```

### If Consumer Doesn't Find Segments
```bash
# Check segment directory
ls -lh /path/to/litkit/workspace/emb_segments/

# Check consumer logs
grep "No new segments" litkit_multi_<JOBID>.out

# Normal during startup; should show activity once producers start
```

## Rollback Plan

If issues arise, revert to previous working version:

```bash
git checkout fix/multi-node-producer-consumer~1
sbatch vector_build_multi.sbatch
```

## Next Steps After Testing

1. Monitor first run to completion
2. Verify GPU utilization improvement
3. Document performance metrics
4. Merge to main branch if successful
