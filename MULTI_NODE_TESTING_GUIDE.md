# Multi-Node Producer-Consumer Testing Guide

## Overview

This guide provides step-by-step instructions for testing the multi-node producer-consumer architecture for litkit's vector indexing pipeline.

## Prerequisites

1. Access to an HPC cluster with SLURM
2. Shared filesystem (Lustre/NFS) accessible from all nodes
3. CUDA-capable GPUs on compute nodes
4. Python 3.10+ with litkit dependencies installed
5. PMC-OA tar shard files available

## Environment Setup

```bash
# Set up workspace on shared filesystem
export LITKIT_WORKSPACE="/path/to/shared/workspace"
export LITKIT_TAR_DIR="/path/to/tar/shards"

# Create necessary directories
mkdir -p ${LITKIT_WORKSPACE}/{sqlite,indices,emb_segments,hf_cache}

# Optional: increase timeouts for shared filesystems
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000
export LITKIT_SAVE_EVERY_SEC=300
```

## Phase 1: Single-Node Baseline Testing

### Test 1.1: Single Producer (No Consumer)

**Purpose**: Verify that producers can write segment files correctly

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=test-producer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

python -m litkit \
    --embed-producer \
    --shard-id 0 \
    --num-shards 1 \
    --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
    --paper-embed-bs 16 \
    --chunk-embed-bs 64 \
    --embed-devices "cuda:0,cuda:1" \
    --embed-workers 2 \
    --build-only
EOF
```

**Expected Results**:
- Segment files written to `${LITKIT_WORKSPACE}/emb_segments/`
- Files follow naming convention: `papers_sh00_<timestamp>_<seq>.npz` and `chunks_sh00_<timestamp>_<seq>.npz`
- SQLite DB populated but `in_index=0` for all rows
- No FAISS index files created

**Validation**:
```bash
# Check segment files
ls -lh ${LITKIT_WORKSPACE}/emb_segments/

# Check DB (should have papers/chunks with in_index=0)
sqlite3 ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*), SUM(in_index) FROM papers; SELECT COUNT(*), SUM(in_index) FROM chunks;"
```

### Test 1.2: Single Consumer (With Pre-existing Segments)

**Purpose**: Verify that consumers can ingest segments correctly

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=test-consumer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

python -m litkit \
    --faiss-writer \
    --consume-segments \
    --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
    --reconcile-only
EOF
```

**Expected Results**:
- All segment files processed and removed
- FAISS index files created/updated
- SQLite DB updated with `in_index=1` for ingested rows
- No segment files remaining in `emb_segments/`

**Validation**:
```bash
# Check that segments are consumed
ls -lh ${LITKIT_WORKSPACE}/emb_segments/

# Check FAISS indices
ls -lh ${LITKIT_WORKSPACE}/indices/

# Verify in_index flags
sqlite3 ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*), SUM(in_index) FROM papers; SELECT COUNT(*), SUM(in_index) FROM chunks;"
```

## Phase 2: Two-Node Producer-Consumer Testing

### Test 2.1: One Producer + One Consumer (Sequential)

**Purpose**: Test basic producer-consumer workflow

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=test-seq-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Node 0: Producer
srun -N1 -n1 --nodelist=$SLURM_NODELIST --exclusive \
  python -m litkit \
    --embed-producer \
    --shard-id 0 \
    --num-shards 1 \
    --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
    --paper-embed-bs 16 \
    --chunk-embed-bs 64 \
    --embed-devices "cuda:0,cuda:1" \
    --embed-workers 2 \
    --build-only &

PRODUCER_PID=$!

# Wait for producer to finish
wait $PRODUCER_PID

# Node 1: Consumer
srun -N1 -n1 --nodelist=$SLURM_NODELIST --exclusive \
  python -m litkit \
    --faiss-writer \
    --consume-segments \
    --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
    --reconcile-only

echo "Sequential test completed"
EOF
```

### Test 2.2: One Producer + One Consumer (Concurrent)

**Purpose**: Test concurrent operation with proper coordination

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=test-concurrent-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Start producer on node 0
srun -N1 -n1 -w $(scontrol show hostname $SLURM_NODELIST | head -1) \
  python -m litkit \
    --embed-producer \
    --shard-id 0 \
    --num-shards 1 \
    --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
    --paper-embed-bs 16 \
    --chunk-embed-bs 64 \
    --embed-devices "cuda:0,cuda:1" \
    --embed-workers 2 \
    --build-only &

PRODUCER_PID=$!

# Start consumer on node 1 (polls for segments)
srun -N1 -n1 -w $(scontrol show hostname $SLURM_NODELIST | tail -1) \
  bash -c '
    while kill -0 '"$PRODUCER_PID"' 2>/dev/null || [ -n "$(ls '"${LITKIT_WORKSPACE}/emb_segments/"'*.npz 2>/dev/null)" ]; do
      python -m litkit \
        --faiss-writer \
        --consume-segments \
        --embed-outdir "'"${LITKIT_WORKSPACE}/emb_segments"'" \
        --reconcile-only
      sleep 30
    done
  ' &

CONSUMER_PID=$!

# Wait for both to finish
wait $PRODUCER_PID
wait $CONSUMER_PID

echo "Concurrent test completed"
EOF
```

**Expected Results**:
- Producer writes segments while consumer ingests them
- No segment accumulation (consumer keeps up)
- All data successfully indexed
- No race conditions or corruption

## Phase 3: Multi-Node Stress Testing

### Test 3.1: Multiple Producers + Single Consumer

**Purpose**: Test scalability with N producers

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=test-multi-producer
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00

NUM_PRODUCERS=3
NODES=($(scontrol show hostname $SLURM_NODELIST))

# Start N producers
for i in $(seq 0 $((NUM_PRODUCERS-1))); do
  srun -N1 -n1 -w ${NODES[$i]} \
    python -m litkit \
      --embed-producer \
      --shard-id $i \
      --num-shards $NUM_PRODUCERS \
      --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
      --paper-embed-bs 16 \
      --chunk-embed-bs 64 \
      --embed-devices "cuda:0,cuda:1" \
      --embed-workers 2 \
      --build-only &
  
  PRODUCER_PIDS[$i]=$!
done

# Start consumer on last node
srun -N1 -n1 -w ${NODES[-1]} \
  bash -c '
    while pgrep -P '"$$"' >/dev/null || [ -n "$(ls '"${LITKIT_WORKSPACE}/emb_segments/"'*.npz 2>/dev/null)" ]; do
      python -m litkit \
        --faiss-writer \
        --consume-segments \
        --embed-outdir "'"${LITKIT_WORKSPACE}/emb_segments"'" \
        --reconcile-only
      sleep 30
    done
  ' &

CONSUMER_PID=$!

# Wait for all producers
for pid in ${PRODUCER_PIDS[@]}; do
  wait $pid
done

# Wait for consumer to finish
wait $CONSUMER_PID

echo "Multi-producer test completed"
EOF
```

## Monitoring and Debugging

### Real-time Monitoring

```bash
# Watch segment directory
watch -n 5 'ls -lh ${LITKIT_WORKSPACE}/emb_segments/ | head -20'

# Monitor DB stats
watch -n 10 'sqlite3 ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*) as total, SUM(in_index) as indexed FROM papers; \
   SELECT COUNT(*) as total, SUM(in_index) as indexed FROM chunks;"'

# Check FAISS index sizes
watch -n 30 'ls -lh ${LITKIT_WORKSPACE}/indices/'
```

### Debug Logging

```bash
# Enable verbose logging
export LITKIT_SEGMENT_FSYNC_FILE=1
export LITKIT_SEGMENT_FSYNC_DIR=1

# Increase checkpoint frequency for testing
export LITKIT_CKPT_EVERY=100
```

### Common Issues and Solutions

#### Issue 1: Segments Accumulating

**Symptom**: Segment files pile up, consumer not keeping pace

**Solution**:
```bash
# Check consumer logs for errors
grep ERROR litkit_multi_*.err

# Manually trigger consumer if it stopped
python -m litkit --faiss-writer --consume-segments \
  --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" --reconcile-only
```

#### Issue 2: DB Lock Timeouts

**Symptom**: "database is locked" errors

**Solution**:
```bash
# Increase timeout
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=600000  # 10 minutes

# Switch to TRUNCATE journal mode for NFS
python -m litkit --sqlite-journal-mode TRUNCATE ...
```

#### Issue 3: Incomplete Indexing

**Symptom**: Some papers/chunks have `in_index=0` after completion

**Solution**:
```bash
# Run backfill/reconciliation
python -m litkit --faiss-writer --reconcile-only

# Check for orphaned segments
ls ${LITKIT_WORKSPACE}/emb_segments/*.failed
```

## Performance Benchmarks

### Expected Throughput

| Configuration | Papers/Hour | Chunks/Hour |
|--------------|-------------|-------------|
| 1 Node (2 GPUs) | ~50,000 | ~500,000 |
| 3 Producers + 1 Consumer | ~120,000 | ~1,200,000 |
| 7 Producers + 1 Consumer | ~200,000 | ~2,000,000 |

### Bottleneck Analysis

```bash
# CPU usage
srun --pty top

# GPU usage  
srun --pty nvidia-smi dmon

# Disk I/O
srun --pty iostat -x 5

# Network (for NFS)
srun --pty iftop
```

## Validation Checklist

After each test phase:

- [ ] All segment files consumed (emb_segments/ empty)
- [ ] FAISS indices created and non-empty
- [ ] SQLite `in_index=1` for all papers and chunks
- [ ] No `.failed` or `.ingesting` files remaining
- [ ] FAISS ntotal matches DB row counts
- [ ] No error messages in logs
- [ ] Query test returns sensible results

## Query Validation Test

```bash
# Test a simple query to verify the index works
python -m litkit "What is HIV?" \
  --top-papers 100 \
  --top-chunks 20 \
  --no-llm

# Expected: Returns 20 relevant chunks about HIV
```

## Cleanup

```bash
# Remove test artifacts
rm -rf ${LITKIT_WORKSPACE}/emb_segments/*
rm -f ${LITKIT_WORKSPACE}/sqlite/.shard_*_complete

# Optional: full reset for next test
rm -rf ${LITKIT_WORKSPACE}/{sqlite,indices}/*
```

## Next Steps

1. Run Phase 1 tests to establish baseline
2. Proceed to Phase 2 if baseline passes
3. Scale up to Phase 3 with multiple producers
4. Document any issues encountered
5. Iterate on fixes as needed
