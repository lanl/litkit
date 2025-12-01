# HPC Deployment Guide for Multi-Node Litkit

## Overview

This guide provides step-by-step instructions for deploying and testing the multi-node producer-consumer architecture on an HPC cluster using Charliecloud containers.

## HPC Configuration

### Cluster Details
- **Location**: `/path/to/litkit`
- **PMC-OA Data**: `/path/to/PMC-OA`
- **Partition**: `gpu-v100` (V100 GPUs)
- **Container Runtime**: Charliecloud 0.42
- **Storage**: Lustre only (no node-local storage)

### Container Setup
- **Image**: `sqfs/litkit-v0.3.33-aarch64-lean.sqfs`
- **CDI Spec**: `/path/to/cdi-v100/nvidia.json`
- **CUDA Host**: `/path/to/cuda-12.5-host/cuda-12.5`
- **HF Cache**: `/path/to/hf_cache_persist`

## Git Workflow: Laptop ↔ HPC

### Phase 1: Push Analysis from Laptop to HPC

**On Laptop**:
```bash
cd /path/to/litkit

# Verify current branch
git branch
# Should show: * fix/multi-node-producer-consumer

# Check what's committed
git log --oneline -5

# Push to HPC
git push origin fix/multi-node-producer-consumer
```

**On HPC**:
```bash
# SSH to HPC
ssh cluster.example.com

# Navigate to repo
cd /path/to/litkit

# Fetch latest changes
git fetch origin

# Checkout the branch (first time)
git checkout fix/multi-node-producer-consumer

# Or pull updates (subsequent times)
git pull origin fix/multi-node-producer-consumer

# Verify files are present
ls -lh *.md *.sbatch
# Should see: PRODUCER_CONSUMER_ANALYSIS.md, MULTI_NODE_TESTING_GUIDE.md,
#            SUMMARY.md, HPC_DEPLOYMENT_GUIDE.md, vector_build_multi.sbatch
```

### Phase 2: Implement Bug Fixes (Laptop)

**On Laptop** (after implementing fixes in src/litkit/cli.py):
```bash
cd /path/to/litkit

# Check what changed
git status
git diff src/litkit/cli.py

# Add and commit
git add src/litkit/cli.py
git commit -m "Fix: Implement atomic segment writing and coordination protocol

- Add state machine for segment files (.tmp -> .writing -> final)
- Defer DB commit until after segment is written and fsynced
- Add ProducerCoordinator and ConsumerCoordinator classes
- Implement completion markers for producer-consumer coordination
- Enhanced error recovery with retry limits and .failed state

Addresses bugs identified in PRODUCER_CONSUMER_ANALYSIS.md"

# Push to HPC
git push origin fix/multi-node-producer-consumer
```

**On HPC** (to get the fixes):
```bash
cd /path/to/litkit

# Pull the fixes
git pull origin fix/multi-node-producer-consumer

# Verify the changes
git log -1 --stat
git diff HEAD~1 src/litkit/cli.py | head -50
```

### Phase 3: Test and Iterate

**On HPC** (after testing, if you make HPC-specific tweaks):
```bash
cd /path/to/litkit

# Example: tune SLURM script
vim vector_build_multi.sbatch

# Commit the changes
git add vector_build_multi.sbatch
git commit -m "HPC: Increase GPU memory allocation for V100s"

# Push back to laptop
git push origin fix/multi-node-producer-consumer
```

**On Laptop** (to sync HPC changes):
```bash
cd /path/to/litkit

# Pull the cluster's changes
git pull origin fix/multi-node-producer-consumer

# Review what changed
git log -1 --stat
```

## Pre-Testing Checklist

Before running multi-node tests on HPC:

- [ ] Latest code pushed from laptop to HPC
- [ ] Branch `fix/multi-node-producer-consumer` checked out on HPC
- [ ] Container image exists: `ls -lh /path/to/litkit/sqfs/litkit-v0.3.33-aarch64-lean.sqfs`
- [ ] CDI spec exists: `ls -lh /path/to/cdi-v100/nvidia.json`
- [ ] CUDA host libs exist: `ls -lh /path/to/cuda-12.5-host/cuda-12.5/`
- [ ] HF cache exists: `ls -lh /path/to/hf_cache_persist/`
- [ ] .safetensors converted: `./setup_safetensors.sh` (run once)
- [ ] Workspace clean: `rm -rf workspace/{sqlite,indices,emb_segments}/*` (optional)

## Testing on HPC

### Test 1: Interactive Single-Node Test

**Purpose**: Verify container setup and basic functionality

```bash
# On HPC, request interactive V100 node
salloc -N1 -t 10:00:00 -p gpu-v100 --no-shell

# Get the node name
squeue --me -h -o "%N"
# Example: gpu-node1

# SSH to the node
ssh gpu-node1

# Navigate to litkit
cd /path/to/litkit

# Load Charliecloud
module purge
module load charliecloud/0.42

# Set up environment
export CDI_SPEC_DIR=/path/to/cdi-v100
export CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export IMG="$(pwd)/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export HF_HOST=/path/to/hf_cache_persist

# Test: Verify PyTorch + CUDA
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  "$IMG" -- \
  /root/.local/share/uv/tools/litkit/bin/python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Test: Run litkit version check
ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$IMG" -- \
  litkit --version

# When done, release allocation
exit  # from node
scancel <jobid>  # release allocation
```

### Test 2: Two-Node SLURM Job

**Purpose**: Test producer-consumer with minimal setup

```bash
# On HPC login node
cd /path/to/litkit

# Clean workspace (optional)
rm -rf workspace/{sqlite,indices,emb_segments}/*

# Submit 2-node job
sbatch -N 2 vector_build_multi.sbatch

# Check job status
squeue --me

# Get job ID
JOBID=$(squeue --me -h -o "%i" | tail -1)
echo "Job ID: $JOBID"

# Monitor output in real-time
tail -f litkit_multi_${JOBID}.out

# Check for errors
tail -f litkit_multi_${JOBID}.err

# After job completes, check results
ls -lh workspace/emb_segments/  # Should be empty (all consumed)
ls -lh workspace/indices/        # Should have papers.index and chunks.index
sqlite3 workspace/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*), SUM(in_index) FROM papers; SELECT COUNT(*), SUM(in_index) FROM chunks;"
```

### Test 3: Multi-Node Stress Test

**Purpose**: Test with 4+ nodes, more data

```bash
# On HPC
cd /path/to/litkit

# Clean workspace
rm -rf workspace/{sqlite,indices,emb_segments}/*

# Submit 4-node job (3 producers + 1 consumer)
sbatch -N 4 vector_build_multi.sbatch

# Monitor as above
JOBID=$(squeue --me -h -o "%i" | tail -1)
tail -f litkit_multi_${JOBID}.out

# Check GPU utilization while running
for n in $(scontrol show hostnames $(squeue -j $JOBID -h -o %N)); do
  srun --jobid=$JOBID --overlap -N1 -n1 -w $n -c1 --cpu-bind=none \
    bash -lc 'echo === $(hostname) ===; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; echo'
done
```

## Monitoring and Debugging

### Real-Time Monitoring

```bash
# Watch segment directory
watch -n 5 'ls -lh /path/to/litkit/workspace/emb_segments/ | head -20'

# Monitor DB stats
watch -n 10 'sqlite3 /path/to/litkit/workspace/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*) as total, SUM(in_index) as indexed FROM papers; \
   SELECT COUNT(*) as total, SUM(in_index) as indexed FROM chunks;"'

# Check FAISS index sizes
watch -n 30 'ls -lh /path/to/litkit/workspace/indices/'

# Monitor job status
watch -n 5 'squeue --me'
```

### Check Job Logs

```bash
# Find recent job ID
JOBID=$(sacct -X -n --name=litkit_multi --starttime=now-7days -o JobID | tail -1)

# View output
less litkit_multi_${JOBID}.out

# View errors
less litkit_multi_${JOBID}.err

# Check runtime and status
sacct -X -j "$JOBID" -o JobID,JobName%30,Partition,State,ExitCode,Start,End,Elapsed,Timelimit
```

### Common Issues

#### Issue 1: "Module not found: charliecloud"

```bash
# List available modules
module avail charliecloud

# Load correct version
module load charliecloud/0.42
```

#### Issue 2: "CDI device not found"

```bash
# Verify CDI spec exists
ls -lh /path/to/cdi-v100/nvidia.json

# List available CDI devices
nvidia-ctk cdi list --spec-dir=/path/to/cdi-v100
```

#### Issue 3: "Database is locked"

```bash
# Check if another process is using the DB
lsof workspace/sqlite/litkit.sqlite3

# Increase timeout (already set in script, but can override)
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=600000  # 10 minutes
```

#### Issue 4: Segments accumulating

```bash
# Check consumer logs
grep "Consumer" litkit_multi_*.out

# Manually trigger consumer if needed
cd /path/to/litkit
module load charliecloud/0.42

# Run consumer manually (interactive node required)
salloc -N1 -t 1:00:00 -p gpu-v100 --no-shell
ssh <node>

# ... set up environment as in Test 1 ...
# ... run consumer with ch-run ...
```

## Validation After Testing

After each test run:

```bash
cd /path/to/litkit

# 1. Check that all segments consumed
ls -lh workspace/emb_segments/
# Should be empty or only .failed files

# 2. Check FAISS indices exist and are non-empty
ls -lh workspace/indices/
# Should have papers.index and chunks.index with reasonable sizes

# 3. Verify in_index flags
sqlite3 workspace/sqlite/litkit.sqlite3 <<EOF
SELECT 'Papers:', COUNT(*) as total, SUM(in_index) as indexed FROM papers;
SELECT 'Chunks:', COUNT(*) as total, SUM(in_index) as indexed FROM chunks;
SELECT 'Unindexed papers:', COUNT(*) FROM papers WHERE in_index = 0;
SELECT 'Unindexed chunks:', COUNT(*) FROM chunks WHERE in_index = 0;
EOF

# 4. Test a query
cat > workspace/test_question.txt <<EOF
What is the treatment for HIV?
EOF

# Run retrieval test (requires valid API key)
KEY="$(head -n1 ~/.llm_api_key)"
BASE="https://llm.example.com/v1"

ch-run \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs=/path/to/cdi-v100 \
  --cdi=nvidia.com/gpu=all \
  --bind /path/to/cuda-12.5-host/cuda-12.5/targets/sbsa-linux/lib \
  --bind /path/to/cuda-12.5-host/cuda-12.5/lib64 \
  --bind "$(pwd)/workspace:/workspace" \
  --bind /path/to/PMC-OA:/path/to/PMC-OA \
  --bind /path/to/hf_cache_persist:/app/hf_cache \
  --set-env="LD_LIBRARY_PATH=/path/to/cuda-12.5-host/cuda-12.5/targets/sbsa-linux/lib:/path/to/cuda-12.5-host/cuda-12.5/lib64" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="LITKIT_TAR_DIR=/path/to/PMC-OA" \
  --set-env="HF_HOME=/app/hf_cache" \
  sqfs/litkit-v0.3.33-aarch64-lean.sqfs -- \
  litkit --no-llm --question-file /workspace/test_question.txt
```

## Performance Expectations

| Configuration | Expected Throughput |
|--------------|---------------------|
| 1 producer + 1 consumer | ~50K papers/hour |
| 3 producers + 1 consumer | ~120K papers/hour |
| 7 producers + 1 consumer | ~200K papers/hour |

## Next Steps After Testing

1. **If tests pass**:
   - Document results
   - Merge branch to main
   - Create production deployment checklist

2. **If tests fail**:
   - Review logs for error patterns
   - Check PRODUCER_CONSUMER_ANALYSIS.md for known issues
   - Iterate on fixes
   - Push updates from laptop → HPC
   - Re-test

## Quick Reference: Complete Git Sync Cycle

```bash
# === ON LAPTOP: Make changes ===
cd /path/to/litkit
git checkout fix/multi-node-producer-consumer
# ... edit files ...
git add <files>
git commit -m "Description of changes"
git push origin fix/multi-node-producer-consumer

# === ON HPC: Get changes ===
ssh cluster.example.com
cd /path/to/litkit
git pull origin fix/multi-node-producer-consumer
# ... test ...

# === ON HPC: Push changes back ===
git add <files>
git commit -m "HPC-specific changes"
git push origin fix/multi-node-producer-consumer

# === ON LAPTOP: Get HPC changes ===
cd /path/to/litkit
git pull origin fix/multi-node-producer-consumer
```

## Support

For issues or questions:
1. Review PRODUCER_CONSUMER_ANALYSIS.md for bug details
2. Check MULTI_NODE_TESTING_GUIDE.md for testing procedures
3. Review job logs in `litkit_multi_*.out` and `litkit_multi_*.err`
4. Check HPC module availability: `module avail`

---

**Last Updated**: December 1, 2025  
**Branch**: `fix/multi-node-producer-consumer`  
**Status**: Ready for HPC testing
