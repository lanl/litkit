# Rebuild Container on HPC with New --init-indices-only Flag

## Problem
The container `litkit-v0.3.33-aarch64-lean.sqfs` doesn't have the new `--init-indices-only` flag.

## Solution: Build on Compute Node (just is only available on gpu-v100 partition)

### Step 1: Pull Latest Code (on login node)
```bash
cd /path/to/litkit
git fetch origin
git checkout fix/multi-node-producer-consumer
git pull origin fix/multi-node-producer-consumer
```

### Step 2: Request Interactive Compute Node
```bash
# Request a node from gpu-v100 partition
salloc -p gpu-v100 -N 1 --time=2:00:00 --cpus-per-task=16

# Wait for allocation message showing node name (e.g., "Granted job allocation 16797XXX on node gpu-node1")
```

### Step 3: SSH to Allocated Node
```bash
# Replace gpu-node1 with your actual allocated node name
ssh gpu-node1
```

### Step 4: Build Container (on compute node)
```bash
cd /path/to/litkit

# Verify just is available
which just
just --version

# Load required modules
module purge
module load charliecloud/0.42

# Build the container using justfile
just build
```

This will create: `sqfs/litkit-v0.3.34-aarch64-lean.sqfs`

**Note:** The build takes 30-60 minutes. The `just build` command:
- Stages pre-built Torch wheels if available
- Builds the container image with `ch-image`
- Converts to SquashFS format
- Uses the version tag from the justfile (currently v0.3.33, but you can edit it to v0.3.34)

### Step 5: Exit Compute Node
```bash
exit  # Return to login node
exit  # Release the salloc allocation (or Ctrl+D)
```

### Step 6: Update SLURM Script (on login node)
Edit `vector_build_multi.sbatch` and change:
```bash
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
```
to:
```bash
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.34-aarch64-lean.sqfs"
```

### Step 7: Test Bootstrap Locally (on login node)
Before submitting the full job, test that the new flag works:
```bash
ch-run sqfs/litkit-v0.3.34-aarch64-lean.sqfs -- \
  litkit --init-indices-only --papers-index hnsw --chunks-index flat
```

You should see it create empty indices and exit in < 30 seconds.

### Step 8: Submit Job
```bash
sbatch vector_build_multi.sbatch
```

## Notes
- `just` is only available on gpu-v100 compute nodes, not on login nodes
- The justfile is designed for building on HPC and handles all the `ch-image` commands
- Building may take 30-60 minutes depending on whether a pre-built Torch wheel is available
- Before building, you may want to update the version in `justfile` from `v0.3.33` to `v0.3.34` (line ~37)
