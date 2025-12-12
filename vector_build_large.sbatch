#!/bin/bash
#SBATCH --job-name=litkit_large
#SBATCH --partition=gpu-v100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --output=litkit_large_%j.out
#SBATCH --error=litkit_large_%j.err
#SBATCH --export=NONE

set -euo pipefail

# Load Charliecloud module
module purge
module load charliecloud/0.42 2>/dev/null || true

# Pick an aarch64 ch-run
want_arch="$(uname -m)"
echo "[arch] Compute node architecture: $want_arch"

declare -a CHRUN_CANDIDATES=()
command -v ch-run >/dev/null 2>&1 && CHRUN_CANDIDATES+=("$(command -v ch-run)")
for p in \
  "/opt/site/centos8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "/opt/site/rhel8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "/opt/site-nv/rhel8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "/usr/bin/ch-run" "/usr/local/bin/ch-run"
do
  [[ -x "$p" ]] && CHRUN_CANDIDATES+=("$p")
done

CHRUN=""
for p in "${CHRUN_CANDIDATES[@]}"; do
  if file -L "$p" 2>/dev/null | grep -qi "$want_arch"; then
    CHRUN="$p"
    echo "[charliecloud] Found arch-correct ch-run: $CHRUN"
    break
  fi
done

if [[ -z "$CHRUN" ]]; then
  echo "FATAL: no arch-correct ch-run ($want_arch) found." >&2
  exit 2
fi

# HPC-specific paths
export LITKIT_REPO="/path/to/litkit"
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"

# Large test manifest: ~30 GB across 3 files (10 GB each)
export LITKIT_TAR_MANIFEST="${LITKIT_WORKSPACE}/large_test.manifest"

# Container and CUDA setup for V100 nodes
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

# Clean up previous run artifacts
rm -rf "${LITKIT_WORKSPACE}/emb_segments"/*
rm -rf "${LITKIT_WORKSPACE}/sqlite"/*
rm -rf "${LITKIT_WORKSPACE}/indices"/*

# Ensure directories exist
mkdir -p "${LITKIT_WORKSPACE}/emb_segments"
mkdir -p "${LITKIT_WORKSPACE}/sqlite"
mkdir -p "${LITKIT_WORKSPACE}/indices"

# Lustre-optimized settings
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000
export LITKIT_SAVE_EVERY_SEC=300

echo "============================================================"
echo "Large-Scale Multi-Node LitKit Build"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Workspace: $LITKIT_WORKSPACE"
echo "TAR manifest: $LITKIT_TAR_MANIFEST"
echo "Container: $IMG"
echo "Time limit: 10 hours"
echo ""
echo "Expected: ~30 GB data, 3 producers + 1 consumer"
echo "All 6 producer GPUs should be active simultaneously!"
echo "============================================================"

# Function to run producer
run_producer() {
    local node_id=$1
    local target_node=$2
    echo "[Producer $node_id] Starting on $target_node"
    
    srun -N1 -n1 -w "$target_node" "$CHRUN" \
        --unset-env='*' \
        --set-env=HOME=/root \
        --cdi-dirs="$CDI_SPEC_DIR" \
        --cdi=nvidia.com/gpu=all \
        --bind "$CUDA_LIBA:$CUDA_LIBA" \
        --bind "$CUDA_LIBB:$CUDA_LIBB" \
        --bind /path/to/PMC-OA:/path/to/PMC-OA \
        --bind "${LITKIT_WORKSPACE}:/workspace" \
        --bind "$HF_HOST:/app/hf_cache" \
        --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
        --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        --set-env="LITKIT_WORKSPACE=/workspace" \
        --set-env="HF_HOME=/app/hf_cache" \
        --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
        --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}" \
        --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}" \
        "$IMG" -- \
        litkit \
            --embed-producer \
            --build-only \
            --shard-id $node_id \
            --num-shards $((SLURM_NNODES - 1)) \
            --embed-outdir "/workspace/emb_segments" \
            --paper-embed-bs 16 \
            --chunk-embed-bs 64 \
            --embed-devices "cuda:0,cuda:1" \
            --embed-workers 2 \
            --tar-manifest "/workspace/large_test.manifest"
    
    echo "[Producer $node_id] Completed on $target_node"
}

# Get list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_NODELIST))
echo "Allocated nodes: ${NODELIST[*]}"

# Consumer node is the last one
CONSUMER_NODE="${NODELIST[$((SLURM_NNODES - 1))]}"
NUM_PRODUCERS=$((SLURM_NNODES - 1))

echo ""
echo "=== Phase 1: Bootstrap Empty FAISS Indices ==="
echo "Running on consumer node: $CONSUMER_NODE"

srun -N1 -n1 -w "$CONSUMER_NODE" "$CHRUN" \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind /path/to/PMC-OA:/path/to/PMC-OA \
    --bind "${LITKIT_WORKSPACE}:/workspace" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="LITKIT_WORKSPACE=/workspace" \
    --set-env="HF_HOME=/app/hf_cache" \
    --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
    --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}" \
    --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}" \
    "$IMG" -- \
    litkit \
        --faiss-writer \
        --init-indices-only \
        --papers-index hnsw \
        --chunks-index flat

echo "=== Bootstrap Complete ==="
echo ""

# Wait for filesystem sync
sleep 5

echo "=== Phase 2: Starting $NUM_PRODUCERS Producers ==="
echo "Producer nodes: ${NODELIST[@]:0:$NUM_PRODUCERS}"
echo ""
echo ">>> Monitor GPU usage with:"
echo ">>> for n in ${NODELIST[@]:0:$NUM_PRODUCERS}; do ssh \$n nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done"
echo ""

# Start all producers in background
for ((i=0; i<NUM_PRODUCERS; i++))
do
    run_producer $i "${NODELIST[$i]}" &
done

# Wait for producers to start
sleep 15

echo "=== Phase 3: Starting Consumer ==="
echo "Consumer node: $CONSUMER_NODE"

srun -N1 -n1 -w "$CONSUMER_NODE" "$CHRUN" \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind /path/to/PMC-OA:/path/to/PMC-OA \
    --bind "${LITKIT_WORKSPACE}:/workspace" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="LITKIT_WORKSPACE=/workspace" \
    --set-env="HF_HOME=/app/hf_cache" \
    --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
    --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}" \
    --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}" \
    "$IMG" -- \
    litkit \
        --faiss-writer \
        --consume-only \
        --embed-outdir "/workspace/emb_segments" \
        --tar-manifest "/workspace/large_test.manifest" \
        --papers-index hnsw \
        --chunks-index flat \
        --num-shards $NUM_PRODUCERS

# Wait for all background jobs
echo "Waiting for all producers to complete..."
wait

echo ""
echo "============================================================"
echo "Large-Scale Build Complete"
echo "============================================================"
echo "Check workspace: $LITKIT_WORKSPACE"
echo "SQLite: ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3"
echo "Indices: ${LITKIT_WORKSPACE}/indices/"
echo "============================================================"
