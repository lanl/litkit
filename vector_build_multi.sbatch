#!/bin/bash
#SBATCH --job-name=litkit_multi
#SBATCH --partition=gpu-v100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=litkit_multi_%j.out
#SBATCH --error=litkit_multi_%j.err
#SBATCH --export=NONE

set -euo pipefail

# Load Charliecloud module (but don't trust it blindly)
module purge
module load charliecloud/0.42 2>/dev/null || true

# Pick an aarch64 ch-run (avoid "Exec format error")
want_arch="$(uname -m)"
echo "[arch] Compute node architecture: $want_arch"

declare -a CHRUN_CANDIDATES=()
command -v ch-run >/dev/null 2>&1 && CHRUN_CANDIDATES+=("$(command -v ch-run)")
# Common site layouts by arch
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
  echo "FATAL: no arch-correct ch-run ($want_arch) found. Searched: ${CHRUN_CANDIDATES[*]-<none>}" >&2
  exit 2
fi

# HPC-specific paths
export LITKIT_REPO="/path/to/litkit"
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"
export LITKIT_TAR_DIR="/path/to/PMC-OA"

# Container and CUDA setup for V100 nodes
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

# Ensure directories exist on Lustre
mkdir -p "${LITKIT_WORKSPACE}/emb_segments"
mkdir -p "${LITKIT_WORKSPACE}/sqlite"
mkdir -p "${LITKIT_WORKSPACE}/indices"

# Lustre-optimized settings
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000
export LITKIT_SAVE_EVERY_SEC=300

echo "=== Multi-Node Litkit Build Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Workspace: $LITKIT_WORKSPACE"
echo "TAR dir: $LITKIT_TAR_DIR"
echo "Container: $IMG"
echo "======================================"

# Function to run producer in Charliecloud container
run_producer() {
    local node_id=$1
    echo "[Producer $node_id] Starting on $(hostname)"
    
    srun -N1 -n1 "$CHRUN" \
        --unset-env='*' \
        --set-env=HOME=/root \
        --cdi-dirs="$CDI_SPEC_DIR" \
        --cdi=nvidia.com/gpu=all \
        --bind "$CUDA_LIBA:$CUDA_LIBA" \
        --bind "$CUDA_LIBB:$CUDA_LIBB" \
        --bind "${LITKIT_TAR_DIR}:${LITKIT_TAR_DIR}" \
        --bind "${LITKIT_WORKSPACE}:/workspace" \
        --bind "$HF_HOST:/app/hf_cache" \
        --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
        --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        --set-env="LITKIT_WORKSPACE=/workspace" \
        --set-env="LITKIT_TAR_DIR=${LITKIT_TAR_DIR}" \
        --set-env="HF_HOME=/app/hf_cache" \
        --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
        --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}" \
        --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}" \
        "$IMG" -- \
        litkit \
            --embed-producer \
            --shard-id $node_id \
            --num-shards $((SLURM_NNODES - 1)) \
            --embed-outdir "/workspace/emb_segments" \
            --paper-embed-bs 16 \
            --chunk-embed-bs 64 \
            --embed-devices "cuda:0,cuda:1" \
            --embed-workers 2 \
            --build-only
    
    echo "[Producer $node_id] Completed"
}

# Function to run consumer in Charliecloud container
run_consumer() {
    echo "[Consumer] Starting on $(hostname)"
    
    # Consumer polls and ingests segments until all producers complete
    local poll_count=0
    local max_polls=1000  # ~8 hours at 30s intervals
    
    while [ $poll_count -lt $max_polls ]; do
        srun -N1 -n1 "$CHRUN" \
            --unset-env='*' \
            --set-env=HOME=/root \
            --cdi-dirs="$CDI_SPEC_DIR" \
            --cdi=nvidia.com/gpu=all \
            --bind "$CUDA_LIBA:$CUDA_LIBA" \
            --bind "$CUDA_LIBB:$CUDA_LIBB" \
            --bind "${LITKIT_TAR_DIR}:${LITKIT_TAR_DIR}" \
            --bind "${LITKIT_WORKSPACE}:/workspace" \
            --bind "$HF_HOST:/app/hf_cache" \
            --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
            --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
            --set-env="LITKIT_WORKSPACE=/workspace" \
            --set-env="LITKIT_TAR_DIR=${LITKIT_TAR_DIR}" \
            --set-env="HF_HOME=/app/hf_cache" \
            --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
            --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}" \
            --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}" \
            "$IMG" -- \
            litkit \
                --faiss-writer \
                --consume-segments \
                --embed-outdir "/workspace/emb_segments" \
                --reconcile-only
        
        # Check if any producers still running or segments remaining
        local producers_running=$(jobs -r | grep -c "run_producer" || true)
        local segments_remaining=$(ls "${LITKIT_WORKSPACE}/emb_segments"/*.npz 2>/dev/null | wc -l)
        
        if [ $producers_running -eq 0 ] && [ $segments_remaining -eq 0 ]; then
            echo "[Consumer] All producers complete and all segments ingested"
            break
        fi
        
        echo "[Consumer] Poll $poll_count: $producers_running producers running, $segments_remaining segments remaining"
        sleep 30
        poll_count=$((poll_count + 1))
    done
    
    echo "[Consumer] Completed"
}

# Run producers on all nodes except the last one
NUM_PRODUCERS=$((SLURM_NNODES - 1))
echo "Starting $NUM_PRODUCERS producers..."

for ((i=0; i<NUM_PRODUCERS; i++))
do
    run_producer $i &
done

# Wait a bit for producers to start
sleep 10

# Run consumer on the last node
echo "Starting consumer..."
run_consumer &

# Wait for all background jobs to finish
echo "Waiting for all jobs to complete..."
wait

echo "=== Multi-Node Litkit Build Complete ==="
echo "Check workspace: $LITKIT_WORKSPACE"
echo "SQLite: ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3"
echo "Indices: ${LITKIT_WORKSPACE}/indices/"
