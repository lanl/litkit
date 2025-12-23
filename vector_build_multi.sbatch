#!/bin/bash
#SBATCH --job-name=litkit_multi
#SBATCH --partition=gpu-v100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
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
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"  # Shared NFS workspace (consumer reads from here)

# LOCAL STAGING: Use node-local SSD to avoid NFS metadata storm
# Evidence: 47× faster for small-file creates (see BOTTLENECK.md)
export LOCAL_SSD="/local/scratch"
export USE_LOCAL_STAGING="${USE_LOCAL_STAGING:-1}"  # Set to 0 to disable

# MANIFEST selection (edit this ONE LINE to change which manifest is used):
#   tiny_test.manifest    (~few MB, 5-10 min) - for debugging merge logic
#   medium_test.manifest  (~7 GB, 30-45 min)
#   large_test.manifest   (~185 GB, 3-5 hours)
MANIFEST_FILE="${MANIFEST:-medium_test.manifest}"
export LITKIT_TAR_MANIFEST="${LITKIT_WORKSPACE}/${MANIFEST_FILE}"

# Container and CUDA setup for V100 nodes
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.34-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

# Parallel parsing workers per producer (use available CPUs)
export PARSE_WORKERS=16

# Debug mode for parallelism verification (set to 0 for quiet production runs)
export LITKIT_DEBUG="${LITKIT_DEBUG:-1}"

# Ensure directories exist on Lustre
mkdir -p "${LITKIT_WORKSPACE}/emb_segments"
mkdir -p "${LITKIT_WORKSPACE}/sqlite"
mkdir -p "${LITKIT_WORKSPACE}/indices"

# Lustre-optimized settings
export LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000
export LITKIT_SAVE_EVERY_SEC=300

# Threading clamps to prevent oversubscription (OMP, MKL, etc.)
# With --parse-workers 16, hidden threading would oversubscribe the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Fail-fast: Multi-node build requires at least 2 nodes (1 producer + 1 consumer)
if [[ "$SLURM_NNODES" -lt 2 ]]; then
    echo "FATAL: vector_build_multi.sbatch requires at least 2 nodes (-N2 or higher)." >&2
    echo "       With -N1, there are 0 producers and the consumer waits forever." >&2
    echo "       Use vector_build_single.sbatch for single-node builds." >&2
    exit 2
fi

### ----- CLEAN STALE WRITER GUARD (SLURM RESTART SAFE) ------------------------
# If a prior job crashed on a different node, the guard file may persist.
# Cross-host auto-eviction is disabled for safety (we can't verify remote PID).
# This unconditional cleanup is safe BECAUSE:
#   1. This runs at job START (no concurrent writer for this workspace)
#   2. Any prior writer is dead (SLURM guarantees our allocation is exclusive)
# WARNING: Do NOT add this mid-job or if running concurrent writers.
rm -f "${LITKIT_WORKSPACE}/.writer_guard" 2>/dev/null || true

echo "=== Multi-Node Litkit Build Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Nodelist: $SLURM_NODELIST"
echo "Workspace: $LITKIT_WORKSPACE"
echo "TAR manifest: $LITKIT_TAR_MANIFEST"
echo "Container: $IMG"
echo "Parse workers per producer: $PARSE_WORKERS"
echo "Chunks index: flat (no IVF-PQ training)"
echo "======================================"

# Function to run producer in Charliecloud container
# If USE_LOCAL_STAGING=1, writes to node-local SSD then rsyncs to NFS
run_producer() {
    local node_id=$1
    local target_node=$2
    local num_shards=$((SLURM_NNODES - 1))
    
    echo "[Producer $node_id] Starting on $target_node with $PARSE_WORKERS parse workers"
    echo "[Producer $node_id] Local staging: USE_LOCAL_STAGING=${USE_LOCAL_STAGING}"
    
    # Compute paths OUTSIDE srun to avoid shell escaping issues
    local LOCAL_STAGE_HOST=""
    local BIND_LOCAL_ARG=""
    local LITKIT_WS_CONTAINER="/workspace"
    local EMBED_OUTDIR_CONTAINER="/workspace/emb_segments"
    local NFS_SQLITE_DEST="${LITKIT_WORKSPACE}/sqlite"
    local NFS_SEGMENTS_DEST="${LITKIT_WORKSPACE}/emb_segments"
    
    if [[ "$USE_LOCAL_STAGING" == "1" ]]; then
        LOCAL_STAGE_HOST="${LOCAL_SSD}/litkit_${SLURM_JOB_ID}_shard${node_id}"
        BIND_LOCAL_ARG="--bind ${LOCAL_STAGE_HOST}:/local_stage"
        LITKIT_WS_CONTAINER="/local_stage"
        EMBED_OUTDIR_CONTAINER="/local_stage/emb_segments"
        # Use shard-specific NFS destinations to avoid collisions
        NFS_SQLITE_DEST="${LITKIT_WORKSPACE}/sqlite/shard_${node_id}"
        NFS_SEGMENTS_DEST="${LITKIT_WORKSPACE}/emb_segments/shard_${node_id}"
    fi
    
    # Export variables for srun to propagate
    export PRODUCER_NODE_ID="$node_id"
    export PRODUCER_NUM_SHARDS="$num_shards"
    export PRODUCER_LOCAL_STAGE="$LOCAL_STAGE_HOST"
    export PRODUCER_BIND_LOCAL="$BIND_LOCAL_ARG"
    export PRODUCER_LITKIT_WS="$LITKIT_WS_CONTAINER"
    export PRODUCER_EMBED_OUTDIR="$EMBED_OUTDIR_CONTAINER"
    export PRODUCER_NFS_SQLITE_DEST="$NFS_SQLITE_DEST"
    export PRODUCER_NFS_SEGMENTS_DEST="$NFS_SEGMENTS_DEST"
    
    # FIX: Request CPUs for srun step
    srun -N1 -n1 -w "$target_node" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --cpu-bind=cores \
        --export=ALL \
        bash -c '
            set -euo pipefail
            
            # Setup local staging if enabled
            if [[ -n "$PRODUCER_LOCAL_STAGE" ]]; then
                echo "[staging] Using local SSD: $PRODUCER_LOCAL_STAGE"
                mkdir -p "$PRODUCER_LOCAL_STAGE/sqlite" "$PRODUCER_LOCAL_STAGE/emb_segments"
                # Clean any stale writer guard from prior failed runs
                rm -f "$PRODUCER_LOCAL_STAGE/.writer_guard" 2>/dev/null || true
            else
                echo "[staging] Using NFS directly (no local staging)"
            fi
            
            # Validate manifest exists before running
            if [[ ! -f "${LITKIT_WORKSPACE}/${MANIFEST_FILE}" ]]; then
                echo "FATAL: Manifest not found: ${LITKIT_WORKSPACE}/${MANIFEST_FILE}" >&2
                exit 1
            fi
            
            # Build ch-run command array to handle optional --bind correctly
            CHRUN_ARGS=(
                "$CHRUN"
                --unset-env="*"
                --set-env=HOME=/root
                --cdi-dirs="$CDI_SPEC_DIR"
                --cdi=nvidia.com/gpu=all
                --bind "$CUDA_LIBA:$CUDA_LIBA"
                --bind "$CUDA_LIBB:$CUDA_LIBB"
                --bind /path/to/PMC-OA:/path/to/PMC-OA
                --bind "${LITKIT_WORKSPACE}:/workspace"
                --bind "$HF_HOST:/app/hf_cache"
            )
            
            # Add local staging bind ONLY if staging is enabled
            if [[ -n "$PRODUCER_BIND_LOCAL" ]]; then
                # shellcheck disable=SC2206
                CHRUN_ARGS+=($PRODUCER_BIND_LOCAL)
            fi
            
            CHRUN_ARGS+=(
                --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB"
                --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                --set-env="LITKIT_WORKSPACE=$PRODUCER_LITKIT_WS"
                --set-env="HF_HOME=/app/hf_cache"
                --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1"
                --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}"
                --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}"
                --set-env="LITKIT_DEBUG=${LITKIT_DEBUG}"
                --set-env="OMP_NUM_THREADS=1"
                --set-env="MKL_NUM_THREADS=1"
                --set-env="OPENBLAS_NUM_THREADS=1"
                --set-env="TOKENIZERS_PARALLELISM=false"
                "$IMG"
                --
                bash -c "
                    echo \"[diag:producer] Cpus_allowed=\$(grep Cpus_allowed_list /proc/self/status | cut -f2)\"
                    python3 -c \"import os; aff=os.sched_getaffinity(0); print(\\\"[diag:producer] affinity_cpus=\\\", len(aff))\"
                    echo \"[diag:producer] GPUs: \$(nvidia-smi -L 2>/dev/null | head -2 || echo nvidia-smi-not-found)\"
                    echo \"[diag:producer] LITKIT_WORKSPACE=\$LITKIT_WORKSPACE\"
                    
                    # Validate manifest inside container
                    if [[ ! -f \"/workspace/${MANIFEST_FILE}\" ]]; then
                        echo \"FATAL: Manifest not found inside container: /workspace/${MANIFEST_FILE}\" >&2
                        exit 1
                    fi
                    
                    exec litkit \\
                        --embed-producer \\
                        --build-only \\
                        --shard-id ${PRODUCER_NODE_ID} \\
                        --num-shards ${PRODUCER_NUM_SHARDS} \\
                        --embed-outdir \"${PRODUCER_EMBED_OUTDIR}\" \\
                        --paper-embed-bs 16 \\
                        --chunk-embed-bs 64 \\
                        --embed-devices \"cuda:0,cuda:1\" \\
                        --embed-workers 2 \\
                        --parse-workers ${PARSE_WORKERS} \\
                        --tar-manifest \"/workspace/${MANIFEST_FILE}\"
                "
            )
            
            # Execute ch-run
            "${CHRUN_ARGS[@]}"
            
            # Rsync local results back to NFS (if staging was used)
            if [[ -n "$PRODUCER_LOCAL_STAGE" ]]; then
                echo "[staging] Syncing local outputs to NFS (shard-specific destinations)..."
                mkdir -p "$PRODUCER_NFS_SQLITE_DEST" "$PRODUCER_NFS_SEGMENTS_DEST"
                rsync -a --delay-updates "$PRODUCER_LOCAL_STAGE/sqlite/" "$PRODUCER_NFS_SQLITE_DEST/"
                rsync -a --delay-updates "$PRODUCER_LOCAL_STAGE/emb_segments/" "$PRODUCER_NFS_SEGMENTS_DEST/"
                echo "[staging] Cleaning up local staging dir..."
                rm -rf "$PRODUCER_LOCAL_STAGE"
                echo "[staging] Rsync complete for shard ${PRODUCER_NODE_ID}"
            fi
        '
    
    echo "[Producer $node_id] Completed"
}


# Get list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_NODELIST))
echo "Allocated nodes: ${NODELIST[*]}"

# Bootstrap phase: Create empty FAISS indices if they don't exist
CONSUMER_NODE="${NODELIST[$((SLURM_NNODES - 1))]}"
echo ""
echo "=== Bootstrapping Empty FAISS Indices ==="
echo "Running fast bootstrap on consumer node: $CONSUMER_NODE"
echo "Using --chunks-index flat (no IVF-PQ training)"
echo "This should complete in < 30 seconds..."

# FIX: Request CPUs for srun step
srun -N1 -n1 -w "$CONSUMER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    --export=ALL \
    "$CHRUN" \
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
    --set-env="LITKIT_DEBUG=${LITKIT_DEBUG}" \
    --set-env="OMP_NUM_THREADS=1" \
    --set-env="MKL_NUM_THREADS=1" \
    --set-env="OPENBLAS_NUM_THREADS=1" \
    --set-env="TOKENIZERS_PARALLELISM=false" \
    "$IMG" -- \
    bash -c '
        echo "[diag:bootstrap] Cpus_allowed=$(grep Cpus_allowed_list /proc/self/status | cut -f2)"
        python3 -c "import os; aff=os.sched_getaffinity(0); print(\"[diag:bootstrap] affinity_cpus=\", len(aff))"
        exec litkit \
            --faiss-writer \
            --init-indices-only \
            --papers-index hnsw \
            --chunks-index flat
    '

echo "=== Bootstrap Complete (empty indices created) ==="
echo ""

# Wait for filesystem sync
sleep 5

# Run producers on all nodes except the last one
NUM_PRODUCERS=$((SLURM_NNODES - 1))
echo "Starting $NUM_PRODUCERS producers on nodes: ${NODELIST[@]:0:$NUM_PRODUCERS}"

for ((i=0; i<NUM_PRODUCERS; i++))
do
    run_producer $i "${NODELIST[$i]}" &
done

# Wait a bit for producers to start
sleep 10

# Run consumer on the last node (already set above in bootstrap)
# NOTE: With local staging, segments are in /workspace/emb_segments/shard_X/
# The consumer scans --embed-outdir recursively, so this still works.
echo "Starting consumer on node: $CONSUMER_NODE"
# FIX: Request CPUs for srun step
srun -N1 -n1 -w "$CONSUMER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    --export=ALL \
    "$CHRUN" \
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
    --set-env="LITKIT_DEBUG=${LITKIT_DEBUG}" \
    --set-env="OMP_NUM_THREADS=1" \
    --set-env="MKL_NUM_THREADS=1" \
    --set-env="OPENBLAS_NUM_THREADS=1" \
    --set-env="TOKENIZERS_PARALLELISM=false" \
    "$IMG" -- \
    bash -c '
        echo "[diag:consumer] Cpus_allowed=$(grep Cpus_allowed_list /proc/self/status | cut -f2)"
        python3 -c "import os; aff=os.sched_getaffinity(0); print(\"[diag:consumer] affinity_cpus=\", len(aff))"
        echo "[diag:consumer] Segment directories:"
        ls -la /workspace/emb_segments/ 2>/dev/null || echo "  (none yet)"
        exec litkit \
            --faiss-writer \
            --consume-only \
            --embed-outdir "/workspace/emb_segments" \
            --tar-manifest "/workspace/'"${MANIFEST_FILE}"'" \
            --papers-index hnsw \
            --chunks-index flat \
            --num-shards '"$((SLURM_NNODES - 1))"'
    '

# Wait for all background jobs to finish
echo "Waiting for all jobs to complete..."
wait

echo "=== Multi-Node Litkit Build Complete ==="
echo "Check workspace: $LITKIT_WORKSPACE"
echo "SQLite: ${LITKIT_WORKSPACE}/sqlite/litkit.sqlite3"
echo "Indices: ${LITKIT_WORKSPACE}/indices/"
