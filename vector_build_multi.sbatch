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
#
# Key design principles:
#   - Race-safe: no shared global exports; everything passed via srun --export=...
#   - Quoting-safe: no word-splitting of "--bind ..." strings
#   - Explicit: no dependence on "host vars magically existing" inside srun shell
#
run_producer() {
    local node_id="$1"
    local target_node="$2"
    local num_shards="$((SLURM_NNODES - 1))"

    echo "[Producer ${node_id}] Starting on ${target_node} with ${PARSE_WORKERS} parse workers"
    echo "[Producer ${node_id}] Local staging: USE_LOCAL_STAGING=${USE_LOCAL_STAGING}"

    # Host-side sanity checks (fail early, deterministically)
    if [[ ! -f "${LITKIT_WORKSPACE}/${MANIFEST_FILE}" ]]; then
        echo "FATAL: Manifest not found on host: ${LITKIT_WORKSPACE}/${MANIFEST_FILE}" >&2
        return 1
    fi

    # Compute local staging path (host path on the compute node)
    local local_stage_host=""
    if [[ "${USE_LOCAL_STAGING}" == "1" ]]; then
        local_stage_host="${LOCAL_SSD}/litkit_${SLURM_JOB_ID}_shard${node_id}"
    fi

    # Run on the target node. Everything needed is passed explicitly via --export.
    srun -N1 -n1 -w "${target_node}" \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        --cpu-bind=cores \
        --export=ALL,PRODUCER_NODE_ID="${node_id}",PRODUCER_NUM_SHARDS="${num_shards}",PRODUCER_LOCAL_STAGE_HOST="${local_stage_host}",PRODUCER_USE_LOCAL_STAGING="${USE_LOCAL_STAGING}",PRODUCER_MANIFEST_FILE="${MANIFEST_FILE}",PRODUCER_PARSE_WORKERS="${PARSE_WORKERS}",PRODUCER_JOB_ID="${SLURM_JOB_ID}",CHRUN="${CHRUN}",IMG="${IMG}",CDI_SPEC_DIR="${CDI_SPEC_DIR}",CUDA_LIBA="${CUDA_LIBA}",CUDA_LIBB="${CUDA_LIBB}",HF_HOST="${HF_HOST}",LITKIT_WORKSPACE="${LITKIT_WORKSPACE}",LITKIT_SQLITE_BUSY_TIMEOUT_MS="${LITKIT_SQLITE_BUSY_TIMEOUT_MS}",LITKIT_SAVE_EVERY_SEC="${LITKIT_SAVE_EVERY_SEC}",LITKIT_DEBUG="${LITKIT_DEBUG}" \
        bash -lc '
            set -euo pipefail

            # Decide container workspace and embed outdir
            CONTAINER_NFS_WS="/workspace"
            if [[ "${PRODUCER_USE_LOCAL_STAGING}" == "1" ]]; then
                CONTAINER_WS="/local_stage"
                EMBED_OUTDIR="${CONTAINER_WS}/emb_segments"
            else
                CONTAINER_WS="${CONTAINER_NFS_WS}"
                EMBED_OUTDIR="${CONTAINER_NFS_WS}/emb_segments"
            fi

            # Prep local staging (host FS) if enabled
            if [[ "${PRODUCER_USE_LOCAL_STAGING}" == "1" ]]; then
                echo "[Producer ${PRODUCER_NODE_ID}] [staging] Using node-local SSD: ${PRODUCER_LOCAL_STAGE_HOST}"
                mkdir -p "${PRODUCER_LOCAL_STAGE_HOST}/sqlite" "${PRODUCER_LOCAL_STAGE_HOST}/emb_segments"
                rm -f "${PRODUCER_LOCAL_STAGE_HOST}/.writer_guard" 2>/dev/null || true
            else
                echo "[Producer ${PRODUCER_NODE_ID}] [staging] Using NFS directly (no local staging)"
            fi

            # Build ch-run argv as a real array (no fragile word-splitting)
            chrun_argv=(
                "${CHRUN}"
                --unset-env="*"
                --set-env=HOME=/root
                --cdi-dirs="${CDI_SPEC_DIR}"
                --cdi=nvidia.com/gpu=all
                --bind "${CUDA_LIBA}:${CUDA_LIBA}"
                --bind "${CUDA_LIBB}:${CUDA_LIBB}"
                --bind /path/to/PMC-OA:/path/to/PMC-OA
                --bind "${LITKIT_WORKSPACE}:${CONTAINER_NFS_WS}"
                --bind "${HF_HOST}:/app/hf_cache"
            )

            # Optional bind for local staging (host path -> /local_stage in container)
            if [[ "${PRODUCER_USE_LOCAL_STAGING}" == "1" ]]; then
                chrun_argv+=( --bind "${PRODUCER_LOCAL_STAGE_HOST}:/local_stage" )
            fi

            # Environment + container entrypoint
            chrun_argv+=(
                --set-env="LD_LIBRARY_PATH=${CUDA_LIBA}:${CUDA_LIBB}"
                --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                --set-env="LITKIT_WORKSPACE=${CONTAINER_WS}"
                --set-env="HF_HOME=/app/hf_cache"
                --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1"
                --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=${LITKIT_SQLITE_BUSY_TIMEOUT_MS}"
                --set-env="LITKIT_SAVE_EVERY_SEC=${LITKIT_SAVE_EVERY_SEC}"
                --set-env="LITKIT_DEBUG=${LITKIT_DEBUG}"
                --set-env="OMP_NUM_THREADS=1"
                --set-env="MKL_NUM_THREADS=1"
                --set-env="OPENBLAS_NUM_THREADS=1"
                --set-env="TOKENIZERS_PARALLELISM=false"
                "${IMG}"
                --
                bash -lc
            )

            # Build litkit command (heredoc avoids nested quoting issues)
            litkit_cmd=$(cat <<EOF
echo "[diag:producer] host=\$(hostname)"
echo "[diag:producer] Cpus_allowed=\$(grep Cpus_allowed_list /proc/self/status | cut -f2)"
python3 -c "import os; aff=os.sched_getaffinity(0); print(\"[diag:producer] affinity_cpus=\", len(aff))"
echo "[diag:producer] GPUs:"; (nvidia-smi -L 2>/dev/null | head -10) || true
echo "[diag:producer] LITKIT_WORKSPACE=\$LITKIT_WORKSPACE"
test -f "${CONTAINER_NFS_WS}/${PRODUCER_MANIFEST_FILE}" || { echo "FATAL: manifest missing in container: ${CONTAINER_NFS_WS}/${PRODUCER_MANIFEST_FILE}" >&2; exit 2; }

exec litkit \
  --embed-producer \
  --build-only \
  --shard-id "${PRODUCER_NODE_ID}" \
  --num-shards "${PRODUCER_NUM_SHARDS}" \
  --embed-outdir "${EMBED_OUTDIR}" \
  --paper-embed-bs 16 \
  --chunk-embed-bs 64 \
  --embed-devices "cuda:0,cuda:1" \
  --embed-workers 2 \
  --parse-workers "${PRODUCER_PARSE_WORKERS}" \
  --tar-manifest "${CONTAINER_NFS_WS}/${PRODUCER_MANIFEST_FILE}"
EOF
)

            set +e
            "${chrun_argv[@]}" "${litkit_cmd}"
            rc=$?
            set -e
            echo "[Producer ${PRODUCER_NODE_ID}] litkit exit code: ${rc}"

            # Sync local results back to NFS only if litkit succeeded
            # Rsync to FLAT emb_segments dir (segment filenames are shard-unique)
            if [[ "${PRODUCER_USE_LOCAL_STAGING}" == "1" ]]; then
                if [[ "${rc}" -eq 0 ]]; then
                    echo "[Producer ${PRODUCER_NODE_ID}] [staging] Syncing local outputs -> NFS (atomic-ish via --delay-updates)"
                    mkdir -p "${LITKIT_WORKSPACE}/sqlite" "${LITKIT_WORKSPACE}/emb_segments"
                    rsync -a --delay-updates "${PRODUCER_LOCAL_STAGE_HOST}/sqlite/" "${LITKIT_WORKSPACE}/sqlite/"
                    rsync -a --delay-updates "${PRODUCER_LOCAL_STAGE_HOST}/emb_segments/" "${LITKIT_WORKSPACE}/emb_segments/"
                    echo "[Producer ${PRODUCER_NODE_ID}] [staging] Cleanup local staging dir"
                    rm -rf "${PRODUCER_LOCAL_STAGE_HOST}"
                else
                    echo "[Producer ${PRODUCER_NODE_ID}] [staging] Not rsyncing (litkit failed). Preserving: ${PRODUCER_LOCAL_STAGE_HOST}" >&2
                fi
            fi

            exit "${rc}"
        '

    echo "[Producer ${node_id}] Completed on ${target_node}"
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
