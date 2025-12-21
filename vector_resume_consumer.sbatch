#!/bin/bash
#SBATCH --job-name=litkit_resume
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=litkit_resume_%j.out
#SBATCH --error=litkit_resume_%j.err
#SBATCH --export=NONE

# ============================================================================
# Resume script: Ingest remaining embedding segments after a timeout/failure
# 
# Use this when:
#   - A multi-node build timed out during segment ingestion
#   - The DB merge completed but segment files remain
#   - validate_build.sh shows "unconsumed segment files"
#
# This script:
#   1. Creates fake producer completion markers
#   2. Runs the consumer in consume-only mode
#   3. Ingests all remaining .npz segment files
# ============================================================================

set -euo pipefail

# Load Charliecloud
module purge
module load charliecloud/0.42 2>/dev/null || true

# Find arch-correct ch-run
want_arch="$(uname -m)"
CHRUN=""
for p in \
  "/opt/site/aarch64/rhel8/charliecloud/0.42/bin/ch-run" \
  "/opt/site/centos8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "$(command -v ch-run 2>/dev/null || true)"
do
  if [[ -x "$p" ]] && file -L "$p" 2>/dev/null | grep -qi "$want_arch"; then
    CHRUN="$p"
    break
  fi
done
[[ -n "$CHRUN" ]] || { echo "FATAL: no arch-correct ch-run found" >&2; exit 2; }
echo "[resume] Using ch-run: $CHRUN"

# Paths (same as vector_build_multi.sbatch)
export LITKIT_REPO="/path/to/litkit"
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.34-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

SEG_DIR="${LITKIT_WORKSPACE}/emb_segments"

# Count remaining segment files
n_segs=$(find "$SEG_DIR" -name '*.npz' 2>/dev/null | wc -l)
echo "[resume] Found $n_segs segment files to ingest"

if [[ "$n_segs" -eq 0 ]]; then
    echo "[resume] No segment files found. Nothing to do."
    echo "[resume] Run ./validate_build.sh to check status."
    exit 0
fi

# Detect how many shards were used (from segment filenames)
n_shards=$(find "$SEG_DIR" -name '*.npz' -exec basename {} \; 2>/dev/null \
           | sed -n 's/.*_sh\([0-9]\+\)_.*/\1/p' | sort -u | wc -l)
[[ "$n_shards" -ge 1 ]] || n_shards=2
echo "[resume] Detected $n_shards shard(s) from segment filenames"

### ----- CLEAN STALE WRITER GUARD (SLURM RESTART SAFE) ------------------------
# If a prior job crashed on a different node, the guard file may persist.
# Cross-host auto-eviction is disabled for safety (we can't verify remote PID).
# This unconditional cleanup is safe BECAUSE:
#   1. This runs at job START (no concurrent writer for this workspace)
#   2. Any prior writer is dead (SLURM guarantees our allocation is exclusive)
# WARNING: Do NOT add this mid-job or if running concurrent writers.
rm -f "${LITKIT_WORKSPACE}/.writer_guard" 2>/dev/null || true

# Create fake completion markers so consume-only doesn't wait
echo "[resume] Creating producer completion markers..."
for ((i=0; i<n_shards; i++)); do
    marker="$SEG_DIR/.shard_$(printf '%02d' $i)_complete"
    if [[ ! -f "$marker" ]]; then
        echo "{\"shard_id\": $i, \"num_shards\": $n_shards, \"resumed\": true}" > "$marker"
        echo "[resume] Created $marker"
    else
        echo "[resume] Marker already exists: $marker"
    fi
done

echo ""
echo "=== Resume Consumer Starting ==="
echo "Segments to ingest: $n_segs"
echo "Shards: $n_shards"
echo "Workspace: $LITKIT_WORKSPACE"
echo "================================"
echo ""

# Run the consumer
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
    --set-env="LITKIT_SQLITE_BUSY_TIMEOUT_MS=300000" \
    --set-env="LITKIT_SAVE_EVERY_SEC=300" \
    "$IMG" -- \
    litkit \
        --faiss-writer \
        --consume-only \
        --tar-manifest "/workspace/medium_test.manifest" \
        --embed-outdir "/workspace/emb_segments" \
        --papers-index hnsw \
        --chunks-index flat \
        --num-shards "$n_shards"

echo ""
echo "=== Resume Complete ==="
echo "Run: ./validate_build.sh"
