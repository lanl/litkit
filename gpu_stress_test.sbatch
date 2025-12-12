#!/bin/bash
#SBATCH --job-name=gpu_stress
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_stress_%j.out
#SBATCH --error=gpu_stress_%j.err
#SBATCH --export=NONE

set -euo pipefail

echo "=============================================="
echo "GPU Stress Test for LitKit"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="

# Load Charliecloud
module purge
module load charliecloud/0.42 2>/dev/null || true

# Find arch-correct ch-run
want_arch="$(uname -m)"
echo "[arch] Compute node architecture: $want_arch"

CHRUN=""
for p in \
  "/opt/site/centos8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "/opt/site/rhel8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "/opt/site-nv/rhel8/${want_arch}/charliecloud/0.42/bin/ch-run" \
  "$(command -v ch-run 2>/dev/null || true)"
do
  if [[ -x "$p" ]] && file -L "$p" 2>/dev/null | grep -qi "$want_arch"; then
    CHRUN="$p"
    echo "[charliecloud] Found: $CHRUN"
    break
  fi
done

if [[ -z "$CHRUN" ]]; then
  echo "FATAL: no arch-correct ch-run found" >&2
  exit 2
fi

# HPC paths
export LITKIT_REPO="/path/to/litkit"
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

echo ""
echo "[config] Container: $IMG"
echo "[config] CUDA libs: $CUDA_LIBA"
echo "[config] HF cache: $HF_HOST"
echo ""

# Run the stress test inside the container
"$CHRUN" \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind "${LITKIT_REPO}:/app/litkit" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="HF_HOME=/app/hf_cache" \
    --set-env="PYTHONPATH=/app/litkit/src" \
    "$IMG" -- \
    python /app/litkit/gpu_stress_test.py \
        --sentences 100000 \
        --devices "cuda:0,cuda:1" \
        --batch-size 64 \
        --rounds 3 \
        --single-gpu-baseline

echo ""
echo "=============================================="
echo "GPU Stress Test Complete"
echo "=============================================="
