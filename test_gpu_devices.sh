#!/bin/bash
# Test script to verify GPU device detection in container

set -euo pipefail

# Use the same variables as the SLURM script
export LITKIT_REPO="/path/to/litkit"
export LITKIT_WORKSPACE="${LITKIT_REPO}/workspace"
export IMG="${LITKIT_REPO}/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
export CDI_SPEC_DIR="/path/to/cdi-v100"
export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="${CUDA_BASE}/targets/sbsa-linux/lib"
export CUDA_LIBB="${CUDA_BASE}/lib64"
export HF_HOST="/path/to/hf_cache_persist"

CHRUN="/opt/site/aarch64/rhel8/charliecloud/0.42/bin/ch-run"

echo "=== Testing GPU Device Detection ==="
echo ""

echo "1. Check CUDA availability in container:"
$CHRUN \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind "${LITKIT_WORKSPACE}:/workspace" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="HF_HOME=/app/hf_cache" \
    "$IMG" -- \
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "2. Test litkit device resolution with 'auto':"
$CHRUN \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind "${LITKIT_WORKSPACE}:/workspace" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="HF_HOME=/app/hf_cache" \
    "$IMG" -- \
    python3 -c "
from litkit.embeddings.devices import resolve_embed_devices
devices = resolve_embed_devices('auto')
print(f'Auto-resolved devices: {devices}')
"

echo ""
echo "3. Test litkit device resolution with 'cuda:0,cuda:1':"
$CHRUN \
    --unset-env='*' \
    --set-env=HOME=/root \
    --cdi-dirs="$CDI_SPEC_DIR" \
    --cdi=nvidia.com/gpu=all \
    --bind "$CUDA_LIBA:$CUDA_LIBA" \
    --bind "$CUDA_LIBB:$CUDA_LIBB" \
    --bind "${LITKIT_WORKSPACE}:/workspace" \
    --bind "$HF_HOST:/app/hf_cache" \
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
    --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --set-env="HF_HOME=/app/hf_cache" \
    "$IMG" -- \
    python3 -c "
from litkit.embeddings.devices import resolve_embed_devices
devices = resolve_embed_devices('cuda:0,cuda:1')
print(f'Explicit devices: {devices}')
"

echo ""
echo "=== Test Complete ==="
