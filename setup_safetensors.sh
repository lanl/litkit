#!/usr/bin/env bash
set -euo pipefail

# Required: path to the .sqfs image
: "${IMG:?set IMG to your .sqfs path, e.g. $(pwd)/sqfs/litkit-v0.3.33-aarch64-lean.sqfs}"

# Optional inputs (override per node as needed)
: "${HF_HOST:=/path/to/hf_cache_persist}"   # persistent HF cache on host
: "${CUDA_BASE:=/path/to/cuda-12.5-host/cuda-12.5}"
: "${CUDA_LIBA:=${CUDA_BASE}/targets/sbsa-linux/lib}"
: "${CUDA_LIBB:=${CUDA_BASE}/lib64}"
# If you have a CDI spec dir, set it; if not, leave empty and we won't pass --cdi flags.
: "${CDI_SPEC_DIR:=}"   # e.g. /path/to/cdi-v100  OR  /path/to/cdi-grace

# Sanity checks
[ -r "$IMG" ] || { echo "ERROR: IMG not readable: $IMG" >&2; exit 1; }
mkdir -p "$HF_HOST"
ls -1 "$CUDA_LIBA"/libcudart.so.12 "$CUDA_LIBB"/libcublas.so.* >/dev/null \
  || { echo "ERROR: CUDA 12.5 libs not found in CUDA_LIBA/B:
  CUDA_LIBA=$CUDA_LIBA
  CUDA_LIBB=$CUDA_LIBB" >&2; exit 1; }

# Build ch-run flag arrays (conditionally add CDI flags if provided)
bind_flags=(
  --bind "$CUDA_LIBA:$CUDA_LIBA"
  --bind "$CUDA_LIBB:$CUDA_LIBB"
  --bind "$HF_HOST:/host_cache"
)
cdi_flags=()
if [[ -n "${CDI_SPEC_DIR}" && -d "${CDI_SPEC_DIR}" ]]; then
  cdi_flags=( --cdi-dirs="${CDI_SPEC_DIR}" --cdi=nvidia.com/gpu=all )
fi

# Do the conversion with only the pieces we actually need.
ch-run \
  --mount="${CH_MNT:-"$(pwd)/.ch-mnt"}" \
  --unset-env='*' \
  --set-env=HOME=/root \
  "${cdi_flags[@]}" \
  "${bind_flags[@]}" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  "$IMG" -- bash -lc '
set -euo pipefail

# Seed host cache from the image; do not overwrite existing files
mkdir -p /host_cache/hub
cp -an /app/hf_cache/hub/. /host_cache/hub/ || true

# Convert .bin -> .safetensors (idempotent)
PY=/root/.local/share/uv/tools/litkit/bin/python
"$PY" - <<'PY'
import os, torch
from safetensors.torch import save_file

bases = [
  "/host_cache/hub/models--allenai--specter2_base",
  "/host_cache/hub/models--sentence-transformers--all-mpnet-base-v2",
]

for base in bases:
    b = os.path.join(base, "pytorch_model.bin")
    s = os.path.join(base, "model.safetensors")
    if os.path.exists(b) and not os.path.exists(s):
        sd = torch.load(b, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        save_file(sd, s)
        print(f"Converted {b} -> {s} ({os.path.getsize(s)} bytes)")
    else:
        print(f"No conversion needed for {base}")
PY

# Delete .bin if .safetensors exists
for d in \
  /host_cache/hub/models--allenai--specter2_base \
  /host_cache/hub/models--sentence-transformers--all-mpnet-base-v2
do
  if [ -f "$d/model.safetensors" ]; then
    rm -f "$d/pytorch_model.bin" "$d"/pytorch_model-*.bin "$d/pytorch_model.bin.index.json" || true
    echo "Removed .bin artifacts in $d"
  else
    echo "SKIP: $d has no model.safetensors; not deleting .bin files." >&2
  fi
done
'
