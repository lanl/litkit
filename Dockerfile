# ===========================================
# Stage 1: Build or reuse Torch 2.5.1 wheel
# (CUDA 12.5 toolchain; aarch64)
# ===========================================
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS torchwheel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# APT with retries + Python 3.12 toolchain
RUN set -eux; \
  echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries; \
  for i in 1 2; do \
    apt-get update && apt-get install -y --no-install-recommends \
      tzdata \
      curl ca-certificates git software-properties-common \
      ccache cmake ninja-build pkg-config libopenblas-dev libomp-dev && break; \
    echo "APT transient failure; retrying ($i/2)..." >&2; sleep 3; \
  done; \
  add-apt-repository -y ppa:deadsnakes/ppa; \
  for i in 1 2; do \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv && break; \
    echo "APT transient failure; retrying ($i/2)..." >&2; sleep 3; \
  done; \
  rm -rf /var/lib/apt/lists/*

# CUDA toolchain env (for build stage)
ENV CUDA_HOME=/usr/local/cuda-12.5
RUN ln -sfn ${CUDA_HOME} /usr/local/cuda || true
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV CCACHE_DIR=/ccache
RUN mkdir -p /ccache /opt/wheels
COPY wheels/ /opt/wheels/

RUN python3.12 -m ensurepip --upgrade && python3.12 -m pip install --upgrade pip setuptools wheel

# Use prebuilt wheel if present; otherwise build Torch 2.5.1 (+cu125)
RUN set -ex; \
  if ls /opt/wheels/torch-2.5.1*cp312*linux_aarch64.whl >/dev/null 2>&1; then \
    echo "Using prebuilt Torch wheel:"; ls -l /opt/wheels/torch-2.5.1*cp312*linux_aarch64.whl; \
  else \
    echo "No prebuilt Torch wheel found; building from source against CUDA 12.5..."; \
    NPROC="$(nproc)"; \
    MEM_GB="$(awk '/MemTotal/ {printf "%.0f",$2/1024/1024}' /proc/meminfo 2>/dev/null || echo 32)"; \
    PAR="$(( MEM_GB / 4 ))"; [ "$PAR" -lt 8 ] && PAR=8; [ "$PAR" -gt "$NPROC" ] && PAR="$NPROC"; \
    PAR="${PARALLEL_JOBS:-${MAX_JOBS:-$PAR}}"; \
    printf 'cores=%s mem_gb=%s -> parallel=%s\n' "$NPROC" "$MEM_GB" "$PAR"; \
    export MAX_JOBS="$PAR" CMAKE_BUILD_PARALLEL_LEVEL="$PAR" \
           CC=gcc CXX=g++ \
           USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_QNNPACK=0 USE_XNNPACK=1 USE_NCCL=0 \
           USE_CCACHE=1 CCACHE_DIR=/ccache \
           TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0+PTX" \
           PYTORCH_BUILD_VERSION=2.5.1+cu125 PYTORCH_BUILD_NUMBER=0; \
    rm -rf /tmp/pytorch; \
    git clone --recursive --depth 1 --branch v2.5.1 https://github.com/pytorch/pytorch.git /tmp/pytorch; \
    cd /tmp/pytorch; \
    git submodule sync --recursive; \
    git submodule update --init --recursive --depth 1 || git submodule update --init --recursive; \
    python3.12 -m pip install --no-cache-dir -r requirements.txt; \
    python3.12 setup.py bdist_wheel; \
    cp dist/torch-2.5.1+cu125*cp312*linux_aarch64.whl /opt/wheels/; \
    echo "Built wheel(s):"; ls -l /opt/wheels/torch-2.5.1+cu125*cp312*linux_aarch64.whl; \
  fi

# Stable alias for Torch 2.5.1 cp312 aarch64
RUN set -e; cd /opt/wheels; \
  wheel_alias="torch-2.5.1-cp312-cp312-linux_aarch64.whl"; \
  real="$(ls -1 torch-2.5.1*cp312*.whl 2>/dev/null | grep -E '(aarch64|arm64)' | head -n1 || true)"; \
  if [ -z "${real:-}" ]; then \
    echo "ERROR: no torch 2.5.1 cp312 aarch64 wheel in /opt/wheels"; ls -l; exit 1; \
  fi; \
  if [ "$real" != "$wheel_alias" ]; then ln -sfn "$real" "$wheel_alias"; fi; \
  sha256sum "$real" | tee /opt/wheels/torch-2.5.1.SHA256

# Pre-download offline deps mirroring lean pins
RUN python3.12 -m pip download -d /opt/wheels --only-binary=:all: \
    "sympy==1.13.1" \
    "lxml==5.4.0" \
    "numpy<2" \
    "faiss-cpu>=1.8.0,<1.9" \
    "safetensors>=0.4,<1" \
    "sentence-transformers==3.4.1" \
    "transformers==4.57.1" \
    "tokenizers==0.22.1" \
    "huggingface-hub==0.36.0" \
    "scikit-learn==1.7.2" \
    "scipy==1.16.3" \
    "openai>=1,<2"

# Keep only our Torch 2.5.1 wheel (avoid inadvertent 2.6+ pulls)
RUN find /opt/wheels -maxdepth 1 -type f -name 'torch-*.whl' ! -name 'torch-2.5.1*' -print -delete || true

# faiss sanity (aarch64 wheel must exist)
RUN set -e; \
  if ! ls /opt/wheels/faiss_cpu-*-cp312-*.whl >/dev/null 2>&1; then \
    echo "ERROR: faiss_cpu cp312 wheel missing in /opt/wheels"; ls -l /opt/wheels | sed -n '1,200p'; exit 1; \
  fi; \
  ls -1 /opt/wheels/faiss_cpu-*-cp312-*.whl | grep -E '(aarch64|arm64)' >/dev/null || { \
    echo "ERROR: faiss_cpu wheel is not an aarch64 build"; ls -l /opt/wheels/faiss_cpu-*-cp312-*.whl; exit 1; \
  }

# Pillow sanity (cp312 aarch64)
ARG PILLOW_VER=12.0.0
RUN set -e; \
  pillow_whl="$(find /opt/wheels -maxdepth 1 -type f -iname "pillow-${PILLOW_VER}-*cp312-*.whl" | head -n1)"; \
  if [ -z "$pillow_whl" ]; then \
    python3.12 -m pip download -d /opt/wheels --only-binary=:all: "pillow==${PILLOW_VER}"; \
    pillow_whl="$(find /opt/wheels -maxdepth 1 -type f -iname "pillow-${PILLOW_VER}-*cp312-*.whl" | head -n1)"; \
  fi; \
  if [ -z "$pillow_whl" ]; then \
    echo "ERROR: No Pillow ${PILLOW_VER} cp312 wheel present after download."; ls -l /opt/wheels | sed -n '1,200p'; exit 1; \
  fi; \
  echo "$pillow_whl" | grep -Ei '(aarch64|arm64)' >/dev/null || { \
    echo "ERROR: Pillow ${PILLOW_VER} wheel is not an aarch64 build: $pillow_whl"; exit 1; \
  }


# ==============================
# Stage 2: Runtime (NV variant)
# - CUDA 12.5 userspace baked in
# - UV tool env for litkit
# ==============================
FROM ghcr.io/astral-sh/uv:0.9.0-bookworm AS runtime

ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.12

# Bind points & wheel cache
RUN mkdir -p /workspace /data/pmc_oa /data/test_tar_shards /opt/wheels
WORKDIR /workspace

# Bring in wheels
COPY --from=torchwheel /opt/wheels /opt/wheels

# CUDA 12.5 userspace (prevents missing libcudart/libcublas at runtime)
COPY --from=torchwheel /usr/local/cuda-12.5/targets /usr/local/cuda-12.5/targets
COPY --from=torchwheel /usr/local/cuda-12.5/lib64   /usr/local/cuda-12.5/lib64
ENV CUDA_HOME=/usr/local/cuda-12.5
RUN ln -sfn /usr/local/cuda-12.5 /usr/local/cuda || true
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/sbsa-linux/lib:/usr/local/cuda-12.5/targets/aarch64-linux/lib:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:${PATH}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Minimal runtime libs (APT with retry)
RUN set -eux; \
  echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries; \
  for i in 1 2; do \
    apt-get update && apt-get install -y --no-install-recommends \
      tzdata \
      curl ca-certificates git vim \
      libopenblas0 libgomp1 libssl3 libffi8 libstdc++6 && break; \
    echo "APT transient failure; retrying ($i/2)..." >&2; sleep 3; \
  done; \
  rm -rf /var/lib/apt/lists/*

ENV UV_NO_CACHE=1
ENV UV_LINK_MODE=copy

# ---------- Hugging Face cache & model prefetch ----------
ARG SPECTER2_REV=3447645e1def9117997203454fa4495937bfbd83
ARG MPNET_REV=e8c3b32edf5434bc2275fc9bab85f82640a19130
ENV HF_HOME=/app/hf_cache

RUN uv tool install huggingface_hub==0.36.0
RUN hf download allenai/specter2_base --repo-type model \
       --revision ${SPECTER2_REV} \
       --local-dir /app/hf_cache/hub/models--allenai--specter2_base \
       --include "*" \
 && hf download sentence-transformers/all-mpnet-base-v2 --repo-type model \
       --revision ${MPNET_REV} \
       --local-dir /app/hf_cache/hub/models--sentence-transformers--all-mpnet-base-v2 \
       --include "*"

# Deterministic snapshot links to the exact SHAs
RUN set -e; \
  for pair in \
    "models--allenai--specter2_base:${SPECTER2_REV}" \
    "models--sentence-transformers--all-mpnet-base-v2:${MPNET_REV}"; do \
    base="/app/hf_cache/hub/${pair%%:*}"; rev="${pair##*:}"; \
    mkdir -p "$base/snapshots"; \
    ln -sfn .. "$base/snapshots/${rev}"; \
  done

# ---------- Project source ----------
COPY pyproject.toml /app
COPY uv.lock /app
COPY src /app/src
RUN touch /app/README.md

# Make the Torch wheel visible to uv.sources (dereference symlink)
RUN mkdir -p /app/wheels \
 && cp -L /opt/wheels/torch-2.5.1-cp312-cp312-linux_aarch64.whl /app/wheels/

# Create litkit tool env (uv 0.9.0 doesn't support --locked here)
RUN (cd /app; uv tool install --python 3.12 --reinstall .)

# Install from local wheels; Torch without deps to avoid stray versions
RUN uv pip install \
      --python /root/.local/share/uv/tools/litkit/bin/python \
      --upgrade pip setuptools wheel \
 && uv pip install \
      --python /root/.local/share/uv/tools/litkit/bin/python \
      --no-index --find-links=/opt/wheels \
      "sympy==1.13.1" "lxml==5.4.0" "numpy<2" "faiss-cpu>=1.8.0,<1.9" \
 && uv pip install \
      --python /root/.local/share/uv/tools/litkit/bin/python \
      --no-index --no-deps /opt/wheels/torch-2.5.1-cp312-cp312-linux_aarch64.whl

ARG PILLOW_VER=12.0.0
RUN uv pip install \
      --python /root/.local/share/uv/tools/litkit/bin/python \
      --no-index --find-links=/opt/wheels \
      "safetensors>=0.4,<1" \
      "sentence-transformers==3.4.1" \
      "transformers==4.57.1" \
      "tokenizers==0.22.1" \
      "huggingface-hub==0.36.0" \
      "scikit-learn==1.7.2" \
      "scipy==1.16.3" \
      "pillow==${PILLOW_VER}" \
      "openai>=1,<2"

# Optional cache link inside package tree (harmless if unused)
RUN set -e; \
  PKG_SITE="$(/root/.local/share/uv/tools/litkit/bin/python -c 'import site; print(site.getsitepackages()[0])')"; \
  PKG_HUB="${PKG_SITE}/litkit/hf_cache"; \
  mkdir -p "${PKG_HUB}" && ln -sfn /app/hf_cache "${PKG_HUB}/hub"

# SPECTER2 .bin -> .safetensors conversion (guarded)
RUN set -e; \
  if /root/.local/share/uv/tools/litkit/bin/python -c 'import torch' >/dev/null 2>&1; then \
    printf '%s\n' \
    'import os, torch' \
    'base="/app/hf_cache/hub/models--allenai--specter2_base"' \
    'b=os.path.join(base,"pytorch_model.bin")' \
    's=os.path.join(base,"model.safetensors")' \
    'if os.path.exists(b) and not os.path.exists(s):' \
    '    sd=torch.load(b, map_location="cpu", weights_only=True)' \
    '    sd=sd.get("state_dict", sd) if isinstance(sd, dict) else sd' \
    '    from safetensors.torch import save_file as _sf; _sf(sd, s)' \
    "print('SPECTER2: converted .bin -> .safetensors' if os.path.exists(s) else 'SPECTER2: no conversion needed')" \
    > /tmp/conv.py; \
    /root/.local/share/uv/tools/litkit/bin/python /tmp/conv.py; \
    rm -f /tmp/conv.py; \
  else \
    echo 'Torch not importable — unexpected for NV build; skipping conversion.'; \
  fi

# Make python, python3, and litkit discoverable system-wide
RUN ln -sf /root/.local/share/uv/tools/litkit/bin/python /usr/local/bin/python \
 && ln -sf /root/.local/share/uv/tools/litkit/bin/python /usr/local/bin/python3 \
 && ln -sf /root/.local/share/uv/tools/litkit/bin/litkit  /usr/local/bin/litkit

# Defaults
ENV LITKIT_WORKSPACE=/workspace
ENV HF_HOME=/app/hf_cache

# Build-time torch info (guarded)
RUN set -e; \
  printf '%s\n' \
  'try:' \
  '    import torch, sys' \
  "    print(\"Torch:\", torch.__version__, \"CUDA build tag:\", getattr(torch.version, \"cuda\", None))" \
  "    print(\"cuda_available_during_build:\", torch.cuda.is_available())" \
  'except Exception as e:' \
  "    print(\"Torch import check failed:\", e)" \
  > /tmp/info.py; \
  /root/.local/share/uv/tools/litkit/bin/python /tmp/info.py || true; \
  rm -f /tmp/info.py