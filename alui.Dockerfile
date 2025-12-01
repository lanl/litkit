FROM ghcr.io/astral-sh/uv:0.9.0-bookworm AS runtime

# Ensure a CPython 3.12 is available for uv-managed tool envs
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_NO_CACHE=1
ENV UV_LINK_MODE=copy
WORKDIR /workspace

RUN uv python install 3.12
RUN mkdir -p /workspace /data/pmc_oa /data/test_tar_shards /opt/wheels /host_cache

RUN apt-get update && apt-get install -y \
  --no-install-recommends tzdata curl \
  ca-certificates git vim libopenblas0 libgomp1 libssl3 libffi8 libstdc++6

# ---------- Hugging Face cache & model prefetch ----------
ARG SPECTER2_REV=3447645e1def9117997203454fa4495937bfbd83
ARG MPNET_REV=e8c3b32edf5434bc2275fc9bab85f82640a19130

RUN uv tool install huggingface_hub==0.36.0
ENV HF_HOME=/app/hf_cache
RUN hf download allenai/specter2_base --repo-type model \
       --revision ${SPECTER2_REV} \
       --local-dir /app/hf_cache/hub/models--allenai--specter2_base \
       --include "*" \
 && hf download sentence-transformers/all-mpnet-base-v2 --repo-type model \
       --revision ${MPNET_REV} \
       --local-dir /app/hf_cache/hub/models--sentence-transformers--all-mpnet-base-v2 \
       --include "*"

# Make deterministic snapshot links to the exact SHAs
RUN set -e; \
  for pair in \
    "models--allenai--specter2_base:${SPECTER2_REV}" \
    "models--sentence-transformers--all-mpnet-base-v2:${MPNET_REV}"; do \
    base="/app/hf_cache/hub/${pair%%:*}"; rev="${pair##*:}"; \
    mkdir -p "$base/snapshots"; \
    ln -sfn .. "$base/snapshots/${rev}"; \
  done

# ---------- Add source ----------
RUN mkdir -p /app/src/litkit
COPY pyproject.toml /app
# COPY uv.lock /app
# COPY src/litkit /app/src/litkit
RUN touch /app/README.md

# Make the torch wheel visible to uv.sources (dereference the symlink)
# RUN mkdir -p /app/wheels \
#   && cp -L /opt/wheels/torch-2.5.1-cp312-cp312-linux_aarch64.whl /app/wheels/

# Create the litkit tool env and install via uv.lock (+ local torch wheel)
# RUN (cd /app; uv tool install --python 3.12 --reinstall .)

RUN uv venv -p 3.12
RUN uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
RUN uv pip install \
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
      "pillow==12.0.0" \
      "openai>=1,<2"

# Optional cache link into package tree (harmless if unused)
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
    echo 'Torch not importable (no CUDA libs) — skipping SPECTER2 conversion (lean build).'; \
  fi

# Make python, python3, and litkit always available
RUN ln -sf /root/.local/share/uv/tools/litkit/bin/python /usr/local/bin/python \
 && ln -sf /root/.local/share/uv/tools/litkit/bin/python /usr/local/bin/python3 \
 && ln -sf /root/.local/share/uv/tools/litkit/bin/litkit /usr/local/bin/litkit

ENV LITKIT_WORKSPACE=/workspace

# Build-time info (guarded so lean builds don’t fail when CUDA libs aren’t present)
RUN set -e; \
  printf '%s\n' \
  'try:' \
  '    import torch, sys' \
  "    print(\"Torch:\", torch.__version__, \"CUDA build tag:\", getattr(torch.version, \"cuda\", None))" \
  "    print(\"cuda_available_during_build:\", torch.cuda.is_available())" \
  'except Exception as e:' \
  "    print(\"Torch not importable at build time — expected for lean image:\", e)" \
  > /tmp/info.py; \
  /root/.local/share/uv/tools/litkit/bin/python /tmp/info.py || true; \
  rm -f /tmp/info.py
