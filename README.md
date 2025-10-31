# litkit

A scalable **two‑stage RAG** pipeline for the PMC‑OA corpus.

This project implements an **offline** retrieval‑augmented generation workflow over JATS/NXML corpora and builds two FAISS indices:

- **Papers:** SPECTER2 embeddings (title+abstract) → HNSW index  
- **Chunks:** SBERT `all-mpnet-base-v2` embeddings → IVF‑PQ (or FLAT for small data)

## Key features

- Resumable/batched ingest (SQLite)
- ANN defaults for HPC / air‑gapped runs
- Token‑budgeted LLM prompts
- Robust FAISS index reporting (avoids ambiguous prints)
- Sharding for multi‑process ingestion on shared filesystems

---

## Quick start — MacBook (Apple Silicon: M1/M2/M3) [CPU or Metal (MPS)]

1. Create and activate a Python env (conda/mamba/venv) with:

   ```bash
   # Python ≥ 3.10 (3.12 preferred)
   pip install "faiss-cpu" "transformers" "sentence-transformers" "torch" "lxml" "openai" "numpy<2"
   # sqlite3 is stdlib
   ```

   **Devices & acceleration (facts):**

   - CUDA multi‑GPU distribution is **NVIDIA‑only**.  
     On Apple Silicon (MPS) and CPU‑only hosts, encoding uses a single device.
   - The **EmbeddingPool** uses one process per CUDA device; it does **not** spawn workers on MPS.
   - `--embed-devices auto` on Apple yields `mps` and a single worker. Multi‑GPU sharding (`cuda:0,cuda:1,…`) requires NVIDIA.

2. Place **local HF snapshots** (for offline/air‑gapped runs) under `./hf_cache/hub/`:

   - `allenai/specter2_base`  
   - `sentence-transformers/all-mpnet-base-v2`

   To force offline behavior, set:
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

3. Build indices (single‑process writer):

   ```bash
   python -m litkit --faiss-writer --build-only
   ```

   Rebuild from scratch:

   ```bash
   python -m litkit --faiss-writer --rebuild --build-only
   ```

4. Query (defaults to local OpenAI‑compatible endpoint for testing or whatever you pass via env/CLI):

   ```bash
   python -m litkit "What is BioNetGen?"
   ```

   Point at a local OpenAI‑compatible endpoint (e.g., LM Studio):

   ```bash
   export OPENAI_BASE_URL="http://localhost:1234/v1"
   export OPENAI_API_KEY="no-auth"
   python -m litkit --llm-model gpt-oss:20b "What is BioNetGen?"
   ```

   Pass a question from a file:

   ```bash
   python -m litkit --question-file question.txt
   ```

5. Tuning / Troubleshooting

   - If you hit a context‑length error on small‑context local models (e.g., 4k tokens), either reduce retrieval sizes:
     ```
     --top-chunks 10 --overshoot 10
     ```
     or lower the prompt budget. The script auto‑trims and retries (×4) when overflow is detected.
   - `--top-chunks` controls how many chunks reach the LLM (default: 30).
   - `--overshoot` widens ANN search before paper‑level filtering (default: 20); reduce if prompt size is an issue.
   - `--ckpt-every` controls resume checkpoints (default: 500 files).

---

## SQLite journal mode (HPC filesystems)

- **Default:** `TRUNCATE`. On shared HPC filesystems (NFS/Lustre), `WAL` can cause “database is locked” or poor behavior under preemption. Use `WAL` **only** on node‑local storage with a single writer.

- **How to set:**
  ```bash
  --sqlite-journal-mode {TRUNCATE|WAL}
  ```

- **Guidance:**
  - Shared storage → `TRUNCATE`.
  - Node‑local NVMe (single writer) → `WAL` is acceptable and faster.

- **What the code does:**
  - Applies `PRAGMA journal_mode`.
  - Sets `wal_autocheckpoint=1000` only for `WAL`.
  - Logs: `[db] journal_mode set to wal|truncate|delete`.

- **Quick check:**
  ```bash
  python -m litkit --sqlite-journal-mode TRUNCATE --build-only
  # Expect: [db] journal_mode set to truncate

  python -m litkit --sqlite-journal-mode WAL --build-only
  # Expect: [db] journal_mode set to wal
  ```

---

## Chunking

- Greedily pack paragraphs to ~`CHUNK_TARGET_CHARS`. Small tail merged to meet `CHUNK_MIN_CHARS`.
- Character overlap (default `CHUNK_OVERLAP_CHARS=200`) reduces claim splitting.

**Tunables (CLI):**
```
--chunk-target-chars  (default 1200)
--chunk-min-chars     (default 300)
--chunk-overlap       (default 200)
```

---

## Busy timeout (locks on shared filesystems)

- **Default:** `120000 ms` (`--sqlite-busy-timeout-ms`), applied to both Python connect and SQLite `PRAGMA busy_timeout`.

**Recommendations:**
- Shared NFS/Lustre with multiple writers: 120–300 s
- Node‑local NVMe (single writer): 10–30 s
- If you see “database is locked,” raise the timeout and ensure `TRUNCATE` on shared FS.

---

## Quick start — HPC (ARM CPUs + NVIDIA GPUs) [CUDA]

1. Use a CUDA‑enabled PyTorch build and `faiss-gpu` or fallback `faiss-cpu`. Device selection: `cuda` → `mps` → `cpu`.
2. Ensure the **offline HF snapshots** exist (see Mac instructions).
3. Multi‑process ingestion on shared FS:
   - Exactly **one writer** (adds vectors to FAISS, saves indices).
   - Others run as **non‑writers** (DB rows only), e.g.:
     ```bash
     # writer on shard 0
     python -m litkit --faiss-writer --shard-id 0 --num-shards 8 --build-only
     # readers on shards 1..7
     python -m litkit --shard-id 1 --num-shards 8 --build-only
     ```
4. Query as usual. If you have OpenAI access, e.g.:
   ```bash
   export OPENAI_API_KEY="sk-..."
   python -m litkit --llm-model o3 "Summarize X"
   ```

**GPU batch‑size presets (baseline):**
- **V100 32 GB**: `--paper-embed-bs 48`, `--chunk-embed-bs 128`
- **H100 80 GB**: `--paper-embed-bs 96`, `--chunk-embed-bs 256`

---

## HPC / cluster recipes (tar shards, single writer, resumable)

**Prereqs (one‑time per project)**

```bash
export LITKIT_WORKSPACE=/lustre/$USER/litkit    # writable base
export LITKIT_TAR_PRESCAN=0                     # default on HPC; keep off for huge shards
export LITKIT_TAR_RENDER_SEC=30                 # progress refresh cadence
export LITKIT_TAR_PCT_STP=5                     # +5% milestones
export LITKIT_THREADS=32                        # cap BLAS/FAISS threads
```

**Recommended shard layout**
```
/lustre/$USER/pmcoa_tar_shards/
    shard_0001.tar.gz
    shard_0002.tar.gz
    ...
```

**Single FAISS writer (build indices from tar shards)**
```bash
python -m litkit \
  --faiss-writer --build-only --rebuild \
  --tar-dir /lustre/$USER/pmcoa_tar_shards \
  --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000 \
  --papers-index hnsw --hnsw-m 32 --efconstruction 200 --efsearch 128 \
  --chunks-index ivfpq --ivf-nlist 16384 --pq-m 64 --nprobe 64 \
  --paper-embed-bs 24 --chunk-embed-bs 24
```

**Scale‑out readers (DB only)**
```bash
for s in 1 2 3 4 5 6 7; do
  python -m litkit \
    --build-only --shard-id $s --num-shards 8 \
    --tar-dir /lustre/$USER/pmcoa_tar_shards \
    --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000 &
done
wait
```

**Query against a local endpoint**
```bash
python -m litkit \
  --offline \
  --llm-model gpt-oss:20b \
  --openai-base-url http://localhost:1234/v1 \
  --openai-api-key no-auth \
  --nprobe 64 \
  "How is mathematical modeling useful in the study of HIV dynamics?"
```

**Operational tips**

- `--ckpt-every` controls commit+checkpoint cadence (default 500).
- Index saves are rate‑limited by `LITKIT_SAVE_EVERY_SEC` (default 120).
- `LITKIT_PROGRESS_MODE=tty` for single‑line bars.
- Tar streaming resumes per‑shard via a persisted member counter.

---

## Writable work directory (`LITKIT_WORKSPACE`)

By default, artifacts go next to the script (`sqlite/`, `indices/`, `hf_cache/`, `emb_segments/`).  
On HPC, set:

```bash
export LITKIT_WORKSPACE=/lustre/$USER/litkit
```

Then litkit creates (under `$LITKIT_WORKSPACE`):

- `sqlite/`   (SQLite DB, checkpoints, locks)  
- `indices/`  (FAISS indices)  
- `hf_cache/` (local HF snapshots)  
- `emb_segments/` (embedding segments)

---

## Guardrails & safety (o‑series vs local OSS)

- OpenAI o‑series (e.g., `o3`) always enforce OpenAI safety policies.
- For fully offline or unguardrailed runs, use a **local** OpenAI‑compatible endpoint.
- Prompts default to strict RAG: **use only provided context** and bracketed citations.

---

## Reproducible locking with `uv.lock`

A `uv.lock` file (derived from `pyproject.toml`) is used for reproducible container builds.

**On a non‑GPU frontend:**
```bash
cd /path/to/litkit
python3.12 -m pip install --user uv
~/.local/bin/uv lock --python 3.12
```
Commit the resulting `uv.lock`.

---

## SPECTER2: `.bin` → `.safetensors`

Recent `transformers` releases block unsafe `torch.load` on `.bin` (older Torch) due to CVEs. We avoid `torch.load` at runtime by using **safetensors**.

- Set:
  ```bash
  export TRANSFORMERS_USE_SAFE_TENSORS=1
  ```
- Convert once:
  - Use the provided `setup_safetensors.sh` to convert `pytorch_model.bin` → `model.safetensors` for **SPECTER2** in your HF cache.
  - After conversion, the embedder loads `model.safetensors` and never calls `torch.load` on `.bin`.

---

## HPC cluster — GH200 & V100: install & run

Two container flavors:

- **lean** (recommended): no CUDA userspace inside; bind site CUDA libs at runtime via Charliecloud + CDI. Most portable.
- **nv**: CUDA userspace baked into the image; simpler to run, but cluster driver must be ≥ the baked CUDA version.

### Common prerequisites

- Charliecloud ≥ 0.42
- Generate a CDI spec once per GPU type:
  ```bash
  # GH200 (Grace Hopper) example
  mkdir -p /path/to/cdi-grace
  nvidia-ctk cdi generate --format=json \
    --output=/path/to/cdi-grace/nvidia.json
  nvidia-ctk cdi list --spec-dir=/path/to/cdi-grace
  ```
- Host CUDA 12.5 userspace (for lean flavor):
  ```bash
  ch-image pull nvidia/cuda:12.5.0-devel-ubuntu22.04
  DEST=/path/to/cuda-12.5-host
  mkdir -p "$DEST"
  ch-run nvidia/cuda:12.5.0-devel-ubuntu22.04 -- bash -lc \
    'tar -C /usr/local -cf - cuda-12.5' | tar -C "$DEST" -xvf -
  ```

Define (per node type):
```bash
# GH200
export CDI_SPEC_DIR="/path/to/cdi-grace"
# V100
export CDI_SPEC_DIR="/path/to/cdi-v100"

export CUDA_BASE="/path/to/cuda-12.5-host/cuda-12.5"
export CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
export CUDA_LIBB="$CUDA_BASE/lib64"
```

### Smoke test: CUDA inside the container (lean flavor)

```bash
module purge
module load charliecloud/0.42
module load cuda/12.5.0

IMG=/path/to/litkit/sqfs/litkit-v0.3.33-aarch64-lean.sqfs

ch-run "$IMG" \
  --unset-env='*' \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -- /root/.local/share/uv/tools/litkit/bin/python - <<'PY'
import torch, sys, os
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0),
          "CC:", torch.cuda.get_device_capability(0))
PY
```

If you see `CUDA available: False`, you didn’t bind CUDA libs or CDI correctly.

### Build vector store (test run)

```bash
HF_HOST=/path/to/hf_cache_persist
mkdir -p "$HF_HOST"

# Convert SPECTER2 .bin -> .safetensors once (required)
./setup_safetensors.sh

ch-run "$IMG" \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "/path/to/test_tar_shards:/path/to/test_tar_shards" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "$HF_HOST:/app/hf_cache" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="HF_HOME=/app/hf_cache" \
  --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
  -- litkit --faiss-writer --build-only --rebuild --yes \
            --tar-dir /path/to/test_tar_shards
```

### Query with hosted LLM API (OpenAI‑compatible)

**Do not forget `/v1`** on the base URL. You must also bind a host CA bundle into the container; Charliecloud 0.42 does **not** support `:ro` suffix on binds, so avoid it.

```bash
KEY="$(head -n1 ~/.llm_api_key)"
BASE="https://llm.example.com/v1"

# Choose an existing CA bundle on the host. On RHEL-like HPC frontends one of these exists:
#   /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
#   /etc/pki/tls/certs/ca-bundle.crt
HOST_CA="/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"  # adjust if needed

cp question.txt workspace/

ch-run "$IMG" \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs="$CDI_SPEC_DIR" --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --bind "$(pwd)/workspace:/workspace" \
  --bind "/path/to/test_tar_shards:/path/to/test_tar_shards" \
  --bind "$HF_HOST:/app/hf_cache" \
  --bind "$HOST_CA:/workspace/site-ca.pem" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  --set-env="SSL_CERT_FILE=/workspace/site-ca.pem" \
  --set-env="REQUESTS_CA_BUNDLE=/workspace/site-ca.pem" \
  --set-env="CURL_CA_BUNDLE=/workspace/site-ca.pem" \
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --set-env="LITKIT_WORKSPACE=/workspace" \
  --set-env="LITKIT_TAR_DIR=/path/to/test_tar_shards" \
  --set-env="HF_HOME=/app/hf_cache" \
  --set-env="HF_HUB_OFFLINE=1" \
  --set-env="TRANSFORMERS_OFFLINE=1" \
  --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1" \
  --set-env="LITKIT_OPENAI_TIMEOUT_SEC=60" \
  --set-env="OPENAI_BASE_URL=$BASE" \
  --set-env="OPENAI_API_KEY=$KEY" \
  -- litkit \
      --llm-model gpt-oss-120b \
      --openai-base-url "$BASE" \
      --openai-api-key  "$KEY" \
      --question-file /workspace/question.txt
```

**HPC gotchas:**

- TLS errors like `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain` mean the CA bundle isn’t visible **inside** the container. Bind the host bundle and set the three env vars exactly as above.
- If you see `Endpoint: http://localhost:1234/v1` in logs, your base URL isn’t propagating. Set both env **and** `--openai-base-url`.
- If you see `libcudart.so.12` or `libcublas.so.* not found`, you didn’t bind CUDA userspace (lean flavor). Bind `CUDA_LIBA` and `CUDA_LIBB` and set `LD_LIBRARY_PATH`, or use the **nv** image.

### V100 quick smoke test

```bash
salloc -N1 -t 10:00:00 -p gpu-v100 --no-shell
ssh gpu-node1  # or gpu-node2/gpu-node3/gpu-node4

IMG=/path/to/litkit/sqfs/litkit-v0.3.33-aarch64-lean.sqfs
CUDA_BASE=/path/to/cuda-12.5-host/cuda-12.5
CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
CUDA_LIBB="$CUDA_BASE/lib64"

ch-run "$IMG" \
  --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs=/path/to/cdi-v100 \
  --cdi=nvidia.com/gpu=all \
  --bind "$CUDA_LIBA:$CUDA_LIBA" \
  --bind "$CUDA_LIBB:$CUDA_LIBB" \
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB" \
  -- /root/.local/share/uv/tools/litkit/bin/python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA build tag:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

---

## Project metadata

- `pyproject.toml` pins `torch==2.5.1.*` and forces a local wheel via `tool.uv.sources` for the aarch64 CUDA 12.5 build.
- Core deps: `openai>=1,<2`, `transformers>=4.44,<5`, `sentence-transformers>=3.1,<4`, `faiss-cpu>=1.8,<1.9`.
- Container builds copy HF snapshots and (optionally) perform guarded `.bin` → `.safetensors` conversion for SPECTER2.

---

## License

Proprietary. 
