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
   # Python ≥ 3.10
   pip install faiss-cpu transformers sentence-transformers torch lxml openai numpy
   # sqlite3 is stdlib
   ```

   **Devices & acceleration (important)**

   - CUDA multi‑GPU distribution is supported and used **only** on NVIDIA GPUs.  
     On Apple Silicon (MPS) and CPU‑only hosts, encoding runs on a single device.
   - The **EmbeddingPool** uses one process per CUDA device; it does **not** spawn workers on MPS.
   - If you pass `--embed-devices auto` on Apple, you will see `mps` and a single worker. Multi‑GPU sharding (`cuda:0,cuda:1,…`) requires NVIDIA GPUs.

2. Place **local HF snapshots** (air‑gapped) under `./hf_cache/hub/`:

   - `allenai/specter2_base`  
   - `sentence-transformers/all-mpnet-base-v2`

   The CLI sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` automatically.

3. Build indices (single‑process writer):

   ```bash
   python -m litkit --faiss-writer --build-only
   ```

   Rebuild from scratch:

   ```bash
   python -m litkit --faiss-writer --rebuild --build-only
   ```

4. Query (uses local LLM endpoint by default for testing):

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

   - If you see a context‑length error on small‑context local models (e.g., 4096 tokens), either reduce retrieval sizes:
     ```
     --top-chunks 10 --overshoot 10
     ```
     or lower the prompt budget (built‑in default for OSS models is conservative).
   - `--top-chunks` controls how many chunks are passed to the LLM (default: 30).
   - `--overshoot` widens ANN search before filtering to shortlisted papers (default: 20), which can increase recall but may increase prompt size; reduce it if you hit limits.
   - The script auto‑trims and retries (×4) when an overflow is detected.
   - `--ckpt-every` controls how often we persist resume progress (default: 500 files). Increase on slow parallel filesystems to reduce metadata churn.

---

## SQLite journal mode (HPC filesystems)

- **Default:** `TRUNCATE`. On shared HPC filesystems (e.g., NFS/Lustre), `WAL` can cause “database is locked” errors or poor behavior under preemption. Use `WAL` **only** on node‑local storage (e.g., NVMe) with a single writer.

- **How to set:**
  ```bash
  # CLI
  --sqlite-journal-mode {TRUNCATE|WAL}
  ```
  (Other modes like `DELETE`/`MEMORY`/`OFF` are intentionally unsupported.)

- **HPC guidance:**
  - Use `TRUNCATE` when the DB lives on shared storage.
  - If you place the DB on node‑local NVMe/scratch and have a single writer, `WAL` is acceptable and faster.

- **What the code does:**
  - Applies `PRAGMA journal_mode` to the chosen mode.
  - Sets `wal_autocheckpoint=1000` only when in `WAL` mode.
  - Logs the effective choice as: `[db] journal_mode set to wal|truncate|delete`.

- **Quick check:**
  ```bash
  python -m litkit --sqlite-journal-mode TRUNCATE --build-only
  # Expect: [db] journal_mode set to truncate

  python -m litkit --sqlite-journal-mode WAL --build-only
  # Expect: [db] journal_mode set to wal
  ```

- If you see “database is locked” or odd stalls on a shared FS, switch to `TRUNCATE`.

---

## Chunking

- Greedily pack paragraphs to ~`CHUNK_TARGET_CHARS`. A small tail chunk is merged into the previous one so most chunks meet `CHUNK_MIN_CHARS`.
- Add a small character overlap between consecutive chunks (default `CHUNK_OVERLAP_CHARS=200`) to reduce claim‑splitting across boundaries.

**Tunables (CLI):**
```
--chunk-target-chars  (default 1200)
--chunk-min-chars     (default 300)
--chunk-overlap       (default 200)
```
For biomedical prose, the defaults are robust; raise overlap to 200–250 if you observe cross‑paragraph claims being split in answers.

---

## Busy timeout (locks on shared filesystems)

- The script sets both Python's connect() timeout and SQLite `PRAGMA busy_timeout` to the same value. **Default:** `120000 ms` (120 s).

**Override:**
- Env: `LITKIT_SQLITE_BUSY_TIMEOUT_MS=180000`  
- CLI: `--sqlite-busy-timeout-ms 180000`

**Recommendations:**
- Shared NFS/Lustre with multiple writers: 120–300 s
- Node‑local NVMe (single writer): 10–30 s is usually fine
- Keep `journal_mode=TRUNCATE` on shared FS. Use `WAL` only on node‑local storage.

**Symptom & fix:** If you see “database is locked” under load, raise `--sqlite-busy-timeout-ms` and ensure you are using `TRUNCATE` on shared storage.

---

## Quick start — HPC (ARM CPUs + NVIDIA GPUs) [CUDA]

1. Use a CUDA‑enabled PyTorch build and `faiss-gpu` or `faiss-cpu` as available. The script auto‑detects device:
   `cuda` (NVIDIA GPU) → `mps` (Apple Metal) → `cpu` (fallback).

2. Ensure the same **offline HF snapshots** exist on the node(s) (see above).

3. For multi‑process ingestion on a shared filesystem:
   - Choose exactly **one writer** (adds vectors to FAISS, saves indices).
   - Others run as **non‑writers** (DB rows only), e.g., with sharding:

     ```bash
     # writer on shard 0
     python -m litkit --faiss-writer --shard-id 0 --num-shards 8 --build-only
     # readers on shards 1..7
     python -m litkit --shard-id 1 --num-shards 8 --build-only
     # ... repeat for 2..7
     ```

   The writer creates/updates indices while all processes insert into SQLite.

4. Query as usual (o3 model recommended for production if online):

   ```bash
   export OPENAI_API_KEY="sk-..."   # if you’re using OpenAI o3 online
   python -m litkit --llm-model o3 "Summarize X"
   ```

**GPU batch‑size presets** (tested on CUDA; override via `--paper-embed-bs` / `--chunk-embed-bs`)

- **V100 32 GB**  
  `--paper-embed-bs 48`  
  `--chunk-embed-bs 128`

- **H100 80 GB**  
  `--paper-embed-bs 96`  
  `--chunk-embed-bs 256`

Notes:
- `all-mpnet-base-v2` and SPECTER2 use `seq=512`; these presets balance throughput vs. headroom.
- If you see OOM on V100, drop chunk batch to 96 or 64. H100 can usually go higher (e.g., 192–256+).

---

## HPC / cluster recipes (tar shards, single writer, resumable)

**Prereqs (one time per project)**

```bash
# Writable base (shared or node-local). All sqlite/, indices/, hf_cache/ go here.
export LITKIT_HOME=/lustre/$USER/brag

# Optional: make logs quieter on giant tar.gz shards
export LITKIT_TAR_RENDER_SEC=30      # progress refresh every N seconds (default: 5)
export LITKIT_TAR_PCT_STP=5          # also print each +5% milestone (0=off)

# Optional: skip tar member prescan if counting is expensive
# (default is LITKIT_TAR_PRESCAN=0 on HPC)
export LITKIT_TAR_PRESCAN=0

# Optional: cap math/BLAS/FAISS threads to keep nodes polite
export LITKIT_THREADS=32
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
srun -N1 -n1 python -m litkit   --faiss-writer --build-only --rebuild   --tar-dir /lustre/$USER/pmcoa_tar_shards   --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000   --papers-index hnsw --hnsw-m 32 --efconstruction 200 --efsearch 128   --chunks-index ivfpq --ivf-nlist 16384 --pq-m 64 --nprobe 64   --paper-embed-bs 24 --chunk-embed-bs 24
```

Notes:
- `journal_mode=TRUNCATE` is safest on NFS/Lustre. `WAL` is fine only on node‑local NVMe.
- `ivf-nlist` is an upper bound; code auto‑reduces to respect training samples (~40× rule).
- `nprobe` defaults to ~√(nlist) if omitted; 64 is a good starting point.
- Batch sizes: CPU/MPS ~16–32; CUDA GPUs ~64–128 (adjust to your memory).

**Scale‑out readers (DB only, no FAISS mutation)**

```bash
# Example: 8 shards total – writer is shard 0; launch readers on 1..7
for s in 1 2 3 4 5 6 7; do
  srun -N1 -n1 --exclusive python -m litkit     --build-only --shard-id $s --num-shards 8     --tar-dir /lustre/$USER/pmcoa_tar_shards     --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000 &
done
wait
```

**SLURM job‑array variant (one writer + array readers)**

Writer:
```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH -J litkit-writer -N 1 -n 1 -c 16
srun python -m litkit   --faiss-writer --build-only --rebuild   --tar-dir /lustre/$USER/pmcoa_tar_shards   --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000   --papers-index hnsw --hnsw-m 32 --efconstruction 200 --efsearch 128   --chunks-index ivfpq --ivf-nlist 16384 --pq-m 64 --nprobe 64   --paper-embed-bs 24 --chunk-embed-bs 24
EOF
```

Readers (array 1..7 for an 8‑way split):
```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH -J litkit-readers -N 1 -n 1 -c 16
#SBATCH --array=1-7
srun python -m litkit   --build-only --shard-id ${SLURM_ARRAY_TASK_ID} --num-shards 8   --tar-dir /lustre/$USER/pmcoa_tar_shards   --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000
EOF
```

**Query (offline or local OpenAI‑compatible endpoint)**

```bash
# Example: local endpoint (LM Studio) on http://localhost:1234/v1
python -m litkit   --offline   --llm-model gpt-oss:20b   --openai-base-url http://localhost:1234/v1   --openai-api-key no-auth   --nprobe 64   "How is mathematical modeling useful in the study of HIV dynamics?"
```

**Operational tips**

- Checkpointing: `--ckpt-every` controls commit+checkpoint cadence (default 500).
- Save throttling: index saves are rate‑limited (`LITKIT_SAVE_EVERY_SEC`, default 120).
- Progress mode: logs append by default; set `LITKIT_PROGRESS_MODE=tty` for single‑line bars.
- Resumes: tar streaming resumes per‑shard via a persisted member counter.

---

## Writable work directory (`LITKIT_HOME`)

By default, the script writes to folders next to the script (`sqlite/`, `indices/`, `hf_cache/`).  
On HPC or shared environments, you can redirect these writable artifacts by setting:

```bash
export LITKIT_HOME=/lustre/$USER/brag   # or any writable Lustre/NFS path
```

When set, the following directories are created under `$LITKIT_HOME`:

- `sqlite/`   (SQLite DB, checkpoints, locks)  
- `indices/`  (FAISS indices)  
- `hf_cache/` (optional local HF snapshots for offline runs)

**Important:**

- `LITKIT_HOME` should point to a writable Lustre/NFS (shared) or node‑local path.
- On shared filesystems, prefer `--sqlite-journal-mode TRUNCATE` (see SQLite guidance).

---

## Guardrails & safety (o3 vs local OSS)

- OpenAI’s hosted o‑series models (e.g., `o3`) always enforce OpenAI safety policies. There is no API flag to disable guardrails.
- For fully offline or unguardrailed runs, use a **local** OpenAI‑compatible endpoint.
- Prompt discipline: By default the script runs in **strict RAG** mode and prompts tell the model to use **only** the provided context and to include bracketed citations.

---

## Notes

- **Overshoot:** during chunk retrieval we search more candidates (`k * overshoot`) then filter down to the shortlisted candidate papers; improves recall.
- **Multiple data subdirectories:** pass the **parent** (e.g., `./data`). The script recurses into `pmc_oa_xml_dir01`, `pmc_oa_xml_dir02`, etc.
- **Air‑gap:** the script is robust to missing internet; all models must be present locally (HF snapshot paths). If a snapshot is missing, it throws a clear `FileNotFoundError` with guidance.
