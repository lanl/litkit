#!/usr/bin/env bash
#SBATCH -J litkit-build
#SBATCH -p gpu-v100                 # gpu-v100 (V100, ARM) | use -p gpu-gh200 for GH200
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=NONE              # do NOT inherit login env; avoid x86_64 tools on ARM nodes

set -euo pipefail

###############################################################################
# PARTITION vs TARGET:
# The -p above selects hardware. TARGET below only tweaks bindings/paths.
###############################################################################

### ----- USER KNOBS (aligned with cli.py) -------------------------------------
: "${TARGET:=v100}"                          # v100 | gh
case "$TARGET" in
  v100) req_part="gpu-v100" ;;
  gh)   req_part="gpu-gh200" ;;
  *)    echo "FATAL: TARGET must be v100|gh" >&2; exit 2 ;;
esac
curr_part="${SLURM_JOB_PARTITION:-${SLURM_JOB_PARTITION_NAME:-}}"
if [[ -n "$curr_part" && "$curr_part" != "$req_part" ]]; then
  echo "FATAL: TARGET=$TARGET requires -p $req_part (job has -p $curr_part)." >&2
  echo "Submit like:  sbatch -p $req_part TARGET=$TARGET vector_build_single.sbatch" >&2
  exit 2
fi

# Build mode
: "${REBUILD:=1}"                            # 1 = --rebuild; 0 = --update
: "${YES:=1}"                                # 1 = auto-confirm rebuild prompts

# Corpus source (exactly one)
: "${MANIFEST:=/path/to/litkit/workspace/small_test.manifest}"
: "${TAR_DIR:=}"

# Retrieval/ANN controls
: "${TOP_PAPERS:=}"
: "${TOP_CHUNKS:=30}"
: "${OVERSHOOT:=20}"
: "${HNSW_RECALL:=normal}"
: "${NPROBE:=}"
: "${PER_PAPER_CAP:=3}"
: "${MIN_CHUNKS_PER_PAPER:=2.0}"
: "${LEXICAL_CAP:=}"
: "${LEXICAL_LIMIT:=200}"
: "${ALLOW_GLOBAL_LEXICAL:=0}"

# Chunking (characters)
: "${CHUNK_TARGET_CHARS:=1200}"
: "${CHUNK_MIN_CHARS:=300}"
: "${CHUNK_OVERLAP:=200}"

# Indices and parameters
: "${PAPERS_INDEX:=hnsw}"
: "${HNSW_M:=32}"
: "${HNSW_EF_CONSTRUCTION:=200}"
: "${HNSW_EF_SEARCH:=128}"

: "${CHUNKS_INDEX:=ivfpq}"
: "${IVF_NLIST:=16384}"
: "${PQ_M:=64}"

# Sharding / batching / checkpoint
: "${SHARD_ID:=0}"
: "${NUM_SHARDS:=1}"
: "${PAPER_BATCH:=20000}"
: "${CHUNK_BATCH:=20000}"
: "${CKPT_EVERY:=500}"

# Embedding workers/devices and batch sizes
: "${EMBED_WORKERS:=1}"
: "${EMBED_DEVICES:=auto}"
: "${PAPER_EMBED_BS:=16}"
: "${CHUNK_EMBED_BS:=64}"

# SQLite/IO behavior
: "${SQLITE_JOURNAL_MODE:=TRUNCATE}"         # TRUNCATE|WAL
: "${SQLITE_BUSY_TIMEOUT_MS:=120000}"
: "${QUIET:=0}"

# Debug mode for parallelism verification (set to 0 for quiet production runs)
: "${LITKIT_DEBUG:=1}"

# Role flags
: "${FAISS_WRITER:=1}"
: "${EMBED_PRODUCER:=0}"
: "${CONSUME_SEGMENTS:=0}"
: "${EMBED_OUTDIR:=}"

# Offline/HF behavior
: "${OFFLINE:=1}"
: "${DO_SAFE_CONVERT:=1}"
: "${CONVERT_BASES:=models--allenai--specter2_base models--sentence-transformers--all-mpnet-base-v2}"

# LLM (not used for builds)
: "${NO_LLM:=1}"
: "${LLM_MODEL:=}"
: "${MAX_OUT_TOKENS:=}"
: "${OPENAI_BASE_URL:=}"
: "${OPENAI_API_KEY:=}"

# Free-form passthrough
: "${EXTRA_ARGS:=}"

### ----- CLUSTER PATHS --------------------------------------------------------
BASE="/path/to"
LITKIT_DIR="$BASE/litkit"
IMG="$LITKIT_DIR/sqfs/litkit-v0.3.34-aarch64-lean.sqfs"   # keep in sync with your image
if [[ "$TARGET" == "gh" ]]; then
  CDI_SPEC_DIR="$BASE/cdi-grace"
else
  CDI_SPEC_DIR="$BASE/cdi-v100"
fi

### ----- CLEAN ENV ON NODE ----------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module reset >/dev/null 2>&1 || module purge
fi

# Optional: site module (may point to wrong arch; we don't trust it blindly)
module load charliecloud/0.42 2>/dev/null || true

### ----- PICK AN aarch64 ch-run (no Exec format error) ------------------------
: "${CHR_BIN:=}"                   # allow override: CHR_BIN=/full/path/to/aarch64/ch-run
want_arch="$(uname -m)"            # expected 'aarch64' on gpu-v100 / GH200

declare -a CHRUN_CANDIDATES=()
[[ -n "${CHR_BIN}" && -x "${CHR_BIN}" ]] && CHRUN_CANDIDATES+=("${CHR_BIN}")
command -v ch-run >/dev/null 2>&1 && CHRUN_CANDIDATES+=("$(command -v ch-run)")
# common site layouts by arch
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
    CHRUN="$p"; break
  fi
done

if [[ -z "$CHRUN" ]]; then
  echo "FATAL: no arch-correct ch-run ($want_arch) found. Searched: ${CHRUN_CANDIDATES[*]-<none>}" >&2
  exit 2
fi
echo "[charliecloud] ch-run=$CHRUN ($(file -L "$CHRUN" | sed 's/.*ELF/ELF/'))"

### ----- CUDA AUTODISCOVERY (prefer 12.5; strict knob) ------------------------
: "${CUDA_VERSION_PIN:=12.5.0}"             # default preference
: "${CUDA_STRICT:=0}"                        # set to 1 to hard-fail if pin not met
: "${STAGED_CUDA_BASE:=$BASE/cuda-12.5-host/cuda-12.5}"

# Try to load pinned CUDA first; ignore errors (we verify below).
if [[ -n "$CUDA_VERSION_PIN" ]]; then
  module load "cuda/$CUDA_VERSION_PIN" 2>/dev/null || true
else
  module load cuda 2>/dev/null || true
fi

cands=()
[[ -d "$STAGED_CUDA_BASE" ]] && cands+=("$STAGED_CUDA_BASE")
[[ -n "${CUDA_HOME:-}"  && -d "$CUDA_HOME"  ]] && cands+=("$CUDA_HOME")
[[ -n "${EBROOTCUDA:-}" && -d "$EBROOTCUDA" ]] && cands+=("$EBROOTCUDA")
command -v nvcc >/dev/null 2>&1 && cands+=("$(dirname "$(dirname "$(command -v nvcc)")")")

CUDA_BASE=""
for c in "${cands[@]}"; do
  [[ "$c" == *"/x86_64/"* ]] && continue
  if [[ -d "$c/targets/sbsa-linux/lib" || -d "$c/targets/aarch64-linux/lib" ]]; then
    CUDA_BASE="$c"; break
  fi
done
if [[ -z "$CUDA_BASE" ]]; then
  echo "FATAL: no suitable ARM CUDA found. Checked: ${cands[*]-<none>}" >&2
  exit 2
fi

# Strict pin check if requested
if [[ "$CUDA_STRICT" == "1" && -n "$CUDA_VERSION_PIN" ]]; then
  if ! [[ "$CUDA_BASE" == *"/$CUDA_VERSION_PIN"* ]]; then
    echo "FATAL: CUDA_STRICT=1 but selected CUDA_BASE=$CUDA_BASE != *$CUDA_VERSION_PIN*" >&2
    exit 2
  fi
fi

if [[ -d "$CUDA_BASE/targets/sbsa-linux/lib" ]]; then
  CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"
else
  CUDA_LIBA="$CUDA_BASE/targets/aarch64-linux/lib"
fi
CUDA_LIBB="$CUDA_BASE/lib64"
echo "[cuda] CUDA_BASE=$CUDA_BASE"
echo "[cuda] CUDA_LIBA=$CUDA_LIBA"
echo "[cuda] CUDA_LIBB=$CUDA_LIBB"

HF_HOST="$BASE/hf_cache_persist"
WORKSPACE="$LITKIT_DIR/workspace"
EMB_SEG_DIR="$WORKSPACE/emb_segments"
mkdir -p "$HF_HOST" "$WORKSPACE" "$EMB_SEG_DIR" "$CDI_SPEC_DIR"

### ----- SANITY ---------------------------------------------------------------
[[ -f "$IMG" ]] || { echo "FATAL: image not found: $IMG" >&2; exit 2; }
ls -1 "$CUDA_LIBA"/libcudart.so.* "$CUDA_LIBB"/libcublas.so.* >/dev/null 2>&1 || {
  cat >&2 <<EOF
FATAL: CUDA libs missing.
  CUDA_BASE = $CUDA_BASE
  CUDA_LIBA = $CUDA_LIBA (needs libcudart.so.*)
  CUDA_LIBB = $CUDA_LIBB (needs libcublas.so.*)
Fix: load a CUDA module or set STAGED_CUDA_BASE / CUDA_VERSION_PIN / CUDA_STRICT=1.
EOF
  exit 2
}

# Source of tars must be present
if [[ -z "${MANIFEST}" && -z "${TAR_DIR}" ]]; then
  echo "FATAL: set MANIFEST=/abs/path.manifest or TAR_DIR=/abs/path/to/shards" >&2
  exit 2
fi
[[ -z "${MANIFEST}" || -f "${MANIFEST}" ]] || { echo "FATAL: manifest not found: ${MANIFEST}" >&2; exit 2; }
[[ -z "${TAR_DIR}"  || -d "${TAR_DIR}"  ]]  || { echo "FATAL: tar dir not found: ${TAR_DIR}" >&2; exit 2; }

### ----- CDI SPEC (GPU) -------------------------------------------------------
if [[ ! -f "$CDI_SPEC_DIR/nvidia.json" ]]; then
  echo "[CDI] Generating NVIDIA CDI spec in $CDI_SPEC_DIR"
  nvidia-ctk cdi generate --format=json --output="$CDI_SPEC_DIR/nvidia.json"
fi
nvidia-ctk cdi list --spec-dir="$CDI_SPEC_DIR" || true

### ----- COMMON ch-run ARGS ---------------------------------------------------
CHR=(
  "$CHRUN"
  --unset-env='*'
  --set-env=HOME=/root
  --cdi-dirs="$CDI_SPEC_DIR"
  --cdi=nvidia.com/gpu=all
  --bind "$CUDA_LIBA:$CUDA_LIBA"
  --bind "$CUDA_LIBB:$CUDA_LIBB"
  --bind "$HF_HOST:/app/hf_cache"
  --bind "$WORKSPACE:/workspace"
  --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB"
  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --set-env="LITKIT_WORKSPACE=/workspace"
  --set-env="HF_HOME=/app/hf_cache"
  --set-env="TRANSFORMERS_USE_SAFE_TENSORS=1"
  --set-env="LITKIT_DEBUG=${LITKIT_DEBUG}"
)

# Use manifest to find tar shards
if [[ -n "${MANIFEST}" ]]; then
  MANI_ABS="$(readlink -f "$MANIFEST" 2>/dev/null \
    || python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$MANIFEST")"
  CHR+=( --bind "$(dirname "$MANI_ABS"):$(dirname "$MANI_ABS")" )
  mapfile -t __tar_dirs < <(grep -vE '^\s*(#|$)' "$MANI_ABS" | awk '{print $1}' \
                            | xargs -r -n1 dirname | sort -u)
  for d in "${__tar_dirs[@]}"; do
    [[ -d "$d" ]] && CHR+=( --bind "$d:$d" )
  done
fi

# Directory with tar shards
if [[ -n "${TAR_DIR}" ]]; then
  TAR_ABS="$(readlink -f "$TAR_DIR" 2>/dev/null \
    || python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$TAR_DIR")"
  CHR+=( --bind "$TAR_ABS:$TAR_ABS" )
fi

### ----- LITKIT (cli.py) FLAGS ------------------------------------------------
LITKIT_ARGS=()
[[ "$QUIET" == "1" ]] && LITKIT_ARGS+=( --quiet )

LITKIT_ARGS+=( --hnsw-recall "$HNSW_RECALL" --top-chunks "$TOP_CHUNKS" --overshoot "$OVERSHOOT" )
[[ -n "$TOP_PAPERS" ]]  && LITKIT_ARGS+=( --top-papers "$TOP_PAPERS" )
[[ -n "$NPROBE" ]]      && LITKIT_ARGS+=( --nprobe "$NPROBE" )
LITKIT_ARGS+=( --per-paper-cap "$PER_PAPER_CAP" --min-chunks-per-paper "$MIN_CHUNKS_PER_PAPER" )
[[ -n "$LEXICAL_CAP" ]] && LITKIT_ARGS+=( --lexical-cap "$LEXICAL_CAP" )
LITKIT_ARGS+=( --lexical-limit "$LEXICAL_LIMIT" )
[[ "$ALLOW_GLOBAL_LEXICAL" == "1" ]] && LITKIT_ARGS+=( --allow-global-lexical )

LITKIT_ARGS+=( --chunk-target-chars "$CHUNK_TARGET_CHARS" --chunk-min-chars "$CHUNK_MIN_CHARS" --chunk-overlap "$CHUNK_OVERLAP" )

LITKIT_ARGS+=( --papers-index "$PAPERS_INDEX" --hnsw-m "$HNSW_M" --efconstruction "$HNSW_EF_CONSTRUCTION" --efsearch "$HNSW_EF_SEARCH" )
LITKIT_ARGS+=( --chunks-index "$CHUNKS_INDEX" --ivf-nlist "$IVF_NLIST" --pq-m "$PQ_M" )

LITKIT_ARGS+=( --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" )
LITKIT_ARGS+=( --paper-batch "$PAPER_BATCH" --chunk-batch "$CHUNK_BATCH" --ckpt-every "$CKPT_EVERY" )

LITKIT_ARGS+=( --embed-workers "$EMBED_WORKERS" --embed-devices "$EMBED_DEVICES" --paper-embed-bs "$PAPER_EMBED_BS" --chunk-embed-bs "$CHUNK_EMBED_BS" )

LITKIT_ARGS+=( --sqlite-journal-mode "$SQLITE_JOURNAL_MODE" --sqlite-busy-timeout-ms "$SQLITE_BUSY_TIMEOUT_MS" )

if [[ -n "${MANIFEST}" ]]; then
  LITKIT_ARGS+=( --tar-manifest "$MANI_ABS" )
else
  LITKIT_ARGS+=( --tar-dir "$TAR_ABS" )
fi

[[ "$OFFLINE" == "1" ]] && LITKIT_ARGS+=( --offline )

[[ "$FAISS_WRITER" == "1"     ]] && LITKIT_ARGS+=( --faiss-writer )
[[ "$EMBED_PRODUCER" == "1"   ]] && LITKIT_ARGS+=( --embed-producer )
[[ "$CONSUME_SEGMENTS" == "1" ]] && LITKIT_ARGS+=( --consume-segments )
[[ -n "$EMBED_OUTDIR"         ]] && LITKIT_ARGS+=( --embed-outdir "$EMBED_OUTDIR" )

if [[ "$REBUILD" == "1" ]]; then
  LITKIT_ARGS+=( --rebuild )
  [[ "$YES" == "1" ]] && LITKIT_ARGS+=( -y )
else
  LITKIT_ARGS+=( --update )
fi
LITKIT_ARGS+=( --build-only )

[[ "$NO_LLM" == "1" ]] && LITKIT_ARGS+=( --no-llm )
[[ -n "$LLM_MODEL"      ]] && LITKIT_ARGS+=( --llm-model "$LLM_MODEL" )
[[ -n "$MAX_OUT_TOKENS" ]] && LITKIT_ARGS+=( --max-out-tokens "$MAX_OUT_TOKENS" )
[[ -n "$OPENAI_BASE_URL" ]] && LITKIT_ARGS+=( --openai-base-url "$OPENAI_BASE_URL" )
[[ -n "$OPENAI_API_KEY"  ]] && LITKIT_ARGS+=( --openai-api-key "$OPENAI_API_KEY" )

[[ -n "$EXTRA_ARGS" ]] && LITKIT_ARGS+=( $EXTRA_ARGS )

### ----- OPTIONAL: seed/convert HF cache to .safetensors ----------------------
if [[ "$DO_SAFE_CONVERT" == "1" ]]; then
  echo "[prep] Seeding HF cache from image and converting to .safetensors..."
  CHR_SEED=(
    "$CHRUN"
    --unset-env='*'
    --set-env=HOME=/root
    --cdi-dirs="$CDI_SPEC_DIR"
    --cdi=nvidia.com/gpu=all
    --bind "$CUDA_LIBA:$CUDA_LIBA"
    --bind "$CUDA_LIBB:$CUDA_LIBB"
    --bind "$HF_HOST:/host_cache"
    --set-env="LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB"
    --set-env="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  )
  "${CHR_SEED[@]}" "$IMG" -- bash -lc '
    set -euo pipefail
    mkdir -p /host_cache/hub
    if [ -d /app/hf_cache/hub ]; then
      cp -an /app/hf_cache/hub/. /host_cache/hub/ || true
    fi
    PY=/root/.local/share/uv/tools/litkit/bin/python
    cat > /tmp/convert.py <<'"'"'PYEOF'"'"'
import os, torch
from safetensors.torch import save_file
bases = os.environ.get("CONVERT_BASES","").split()
root = "/host_cache/hub"
def convert_one(base):
    b = os.path.join(base, "pytorch_model.bin")
    s = os.path.join(base, "model.safetensors")
    if os.path.exists(s):
        print(f"[ok] safetensors already present: {base}")
        return
    if not os.path.exists(b):
        print(f"[skip] no .bin at {base}")
        return
    sd = torch.load(b, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    save_file(sd, s)
    print(f"[conv] {b} -> {s} ({os.path.getsize(s)} bytes)")
for rel in bases:
    base = os.path.join(root, rel)
    if os.path.isdir(base):
        convert_one(base)
    else:
        print(f"[miss] cache dir not found: {base}")
for rel in bases:
    base = os.path.join(root, rel)
    s = os.path.join(base, "model.safetensors")
    if os.path.exists(s):
        for name in ["pytorch_model.bin", "pytorch_model.bin.index.json"]:
            p = os.path.join(base, name)
            if os.path.exists(p): os.remove(p)
        for fn in os.listdir(base):
            if fn.startswith("pytorch_model-") and fn.endswith(".bin"):
                try: os.remove(os.path.join(base, fn))
                except OSError: pass
        print(f"[clean] removed .bin artifacts in {base}")
    else:
        print(f"[keep] no safetensors in {base}; leaving .bin files.")
PYEOF
    export CONVERT_BASES="'"$CONVERT_BASES"'"
    "$PY" /tmp/convert.py
  '
fi

### ----- CLEAN STALE WRITER GUARD (SLURM RESTART SAFE) ------------------------
# If a prior job crashed on a different node, the guard file may persist.
# Cross-host auto-eviction is disabled for safety (we can't verify remote PID).
# This unconditional cleanup is safe BECAUSE:
#   1. This runs at job START (no concurrent writer for this workspace)
#   2. Any prior writer is dead (SLURM guarantees our allocation is exclusive)
# WARNING: Do NOT add this mid-job or if running concurrent writers.
rm -f "${WORKSPACE:-$LITKIT_WORKSPACE}/.writer_guard" 2>/dev/null || true

### ----- RUN BUILD ------------------------------------------------------------
echo "[info] Starting build in Charliecloud..."
"${CHR[@]}" "$IMG" -- litkit "${LITKIT_ARGS[@]}"
echo "[done] Vector store build completed."
