#!/usr/bin/env bash
#SBATCH -J litkit-build
#SBATCH -p gpu-v100
#SBATCH -N 2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --gres-flags=enforce-binding
#SBATCH -t 10:00:00
#SBATCH -o /path/to/litkit/workspace/log/%x.%j.out
#SBATCH -e /path/to/litkit/workspace/log/%x.%j.err
#SBATCH --export=ALL

# ==============================================================================
# Env knobs (optional):
#   ALLOW_GUARD_STEAL=1           # force-remove faiss_writer.guard
#   GUARD_STALE_SEC=86400         # guard staleness threshold (default 24h)
#   FORCE_NSHARDS=<int>           # override persisted NSHARDS (dangerous)
#   FORCE_WRITER=1                # force running writer even when indices exist
#   ENABLE_LEXICAL=0              # 1 = Stage-1 bootstrap: producers start w/ papers.faiss only
# ==============================================================================

set -euo pipefail
ulimit -n 65535 || true
trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT INT TERM

echo "[script] file=$(readlink -f "$0") md5=$(md5sum "$0" | awk "{print \$1}") ts=$(date -Is)"

# -------------------- Tunables --------------------
: "${TARGET:=v100}"          # v100 | gh
: "${REBUILD:=1}"            # 1=rebuild, 0=update
: "${YES:=1}"
: "${OFFLINE:=1}"
: "${QUIET:=0}"
: "${MANIFEST:=/path/to/litkit/workspace/pmcoa.manifest}"
: "${TAR_DIR:=}"

# Topology params
GPUS_PER_NODE_DEFAULT="${SLURM_NTASKS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
: "${GPUS_PER_NODE:=$GPUS_PER_NODE_DEFAULT}"

if [[ "$TARGET" == "gh" ]]; then
  : "${PROD_CPUS:=12}" ; : "${PROD_OMP:=6}"
  : "${WRITER_CPUS:=${PROD_CPUS}}"
  : "${PAPER_EMBED_BS:=128}" ; : "${CHUNK_EMBED_BS:=640}" ; : "${LITKIT_EMBED_SEGMENT_SIZE:=524288}"
else
  : "${PROD_CPUS:=12}" ; : "${PROD_OMP:=6}"
  : "${WRITER_CPUS:=${PROD_CPUS}}"
  : "${PAPER_EMBED_BS:=32}" ; : "${CHUNK_EMBED_BS:=224}" ; : "${LITKIT_EMBED_SEGMENT_SIZE:=131072}"
fi
(( PROD_OMP > PROD_CPUS )) && PROD_OMP="$PROD_CPUS"

: "${SQLITE_JOURNAL_MODE:=TRUNCATE}"
: "${SQLITE_BUSY_TIMEOUT_MS:=900000}"
: "${HNSW_RECALL:=normal}" : "${TOP_CHUNKS:=30}" : "${OVERSHOOT:=20}"
: "${PAPERS_INDEX:=hnsw}" : "${HNSW_M:=32}" : "${HNSW_EF_CONSTRUCTION:=200}" : "${HNSW_EF_SEARCH:=128}"
: "${CHUNKS_INDEX:=ivfpq}" : "${PQ_M:=64}" : "${NPROBE:=}"
if [[ "$TARGET" == "gh" ]]; then : "${IVF_NLIST:=32768}"; else : "${IVF_NLIST:=16384}"; fi

# Producer gate (max seconds to wait for real FAISS indices)
: "${GATE_WAIT_MAX:=14400}"

# -------------------- Paths & container --------------------
BASE="/path/to"
LITKIT_DIR="$BASE/litkit"
IMG="$LITKIT_DIR/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
CDI_SPEC_DIR="$BASE/$([[ "$TARGET" == "gh" ]] && echo cdi-grace || echo cdi-v100)"
HF_HOST="$BASE/hf_cache_persist"
WORKSPACE="$LITKIT_DIR/workspace"
LOG_DIR="$WORKSPACE/log/${SLURM_JOB_ID:-manual}"

# NEW: persist uv/pip tool homes in addition to caches
UV_CACHE_HOST="$BASE/uv_cache"
PIP_CACHE_HOST="$BASE/pip_cache"
UV_HOME_HOST="$BASE/uv_home"       # binds to /root/.local/share/uv
PIP_HOME_HOST="$BASE/pip_home"     # binds to /root/.local/share/pip

mkdir -p "$HF_HOST" "$WORKSPACE" "$LOG_DIR" "$WORKSPACE/indices" "$WORKSPACE/sqlite" "$WORKSPACE/tmp" "$WORKSPACE/emb_segments" \
         "$UV_CACHE_HOST" "$PIP_CACHE_HOST" "$UV_HOME_HOST" "$PIP_HOME_HOST"
umask 002; chmod g+s "$WORKSPACE" "$WORKSPACE"/{log,sqlite,indices,emb_segments} "$UV_CACHE_HOST" "$PIP_CACHE_HOST" "$UV_HOME_HOST" "$PIP_HOME_HOST"

export WORKSPACE LOG_DIR GATE_WAIT_MAX

# Charliecloud
module -q load charliecloud/0.42 2>/dev/null || true
export LMOD_DISABLE_SAME_NAME_AUTOSWAP=yes
unset CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS SLURM_STEP_GPUS
find_chrun(){ for p in $(command -v ch-run || true) \
  "/opt/site/rhel8/$(uname -m)/charliecloud/0.42/bin/ch-run" \
  "/usr/bin/ch-run" "/usr/local/bin/ch-run"; do
  [[ -x "$p" ]] && file -L "$p" | grep -qi "$(uname -m)" && { echo "$p"; return; }
done; return 1; }
CHRUN="$(find_chrun)" || { echo "FATAL: ch-run not found for $(uname -m)" >&2; exit 2; }
echo "[charliecloud] ch-run=$CHRUN ($(file -L "$CHRUN" | sed 's/.*ELF/ELF/'))"

# CUDA libs from host (required by PyTorch CUDA in container)
: "${CUDA_VERSION_PIN:=12.5.0}"
: "${STAGED_CUDA_BASE:=$BASE/cuda-12.5-host/cuda-12.5}"
module -q load "cuda/$CUDA_VERSION_PIN" 2>/dev/null || true
CUDA_BASE=""
for c in "$STAGED_CUDA_BASE" "${CUDA_HOME:-}" "${EBROOTCUDA:-}"; do [[ -d "${c:-/nope}" ]] && CUDA_BASE="$c" && break; done
command -v nvcc >/dev/null 2>&1 && CUDA_BASE="${CUDA_BASE:-$(dirname "$(dirname "$(command -v nvcc)")")}"
[[ -d "$CUDA_BASE" ]] || { echo "FATAL: CUDA_BASE not found" >&2; exit 2; }
if [[ -d "$CUDA_BASE/targets/sbsa-linux/lib" ]]; then CUDA_LIBA="$CUDA_BASE/targets/sbsa-linux/lib"; else CUDA_LIBA="$CUDA_BASE/targets/aarch64-linux/lib"; fi
CUDA_LIBB="$CUDA_BASE/lib64"
ls -1 "$CUDA_LIBA"/libcudart.so.* "$CUDA_LIBB"/libcublas.so.* >/dev/null 2>&1 || { echo "FATAL: CUDA libs missing" >&2; exit 2; }

# Inputs
if [[ -n "${MANIFEST}" && -n "${TAR_DIR}" ]]; then echo "FATAL: set only one of MANIFEST or TAR_DIR" >&2; exit 2; fi
if [[ -z "${MANIFEST}" && -z "${TAR_DIR}" ]]; then echo "FATAL: set MANIFEST=/abs/file.manifest or TAR_DIR=/abs/dir" >&2; exit 2; fi
if [[ -n "${MANIFEST}" ]]; then
  MANI_ABS="$(readlink -f "$MANIFEST" 2>/dev/null || true)"; [[ -z "$MANI_ABS" ]] && MANI_ABS="$(cd "$(dirname "$MANIFEST")" && pwd -P)/$(basename "$MANIFEST")"
  mapfile -t TAR_PARENTS < <(grep -vE '^\s*(#|$)' "$MANI_ABS" | awk '{print $1}' | xargs -r -n1 dirname | sort -u)
else
  TAR_ABS="$(readlink -f "$TAR_DIR" 2>/dev/null || true)"; [[ -z "$TAR_ABS" ]] && TAR_ABS="$(cd "$TAR_DIR" && pwd -P)"
fi

echo "[logs] writing step logs under: $LOG_DIR"
echo "[alloc] NODES=${SLURM_NNODES:-?} PART=${SLURM_JOB_PARTITION:-?}"
echo "[alloc] SBATCH gpn=${SLURM_GPUS_ON_NODE:-?} ntasks-per-node=${SLURM_NTASKS_PER_NODE:-?} GPUS_PER_NODE=${GPUS_PER_NODE}"

# ---------- Guard preflight ----------
: "${GUARD_STALE_SEC:=86400}"
GUARD="$WORKSPACE/sqlite/faiss_writer.guard"
if [[ -e "$GUARD" ]]; then
  read -r GUARD_PID GUARD_HOST GUARD_TS < <(head -n1 "$GUARD" 2>/dev/null || echo "0 unknown 0")
  NOW=$(date +%s || echo 0)
  AGE=$(( NOW - ${GUARD_TS:-0} ))
  echo "[guard] found: pid=${GUARD_PID:-?} host=${GUARD_HOST:-?} ts=${GUARD_TS:-?} age=${AGE:-?}s path=$GUARD"
  if [[ "${ALLOW_GUARD_STEAL:-0}" == "1" ]]; then
    echo "[guard] ALLOW_GUARD_STEAL=1 -> removing guard"
    rm -f "$GUARD"
  else
    if (( AGE >= GUARD_STALE_SEC )); then
      echo "[guard] stale (> ${GUARD_STALE_SEC}s) -> removing guard"
      rm -f "$GUARD"
    else
      echo "[guard] appears active (age ${AGE}s < ${GUARD_STALE_SEC}s). If this is wrong: set ALLOW_GUARD_STEAL=1 or raise GUARD_STALE_SEC. Blocking." >&2
      exit 2
    fi
  fi
fi

# ---------- Sharding with persistence ----------
SHARD_FILE="$WORKSPACE/.num_shards"
TOTAL_TASKS="$(( (${SLURM_NNODES:-1}) * (${SLURM_NTASKS_PER_NODE:-$GPUS_PER_NODE}) ))"
COMPUTED_NSHARDS=$TOTAL_TASKS
if [[ -n "${FORCE_NSHARDS:-}" ]]; then
  NSHARDS="$FORCE_NSHARDS"
  echo "$NSHARDS" > "$SHARD_FILE"
  echo "[shards] FORCE_NSHARDS=$NSHARDS persisted -> $SHARD_FILE"
elif [[ -f "$SHARD_FILE" ]]; then
  read -r PERSISTED < "$SHARD_FILE" || PERSISTED=""
  [[ -z "$PERSISTED" ]] && { echo "FATAL: $SHARD_FILE exists but empty" >&2; exit 2; }
  NSHARDS="$PERSISTED"
  if (( NSHARDS != COMPUTED_NSHARDS )); then
    echo "[shards] mismatch: persisted NSHARDS=$NSHARDS vs current COMPUTED=$COMPUTED_NSHARDS (TOTAL_TASKS=$TOTAL_TASKS). Refuse to run." >&2
    echo "         Fix by matching resources or export FORCE_NSHARDS=$NSHARDS if you accept the risk." >&2
    exit 2
  fi
  echo "[shards] using persisted NSHARDS=$NSHARDS from $SHARD_FILE"
else
  NSHARDS="$COMPUTED_NSHARDS"
  echo "$NSHARDS" > "$SHARD_FILE"
  echo "[shards] first run: persisted NSHARDS=$NSHARDS -> $SHARD_FILE"
fi

# ---------- Writer auto-skip on resume ----------
indices_ok=0
if [[ -s "$WORKSPACE/indices/papers.faiss" && -s "$WORKSPACE/indices/chunks.faiss" ]]; then
  indices_ok=1
fi
START_WRITER=1
if [[ "${FORCE_WRITER:-0}" == "1" ]]; then
  START_WRITER=1
elif [[ "$REBUILD" == "0" && $indices_ok -eq 1 ]]; then
  START_WRITER=0
fi

export START_WRITER NSHARDS GATE_WAIT_MAX

# Cap per-step cores
JOB_CPT="${SLURM_CPUS_PER_TASK:-0}"
W_C=$WRITER_CPUS; P_C=$PROD_CPUS
(( JOB_CPT > 0 && W_C > JOB_CPT )) && W_C="$JOB_CPT"
(( JOB_CPT > 0 && P_C > JOB_CPT )) && P_C="$JOB_CPT"

# Common CLI fragments
COMMON_DB="--sqlite-journal-mode $SQLITE_JOURNAL_MODE --sqlite-busy-timeout-ms $SQLITE_BUSY_TIMEOUT_MS"
COMMON_BS="--paper-embed-bs $PAPER_EMBED_BS --chunk-embed-bs $CHUNK_EMBED_BS"
COMMON_IDX="--papers-index $PAPERS_INDEX --hnsw-m $HNSW_M --efconstruction $HNSW_EF_CONSTRUCTION --efsearch $HNSW_EF_SEARCH --chunks-index $CHUNKS_INDEX --ivf-nlist $IVF_NLIST --pq-m $PQ_M"
[[ -n "$NPROBE" ]] && COMMON_IDX="$COMMON_IDX --nprobe $NPROBE"
if [[ "$QUIET" == "1" ]]; then QUIET_FLAG="--quiet"; else QUIET_FLAG=""; fi
if [[ "$OFFLINE" == "1" ]]; then OFFLINE_FLAG="--offline"; else OFFLINE_FLAG=""; fi

# Bind shards
BIND_TARS=""
if [[ -n "${MANI_ABS:-}" ]]; then
  BIND_TARS+=" --bind '$(dirname "$MANI_ABS")':'$(dirname "$MANI_ABS")'"
  for d in "${TAR_PARENTS[@]:-}"; do [[ -d "$d" ]] && BIND_TARS+=" --bind '$d':'$d'"; done
else
  BIND_TARS+=" --bind '$TAR_ABS':'$TAR_ABS'"
fi

# Base ch-run strings (+ shared caches + tool homes)
CHR_BASE_STR="$CHRUN --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs='$CDI_SPEC_DIR' --cdi=nvidia.com/gpu=all \
  --bind '$CUDA_LIBA':'$CUDA_LIBA' --bind '$CUDA_LIBB':'$CUDA_LIBB' \
  --bind '$HF_HOST':/app/hf_cache --bind '$WORKSPACE':/workspace $BIND_TARS \
  --bind '$UV_CACHE_HOST':/root/.cache/uv --bind '$PIP_CACHE_HOST':/root/.cache/pip \
  --bind '$UV_HOME_HOST':/root/.local/share/uv --bind '$PIP_HOME_HOST':/root/.local/share/pip \
  --set-env='LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB' \
  --set-env='PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --set-env='LITKIT_WORKSPACE=/workspace' \
  --set-env='HF_HOME=/app/hf_cache' \
  --set-env='TRANSFORMERS_USE_SAFE_TENSORS=1' \
  --set-env='HF_DATASETS_OFFLINE=1' \
  --set-env='HF_HUB_OFFLINE=1' \
  --set-env='PYTHONUNBUFFERED=1' \
  --set-env='TOKENIZERS_PARALLELISM=false' \
  --set-env='CUDA_DEVICE_ORDER=PCI_BUS_ID' \
  --set-env='LITKIT_EMBED_SEGMENT_SIZE=$LITKIT_EMBED_SEGMENT_SIZE' \
  --set-env='UV_CACHE_DIR=/root/.cache/uv' \
  --set-env='PIP_CACHE_DIR=/root/.cache/pip' \
  --set-env='TORCH_COMPILE_DISABLE=1' \
  --set-env='TORCHDYNAMO_DISABLE=1'"

# Distinct CPU threading per role
CHR_WRITER_STR="$CHR_BASE_STR \
  --set-env='OMP_NUM_THREADS=$W_C' --set-env='MKL_NUM_THREADS=$W_C' --set-env='OPENBLAS_NUM_THREADS=$W_C' \
  --set-env='SQLITE_TMPDIR=/workspace/tmp'"

CHR_PROD_STR="$CHR_BASE_STR \
  --set-env='OMP_NUM_THREADS=$PROD_OMP' --set-env='MKL_NUM_THREADS=$PROD_OMP' --set-env='OPENBLAS_NUM_THREADS=$PROD_OMP'"

# -------------------- Flags for writer & producers --------------------
WR_FLAGS="$QUIET_FLAG $OFFLINE_FLAG $COMMON_DB $COMMON_BS $COMMON_IDX \
  --hnsw-recall $HNSW_RECALL --top-chunks $TOP_CHUNKS --overshoot $OVERSHOOT \
  --faiss-writer --consume-segments --embed-workers 0 --no-llm \
  --shard-id 0 --num-shards $NSHARDS"
if [[ -n "${MANI_ABS:-}" ]]; then WR_FLAGS="$WR_FLAGS --tar-manifest '$MANI_ABS'"; else WR_FLAGS="$WR_FLAGS --tar-dir '$TAR_ABS'"; fi
if [[ "$REBUILD" == "1" ]]; then WR_FLAGS="$WR_FLAGS --rebuild"; [[ "$YES" == "1" ]] && WR_FLAGS="$WR_FLAGS -y"; else WR_FLAGS="$WR_FLAGS --update"; fi

PROD_FLAGS_BASE="$QUIET_FLAG $OFFLINE_FLAG $COMMON_BS \
  --no-llm --embed-producer --embed-workers 1 \
  --num-shards $NSHARDS"
if [[ -n "${MANI_ABS:-}" ]]; then PROD_FLAGS_BASE="$PROD_FLAGS_BASE --tar-manifest '$MANI_ABS'"; else PROD_FLAGS_BASE="$PROD_FLAGS_BASE --tar-dir '$TAR_ABS'"; fi

# ---- Bootstrap mode to avoid empty ANN deadlock ----
: "${ENABLE_LEXICAL:=0}"  # leave 0 until chunks.{faiss,trained.json} exist
export ENABLE_LEXICAL
if [[ "$ENABLE_LEXICAL" == "1" ]]; then
  PROD_FLAGS_BASE+=" --allow-global-lexical --min-chunks-per-paper 0"
fi

# -------------------- tmp roots (per-rank scratch) --------------------
TMP_ROOT="$WORKSPACE/tmp/emb_segments/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMP_ROOT"
export TMP_ROOT

echo "[plan] nodes=${SLURM_NNODES} gpn=${GPUS_PER_NODE} TOTAL_TASKS=$TOTAL_TASKS NSHARDS=$NSHARDS PRODUCERS=$((NSHARDS-1)) START_WRITER=$START_WRITER OFFLINE=$OFFLINE ENABLE_LEXICAL=$ENABLE_LEXICAL"

# -------------------- Preflight import (torch/faiss) --------------------
echo "[preflight] import torch/faiss on each node (OFFLINE=$OFFLINE)"
set +e
srun --mpi=none -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --gpus-per-task=1 --gpu-bind=per_task:1 -c 1 --cpu-bind=cores --kill-on-bad-exit=1 \
  bash -c "
    set -eo pipefail
    echo \"[preflight:\$HOSTNAME] starting\"
    $CHR_BASE_STR $IMG -- /usr/bin/env python3 -c 'import sys; ok=True
try:
    import torch; print(\"torch:\", getattr(torch, \"__version__\", \"unknown\"))
except Exception as e:
    ok=False; print(\"torch_import_error:\", e)
try:
    import faiss; print(\"faiss:\", getattr(faiss, \"__version__\", \"unknown\"))
except Exception as e:
    ok=False; print(\"faiss_import_error:\", e)
sys.exit(0 if ok else 84)'
  "
PRE_RC=$?
set -e
if [[ $PRE_RC -ne 0 ]]; then
  if [[ "$OFFLINE" == "1" ]]; then
    echo "[preflight] FAILED with OFFLINE=1. faiss/torch not importable on at least one node. Re-run with OFFLINE=0 once to warm caches, then retry offline." >&2
    exit $PRE_RC
  else
    echo "[preflight] non-zero rc=$PRE_RC with OFFLINE=0; proceeding (caches should be warmed now)."
  fi
fi

# Record job start epoch to defeat stale indices
JOB_TS=$(date +%s)
export JOB_TS

echo "[phase] unified-step: start $(date -Is)"

srun --mpi=none \
     --nodes="${SLURM_NNODES}" \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-$GPUS_PER_NODE}" \
     --gpus-per-task=1 --gpu-bind=per_task:1 \
     --distribution=block:block \
     -c "$P_C" --cpu-bind=cores --kill-on-bad-exit=1 \
     --output="$LOG_DIR/unified-%N-%t.%j.log" --error="$LOG_DIR/unified-%N-%t.%j.log" --open-mode=append \
     bash -c '
       set -eo pipefail
       SID=${SLURM_PROCID:?SLURM_PROCID missing}

       CVD="${CUDA_VISIBLE_DEVICES-}"
       [[ -z "${CVD}" ]] && CVD="${SLURM_LOCALID:-0}"
       NVVD="${NVIDIA_VISIBLE_DEVICES-${CVD}}"

       echo "[rank $SID] host=$HOSTNAME CVD=${CVD} start=$(date -Is)"

       if [[ "'"$START_WRITER"'" == "1" && "$SID" -eq 0 ]]; then
         echo "[writer] host=$HOSTNAME sid=$SID nshards="'"$NSHARDS"'" start $(date -Is)"
         exec '"$CHR_WRITER_STR"' \
           --set-env=CUDA_VISIBLE_DEVICES="${CVD}" \
           --set-env=NVIDIA_VISIBLE_DEVICES="${NVVD}" \
           '"$IMG"' -- litkit build '"$WR_FLAGS"'
       fi

       # Producers: shard IDs 1..(NSHARDS-1)
       if (( SID >= '"$NSHARDS"' )); then
         echo "[rank $SID] idle (SID >= NSHARDS)"; exit 0
       fi
       if [[ "$SID" -eq 0 ]]; then
         echo "[rank $SID] writer skipped (START_WRITER=0)"; exit 0
       fi

       echo "[producer] host=$HOSTNAME sid=$SID shard-id=$SID CVD=${CVD} start $(date -Is)"

       NEED_CHUNKS=1

       # Wait for freshly-created indices (mtime >= JOB_TS) to avoid stale files from prior runs.
       if [[ "'"$START_WRITER"'" == "1" ]]; then
         (
           SECONDS=0; W="'"$WORKSPACE"'"; LIM="'"$GATE_WAIT_MAX"'"; TS='"$JOB_TS"'
           mtime_ge(){ [[ -e "$1" ]] && { MT=$(stat -c %Y "$1" 2>/dev/null || echo 0); [[ "$MT" -ge "$TS" ]]; }; }

           while true; do
             HAVE_PAPERS=0; mtime_ge "$W/indices/papers.faiss" && HAVE_PAPERS=1
             HAVE_CHUNKS=0
             if mtime_ge "$W/indices/chunks.faiss"; then
               if [[ ! -e "$W/indices/chunks.trained.json" ]] || mtime_ge "$W/indices/chunks.trained.json"; then
                 HAVE_CHUNKS=1
               fi
             fi

             if (( HAVE_PAPERS == 1 )) && { (( NEED_CHUNKS == 0 )) || (( HAVE_CHUNKS == 1 )); }; then
               break
             fi

             (( SECONDS >= LIM )) && { echo "[producer $SID] timeout waiting for fresh indices after ${LIM}s" >&2; exit 124; }
             sleep 2
           done
         )
       fi

       RSEG="'"$TMP_ROOT"'/shard-$SID"; mkdir -p "$RSEG"
       exec '"$CHR_PROD_STR"' \
         --set-env=CUDA_VISIBLE_DEVICES="${CVD}" \
         --set-env=NVIDIA_VISIBLE_DEVICES="${NVVD}" \
         --set-env=TMPDIR="$RSEG" --set-env=SQLITE_TMPDIR="$RSEG" \
         '"$IMG"' -- litkit build '"$PROD_FLAGS_BASE"' --embed-devices cuda:0 --shard-id $SID
     '

UNIFIED_RC=$?
echo "[phase] unified-step: done rc=$UNIFIED_RC at $(date -Is)"
WR_RC=0
PROD_RC=$UNIFIED_RC

# -------------------- Writer finalize --------------------
WR2_FLAGS="$QUIET_FLAG $OFFLINE_FLAG $COMMON_DB $COMMON_BS $COMMON_IDX \
  --hnsw-recall $HNSW_RECALL --top-chunks $TOP_CHUNKS --overshoot $OVERSHOOT \
  --faiss-writer --consume-segments --embed-workers 0 --no-llm \
  --update --shard-id 0 --num-shards $NSHARDS"
if [[ -n "${MANI_ABS:-}" ]]; then WR2_FLAGS="$WR2_FLAGS --tar-manifest '$MANI_ABS'"; else WR2_FLAGS="$WR2_FLAGS --tar-dir '$TAR_ABS'"; fi

echo "[phase] writer-finalize: start $(date -Is)"
srun --mpi=none -N1 -n1 --gpus-per-task=1 --gpu-bind=per_task:1 -c "$W_C" --cpu-bind=cores --kill-on-bad-exit=1 \
     --output="$LOG_DIR/writer-finalize.%j.log" --error="$LOG_DIR/writer-finalize.%j.log" --open-mode=append \
     bash -c "exec $CHR_WRITER_STR \
       --set-env=CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
       --set-env=NVIDIA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
       '$IMG' -- litkit build $WR2_FLAGS"
WR2_RC=$?
echo "[phase] writer-finalize: done rc=$WR2_RC at $(date -Is)"

RC=$(( WR_RC | PROD_RC | WR2_RC ))
echo "[done] build complete with rc=$RC"
exit "$RC"