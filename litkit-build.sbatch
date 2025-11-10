#!/usr/bin/env bash
#SBATCH -J litkit-build
#SBATCH -p gpu-v100
#SBATCH -N 3
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

set -euo pipefail
ulimit -n 65535 || true
trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT INT TERM

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
: "${GATE_WAIT_MAX:=900}"

# -------------------- Paths & container --------------------
BASE="/path/to"
LITKIT_DIR="$BASE/litkit"
IMG="$LITKIT_DIR/sqfs/litkit-v0.3.33-aarch64-lean.sqfs"
CDI_SPEC_DIR="$BASE/$([[ "$TARGET" == "gh" ]] && echo cdi-grace || echo cdi-v100)"
HF_HOST="$BASE/hf_cache_persist"
WORKSPACE="$LITKIT_DIR/workspace"
LOG_DIR="$WORKSPACE/log/${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_HOST" "$WORKSPACE" "$LOG_DIR" "$WORKSPACE/indices" "$WORKSPACE/sqlite" "$WORKSPACE/tmp" "$WORKSPACE/emb_segments"
umask 002; chmod g+s "$WORKSPACE" "$WORKSPACE"/{log,sqlite,indices,emb_segments}

# Charliecloud
module load charliecloud/0.42 2>/dev/null || true
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
module load "cuda/$CUDA_VERSION_PIN" 2>/dev/null || true
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

# Sharding
TOTAL_TASKS="$(( (${SLURM_NNODES:-1}) * (${SLURM_NTASKS_PER_NODE:-$GPUS_PER_NODE}) ))"
if [[ -n "${MANI_ABS:-}" ]]; then
  NTARS=$(grep -vE '^\s*(#|$)' "$MANI_ABS" | wc -l | awk '{print $1}')
else
  NTARS=$(find "$TAR_ABS" -maxdepth 1 -type f -name '*.tar*' | wc -l | awk '{print $1}')
fi
NSHARDS=$(( NTARS < TOTAL_TASKS ? NTARS : TOTAL_TASKS ))
(( NSHARDS < 1 )) && { echo "FATAL: no shards detected"; exit 2; }
echo "[plan] NTARS=$NTARS TOTAL_TASKS=$TOTAL_TASKS -> NSHARDS=$NSHARDS (producers=$((NSHARDS-1)), writer uses shard 0)"
if (( NSHARDS < TOTAL_TASKS )); then echo "[plan] NOTE: under-sharded (NSHARDS < TOTAL_TASKS). Some GPUs will be idle."; fi

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
[[ "$QUIET" == "1" ]] && QUIET_FLAG="--quiet" || QUIET_FLAG=""
[[ "$OFFLINE" == "1" ]] && OFFLINE_FLAG="--offline" || OFFLINE_FLAG=""

# Bind shards
BIND_TARS=""
if [[ -n "${MANI_ABS:-}" ]]; then
  BIND_TARS+=" --bind '$(dirname "$MANI_ABS")':'$(dirname "$MANI_ABS")'"
  for d in "${TAR_PARENTS[@]:-}"; do [[ -d "$d" ]] && BIND_TARS+=" --bind '$d':'$d'"; done
else
  BIND_TARS+=" --bind '$TAR_ABS':'$TAR_ABS'"
fi

# Base ch-run strings
CHR_BASE_STR="$CHRUN --unset-env='*' \
  --set-env=HOME=/root \
  --cdi-dirs='$CDI_SPEC_DIR' --cdi=nvidia.com/gpu=all \
  --bind '$CUDA_LIBA':'$CUDA_LIBA' --bind '$CUDA_LIBB':'$CUDA_LIBB' \
  --bind '$HF_HOST':/app/hf_cache --bind '$WORKSPACE':/workspace $BIND_TARS \
  --set-env='LD_LIBRARY_PATH=$CUDA_LIBA:$CUDA_LIBB' \
  --set-env='PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --set-env='LITKIT_WORKSPACE=/workspace' \
  --set-env='HF_HOME=/app/hf_cache' \
  --set-env='TRANSFORMERS_USE_SAFE_TENSORS=1' \
  --set-env='PYTHONUNBUFFERED=1' \
  --set-env='TOKENIZERS_PARALLELISM=false' \
  --set-env='CUDA_DEVICE_ORDER=PCI_BUS_ID' \
  --set-env='LITKIT_EMBED_SEGMENT_SIZE=$LITKIT_EMBED_SEGMENT_SIZE'"

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

# -------------------- tmp roots (per-rank scratch) --------------------
TMP_ROOT="$WORKSPACE/tmp/emb_segments/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMP_ROOT"

echo "[phase] unified-step: start $(date -Is) nodes=${SLURM_NNODES} gpn=${GPUS_PER_NODE} (TOTAL_TASKS=$TOTAL_TASKS, NSHARDS=$NSHARDS)"

srun --mpi=none \
     --nodes="${SLURM_NNODES}" \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-$GPUS_PER_NODE}" \
     --gpus-per-task=1 --gpu-bind=per_task:1 \
     --distribution=block:block \
     -c "$P_C" --cpu-bind=cores --kill-on-bad-exit=1 \
     --output="$LOG_DIR/unified-%N-%t.%j.log" --error="$LOG_DIR/unified-%N-%t.%j.log" --open-mode=append \
     bash -lc '
       set -euo pipefail
       SID=$SLURM_PROCID
       # One GPU is made visible per task by Slurm; inside the task it is cuda:0.
       EMB_DEV="cuda:0"

       echo "[rank $SID] host=$HOSTNAME dev=$EMB_DEV CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is)"

       if [[ "$SID" -eq 0 ]]; then
         echo "[writer] $HOSTNAME sid=$SID nshards='"$NSHARDS"' start $(date -Is)"
         exec '"$CHR_WRITER_STR"' \
           --set-env=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
           --set-env=NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
           '"$IMG"' -- litkit '"$WR_FLAGS"'
       fi

       if (( SID >= '"$NSHARDS"' )); then
         echo "[rank $SID] idle (no shard)"; exit 0
       fi

       # Wait until real (non-empty) FAISS indices exist
       echo "[producer] waiting for FAISS indices..."
       timeout ${GATE_WAIT_MAX:-900} bash -lc '"'"'until [[ -s "'"$WORKSPACE"'/indices/papers.faiss" && -s "'"$WORKSPACE"'/indices/chunks.faiss" ]]; do sleep 2; done'"'"'

       RSEG="'"$TMP_ROOT"'/shard-$SID"; mkdir -p "$RSEG"
       echo "[producer] $HOSTNAME sid=$SID tmp=$RSEG dev=$EMB_DEV start $(date -Is)"
       exec '"$CHR_PROD_STR"' \
         --set-env=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
         --set-env=NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
         --set-env=TMPDIR="$RSEG" --set-env=SQLITE_TMPDIR="$RSEG" \
         '"$IMG"' -- litkit '"$PROD_FLAGS_BASE"' --embed-devices "$EMB_DEV" --shard-id $SID
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
srun --mpi=none -N1 -n1 --gpus-per-task=1 --gpu-bind=map_gpu:0 -c "$W_C" --cpu-bind=cores --kill-on-bad-exit=1 \
     --output="$LOG_DIR/writer-finalize.%j.log" --error="$LOG_DIR/writer-finalize.%j.log" --open-mode=append \
     bash -lc "exec $CHR_WRITER_STR \
       --set-env=CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
       --set-env=NVIDIA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
       '$IMG' -- litkit $WR2_FLAGS"
WR2_RC=$?
echo "[phase] writer-finalize: done rc=$WR2_RC at $(date -Is)"

RC=$(( WR_RC | PROD_RC | WR2_RC ))
echo "[done] build complete with rc=$RC"
exit "$RC"