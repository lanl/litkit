# ---- BUILD & RUN NOTES (read me) ----------------------------------------------
# Flavors:
#   - lean:  No CUDA userspace in image. At runtime you must provide host NVIDIA
#            device nodes (via CDI) and host CUDA libs (via module + bind).
#   - fat:   CUDA userspace injected into the squashed image at export time.
#            Simpler to run, but less portable: baked libs must be <= site driver.
#
# What we build here (defaults):
#   FLAVOR=lean, INJECT_NVIDIA=0, DOCKERFILE=Dockerfile.lean
#   -> Torch 2.5.1 (cp312, CUDA 12.5 toolchain) wheel is produced/used offline.
#
# Prereqs on the host:
#   - Charliecloud ≥ 0.42 (ch-image/ch-run/ch-convert/ch-fromhost)
#   - Optional: a Torch wheel staged at $WHEELS_HOST_DIR (speeds builds)
#   - For GPU runs (lean flavor):
#       * CDI specs available (we use --cdi and --cdi-dirs)
#       * A CUDA module whose toolkit version ≤ driver version (e.g. cuda/12.5.0)
#
# Typical workflows:
#
# (A) Build lean image (no libs baked, recommended for portability)
#     FLAVOR=lean INJECT_NVIDIA=0 DOCKERFILE=Dockerfile.lean just release
#
# (B) Run CPU-only (test)
#     just run python -c 'import sys; print(sys.version)'
#     just run litkit --version
#
# (C) Run with GPU on GH200 (HPC) using CDI + host CUDA 12.5
#     module purge; module load cuda/12.5.0
#     # one-time: ensure your CDI specs dir (e.g. /path/to/cdi-hpc) is correct:
#     #   nvidia-ctk cdi list --spec-dir=/path/to/cdi-hpc
#     just run-gpu python - <<'PY'
#     import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
#     PY
#
# (D) Produce a fat image (libs baked, less portable; only if you know targets)
#     FLAVOR=fat DOCKERFILE=Dockerfile.fat INJECT_NVIDIA=1 just build
#     # ch-fromhost --nvidia runs during export; build where the driver matches targets.
#
# Notes:
#   - Driver must be >= CUDA userspace. Example: driver 12.6 can run toolkit 12.5,
#     but not 12.8. Lean+CDI avoids this mismatch by using the site’s module at run time.
#   - The ask/new-index targets assume data binds exist (see run target).
#   - If litkit import fails with "libcudart.so.12 not found", you didn’t bind a toolkit
#     or LD_LIBRARY_PATH inside the container is missing the bound lib dir.
#   - Reset Charliecloud cache: `just reset` (destroys /var/tmp/$USER.ch).
# -------------------------------------------------------------------------------

set shell := ['bash', '-l', '-c']

# ---- Paths & tags ----
arch := env("ARCH", "aarch64")
flavor := env("FLAVOR", "lean")           # fat | lean
tag := "v0.3.35-" + arch + "-" + flavor
name := "litkit"
sqfs-path := "./sqfs" / name + "-" + tag + ".sqfs"

# Select which Dockerfile to use
dockerfile := env("DOCKERFILE", "Dockerfile.lean")

# Whether to inject host NVIDIA libs on export (1=yes, 0=no)
inject-nvidia := env("INJECT_NVIDIA", "0")
run-nvidia := env("RUN_NVIDIA", "0")

# Where we stash team wheels (host)
wheels-host := env("WHEELS_HOST_DIR", "/path/to/wheels/torch")

# Path to Charliecloud bin (override via env)
charlie-bin := env("CHARLIECLOUD_BIN", "/opt/site/aarch64/rhel8/charliecloud/0.42/bin")

# Container-internal working area (bind-mounted)
workspace := "workspace"
sqlite-dir := workspace / "sqlite"
indices-dir := workspace / "indices"
workspace-host := justfile_directory() / workspace

# Data locations (override via env)
test-tar-shards-host := env("TEST_TAR_SHARDS", "/path/to/test_tar_shards")
pmc-oa-host          := env("PMC_OA_DIR",      "/path/to/PMC-OA")

# Secrets/LLM (hosted LLM API defaults)
openai-api-key := env("OPENAI_API_KEY", "")
openai-base-url := env("OPENAI_BASE_URL", "https://llm.example.com")
llm-model := env("LLM_MODEL", "gpt-oss-120b")
ssl-cert-file := env("SSL_CERT_FILE", "/etc/ssl/certs/ca-bundle.crt")
llm-api-key-file := env("LLM_API_KEY_FILE", "~/.llm_api_key")
default-query  := "What is the role of follicular dendritic cells (FDCs) in HIV dynamics under antiretroviral therapy?"

help:
	just -l -u

# ---- Tunables (override via env) ----
hnsw-m         := env("HNSW_M", "32")
efconstruction := env("EFCONSTRUCTION", "200")
efsearch       := env("EFSEARCH", "128")
ivf-nlist      := env("IVF_NLIST", "16384")
pq-m           := env("PQ_M", "64")
nprobe         := env("NPROBE", "64")
paper-bs       := env("PAPER_EMBED_BS", "64")
chunk-bs       := env("CHUNK_EMBED_BS", "128")

# ---- SLURM defaults (override via env) ----
slurm-partition    := env("SLURM_PARTITION", "gpu-v100")
slurm-account      := env("SLURM_ACCOUNT",   "")
slurm-cpus         := env("SLURM_CPUS",      "16")
slurm-time         := env("SLURM_TIME",      "10:00:00")
slurm-mem          := env("SLURM_MEM",       "0")  # "0" -> omit --mem
slurm-gpus-writer  := env("SLURM_GPUS_WRITER",  "1")
slurm-gpus-readers := env("SLURM_GPUS_READERS", "0")

# ======================== Build primitives ========================
_build_image_dir:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p wheels
	# Ensure charliecloud
	if ! command -v ch-image >/dev/null 2>&1; then
	  if [ -x "{{ charlie-bin }}/ch-image" ]; then
	    export PATH="{{ charlie-bin }}:$PATH"
	  else
	    module --ignore_cache load charliecloud >/dev/null 2>&1 || true
	  fi
	fi
	command -v ch-image >/dev/null 2>&1 || { echo "ERROR: ch-image not found (PATH=$PATH)"; exit 1; }
	command -v ch-convert >/dev/null 2>&1 || { echo "ERROR: ch-convert not found (PATH=$PATH)"; exit 1; }
	# Build OCI and convert to unpacked dir
	mkdir -p sqfs
	unset CH_IMAGE_USERNAME CH_IMAGE_PASSWORD CH_IMAGE_AUTH
	ch-image build -f {{ dockerfile }} -t {{ name }}:{{ tag }} .
	ch-convert {{ name }}:{{ tag }} ./{{ name }}:{{ tag }}

# DEBUG --- add the lines of code below later
ensure-uv-lock:
	#!/usr/bin/env bash
	set -euo pipefail
	if [ ! -f uv.lock ]; then
	  echo "NOTE: uv.lock not found; creating placeholder. For reproducible builds, run 'uv lock' and commit the file."
	  : > uv.lock
	fi

# --- stage prebuilt Torch wheel into build context if available ---
stage-torch-wheel:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p wheels
	shopt -s nullglob
	# Prefer explicit env TORCH_WHEEL if provided; else auto-pick from wheels-host
	if [ -n "${TORCH_WHEEL:-}" ] && [ -f "${TORCH_WHEEL}" ]; then
	  src="${TORCH_WHEEL}"
	else
	  cands=( "{{ wheels-host }}/torch-2.5.1"*cp312*linux_aarch64.whl )
	  if [ ${#cands[@]} -gt 0 ]; then src="${cands[0]}"; else src=""; fi
	fi
	if [ -n "${src}" ]; then
	  echo "Staging Torch wheel: ${src}"
	  cp -v "${src}" wheels/
	else
	  echo "WARN: no matching Torch wheel found in {{ wheels-host }}; build will fall back to source compile."
	fi

# ======================== Wheel helpers ========================
use-wheel wheel-path:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p wheels
	cp -v "{{ wheel-path }}" wheels/

save-wheel:
	#!/usr/bin/env bash
	set -euo pipefail
	imgdir="./{{ name }}:{{ tag }}"
	[ -d "$imgdir" ] || { echo "Image dir $imgdir not found; run 'just build-wheel' first."; exit 1; }
	mkdir -p "{{ wheels-host }}"
	shopt -s nullglob
	found=( "$imgdir/opt/wheels/torch-2.5.1"*linux_aarch64.whl )
	[ ${#found[@]} -gt 0 ] || { echo "No torch wheel found in $imgdir/opt/wheels/"; exit 1; }
	for w in "${found[@]}"; do
	  sha="$(sha256sum "$w" | awk '{print $1}')"
	  bn="$(basename "$w")"
	  echo "$sha  $bn" > "$imgdir/opt/wheels/${bn}.SHA256"
	  cp -v "$w" "{{ wheels-host }}/"
	  cp -v "$imgdir/opt/wheels/${bn}.SHA256" "{{ wheels-host }}/${bn}.SHA256"
	done
	echo "Saved wheel(s) to {{ wheels-host }}"

build-wheel: ensure-uv-lock stage-torch-wheel _build_image_dir

export-sqfs:
	#!/usr/bin/env bash
	set -euo pipefail
	if ! command -v ch-fromhost >/dev/null 2>&1; then
	  if [ -x "{{ charlie-bin }}/ch-fromhost" ]; then
	    export PATH="{{ charlie-bin }}:$PATH"
	  else
	    module --ignore_cache load charliecloud >/dev/null 2>&1 || true
	  fi
	fi
	command -v ch-fromhost >/dev/null 2>&1 || { echo "ERROR: ch-fromhost not found (PATH=$PATH)"; exit 1; }
	command -v ch-convert  >/dev/null 2>&1 || { echo "ERROR: ch-convert not found (PATH=$PATH)"; exit 1; }
	mkdir -p sqfs
	if [ "{{ inject-nvidia }}" = "1" ]; then
	  ch-fromhost --nvidia ./{{ name }}:{{ tag }}
	else
	  echo "Skipping host NVIDIA injection (INJECT_NVIDIA={{ inject-nvidia }})"
	fi
	ch-convert ./{{ name }}:{{ tag }} {{ sqfs-path }}
	rm -rf ./{{ name }}:{{ tag }}

release: build-wheel save-wheel export-sqfs
	@# no-op

build: build-wheel export-sqfs

reset:
	#!/usr/bin/env bash
	set -euo pipefail
	if ! command -v ch-image >/dev/null 2>&1; then
	  if [ -x "{{ charlie-bin }}/ch-image" ]; then
	    export PATH="{{ charlie-bin }}:${PATH}"
	  else
	    module --ignore_cache load charliecloud >/dev/null 2>&1 || true
	  fi
	fi
	storage="${CH_IMAGE_STORAGE:-/var/tmp/${USER}.ch}"
	rm -rf "${storage}" "/ram/var/tmp/${USER}.ch" 2>/dev/null || true
	echo "Deleted Charliecloud storage at ${storage} (if it existed)."

# ======================== Preflight ========================
check:
	#!/usr/bin/env bash
	set -euo pipefail
	[ -f {{ sqfs-path }} ] || { echo "Missing image: {{ sqfs-path }}" >&2; exit 1; }
	[ -d "{{ pmc-oa-host }}" ] || { echo "Missing PMC OA dir: {{ pmc-oa-host }}" >&2; exit 1; }
	[ -d "{{ test-tar-shards-host }}" ] || { echo "Missing test tar shards dir: {{ test-tar-shards-host }}" >&2; exit 1; }
	echo "OK: image + data dirs present."

# ======================== Indexing (LLM-free) ========================
new-index-test: check
	just -f {{ justfile() }} run litkit --faiss-writer --build-only --rebuild --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000 --tar-dir /data/test_tar_shards --papers-index hnsw --hnsw-m {{ hnsw-m }} --efconstruction {{ efconstruction }} --efsearch {{ efsearch }} --chunks-index ivfpq --ivf-nlist {{ ivf-nlist }} --pq-m {{ pq-m }} --nprobe {{ nprobe }} --paper-embed-bs {{ paper-bs }} --chunk-embed-bs {{ chunk-bs }}

new-index: check
	just -f {{ justfile() }} run litkit --faiss-writer --build-only --rebuild --sqlite-journal-mode TRUNCATE --sqlite-busy-timeout-ms 180000 --tar-dir /data/pmc_oa --papers-index hnsw --hnsw-m {{ hnsw-m }} --efconstruction {{ efconstruction }} --efsearch {{ efsearch }} --chunks-index ivfpq --ivf-nlist {{ ivf-nlist }} --pq-m {{ pq-m }} --nprobe {{ nprobe }} --paper-embed-bs {{ paper-bs }} --chunk-embed-bs {{ chunk-bs }}

# ======================== Retrieve context and send prompt to LLM ========================
ask +query=default-query:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p "{{ workspace-host }}"
	printf 'litkit --llm-model={{ llm-model }} -- %q\n' "{{ query }}" > "{{ workspace-host }}/.ask.sh"
	chmod +x "{{ workspace-host }}/.ask.sh"
	just -f {{ justfile() }} run /workspace/.ask.sh

# ======================== Retrieve context only (no LLM prompt) ========================
ask-nollm +query=default-query:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p "{{ workspace-host }}"
	printf 'litkit --no-llm -- %q\n' "{{ query }}" > "{{ workspace-host }}/.ask_nollm.sh"
	chmod +x "{{ workspace-host }}/.ask_nollm.sh"
	just -f {{ justfile() }} run /workspace/.ask_nollm.sh

# ======================== Query from file (with diagnostics) ========================
# Reads question from workspace/question.txt by default
# Auto-loads API key from ~/.llm_api_key if OPENAI_API_KEY is not set
ask-file file="question.txt":
	#!/usr/bin/env bash
	set -euo pipefail
	
	# Auto-load API key from key file if not already set
	keyfile="{{ llm-api-key-file }}"
	keyfile="${keyfile/#\~/$HOME}"  # expand ~
	if [[ -z "${OPENAI_API_KEY:-}" ]]; then
	    if [[ -f "$keyfile" ]]; then
	        OPENAI_API_KEY="$(head -n1 "$keyfile" | tr -d '[:space:]')"
	        export OPENAI_API_KEY
	        echo "[info] Loaded API key from $keyfile"
	    fi
	fi
	
	# Check for API key
	if [[ -z "${OPENAI_API_KEY:-}" ]]; then
	    echo "ERROR: OPENAI_API_KEY not set and no key file found"
	    echo ""
	    echo "Option 1 - Create a key file:"
	    echo "  echo 'your-api-key' > ~/.llm_api_key && chmod 600 ~/.llm_api_key"
	    echo "  just ask-file"
	    echo ""
	    echo "Option 2 - Set environment variable:"
	    echo "  export OPENAI_API_KEY=\"\$(cat ~/.llm_api_key)\""
	    echo "  just ask-file"
	    exit 1
	fi
	
	qfile="{{ workspace-host }}/{{ file }}"
	[ -f "$qfile" ] || { echo "ERROR: $qfile not found"; exit 1; }
	
	echo "=== DIAGNOSTIC INFO ==="
	echo "Model:    {{ llm-model }}"
	echo "          (override: LLM_MODEL=gpt-oss-20b just ask-file)"
	echo "          (list available: curl -sH \"Authorization: Bearer \$OPENAI_API_KEY\" {{ openai-base-url }}/models | jq -r '.data[].id')"
	echo ""
	echo "Endpoint: {{ openai-base-url }}"
	echo "          (override: OPENAI_BASE_URL=https://api.openai.com/v1 just ask-file)"
	echo ""
	echo "API key:  set (${#OPENAI_API_KEY} chars)"
	echo "          (override: OPENAI_API_KEY=... just ask-file)"
	echo "          (key file: {{ llm-api-key-file }})"
	echo ""
	echo "SSL cert: {{ ssl-cert-file }}"
	echo "          (override: SSL_CERT_FILE=/path/to/cert just ask-file)"
	echo ""
	echo "Question: $qfile"
	echo "          (override: just ask-file file=other.txt)"
	echo "========================"
	echo ""
	
	printf 'litkit --llm-model={{ llm-model }} --question-file "/workspace/{{ file }}"\n' > "{{ workspace-host }}/.ask.sh"
	chmod +x "{{ workspace-host }}/.ask.sh"
	just -f {{ justfile() }} run /workspace/.ask.sh

# ======================== Shell ========================
shell:
	just -f {{ justfile() }} run bash

# ======================== Clean indices ========================
[confirm("Remove indices and sqlite?")]
clean:
	#!/usr/bin/env bash
	set -euo pipefail
	rm -rf {{ workspace }}/sqlite {{ workspace }}/indices

# ======================== Run wrapper (Charliecloud) ========================
run *cmd:
	#!/usr/bin/env bash
	set -euo pipefail
	[ -f "{{ sqfs-path }}" ] || { echo "ERROR: image not found: {{ sqfs-path }}. Run 'just build' or 'just release'."; exit 1; }
	# Ensure ch-run is available
	if ! command -v ch-run >/dev/null 2>&1; then
	  if [ -x "{{ charlie-bin }}/ch-run" ]; then
	    export PATH="{{ charlie-bin }}:$PATH"
	  else
	    module --ignore_cache load charliecloud >/dev/null 2>&1 || true
	  fi
	fi
	command -v ch-run >/dev/null 2>&1 || { echo "ERROR: ch-run not found in PATH=${PATH}"; exit 1; }
	# Data dirs
	[ -d "{{ pmc-oa-host }}" ] || { echo "ERROR: PMC OA dir not found on host: {{ pmc-oa-host }}" >&2; exit 1; }
	[ -d "{{ test-tar-shards-host }}" ] || { echo "ERROR: test tar shards dir not found on host: {{ test-tar-shards-host }}" >&2; exit 1; }
	# Workspace subdirs
	mkdir -p "{{ workspace-host }}" "{{ sqlite-dir }}" "{{ indices-dir }}"
	# Threads
	export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-{{ slurm-cpus }}}"
	ch-run "{{ sqfs-path }}" \
	  --unset-env='*' \
	  --set-env="PATH=/root/.local/share/uv/tools/litkit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
	  --set-env="PYTHONPATH=/root/.local/share/uv/tools/litkit/lib/python3.12/site-packages" \
	  --set-env="HOME=/workspace" \
	  --set-env="OPENAI_API_KEY={{ openai-api-key }}" \
	  $( [ -n "{{ openai-base-url }}" ] && echo --set-env="OPENAI_BASE_URL={{ openai-base-url }}" ) \
	  --set-env="LITKIT_OPENAI_TIMEOUT_SEC={{ env("LITKIT_OPENAI_TIMEOUT_SEC", "600") }}" \
	  --set-env="SSL_CERT_FILE=/workspace/host-ca.pem" \
	  --set-env="REQUESTS_CA_BUNDLE=/workspace/host-ca.pem" \
	  --set-env="CURL_CA_BUNDLE=/workspace/host-ca.pem" \
	  --set-env="HF_HOME=/app/hf_cache" \
	  --set-env="LITKIT_WORKSPACE=/workspace" \
	  --set-env="TOKENIZERS_PARALLELISM=false" \
	  --set-env="LC_ALL=C.UTF-8" \
	  --set-env="LANG=C.UTF-8" \
	  --set-env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
	  --set-env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
	  --set-env="OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
	  --bind "{{ workspace-host }}:/workspace" \
	  --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/workspace/host-ca.pem" \
	  --bind "{{ test-tar-shards-host }}:/data/test_tar_shards" \
	  --bind "{{ pmc-oa-host }}:/data/pmc_oa" \
	  --cd /workspace -- bash -lc '\
	    mkdir -p /workspace/hf_cache; \
	    [ -e /workspace/hf_cache/hub ] || ln -sfn /app/hf_cache/hub /workspace/hf_cache/hub; \
	    # Optional, runtime-only conversion for SPECTER2 (lean builds). \
	    if [ -f /app/hf_cache/hub/models--allenai--specter2_base/pytorch_model.bin ] \
	      && [ ! -f /app/hf_cache/hub/models--allenai--specter2_base/model.safetensors ]; then \
	      if python -c "import torch" >/dev/null 2>&1; then \
	        printf "%s\n" \
	          "import os, torch" \
	          "base=\"/app/hf_cache/hub/models--allenai--specter2_base\"" \
	          "b=os.path.join(base,\"pytorch_model.bin\")" \
	          "s=os.path.join(base,\"model.safetensors\")" \
	          "if os.path.exists(b) and not os.path.exists(s):" \
	          "    sd=torch.load(b, map_location=\"cpu\", weights_only=True)" \
	          "    sd=sd.get(\"state_dict\", sd) if isinstance(sd, dict) else sd" \
	          "    from safetensors.torch import save_file as _sf; _sf(sd, s)" \
	          "print(\"SPECTER2: converted .bin→.safetensors\" if os.path.exists(s) else \"SPECTER2: no conversion needed\")" \
	          > /tmp/conv.py; \
	        python /tmp/conv.py; rm -f /tmp/conv.py; \
	      else \
	        echo "Torch not importable (likely no NVIDIA libs); skipping safetensors conversion."; \
	      fi; \
	    fi; \
	    exec "$@"' x {{cmd}}

# ======================== Run wrapper (quoted command) ========================
runq +line:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p "{{ workspace-host }}"
	printf '%s' "{{ line }}" > "{{ workspace-host }}/.runq.cmd"
	entry_exec="{{ workspace-host }}/.runq-exec.sh"
	{
		echo '#!/usr/bin/env bash'
		echo 'set -euo pipefail'
		echo 'exec bash -lc -- "$(cat /workspace/.runq.cmd)"'
	} > "$entry_exec"
	chmod +x "$entry_exec"
	just -f {{ justfile() }} run /workspace/.runq-exec.sh
