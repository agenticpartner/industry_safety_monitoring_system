#!/usr/bin/env bash
# One-shot installer. Runs ON THE JETSON. Idempotent — safe to re-run at any point.
#
#   ./scripts/setup.sh                  check hardware, then install everything
#   ./scripts/setup.sh --check-only     run the hardware probe and stop
#   ./scripts/setup.sh --skip-apt       skip system packages (already installed, or no sudo)
#   ./scripts/setup.sh --no-perf-mode   do not touch nvpmodel / jetson_clocks
#   ./scripts/setup.sh --yes            do not prompt
#
# What it installs, and why each one is here rather than assumed:
#
#   1. system packages   ffmpeg (media + clips), redis-server (event bus), libmosquitto1 (the
#                        tracker dlopens it), cmake/build-essential (parser + llama.cpp)
#   2. build/venv-export torch-cpu + ultralytics + onnx, for .pt -> ONNX only. Deliberately CPU
#                        torch: TensorRT does every inference, so a CUDA torch build would be a
#                        multi-GB download for zero benefit.
#   3. build/venv-services  FastAPI, uvicorn, redis, PyYAML. The services must NOT be installed
#                        into system python — that is where pyservicemaker lives and the pipeline
#                        imports it.
#   4. build/venv-hf     huggingface_hub, used only to pull model weights.
#   5. build/bin/mediamtx  republishes the tiled output as RTSP/WebRTC/HLS.
#   6. models/parser     the custom YOLO output parser (.so) that nvinfer loads.
#   7. .env              created from .env.example, chmod 600, never committed.
#   8. performance mode  MAXN + locked clocks. Benchmarks are meaningless without it.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source scripts/env.sh

CHECK_ONLY=0; SKIP_APT=0; NO_PERF=0; ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    --check-only)   CHECK_ONLY=1 ;;
    --skip-apt)     SKIP_APT=1 ;;
    --no-perf-mode) NO_PERF=1 ;;
    --yes|-y)       ASSUME_YES=1 ;;
    --help|-h)      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a  (--help for usage)"; exit 1 ;;
  esac
done

step() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
# 0. hardware gate
# ---------------------------------------------------------------------------------------------
bash scripts/check_hardware.sh || die "hardware pre-flight failed — see the FAIL lines above"
[ -f build/hardware.env ] && { set -a; . build/hardware.env; set +a; }
[ "$CHECK_ONLY" = 1 ] && { echo; echo "--check-only: stopping here."; exit 0; }

if [ "$ASSUME_YES" != 1 ]; then
  echo
  read -r -p "Proceed with installation into ${ROOT}/build ? [y/N] " reply
  case "$reply" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

# ---------------------------------------------------------------------------------------------
step "[1/8] system packages"
# ---------------------------------------------------------------------------------------------
# libmosquitto1 is not optional despite looking like an MQTT dependency this project has no use
# for: libnvds_nvmultiobjecttracker.so dlopens it, and without it the tracker fails to load with
# "Failed to initilaize low level lib" — a message that names the tracker, not the library.
PKGS=(ffmpeg mediainfo bc curl git make cmake build-essential
      python3-venv python3-yaml redis-server libmosquitto1)
if [ "$SKIP_APT" = 1 ]; then
  echo "    --skip-apt: skipping"
else
  NEED=()
  for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || NEED+=("$p"); done
  if [ ${#NEED[@]} -gt 0 ]; then
    echo "    installing: ${NEED[*]}"
    sudo apt-get update -qq || die "apt-get update failed"
    sudo apt-get install -y -qq "${NEED[@]}" || die "apt-get install failed"
  else
    echo "    all present"
  fi
fi

# ---------------------------------------------------------------------------------------------
step "[2/8] redis"
# ---------------------------------------------------------------------------------------------
# The event bus between the pipeline (system python, RESP over a plain socket) and the services
# (venv, real redis client). Enabled so the demo survives a reboot.
if command -v redis-cli >/dev/null 2>&1; then
  sudo systemctl enable --now redis-server >/dev/null 2>&1 || true
  if redis-cli ping >/dev/null 2>&1; then
    echo "    redis responding on :6379"
  else
    echo "    !! redis installed but not responding — start it with:"
    echo "       sudo systemctl start redis-server"
  fi
else
  echo "    !! redis-cli not found; the event, clip and reasoning services will not run"
fi

# ---------------------------------------------------------------------------------------------
step "[3/8] build/venv-export  (ONNX export)"
# ---------------------------------------------------------------------------------------------
VENV_EXPORT="${ROOT}/build/venv-export"
if [ ! -x "${VENV_EXPORT}/bin/python3" ]; then
  python3 -m venv "$VENV_EXPORT" || die "could not create ${VENV_EXPORT}"
  "${VENV_EXPORT}/bin/pip" install -q --upgrade pip
fi
if ! "${VENV_EXPORT}/bin/python3" -c "import ultralytics, onnx" 2>/dev/null; then
  echo "    installing torch (cpu) + ultralytics + onnx — several minutes"
  "${VENV_EXPORT}/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision || die "torch install failed"
  "${VENV_EXPORT}/bin/pip" install -q -r requirements/export.txt || die "export deps failed"
fi
"${VENV_EXPORT}/bin/python3" - <<'PY'
import torch, ultralytics, onnx
print(f"    torch {torch.__version__} | ultralytics {ultralytics.__version__} | onnx {onnx.__version__}")
PY

# ---------------------------------------------------------------------------------------------
step "[4/8] build/venv-services  (API + event/notify services)"
# ---------------------------------------------------------------------------------------------
# Separate from system python on purpose. The pipeline runs on system python because that is
# where pyservicemaker lives, and nothing may be installed there; app/events.py is dependency-free
# for exactly this reason and speaks RESP over a plain socket.
VENV_SERVICES="${ROOT}/build/venv-services"
if [ ! -x "${VENV_SERVICES}/bin/python3" ]; then
  python3 -m venv "$VENV_SERVICES" || die "could not create ${VENV_SERVICES}"
  "${VENV_SERVICES}/bin/pip" install -q --upgrade pip
fi
"${VENV_SERVICES}/bin/pip" install -q -r requirements/services.txt || die "service deps failed"
"${VENV_SERVICES}/bin/python3" - <<'PY'
import fastapi, uvicorn, redis, yaml
print(f"    fastapi {fastapi.__version__} | uvicorn {uvicorn.__version__} | redis {redis.__version__}")
PY

# ---------------------------------------------------------------------------------------------
step "[5/8] build/venv-hf  (model downloads)"
# ---------------------------------------------------------------------------------------------
VENV_HF="${ROOT}/build/venv-hf"
if [ ! -x "${VENV_HF}/bin/python3" ]; then
  python3 -m venv "$VENV_HF" || die "could not create ${VENV_HF}"
  "${VENV_HF}/bin/pip" install -q --upgrade pip
fi
"${VENV_HF}/bin/pip" install -q -r requirements/hf.txt || die "huggingface_hub install failed"
echo "    huggingface_hub ready"

# ---------------------------------------------------------------------------------------------
step "[6/8] mediamtx  (RTSP / WebRTC / HLS republisher)"
# ---------------------------------------------------------------------------------------------
MEDIAMTX_VER="v1.20.0"
mkdir -p "${ROOT}/build/bin"
if [ ! -x "${ROOT}/build/bin/mediamtx" ]; then
  URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VER}/mediamtx_${MEDIAMTX_VER}_linux_arm64.tar.gz"
  echo "    fetching ${MEDIAMTX_VER}"
  curl -fsSL "$URL" | tar xz -C "${ROOT}/build/bin" mediamtx || die "mediamtx download failed"
fi
echo "    $("${ROOT}/build/bin/mediamtx" --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------------------------
step "[7/8] custom YOLO output parser"
# ---------------------------------------------------------------------------------------------
# nvinfer loads this .so to turn each detector's raw tensor into bounding boxes. Both nvinfer
# configs reference it by path, so the pipeline cannot start without it.
if make -C models/parser >/dev/null 2>&1; then
  echo "    built $(ls models/parser/*.so 2>/dev/null | head -1)"
else
  echo "    !! parser build failed — run 'make -C models/parser' to see the error"
  echo "       (needs the DeepStream headers under ${DS_ROOT}/sources/includes)"
fi

# ---------------------------------------------------------------------------------------------
step "[8/8] credentials file and performance mode"
# ---------------------------------------------------------------------------------------------
if [ ! -f "${ROOT}/.env" ]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
  chmod 600 "${ROOT}/.env"
  echo "    created .env from .env.example — fill in your own tokens (it is gitignored)"
else
  chmod 600 "${ROOT}/.env"
  echo "    .env already present, left untouched"
fi

if [ "$NO_PERF" = 1 ]; then
  echo "    --no-perf-mode: leaving nvpmodel / clocks alone"
else
  # MAXN + locked clocks. Without this the governor throttles mid-run and throughput numbers
  # vary by tens of percent between identical runs.
  sudo nvpmodel -m 0 >/dev/null 2>&1 || true
  sudo jetson_clocks 2>/dev/null || true
  echo "    nvpmodel: $(sudo nvpmodel -q 2>/dev/null | head -1 || echo 'unavailable')"
fi

mkdir -p data logs media/src

# ---------------------------------------------------------------------------------------------
cat <<EOF

==============================================================
 Setup complete.
==============================================================
  export venv    : ${VENV_EXPORT}/bin/python3
  services venv  : ${VENV_SERVICES}/bin/python3
  hf venv        : ${VENV_HF}/bin/python3
  mediamtx       : ${ROOT}/build/bin/mediamtx
  trtexec        : $(command -v trtexec || echo 'NOT ON PATH — engine builds will fail')

 Next, in order:

   1. Put your own tokens in .env            (HF_TOKEN is needed for step 4)
   2. Drop source footage into media/src/, then:
        ./scripts/make_streams.sh 20
   3. Detector models -> ONNX -> TensorRT engines:
        ./build/venv-export/bin/python3 scripts/export_models.py all
        ./scripts/build_engines.sh all
   4. Optional reasoning layer (VLM + agent LLM):
        ./scripts/setup_reasoning.sh llamacpp --serve
        ./scripts/setup_reasoning.sh llm --serve
   5. Bring the demo up:
        ./scripts/demo_up.sh 20

 Full walkthrough: README.md
==============================================================
EOF
