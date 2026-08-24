#!/usr/bin/env bash
# Pre-flight hardware and software probe. Runs standalone, and is also phase 0 of setup.sh.
#
#   ./scripts/check_hardware.sh            full report
#   ./scripts/check_hardware.sh --quiet    only WARN/FAIL lines
#
# Exit code is 0 when nothing FAILed, 1 otherwise, so it can gate a deploy.
#
# Two platforms, one script. `jetson` is the reference target; `dgpu` is x86_64 with a discrete
# NVIDIA card, which is how the Docker image runs. The checks that differ are the ones tied to
# Tegra silicon — DLA, tegrastats, NVDEC device nodes, the board string — and each is gated on
# PLATFORM rather than skipped by accident.
#
# Every check below exists because getting it wrong produces a failure a long way from its cause.
# The tracker's "Failed to initilaize low level lib", nvinfer's "setDimensions: Error Code 3" and
# nv3dsink's "no display found" all look like pipeline bugs and are all environment problems.
# Finding them here costs seconds; finding them at runtime costs an afternoon.
set -uo pipefail
cd "$(dirname "$0")/.."

QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

PASS=0; WARN=0; FAIL=0
C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_BAD=$'\033[31m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_WARN=""; C_BAD=""; C_OFF=""; }

ok()   { PASS=$((PASS+1)); [ "$QUIET" = 1 ] || printf "  ${C_OK}PASS${C_OFF}  %-26s %s\n" "$1" "${2:-}"; }
warn() { WARN=$((WARN+1)); printf "  ${C_WARN}WARN${C_OFF}  %-26s %s\n" "$1" "${2:-}"; }
bad()  { FAIL=$((FAIL+1)); printf "  ${C_BAD}FAIL${C_OFF}  %-26s %s\n" "$1" "${2:-}"; }
head_() { [ "$QUIET" = 1 ] || printf "\n\033[1m%s\033[0m\n" "$1"; }

# Values other scripts can consume: this script writes them to build/hardware.env when it can.
DETECTED_BOARD=""; DETECTED_L4T=""; DETECTED_CUDA_ARCH=""; DETECTED_DISPLAY=""
DETECTED_DLA=0; DETECTED_RAM_GB=0; DETECTED_PLATFORM=""; DETECTED_VRAM_MB=0

[ "$QUIET" = 1 ] || cat <<'BANNER'
==============================================================
 Industry Safety Monitoring — hardware & software pre-flight
==============================================================
BANNER

# ---------------------------------------------------------------------------------------------
head_ "Platform"
# ---------------------------------------------------------------------------------------------
ARCH="$(uname -m)"

# Tegra is identified by the device tree, never by the architecture alone: aarch64 is also every
# ARM server and every Apple VM. /proc/device-tree/model is the authoritative board string there,
# and it is NUL-terminated, hence the tr.
if [ -r /proc/device-tree/model ]; then
  DETECTED_BOARD="$(tr -d '\0' < /proc/device-tree/model)"
fi
if [ -n "$DETECTED_BOARD" ] || [ -r /etc/nv_tegra_release ]; then
  DETECTED_PLATFORM=jetson
elif [ "$ARCH" = "x86_64" ] && command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_PLATFORM=dgpu
else
  DETECTED_PLATFORM=unknown
fi

case "$DETECTED_PLATFORM" in
  jetson) ok "platform" "$ARCH — NVIDIA Jetson (reference target)" ;;
  dgpu)   ok "platform" "$ARCH — discrete NVIDIA GPU" ;;
  *)      bad "platform" "$ARCH with no Tegra device tree and no nvidia-smi — no NVIDIA GPU found" ;;
esac

if [ "$DETECTED_PLATFORM" = jetson ]; then
  if [ -n "$DETECTED_BOARD" ]; then
    case "$DETECTED_BOARD" in
      *AGX*Orin*|*"AGX Orin"*)  ok   "board" "$DETECTED_BOARD" ;;
      *Orin*)                   warn "board" "$DETECTED_BOARD — measured on AGX Orin 64GB; expect a lower stream ceiling here" ;;
      *Thor*|*T264*)            warn "board" "$DETECTED_BOARD — newer than the measured target; CUDA arch and DLA behaviour differ" ;;
      *)                        warn "board" "$DETECTED_BOARD — not an Orin; nothing below is calibrated for it" ;;
    esac
  else
    bad "board" "no /proc/device-tree/model — this does not look like a Jetson"
  fi

  if [ -r /etc/nv_tegra_release ]; then
    DETECTED_L4T="$(awk '{print $2, $5, $6}' /etc/nv_tegra_release | tr -d ',' | head -1)"
    ok "L4T release" "$DETECTED_L4T"
  elif dpkg -s nvidia-jetpack >/dev/null 2>&1; then
    DETECTED_L4T="$(dpkg -s nvidia-jetpack 2>/dev/null | awk '/^Version:/{print $2}')"
    ok "JetPack" "$DETECTED_L4T"
  else
    warn "L4T / JetPack" "could not determine version"
  fi

elif [ "$DETECTED_PLATFORM" = dgpu ]; then
  DETECTED_BOARD="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  DETECTED_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  DETECTED_VRAM_MB="${DETECTED_VRAM_MB:-0}"
  [ -n "$DETECTED_BOARD" ] && ok "gpu" "$DETECTED_BOARD" || bad "gpu" "nvidia-smi returned no device"

  # VRAM is the constraint that does not exist on Jetson, where the 64 GB is unified and the model
  # servers draw from the same pool as the pipeline. Here they compete for a fixed, much smaller
  # budget: ~13 GB for both llama.cpp servers, ~7 GB for the engines and twenty decode contexts.
  if   [ "${DETECTED_VRAM_MB:-0}" -ge 40000 ]; then ok   "VRAM" "$((DETECTED_VRAM_MB / 1024)) GB"
  elif [ "${DETECTED_VRAM_MB:-0}" -ge 20000 ]; then warn "VRAM" "$((DETECTED_VRAM_MB / 1024)) GB — fits the full stack with little headroom; quantise the VLM if llama.cpp fails to allocate"
  elif [ "${DETECTED_VRAM_MB:-0}" -gt 0 ];     then warn "VRAM" "$((DETECTED_VRAM_MB / 1024)) GB — run the vision path only, or reduce the stream count"
  else                                              warn "VRAM" "could not read memory.total"
  fi

  # DeepStream 9.1 documents driver 590+ for dGPU. A lower kernel driver still works when the
  # container supplies its own newer user-mode driver (CUDA forward compatibility, datacenter GPUs
  # only) — measured working on 580.126.20 with an L4. Outside a container it is a real floor.
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  DRV_MAJOR="${DRV%%.*}"
  if [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 590 ] 2>/dev/null; then
    ok "driver" "$DRV"
  elif [ -n "${CUDA_DRIVER_CAPABILITIES:-}" ] || [ -d /usr/local/cuda/compat ]; then
    ok "driver" "$DRV (below the documented 590+, satisfied by CUDA forward compatibility)"
  else
    warn "driver" "$DRV — DeepStream 9.1 documents 590+ for dGPU; run in the container, which carries a newer user-mode driver"
  fi
fi

# ---------------------------------------------------------------------------------------------
head_ "Capacity"
# ---------------------------------------------------------------------------------------------
DETECTED_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
# The two llama.cpp servers hold ~12 GB resident between them; the pipeline, TensorRT engines and
# 20 decode contexts want the rest. 32 GB runs the vision path but is tight with both models up.
if   [ "$DETECTED_RAM_GB" -ge 60 ]; then ok   "system memory" "${DETECTED_RAM_GB} GB"
elif [ "$DETECTED_RAM_GB" -ge 30 ]; then warn "system memory" "${DETECTED_RAM_GB} GB — enough for the pipeline; run one model server at a time"
elif [ "$DETECTED_RAM_GB" -gt 0 ];  then warn "system memory" "${DETECTED_RAM_GB} GB — reduce the stream count and skip the reasoning layer"
else                                     warn "system memory" "could not read /proc/meminfo"
fi

DISK_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
DISK_GB="${DISK_GB:-0}"
# ~8 GB of model weights, ~4 GB of engines and build artifacts, a 4 GB clip budget, plus media.
if   [ "$DISK_GB" -ge 40 ]; then ok   "free disk" "${DISK_GB} GB"
elif [ "$DISK_GB" -ge 25 ]; then warn "free disk" "${DISK_GB} GB — enough to install, tight once clips accumulate"
else                             bad  "free disk" "${DISK_GB} GB — need ~25 GB for models, engines, media and clips"
fi

# ---------------------------------------------------------------------------------------------
head_ "DeepStream / TensorRT"
# ---------------------------------------------------------------------------------------------
DS_ROOT="${DS_ROOT:-/opt/nvidia/deepstream/deepstream}"
if [ -d "$DS_ROOT" ]; then
  DS_VER="$(cat "${DS_ROOT}/version" 2>/dev/null | head -1)"
  [ -n "$DS_VER" ] || DS_VER="$(basename "$(readlink -f "$DS_ROOT")")"
  ok "DeepStream" "${DS_VER:-present} at ${DS_ROOT}"
else
  bad "DeepStream" "not found at ${DS_ROOT} — install the DeepStream SDK, or set DS_ROOT"
fi

if command -v trtexec >/dev/null 2>&1; then
  ok "trtexec" "$(command -v trtexec)"
else
  # It ships inside the TensorRT package but is not always on PATH.
  if [ -x /usr/src/tensorrt/bin/trtexec ]; then
    warn "trtexec" "present at /usr/src/tensorrt/bin/trtexec but NOT on PATH — engine builds will fail"
  else
    bad "trtexec" "not found — TensorRT samples package missing; engines cannot be built"
  fi
fi

TRT_VER="$(dpkg -l 2>/dev/null | awk '/libnvinfer-bin|tensorrt /{print $3; exit}')"
[ -n "$TRT_VER" ] && ok "TensorRT" "$TRT_VER" || warn "TensorRT" "version not determined via dpkg"

# ---------------------------------------------------------------------------------------------
head_ "GStreamer plugins"
# ---------------------------------------------------------------------------------------------
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  ok "gstreamer" "$(gst-inspect-1.0 --version 2>/dev/null | awk '/version/{print $NF; exit}')"
  # Each of these is used directly by app/safety_pipeline.py. A missing one fails at graph
  # construction with "no element named X", which reads like a typo in our code.
  for el in nvurisrcbin nvstreammux nvinfer nvtracker nvdsanalytics nvmultistreamtiler nvdsosd nvvideoconvert; do
    if gst-inspect-1.0 "$el" >/dev/null 2>&1; then
      ok "plugin ${el}" ""
    else
      bad "plugin ${el}" "missing — DeepStream plugins not registered (try: rm -rf ~/.cache/gstreamer-1.0)"
    fi
  done
  # This one is only needed for the local display sink, so its absence is not fatal.
  gst-inspect-1.0 nv3dsink >/dev/null 2>&1 && ok "plugin nv3dsink" "" \
    || warn "plugin nv3dsink" "missing — local display output unavailable, dashboard/RTSP still work"
else
  bad "gstreamer" "gst-inspect-1.0 not found"
fi

# ---------------------------------------------------------------------------------------------
head_ "Python bindings"
# ---------------------------------------------------------------------------------------------
# The pipeline uses pyservicemaker, NOT pyds. They are different bindings shipped by the same SDK
# and only one of them is installed on a default DeepStream 9.x image.
if python3 -c "import pyservicemaker" >/dev/null 2>&1; then
  ok "pyservicemaker" "importable from system python3"
else
  bad "pyservicemaker" "not importable — install the DeepStream Python bindings (${DS_ROOT}/service-maker)"
fi
python3 -c "import yaml" >/dev/null 2>&1 && ok "pyyaml (system)" "" \
  || warn "pyyaml (system)" "missing — setup.sh installs python3-yaml"

# ---------------------------------------------------------------------------------------------
head_ "Accelerators"
# ---------------------------------------------------------------------------------------------
# Compute capability decides the llama.cpp CUDA build flag. Orin is sm_87; guessing wrong makes
# the build succeed and every kernel launch fail.
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
fi
if [ -z "$DETECTED_CUDA_ARCH" ]; then
  case "$DETECTED_BOARD" in
    *Orin*) DETECTED_CUDA_ARCH=87 ;;
    *Thor*|*T264*) DETECTED_CUDA_ARCH=110 ;;
    *Xavier*) DETECTED_CUDA_ARCH=72 ;;
  esac
fi
[ -n "$DETECTED_CUDA_ARCH" ] && ok "CUDA arch" "sm_${DETECTED_CUDA_ARCH}" \
  || warn "CUDA arch" "not determined — set CUDA_ARCH before building llama.cpp"

if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc" "$(nvcc --version | awk '/release/{print $5, $6}' | tr -d ',')"
else
  [ -x /usr/local/cuda/bin/nvcc ] \
    && warn "nvcc" "present at /usr/local/cuda/bin/nvcc but not on PATH" \
    || warn "nvcc" "not found — needed only for the optional reasoning layer (llama.cpp)"
fi

if [ "$DETECTED_PLATFORM" = jetson ]; then
  # DLA is optional: no model has passed the all-or-nothing qualification gate, so the shipped
  # configuration is GPU-only. Reported because qualify_dla.sh needs to know the cores exist.
  DETECTED_DLA=$(ls -d /sys/class/nvdla* /dev/nvhost-nvdla* 2>/dev/null | wc -l | tr -d ' ')
  [ "${DETECTED_DLA:-0}" -gt 0 ] && ok "DLA cores" "$DETECTED_DLA detected (optional — models run on GPU)" \
    || warn "DLA cores" "none detected — scripts/qualify_dla.sh will not apply"

  command -v tegrastats >/dev/null 2>&1 && ok "tegrastats" "$(command -v tegrastats)" \
    || warn "tegrastats" "not found — the dashboard system chart will be empty"

  # NVDEC is what caps the stream count: 20×1080p30 H.265 is 91% of the measured AGX Orin ceiling.
  if [ -e /dev/nvidia0 ] || [ -e /dev/nvhost-nvdec ] || [ -d /sys/class/nvidia-nvdec ]; then
    ok "NVDEC" "present"
  else
    warn "NVDEC" "device node not visible — decode ceiling unverified (run scripts/decode_sweep.sh)"
  fi

else
  # No DLA on any discrete card, and no tegrastats: services/metrics.py reads nvidia-smi instead.
  # `utilization.decoder` is the NVDEC field the dashboard binds to, and it is not universal —
  # it is absent on some consumer cards, where the system chart loses that one series.
  if nvidia-smi --query-gpu=utilization.decoder --format=csv,noheader >/dev/null 2>&1; then
    ok "NVDEC telemetry" "nvidia-smi reports utilization.decoder"
  else
    warn "NVDEC telemetry" "nvidia-smi does not report utilization.decoder — the dashboard NVDEC series will be empty"
  fi

  # NVENC is what the --rtsp-out branch encodes the tiled wall with. A100 and H100 have none, and
  # nvv4l2h264enc fails at graph construction there rather than at run time.
  if nvidia-smi --query-gpu=utilization.encoder --format=csv,noheader >/dev/null 2>&1; then
    ok "NVENC telemetry" "nvidia-smi reports utilization.encoder"
  else
    warn "NVENC" "no encoder telemetry — if this card has no NVENC, run without --rtsp-out"
  fi
fi

# ---------------------------------------------------------------------------------------------
head_ "Runtime dependencies"
# ---------------------------------------------------------------------------------------------
for c in ffmpeg ffprobe curl git make cmake; do
  command -v "$c" >/dev/null 2>&1 && ok "$c" "" || warn "$c" "missing — setup.sh installs it"
done

if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli ping >/dev/null 2>&1; then ok "redis" "responding on :6379"
  else warn "redis" "installed but not responding — sudo systemctl start redis-server"; fi
else
  warn "redis" "not installed — setup.sh installs it (the event bus needs it)"
fi

# The tracker's libnvds_nvmultiobjecttracker.so dlopens libmosquitto. Without it the load fails
# with "Failed to initilaize low level lib", which names the tracker and not the missing library.
if ldconfig -p 2>/dev/null | grep -q libmosquitto; then
  ok "libmosquitto" "present (required by the tracker)"
else
  warn "libmosquitto" "missing — the tracker will fail to load; setup.sh installs it"
fi

# ---------------------------------------------------------------------------------------------
head_ "Display"
# ---------------------------------------------------------------------------------------------
# A socket proves an X server exists, NOT that this user may connect to it: at the login screen
# :0 belongs to the display manager's greeter, root-owned and with no xauth cookie for the
# service user. That distinction is not cosmetic — a set-but-unreachable DISPLAY makes
# nvbufsurftransform fail EGL init and the pipeline never reaches PLAYING, reporting the error
# against nvinfer. So probe with xdpyinfo and report the two cases differently.
_sock_seen=""
for sock in /tmp/.X11-unix/X*; do
  [ -e "$sock" ] || continue
  _sock_seen=":${sock##*/X}"
  if DISPLAY="$_sock_seen" xdpyinfo >/dev/null 2>&1; then
    DETECTED_DISPLAY="$_sock_seen"; break
  fi
done
if [ -n "$DETECTED_DISPLAY" ]; then
  ok "X display" "$DETECTED_DISPLAY (reachable — local preview available)"
elif [ -n "$_sock_seen" ]; then
  warn "X display" "$_sock_seen exists but is NOT reachable by this user (greeter/xauth) — running headless, which is correct; do NOT force DISPLAY_NUM at it"
else
  warn "X display" "none found — headless is fine; the dashboard and RTSP output do not need one"
fi

# ---------------------------------------------------------------------------------------------
# Persist what other scripts need, so nothing has to re-probe or hardcode.
mkdir -p build 2>/dev/null && cat > build/hardware.env <<EOF
# Generated by scripts/check_hardware.sh — regenerate rather than edit.
DETECTED_PLATFORM="${DETECTED_PLATFORM}"
DETECTED_BOARD="${DETECTED_BOARD}"
DETECTED_L4T="${DETECTED_L4T}"
DETECTED_CUDA_ARCH="${DETECTED_CUDA_ARCH}"
DETECTED_DISPLAY="${DETECTED_DISPLAY}"
DETECTED_DLA="${DETECTED_DLA}"
DETECTED_RAM_GB="${DETECTED_RAM_GB}"
DETECTED_VRAM_MB="${DETECTED_VRAM_MB}"
EOF

printf "\n──────────────────────────────────────────────────────────────\n"
printf " %d passed · %d warning · %d failed\n" "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf " ${C_BAD}Fix the FAIL lines before running setup.sh.${C_OFF}\n"
  printf "──────────────────────────────────────────────────────────────\n"
  exit 1
fi
[ "$WARN" -gt 0 ] && printf " Warnings are safe to proceed through; setup.sh resolves most of them.\n"
printf "──────────────────────────────────────────────────────────────\n"
exit 0
