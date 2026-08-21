#!/usr/bin/env bash
# Shared settings for every script in this project. Source, don't execute.
#
# Everything here is either a hardware fact or the demo media contract. There are no host names,
# credentials or machine-specific paths — those live in .env (see .env.example), which is never
# committed.

# --- DeepStream install --------------------------------------------------------------------
DS_ROOT="${DS_ROOT:-/opt/nvidia/deepstream/deepstream}"
DS_SAMPLES="${DS_SAMPLES:-${DS_ROOT}/samples}"

# --- display -------------------------------------------------------------------------------
# Export DISPLAY only when an X server is actually REACHABLE by this user. Otherwise leave it
# unset and let DeepStream run headless.
#
# A set-but-unusable DISPLAY is worse than no DISPLAY at all. nvbufsurftransform tries to open an
# EGL display against it and the pipeline dies with:
#
#     nvbufsurftransform: Could not get EGL display connection
#     nvinfer <pgie_fire> error: Failed to set buffer pool to active
#     Unable to set the pipeline to the playing state.
#
# which reads as an inference fault and is nothing of the kind. Diagnosed on a device whose X
# socket existed at :0 but belonged to the gdm greeter — root-owned, no xauth cookie for the
# service user — so merely finding a socket proves nothing. `xdpyinfo` is the actual test.
#
# Set DISPLAY_NUM in .env to force a specific display (skips the probe entirely).
# Needs `xdpyinfo` (x11-utils, installed by scripts/setup.sh). If it is missing the probe can
# never succeed and every run goes headless — the safe direction to fail, since the cost is a
# lost local preview rather than a dead pipeline.
detect_display() {
  local d n
  for d in /tmp/.X11-unix/X*; do
    [ -e "$d" ] || continue
    n=":${d##*/X}"
    if DISPLAY="$n" xdpyinfo >/dev/null 2>&1; then
      echo "$n"
      return 0
    fi
  done
  return 1
}

if [ -n "${DISPLAY_NUM:-}" ]; then
  export DISPLAY="$DISPLAY_NUM"
else
  unset DISPLAY
  _d="$(detect_display)" && export DISPLAY="$_d"
  unset _d
fi

# --- demo media contract -------------------------------------------------------------------
# H.265 is not a preference. 20 concurrent 1080p30 streams is 91% of the measured AGX Orin NVDEC
# ceiling for H.265 (22 streams); the H.264 ceiling is materially lower and will not hold 20.
STREAM_W=1920
STREAM_H=1080
STREAM_FPS=30
STREAM_CODEC=hevc
GOP=30                   # closed GOP, 1s keyframe interval
MAX_STREAMS="${MAX_STREAMS:-20}"

# --- secrets ---------------------------------------------------------------------------------
# Load HF_TOKEN / TELEGRAM_* from .env at the repo root, falling back to the XDG config dir.
#
# They are read from a file rather than the shell profile because Ubuntu's default .bashrc
# returns early for non-interactive shells, so an `export` there is invisible to exactly the
# `nohup`, `systemd` and `ssh host 'cmd'` invocations that need it.
load_env() {
  local root f
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  for f in "${root}/.env" "${HOME}/.config/industry_safety/env"; do
    [ -f "$f" ] && { set -a; . "$f"; set +a; }
  done
  return 0
}
