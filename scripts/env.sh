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
# nv3dsink needs a real X display and there is no safe default: on a Jetson running a desktop
# session the server is often on :1 rather than :0, and guessing wrong fails with an unhelpful
# "no display found". Set DISPLAY_NUM in .env to pin it; otherwise probe for a live X socket and
# fall back to :0.
detect_display() {
  local d
  for d in /tmp/.X11-unix/X*; do
    [ -e "$d" ] || continue
    echo ":${d##*/X}"
    return 0
  done
  echo ":0"
}
export DISPLAY="${DISPLAY_NUM:-$(detect_display)}"

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
