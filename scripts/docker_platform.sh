#!/usr/bin/env bash
# Detect whether this host needs the DGX Spark compose overlay.
# Source it (`eval "$(scripts/docker_platform.sh)"`) or run it to print KEY=VALUE.
#
#   ISMS_PLATFORM=jetson|dgpu|sbsa|unknown
#   ISMS_SPARK=0|1
set -eu
cd "$(dirname "$0")/.."

ARCH="$(uname -m)"
PLATFORM=unknown
if [ -r /etc/nv_tegra_release ]; then
  PLATFORM=jetson
elif [ "$ARCH" = "x86_64" ] && command -v nvidia-smi >/dev/null 2>&1; then
  PLATFORM=dgpu
elif [ "$ARCH" = "aarch64" ] && command -v nvidia-smi >/dev/null 2>&1; then
  PLATFORM=sbsa
fi

printf 'ISMS_PLATFORM=%s\n' "$PLATFORM"
if [ "$PLATFORM" = sbsa ]; then
  printf 'ISMS_SPARK=1\n'
else
  printf 'ISMS_SPARK=0\n'
fi
