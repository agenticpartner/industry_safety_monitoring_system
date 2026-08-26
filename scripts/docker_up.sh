#!/usr/bin/env bash
# Bring the Docker stack up on this host. Picks docker/compose.spark.yml on DGX Spark / GB10
# so you do not have to remember the overlay.
#
#   ./scripts/docker_up.sh              up -d --build
#   ./scripts/docker_up.sh logs -f app
#   ./scripts/docker_up.sh down
set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(bash scripts/docker_platform.sh)"

FILES=(-f docker/compose.yml)
if [ "${ISMS_SPARK:-0}" = 1 ]; then
  FILES+=(-f docker/compose.spark.yml)
  echo "==> DGX Spark / ARM SBSA — using ${FILES[*]}"
elif [ "${ISMS_PLATFORM:-}" = jetson ]; then
  echo "==> this is a Jetson; the host install is scripts/setup.sh, not Docker."
  echo "    continuing with compose anyway if you asked for it."
fi

if [ "$#" -eq 0 ]; then
  set -- up --build -d
fi

# Compose interpolates ${ISMS_*} from --env-file. Without this, running from the repo root
# (this script's cwd) would miss docker/.env and silently keep file-mode defaults.
ENV_FILE=()
if [ -f docker/.env ]; then
  ENV_FILE=(--env-file docker/.env)
fi

exec docker compose "${ENV_FILE[@]}" "${FILES[@]}" "$@"
