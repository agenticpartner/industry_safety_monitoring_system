#!/usr/bin/env bash
# Push dashboard/ from this checkout into the running app container so a browser refresh
# sees the edit. The image copies dashboard/ in at build time; that tree is what :9080
# serves. Editing the host files does nothing until this runs (or until dashboard/ is
# bind-mounted — see docker/compose.yml).
#
#   ./scripts/reload_ui.sh
#   then Ctrl+Shift+R on http://<spark>:9080/  (plain Ctrl+R often keeps a cached page)
set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(bash scripts/docker_platform.sh)"

FILES=(-f docker/compose.yml)
if [ "${ISMS_SPARK:-0}" = 1 ]; then
  FILES+=(-f docker/compose.spark.yml)
fi

ENV_FILE=()
if [ -f docker/.env ]; then
  ENV_FILE=(--env-file docker/.env)
fi

CID="$(docker compose "${ENV_FILE[@]}" "${FILES[@]}" ps -q app)"
if [ -z "$CID" ]; then
  echo "app container is not running. Start it with:" >&2
  echo "  ./scripts/docker_up.sh --profile sources up -d" >&2
  exit 1
fi

mounted="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/opt/isms/dashboard"}}{{.Source}}{{end}}{{end}}' "$CID")"
if [ -n "$mounted" ]; then
  echo "dashboard/ is already bind-mounted from:"
  echo "  $mounted"
  echo "Save the file, then hard-refresh the browser (Ctrl+Shift+R)."
  exit 0
fi

docker cp dashboard/. "${CID}:/opt/isms/dashboard/"
echo "Copied dashboard/ into the app container."
echo "Hard-refresh the browser (Ctrl+Shift+R). Plain Ctrl+R often keeps a cached page."
echo "Python edits still need: ./scripts/docker_up.sh up -d --force-recreate app"
