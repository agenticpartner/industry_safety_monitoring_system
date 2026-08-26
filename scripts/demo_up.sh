#!/usr/bin/env bash
# Runs ON THE JETSON. Bring the whole dashboard demo up from cold, with a clean slate.
#
#   ./scripts/demo_up.sh                 wipe previous data, 20 streams, live view on
#   ./scripts/demo_up.sh 12              same, 12 streams
#   ./scripts/demo_up.sh 20 --keep       keep the existing incidents instead of wiping
#   ./scripts/demo_up.sh --down          stop everything (leaves the model servers up)
#
# Wiping is the DEFAULT because the usual reason to restart this demo is to show it detecting
# things live. A feed pre-filled with yesterday's incidents makes new alerts impossible to pick
# out, and the KPI tiles describe a run nobody watched.
#
# What it does NOT touch: the two llama.cpp model servers on :8000 and :8001. They are measured
# to cost nothing when idle (Phase 2.5: 517.8 fps at 20 streams with both loaded, identical to
# baseline) and take minutes to reload, so restarting the demo should never restart them.
# `scripts/setup_reasoning.sh llamacpp --serve` and `... llm --serve` bring them back if needed.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

API_PORT="${ISMS_API_PORT:-8080}"
API="http://127.0.0.1:${API_PORT}"

# ---- down ----------------------------------------------------------------------------------
if [ "${1:-}" = "--down" ]; then
  echo "==> stopping pipeline"
  curl -s -m 60 -X POST "${API}/pipeline/stop" >/dev/null 2>&1 || true
  # Fall back to a direct kill: the API is what we are about to stop, so it may already be gone.
  ps -eo pid,args | grep 'safety_pipeline.py' | grep -v 'bash -c' | grep -v grep |
    awk '{print $1}' | xargs -r kill -9 2>/dev/null
  bash scripts/run_services.sh stop
  bash scripts/serve_rtsp.sh stop
  echo "==> down. Model servers on :8000/:8001 left running."
  exit 0
fi

STREAMS="${1:-$MAX_STREAMS}"
KEEP=0
[ "${2:-}" = "--keep" ] && KEEP=1

# ---- pre-flight ----------------------------------------------------------------------------
# Refuse rather than stack a second pipeline on top of a running one. Four concurrent pipelines
# once contaminated a whole benchmark run (each measured the others' contention), which is why
# every entry point here checks first.
if ps -eo args | grep 'safety_pipeline.py' | grep -v 'bash -c' | grep -qv grep; then
  echo "ERROR: a pipeline is already running. ./scripts/demo_up.sh --down first."
  exit 1
fi
[ -f media/cam01.mp4 ] || { echo "ERROR: no media — run scripts/make_streams.sh"; exit 1; }
redis-cli ping >/dev/null 2>&1 || {
  echo "ERROR: redis is not responding — sudo systemctl start redis-server"; exit 1; }

# ---- services ------------------------------------------------------------------------------
if [ "$KEEP" = 1 ]; then
  echo "==> starting services, KEEPING existing incidents"
  bash scripts/run_services.sh start || exit 1
else
  echo "==> starting services and WIPING previous incidents, clips and queued events"
  bash scripts/run_services.sh reset || exit 1
fi

# ---- live view -----------------------------------------------------------------------------
echo "==> starting mediamtx (WebRTC :8889 · HLS :8888)"
bash scripts/serve_rtsp.sh start || exit 1

# ---- pipeline ------------------------------------------------------------------------------
# `--rtsp-out` is a START-TIME flag: the encode branch is part of the pipeline graph and cannot
# be attached later, so live view has to be decided here.
echo "==> starting pipeline: ${STREAMS} streams, live output on"
for _ in $(seq 1 20); do
  curl -s -m 10 "${API}/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -s -m 120 -X POST "${API}/pipeline/start?streams=${STREAMS}&rtsp_out=true" | sed 's/^/    /'
echo

sleep 12
echo "==> live view:"
curl -s -m 15 "${API}/live/status" | python3 -c '
import json, sys
d = json.load(sys.stdin)
line = "    ready=%s publishing=%s" % (d["ready"], d["publishing"])
if d.get("reason"):
    line += "  (%s -> %s)" % (d["reason"], d.get("action") or "")
print(line)
' || echo "    (could not read /live/status)"

echo
# Print the ADDRESSES the dashboard answers on, not this machine's hostname. A hostname only
# works if the viewer's DNS agrees, and it may quietly resolve to something else entirely —
# a stale VPN entry for the same name sent a browser to a dead host and looked like the
# dashboard being down.
echo "==> dashboard:"
for _ip in $(hostname -I 2>/dev/null); do
  case "$_ip" in
    127.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|*:*) continue ;;   # loopback, docker, IPv6
  esac
  echo "      http://${_ip}:${API_PORT}/"
done
echo "      http://$(hostname):${API_PORT}/   (only if this name resolves from the viewer)"
echo "    A clip takes 2-3 s to cut and the VLM adjudicates at ~6 s each, serially —"
echo "    a cold start burst takes a few minutes to fully verify. Alerts appear in <1 s."
