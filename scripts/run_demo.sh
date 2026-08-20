#!/usr/bin/env bash
# Runs ON THE JETSON. The one command to show the demo.
#
#   ./scripts/run_demo.sh [streams] [file|rtsp]
#
#   ./scripts/run_demo.sh              20 streams from local files, on the attached monitor
#   ./scripts/run_demo.sh 20 rtsp      20 streams over RTSP + published to rtsp://<host>:8554/safety
#   ./scripts/run_demo.sh 4            4 streams, handy for a quick visual check
#
# Renders to the attached monitor on the auto-detected X display (scripts/env.sh) and, in rtsp
# mode, also publishes the tiled output so it can be watched from another machine.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

STREAMS="${1:-$MAX_STREAMS}"
MODE="${2:-file}"

command -v python3 >/dev/null || { echo "ERROR: python3 missing"; exit 1; }
[ -f media/cam01.mp4 ] || { echo "ERROR: no media — run scripts/make_streams.sh"; exit 1; }

for m in ppe fire; do
  eng="models/${m}/model/${m}_gpu_b20_fp16.engine"
  [ -f "$eng" ] || { echo "ERROR: $eng missing — run scripts/build_engines.sh"; exit 1; }
done
[ -f models/parser/libnvds_yolo_parser.so ] || {
  echo "ERROR: parser missing — run 'make -C models/parser'"; exit 1; }

# The demo media set must not be the --fast hardlinked benchmark set: identical tiles show
# nothing. Warn rather than refuse, in case that is genuinely what is wanted.
if [ "$(stat -c %i media/cam01.mp4)" = "$(stat -c %i media/cam02.mp4 2>/dev/null)" ]; then
  echo "!! media/cam01 and cam02 are the same file (--fast benchmark set)."
  echo "!! Every tile will look identical. Re-run scripts/make_streams.sh without --fast for the demo."
  echo
fi

ARGS=(--streams "$STREAMS" --source "$MODE" --fps --stats)

# Zones and events are OFF by default in the pipeline so benchmark runs stay comparable with the
# Phase 1 / 2.0 numbers. The DEMO is the product, so it turns both on.
if [ -f configs/analytics/analytics.txt ]; then
  ARGS+=(--zones)
else
  echo "!! configs/analytics/analytics.txt missing — no zone overlays."
  echo "!! Regenerate with: python3 scripts/make_zones.py --generate"
fi

# Events need redis; without it the emitter would just count drops forever, so say so plainly
# rather than letting the demo look like it is recording incidents when it is not.
if redis-cli ping >/dev/null 2>&1; then
  ARGS+=(--events)
  if ! ps -eo args | grep -q '[e]vent_service.py'; then
    echo "==> starting the event service"
    bash scripts/run_services.sh start >/dev/null 2>&1 || true
  fi
else
  echo "!! redis is not running — incidents will NOT be recorded."
  echo "!!   sudo systemctl start redis-server && ./scripts/run_services.sh start"
  echo
fi

if [ "$MODE" = rtsp ]; then
  # Camera sources come from ANOTHER machine — start scripts/serve_rtsp_sources.sh there first.
  # This box only starts the output sink, so the demo view can be watched remotely.
  base=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/demo.yml'))['sources']['rtsp_base'])")
  probe="${base}/cam01"
  echo "==> checking camera source ${probe}"
  if ! timeout 8 ffprobe -v error -rtsp_transport tcp -i "$probe" \
        -show_entries stream=codec_name -of csv=p=0 >/dev/null 2>&1; then
    echo "ERROR: no RTSP source at ${probe}"
    echo "       On the source machine run:  ./scripts/serve_rtsp_sources.sh start ${STREAMS}"
    echo "       and confirm rtsp_base in configs/demo.yml points at its LAN IP."
    exit 1
  fi
  echo "    sources OK"

  bash scripts/serve_rtsp.sh start
  ARGS+=(--rtsp-out)
  trap 'echo; echo "==> stopping RTSP sink"; bash scripts/serve_rtsp.sh stop' EXIT
fi

# DISPLAY comes from scripts/env.sh, which probes for a live X socket.
echo "==> ${STREAMS} streams | ${MODE} | DISPLAY=${DISPLAY}"
echo "    Ctrl-C to stop."
echo
python3 app/safety_pipeline.py "${ARGS[@]}"
