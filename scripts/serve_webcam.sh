#!/usr/bin/env bash
# Publish this machine's V4L2 webcam as one RTSP camera so the pipeline can use it as a slot.
#
# The DeepStream pipeline only ingests file:// or rtsp:// — there is no USB source — so the
# webcam has to become an RTSP camera first. MediaMTX on :8654 (scripts/serve_rtsp_sources.sh /
# compose `sources`) already accepts extra paths; this publishes:
#
#   rtsp://127.0.0.1:${RTSP_PORT:-8654}/webcam
#
#   ./scripts/serve_webcam.sh            foreground (compose / docker)
#   ./scripts/serve_webcam.sh urls [N]   print ISMS_RTSP_URLS with slot N = webcam (default 20)
#
# Capture is MJPEG 1280x720 @ 30fps, scaled to the 1080p30 media contract and encoded H.264
# zerolatency. Verified on DGX Spark /dev/video0 (UVC, MJPEG up to 4K).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${RTSP_PORT:-8654}"
DEV="${VIDEO_DEV:-/dev/video0}"
MOUNT="${WEBCAM_MOUNT:-webcam}"
URL="rtsp://127.0.0.1:${PORT}/${MOUNT}"

urls() {
  local slot="${1:-20}" i
  local out=()
  for i in $(seq 1 20); do
    if [ "$i" -eq "$slot" ]; then
      out+=("$URL")
    else
      out+=("$(printf 'rtsp://127.0.0.1:%s/cam%02d' "$PORT" "$i")")
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

wait_mtx() {
  local i
  echo "==> waiting for MediaMTX on 127.0.0.1:${PORT}"
  for i in $(seq 1 60); do
    if bash -c "echo >/dev/tcp/127.0.0.1/${PORT}" >/dev/null 2>&1; then
      echo "    up"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: nothing listening on :${PORT} — start the sources profile first" >&2
  exit 1
}

publish() {
  [ -e "$DEV" ] || { echo "ERROR: ${DEV} not found"; exit 1; }
  command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not on PATH"; exit 1; }
  wait_mtx
  echo "==> ${DEV} -> ${URL}  (1080p30 H.264)"
  # Restart on drop: unplugging the cam, a USB glitch, or MediaMTX restart should not kill the
  # container. The pipeline's nvurisrcbin reconnects on its own once the path is back.
  while true; do
    ffmpeg -hide_banner -loglevel warning \
      -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i "$DEV" \
      -an \
      -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p" \
      -c:v libx264 -preset veryfast -tune zerolatency -g 30 -bf 0 \
      -f rtsp -rtsp_transport tcp "$URL" || true
    echo "!! publisher exited — retry in 2s"
    sleep 2
  done
}

case "${1:-publish}" in
  urls) urls "${2:-20}" ;;
  publish|"") publish ;;
  *) echo "usage: $0 [publish|urls [slot]]"; exit 1 ;;
esac
