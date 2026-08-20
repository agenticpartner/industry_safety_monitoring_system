#!/usr/bin/env bash
# Runs ON THE JETSON. Phase 1 gate: how many 1080p30 H.265 streams can NVDEC actually decode?
#
#   ./scripts/decode_sweep.sh [steps...]      default: 1 2 4 8 12 16 18 20
#
# Decode ONLY — no inference, no tracker, no OSD, no display. If 20 streams fails here, no amount
# of model tuning fixes it, so this runs before anything else is built.
#
# Method: run N decode chains flat-out (sync=false) and measure wall-clock. Aggregate decode
# throughput = total_frames / elapsed. Realtime needs aggregate >= N * 30 fps.
#
# Note this measures the *ceiling*, not realtime behaviour: with sync=false the decoder runs as
# fast as it can, so headroom shows up as aggregate_fps well above the target. A live 30fps
# source would simply idle the spare capacity.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(1 2 4 8 12 16 18 20)
RESULTS=bench/decode_sweep.csv
TGLOG=bench/tegrastats_decode.log
mkdir -p bench

command -v ffprobe >/dev/null || { echo "ERROR: ffprobe missing — run scripts/setup.sh"; exit 1; }
[ -f media/cam01.mp4 ] || { echo "ERROR: no media — run scripts/make_streams.sh"; exit 1; }

FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets \
         -show_entries stream=nb_read_packets -of csv=p=0 media/cam01.mp4)
echo "==> ${FRAMES} frames per clip; realtime target 30 fps/stream"
echo "stream_count,elapsed_s,total_frames,aggregate_fps,per_stream_fps,target_fps,holds_realtime,headroom_x,nvdec_mhz_peak,gr3d_pct_peak,ram_mb_peak" > "$RESULTS"
: > "$TGLOG"

sudo tegrastats --stop >/dev/null 2>&1 || true

for N in "${STEPS[@]}"; do
  last=$(printf 'media/cam%02d.mp4' "$N")
  [ -f "$last" ] || { echo "!! only $(ls media/cam*.mp4 2>/dev/null | wc -l) clips available, skipping N=$N"; continue; }

  PIPE=""
  for i in $(seq 1 "$N"); do
    f="$(pwd)/$(printf 'media/cam%02d.mp4' "$i")"
    PIPE+=" filesrc location=${f} ! qtdemux ! h265parse ! nvv4l2decoder ! fakesink sync=false async=false"
  done

  TG=$(mktemp)
  sudo tegrastats --interval 200 > "$TG" 2>/dev/null &
  sleep 0.5

  start=$(date +%s.%N)
  gst-launch-1.0 -q $PIPE >/dev/null 2>&1
  rc=$?
  end=$(date +%s.%N)

  sudo tegrastats --stop >/dev/null 2>&1 || true
  sleep 0.2

  if [ $rc -ne 0 ]; then
    echo "  N=$N  PIPELINE FAILED (rc=$rc)"
    echo "$N,,,,,,FAIL,,,," >> "$RESULTS"
    rm -f "$TG"; continue
  fi

  elapsed=$(echo "$end - $start" | bc)
  total=$(( FRAMES * N ))
  agg=$(echo "scale=1; $total / $elapsed" | bc)
  per=$(echo "scale=1; $agg / $N" | bc)
  target=$(( N * 30 ))
  head=$(echo "scale=2; $agg / $target" | bc)
  [ "$(echo "$agg >= $target" | bc)" = 1 ] && verdict=YES || verdict=NO

  # NVDEC0 reports "off" when idle, a clock in MHz when active.
  nvdec=$(grep -oE 'NVDEC0 [0-9]+' "$TG" | awk '{print $2}' | sort -n | tail -1); nvdec=${nvdec:-0}
  gr3d=$(grep -oE 'GR3D_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1); gr3d=${gr3d:-0}
  ram=$(grep -oE 'RAM [0-9]+/' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1); ram=${ram:-0}

  { echo "### N=$N"; cat "$TG"; } >> "$TGLOG"
  rm -f "$TG"

  printf "  N=%-3s %6.1fs   agg %8s fps   per-stream %6s fps   target %5s   realtime:%-3s  headroom %sx  NVDEC %sMHz\n" \
    "$N" "$elapsed" "$agg" "$per" "$target" "$verdict" "$head" "$nvdec"
  echo "$N,$elapsed,$total,$agg,$per,$target,$verdict,$head,$nvdec,$gr3d,$ram" >> "$RESULTS"
done

echo
echo "==> $RESULTS"
column -s, -t < "$RESULTS"
