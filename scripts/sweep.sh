#!/usr/bin/env bash
# Runs ON THE JETSON. Phase 4: end-to-end throughput from 1 to 20 streams.
#
#   ./scripts/sweep.sh [--topology serial|parallel] [--label NAME] [--metadata-only] [steps...]
#
# Runs the REAL pipeline — decode, both detectors, tracker, compliance probe, tiler, OSD —
# headless, and reports the aggregate frames per second the whole chain sustains.
#
# Starting at N=1 shows exactly which stream count a metric turns over at, rather than
# discovering a cliff at 20 with no idea where it started.
#
# MEASUREMENT: each step plays every clip once, to EOS, and is timed with wall-clock; fps is
# (frames_per_clip x N) / elapsed. This replaced an in-probe running average, which silently
# folded ~20 s of TensorRT engine deserialisation into the mean and under-reported throughput by
# up to 7x. Wall-clock to EOS has no such blind spot. It requires sources.loop=false, which the
# script forces via a derived config.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

TOPOLOGY=serial
LABEL=gpu
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --label)    LABEL="$2";    shift 2 ;;
    --metadata-only) EXTRA+=(--no-osd); shift ;;
    *) break ;;
  esac
done
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(1 2 4 8 12 16 18 20)

RESULTS="bench/sweep_${LABEL}_${TOPOLOGY}.csv"
LOGDIR="bench/sweep_${LABEL}_${TOPOLOGY}"
mkdir -p "$LOGDIR"
echo "stream_count,elapsed_s,measured_fps,target_fps,per_stream_fps,holds_realtime,headroom_x,gr3d_pct_peak,nvdec_mhz_peak,dla0,dla1,ram_mb_peak,emc_pct_peak,detections" > "$RESULTS"

command -v ffprobe >/dev/null || { echo "ERROR: ffprobe missing"; exit 1; }
[ -f media/cam01.mp4 ] || { echo "ERROR: no media — run scripts/make_streams.sh"; exit 1; }
FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets \
         -show_entries stream=nb_read_packets -of csv=p=0 media/cam01.mp4)

# Derived config with looping off, so every run terminates at EOS and can be timed.
CFG=/tmp/sweep_cfg.yml
python3 - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open("configs/demo.yml"))
c["sources"]["loop"] = False
yaml.safe_dump(c, open(sys.argv[1], "w"))
PY

echo "==> ${FRAMES} frames/clip · realtime target 30 fps/stream · topology=${TOPOLOGY}"
sudo tegrastats --stop >/dev/null 2>&1 || true

for N in "${STEPS[@]}"; do
  [ -f "$(printf 'media/cam%02d.mp4' "$N")" ] || { echo "!! not enough clips for N=$N"; continue; }
  LOG="${LOGDIR}/n${N}.log"
  TG=$(mktemp)

  sudo tegrastats --interval 500 > "$TG" 2>/dev/null &
  sleep 0.5

  s=$(date +%s.%N)
  timeout --signal=KILL 900 \
    python3 app/safety_pipeline.py --config "$CFG" --streams "$N" --topology "$TOPOLOGY" \
      --source file --no-display --stats "${EXTRA[@]}" > "$LOG" 2>&1
  rc=$?
  e=$(date +%s.%N)

  sudo tegrastats --stop >/dev/null 2>&1 || true
  sleep 0.4

  if [ $rc -ne 0 ]; then
    echo "  N=$N  RUN FAILED (rc=$rc) — see $LOG"
    grep -a -m3 -iE 'error|critical|Traceback' "$LOG" | sed 's/^/      /'
    echo "$N,,,,,ERROR,,,,,,," >> "$RESULTS"
    rm -f "$TG"; continue
  fi

  # A throughput number from a pipeline that detected nothing is worse than useless — it looks
  # like a pass. This exact failure happened: the stock NvSORT tracker config emits zero objects
  # (minTrackerConfidence unreachable without a visual tracker), which made a broken pipeline
  # the FASTEST result in the sweep. Every row must now prove it did real work.
  dets=$(grep -aoE 'detections: [^|]*' "$LOG" | tail -1 | sed 's/detections: //;s/ *$//')
  if [ -z "$dets" ] || [ "$dets" = "none" ]; then
    echo "  N=$N  ZERO DETECTIONS — throughput is meaningless, not recording a pass"
    echo "        check the tracker config and nvinfer thresholds; log: $LOG"
    echo "$N,,,,,NO-DETECTIONS,,,,,,," >> "$RESULTS"
    rm -f "$TG"; continue
  fi

  elapsed=$(echo "$e - $s" | bc)
  fps=$(echo "scale=1; $FRAMES * $N / $elapsed" | bc)
  target=$(( N * 30 ))
  per=$(echo "scale=1; $fps / $N" | bc)
  head=$(echo "scale=2; $fps / $target" | bc)
  [ "$(echo "$fps >= $target" | bc)" = 1 ] && verdict=YES || verdict=NO

  gr3d=$(grep -oE 'GR3D_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)
  nvdec=$(grep -oE 'NVDEC0 [0-9]+' "$TG" | awk '{print $2}' | sort -n | tail -1)
  emc=$(grep -oE 'EMC_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)
  ram=$(grep -oE 'RAM [0-9]+/' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)
  dla0=$(grep -oE 'NVDLA0 [^ ]+' "$TG" | awk '{print $2}' | grep -v off | tail -1); dla0=${dla0:-off}
  dla1=$(grep -oE 'NVDLA1 [^ ]+' "$TG" | awk '{print $2}' | grep -v off | tail -1); dla1=${dla1:-off}
  cp "$TG" "${LOGDIR}/n${N}.tegrastats"; rm -f "$TG"

  printf "  N=%-3s %7.1fs  %8s fps  target %5s  per-stream %5s  realtime:%-4s %sx  GPU %s%%  RAM %sMB\n" \
    "$N" "$elapsed" "$fps" "$target" "$per" "$verdict" "$head" "${gr3d:-?}" "${ram:-?}"
  echo "        detections: $dets"
  echo "$N,$elapsed,$fps,$target,$per,$verdict,$head,${gr3d:-},${nvdec:-},$dla0,$dla1,${ram:-},${emc:-},${dets// /;}" >> "$RESULTS"
done

echo
echo "==> $RESULTS"
column -s, -t < "$RESULTS"
