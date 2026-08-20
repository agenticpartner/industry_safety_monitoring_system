#!/usr/bin/env bash
# Runs ON THE JETSON. Splice a fire clip into the middle of one demo camera.
#
#   ./scripts/splice_fire.sh <fire-source.mp4> [camera-number] [insert-at-seconds]
#   ./scripts/splice_fire.sh --period <seconds> <fire-source.mp4> [camera] [insert-at]
#   ./scripts/splice_fire.sh --restore [camera-number]
#
# `--period N` first LOOPS the camera's own footage out to N seconds, then splices the fire in
# once. Because the pipeline loops each file, the fire then recurs every N seconds instead of
# every ~40 s (the clip's natural length). Put fire on ONE camera with a long period and the
# demo has a rare, findable event rather than three tiles burning continuously — which reads as
# a broken feed, not an emergency.
#
# The MV3DT warehouse footage contains no fire, so the fire detector — which is real, loaded and
# benchmarked — never fires on the demo set. Rather than injecting a synthetic alert and calling
# it a demo, this puts actual fire in front of the actual model: real frames, real inference, a
# real `fire_alert` incident with a real clip.
#
# The fire footage is a different camera angle from the warehouse views, so the spliced tile
# visibly cuts to another scene for ten seconds. That is accepted deliberately — an honest
# "different view, real detection" beats a seamless fake.
#
# The original clip is backed up next to it and `--restore` puts it back, because the demo media
# set is expensive to regenerate (scripts/make_streams.sh re-encodes 20 clips).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

PERIOD=""
if [ "${1:-}" = "--period" ]; then PERIOD="${2:?--period needs seconds}"; shift 2; fi

if [ "${1:-}" = "--restore" ]; then
  CAM=$(printf "media/cam%02d.mp4" "${2:-7}")
  [ -f "${CAM}.orig" ] || { echo "no backup at ${CAM}.orig"; exit 1; }
  mv "${CAM}.orig" "$CAM"
  echo "==> restored $CAM"
  exit 0
fi

SRC="${1:?usage: splice_fire.sh <fire-source.mp4> [camera] [insert-at-s]}"
CAMN="${2:-7}"
AT="${3:-12}"
CAM=$(printf "media/cam%02d.mp4" "$CAMN")
[ -f "$SRC" ] || { echo "ERROR: $SRC not found"; exit 1; }
[ -f "$CAM" ] || { echo "ERROR: $CAM not found"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --period: lengthen the camera's own footage by looping it, so the spliced fire recurs at that
# interval instead of every clip length. Done BEFORE the splice so the fire lands once in the
# long clip, not once per repetition.
if [ -n "$PERIOD" ]; then
  # The base for looping must be the ORIGINAL, or re-running this loops footage that already
  # contains fire and the "one fire per period" guarantee quietly stops holding.
  SRC_LOOP="$CAM"
  [ -f "${CAM}.orig" ] && SRC_LOOP="${CAM}.orig"
  BASE_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC_LOOP")
  REPS=$(python3 -c "import math;print(max(1, math.ceil($PERIOD / $BASE_DUR)))")
  echo "==> --period ${PERIOD}s: looping ${SRC_LOOP} (${BASE_DUR}s) x${REPS}"
  # -stream_loop copies the same encoded frames, so this is cheap and lossless.
  ffmpeg -y -hide_banner -loglevel error -stream_loop $((REPS - 1)) -i "$SRC_LOOP" \
    -t "$PERIOD" -c copy -movflags +faststart "$TMP/long.mp4" || exit 1
  [ -f "${CAM}.orig" ] || cp "$CAM" "${CAM}.orig"
  cp "$TMP/long.mp4" "$CAM"
fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$CAM")
FIRE_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
FIRE_DUR=${FIRE_DUR%.*}
END=$(python3 -c "print(max(0, $AT + $FIRE_DUR))")

echo "==> ${CAM} is ${DUR}s; inserting ${FIRE_DUR}s of fire at ${AT}s (replacing ${AT}-${END}s)"

# NVENC is unavailable to ffmpeg on this build ("Cannot load libnvidia-encode.so.1") even though
# `ffmpeg -encoders` lists hevc_nvenc — the DeepStream pipeline reaches the encoder through a
# different V4L2 path. libx265 ultrafast is used instead; for one 40s clip that is seconds.
ENC=(-c:v libx265 -preset ultrafast -b:v 4M
     -x265-params "keyint=${GOP}:min-keyint=${GOP}:scenecut=0:repeat-headers=1:log-level=none")


# Normalise the fire source to the media contract before concatenating — mismatched resolution or
# frame rate makes the concat filter produce a corrupt or wrongly-timed stream.
echo "    normalising fire source to ${STREAM_W}x${STREAM_H} ${STREAM_FPS}fps"
ffmpeg -y -hide_banner -loglevel error -i "$SRC" \
  -vf "scale=${STREAM_W}:${STREAM_H}:force_original_aspect_ratio=decrease,pad=${STREAM_W}:${STREAM_H}:(ow-iw)/2:(oh-ih)/2,fps=${STREAM_FPS},format=yuv420p" \
  "${ENC[@]}" -an -tag:v hvc1 "$TMP/fire.mp4" || exit 1

echo "    building spliced clip"
# One filter_complex concat with a single re-encode. Stream-copy concat would be faster but the
# three pieces come from different encoder settings, and -c copy across mismatched parameter sets
# yields a file that probes fine and plays wrong.
ffmpeg -y -hide_banner -loglevel error \
  -i "$CAM" -i "$TMP/fire.mp4" \
  -filter_complex \
    "[0:v]trim=0:${AT},setpts=PTS-STARTPTS[a]; \
     [1:v]setpts=PTS-STARTPTS[b]; \
     [0:v]trim=${END}:${DUR},setpts=PTS-STARTPTS[c]; \
     [a][b][c]concat=n=3:v=1:a=0[out]" \
  -map "[out]" "${ENC[@]}" -an -tag:v hvc1 -movflags +faststart "$TMP/out.mp4" || exit 1

[ -f "${CAM}.orig" ] || cp "$CAM" "${CAM}.orig"
mv "$TMP/out.mp4" "$CAM"

echo "==> done"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate \
        -show_entries format=duration -of default=nw=1 "$CAM" | sed 's/^/    /'
echo "    original preserved at ${CAM}.orig  (./scripts/splice_fire.sh --restore ${CAMN})"
