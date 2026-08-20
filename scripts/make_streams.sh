#!/usr/bin/env bash
# Runs ON THE JETSON. Builds the demo media set: N distinct 1080p30 H.265 clips in media/cam*.mp4
#
#   ./scripts/make_streams.sh [count] [seconds]
#   ./scripts/make_streams.sh --fast [count]     hardlink the DS H.265 sample, no re-encode
#
# Source pool = every video in media/src/. If that's empty it falls back to the bundled
# DeepStream 1080p H.265 sample, which is fine for the decode-ceiling sweep (Phase 1) but has
# no PPE or fire content — drop real footage into media/src/ before the visual demo.
#
# --fast exists because the bundled sample_1080p_h265.mp4 already satisfies the media contract
# (1920x1080, 30fps, hevc). NVDEC load depends only on codec/resolution/framerate, not content,
# so for the Phase-1 decode ceiling identical clips measure exactly the same thing as distinct
# ones — and skip ~20 minutes of CPU re-encoding. Tiles will look identical; that's fine for a
# benchmark and wrong for a demo, so --fast is never used for the visual run.
#
# Each output gets a different start offset so tiles are visually distinguishable even when the
# pool has only one clip. Camera labels are NOT burned in — nvdsosd draws those at runtime.
#
# Contract (see project_skill.md): 1920x1080, 30fps, H.265, closed GOP (keyint=30, scenecut off).
# H.265 is not optional: 20 streams does not fit the H.264 NVDEC budget on AGX Orin.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

FAST=0
if [ "${1:-}" = "--fast" ] || [ "${1:-}" = "-f" ]; then FAST=1; shift; fi
COUNT="${1:-$MAX_STREAMS}"
SECONDS_EACH="${2:-60}"
OUT=media
SRC=media/src

mkdir -p "$OUT" "$SRC"

# ---------- fast path: benchmark-only, no re-encode ----------
if [ "$FAST" = 1 ]; then
  SAMPLE="${DS_SAMPLES}/streams/sample_1080p_h265.mp4"
  [ -f "$SAMPLE" ] || { echo "ERROR: $SAMPLE not found"; exit 1; }
  echo "==> --fast: hardlinking ${COUNT} copies of $(basename "$SAMPLE") (decode benchmark only)"
  echo "!!  All tiles will be identical. Do NOT use this media set for the visual demo."
  for i in $(seq 1 "$COUNT"); do
    dst=$(printf "%s/cam%02d.mp4" "$OUT" "$i")
    rm -f "$dst"; ln "$SAMPLE" "$dst" 2>/dev/null || cp "$SAMPLE" "$dst"
  done
  ls -1 "$OUT"/cam*.mp4 | wc -l | xargs echo "==> clips:"
  du -sh "$OUT"
  exit 0
fi

# ---------- build the source pool ----------
mapfile -t POOL < <(find "$SRC" -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.h264' -o -iname '*.h265' -o -iname '*.webm' \) \
  ! -name 'cam[0-9][0-9].mp4' | sort)

if [ ${#POOL[@]} -eq 0 ]; then
  FALLBACK="${DS_SAMPLES}/streams/sample_1080p_h265.mp4"
  [ -f "$FALLBACK" ] || { echo "ERROR: no sources in ${SRC}/ and no DS sample at ${FALLBACK}"; exit 1; }
  POOL=("$FALLBACK")
  echo "!! media/src/ is empty — falling back to the DeepStream traffic sample."
  echo "!! Fine for the decode sweep; it contains NO helmets, vests, or fire."
fi

echo "==> pool: ${#POOL[@]} source clip(s)"
for p in "${POOL[@]}"; do echo "      $p"; done
echo "==> generating ${COUNT} × ${SECONDS_EACH}s 1080p30 H.265 clips"

# Prefer the hardware encoder — this ffmpeg build (7:8.0.1-nvidia) ships hevc_nvenc, which turns
# a ~20-minute CPU transcode of 20 clips into well under a minute. NVENC is a separate engine from
# NVDEC and the GPU, so it does not disturb anything the pipeline needs.
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc; then
  ENC=hevc_nvenc
  ENC_OPTS="-preset p4 -b:v 4M -g ${GOP} -no-scenecut 1"
  echo "==> encoder: hevc_nvenc (hardware)"
else
  ENC=libx265
  ENC_OPTS="-preset ultrafast -b:v 4M -x265-params keyint=${GOP}:min-keyint=${GOP}:scenecut=0:repeat-headers=1"
  echo "==> encoder: libx265 (CPU fallback — slow)"
fi

for i in $(seq 1 "$COUNT"); do
  dst=$(printf "%s/cam%02d.mp4" "$OUT" "$i")
  if [ -f "$dst" ]; then echo "    [$i/$COUNT] $dst (exists, skip)"; continue; fi

  src="${POOL[$(( (i-1) % ${#POOL[@]} ))]}"
  # Rotate the start point so repeated use of one source still yields distinct-looking tiles.
  offset=$(( ((i-1) / ${#POOL[@]}) * 3 ))

  echo "    [$i/$COUNT] $dst  <- $(basename "$src")  +${offset}s"
  ffmpeg -hide_banner -loglevel error -y \
    -stream_loop -1 -ss "$offset" -i "$src" -t "$SECONDS_EACH" \
    -an \
    -vf "scale=${STREAM_W}:${STREAM_H}:force_original_aspect_ratio=decrease,pad=${STREAM_W}:${STREAM_H}:(ow-iw)/2:(oh-ih)/2,fps=${STREAM_FPS},format=yuv420p" \
    -c:v "$ENC" $ENC_OPTS \
    -tag:v hvc1 -movflags +faststart \
    "$dst"
done

echo
echo "==> verifying"
ok=1
for i in $(seq 1 "$COUNT"); do
  f=$(printf "%s/cam%02d.mp4" "$OUT" "$i")
  read -r codec w h fps < <(ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 "$f" | tr ',' ' ')
  if [ "$codec" != "hevc" ] || [ "$w" != "$STREAM_W" ] || [ "$h" != "$STREAM_H" ]; then
    echo "    BAD  $f  ($codec ${w}x${h} $fps)"; ok=0
  fi
done
[ "$ok" = 1 ] && echo "    all ${COUNT} clips are hevc ${STREAM_W}x${STREAM_H}" || { echo "    FAILED"; exit 1; }
du -sh "$OUT"
