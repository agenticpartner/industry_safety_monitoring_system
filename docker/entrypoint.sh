#!/usr/bin/env bash
# Container entrypoint. Everything the Jetson does by hand in README.md "Quick start" steps 4-6,
# done on first boot instead.
#
#   serve     (default)  build engines if needed, bring the services up, hold the container open
#   engines              build the TensorRT engines and exit
#   check                run the hardware pre-flight and exit
#   bash                 drop to a shell
#
# Tunables, all with working defaults:
#
#   ISMS_STREAMS=20        cameras to run
#   ISMS_SOURCE=file       file | rtsp
#   ISMS_AUTOSTART=1       start the pipeline on boot, not just the API
#   ISMS_WIPE=1            wipe previous incidents and clips first
#   ISMS_ENGINE_BATCH=20   TensorRT engine batch. Must match `batch-size` in the nvinfer configs.
#   ISMS_WAIT_VLM_S=2400   how long to keep watching for the VLM in the background. 0 disables.
set -uo pipefail
cd "${ISMS_ROOT:-/opt/isms}"
ROOT="$(pwd)"

STREAMS="${ISMS_STREAMS:-20}"
SOURCE="${ISMS_SOURCE:-file}"
AUTOSTART="${ISMS_AUTOSTART:-1}"
WIPE="${ISMS_WIPE:-1}"
BATCH="${ISMS_ENGINE_BATCH:-20}"
WAIT_VLM_S="${ISMS_WAIT_VLM_S:-0}"

step() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

# -------------------------------------------------------------------------------------------
# Image content vs volume content
# -------------------------------------------------------------------------------------------
# models/ is a volume so the engine build survives a container replace, and a volume is seeded
# from the image exactly once. Everything under models/ that the IMAGE owns — the ONNX, the label
# files, the compiled parser — is therefore refreshed from models.dist on every start, or an
# upgraded image would keep serving the previous image's artifacts forever.
#
# Engines are excluded because they are the one thing here the image does not own; ensure_engines
# decides their fate a few lines below.
refresh_from_image() {
  [ -d models.dist ] || return 0
  ( cd models.dist && find . -type f ! -name '*.engine' -print0 ) \
    | while IFS= read -r -d '' f; do
        mkdir -p "models/$(dirname "$f")"
        cp -a "models.dist/$f" "models/$f"
      done
}

# -------------------------------------------------------------------------------------------
# Engines
# -------------------------------------------------------------------------------------------
# A TensorRT engine is valid only for the exact GPU architecture, TensorRT version and driver
# that built it. `models/` is a volume, so an image moved from an L4 to an A100 would otherwise
# find a plausible-looking engine and deserialize it into a failure a long way from this cause.
# The stamp records what built them; a mismatch rebuilds rather than trusting the filename.
engine_stamp() {
  printf "%s|%s|%s" \
    "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)" \
    "$(trtexec --help 2>&1 | grep -oE 'TensorRT\.trtexec \[TensorRT v[0-9]+\]' | head -1)" \
    "$BATCH"
}

ensure_engines() {
  local stamp_file="models/.engine_stamp" want have
  want="$(engine_stamp)"
  have="$(cat "$stamp_file" 2>/dev/null || true)"

  local missing=0
  for m in ppe fire; do
    [ -f "models/${m}/model/${m}_gpu_b${BATCH}_fp16.engine" ] || missing=1
  done

  if [ "$missing" = 0 ] && [ "$want" = "$have" ]; then
    step "engines present and match this GPU — skipping build"
    return 0
  fi

  if [ "$missing" = 0 ] && [ -n "$have" ] && [ "$want" != "$have" ]; then
    step "engines were built elsewhere — rebuilding"
    printf "    built for : %s\n    running on: %s\n" "$have" "$want"
    rm -f models/*/model/*.engine
  fi

  step "building TensorRT engines (several minutes, once per GPU)"
  for m in ppe fire; do
    [ -f "models/${m}/model/${m}.onnx" ] || die "models/${m}/model/${m}.onnx missing from the image"
  done
  # The engine FILES are the signal, not the exit code. build_engines.sh finishes by piping its
  # results CSV through `column` and returns whatever that returned, so its status describes the
  # pretty-printer rather than the build. Checking the artifacts says the true thing either way.
  bash scripts/build_engines.sh all "$BATCH" fp16
  rc=$?

  for m in ppe fire; do
    [ -f "models/${m}/model/${m}_gpu_b${BATCH}_fp16.engine" ] \
      || die "no engine at models/${m}/model/${m}_gpu_b${BATCH}_fp16.engine (build_engines.sh exited ${rc}) — see the log above"
  done
  printf "%s" "$want" > "$stamp_file"
  step "engines ready"
}

# -------------------------------------------------------------------------------------------
# The reasoning layer runs in its own containers and is slow to arrive: ~12 GB of weights on a
# first boot, then minutes to load. run_services.sh checks for a VLM endpoint ONCE and skips the
# reasoning service if it is not there yet, which on a cold `compose up` it never is.
#
# So the wait happens here, in the background, and the vision path does not participate in it.
# The dashboard and the pipeline come up immediately; the reasoning service joins when the VLM
# answers. Until then incidents stay `unverified` — the documented degraded mode.
join_reasoning_when_ready() {
  local endpoint deadline
  [ "${WAIT_VLM_S:-0}" -gt 0 ] 2>/dev/null || return 0
  endpoint="$(python3 -c "import yaml;print(yaml.safe_load(open('configs/services.yml'))['reasoning']['endpoint'])" 2>/dev/null)" || return 0
  deadline=$(( $(date +%s) + WAIT_VLM_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf --max-time 3 "${endpoint}/models" >/dev/null 2>&1; then
      echo "[reasoning] VLM answered at ${endpoint} — starting the reasoning service"
      bash scripts/run_services.sh reasoning
      return 0
    fi
    sleep 15
  done
  echo "[reasoning] no VLM at ${endpoint} after ${WAIT_VLM_S}s — running vision-only"
}

# -------------------------------------------------------------------------------------------
wait_for_redis() {
  local host port n
  host="$(python3 -c "import yaml;print(yaml.safe_load(open('configs/services.yml'))['events']['redis']['host'])" 2>/dev/null || echo 127.0.0.1)"
  port="$(python3 -c "import yaml;print(yaml.safe_load(open('configs/services.yml'))['events']['redis']['port'])" 2>/dev/null || echo 6379)"
  for n in $(seq 1 60); do
    redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1 && { echo "    redis up at ${host}:${port}"; return 0; }
    sleep 1
  done
  die "redis never came up at ${host}:${port}"
}

# -------------------------------------------------------------------------------------------
case "${1:-serve}" in

  check)
    exec bash scripts/check_hardware.sh
    ;;

  engines)
    bash scripts/check_hardware.sh --quiet || true
    refresh_from_image
    ensure_engines
    exit 0
    ;;

  bash|sh)
    shift
    exec bash "$@"
    ;;

  serve)
    step "pre-flight"
    # Advisory: it writes build/hardware.env, which setup_reasoning.sh reads for CUDA_ARCH. A FAIL
    # here should not stop a container that has already been told to serve.
    bash scripts/check_hardware.sh --quiet || true

    step "refreshing image-owned model files"
    refresh_from_image

    ensure_engines

    step "waiting for redis"
    wait_for_redis

    [ -f media/cam01.mp4 ] || [ "$SOURCE" = rtsp ] \
      || die "no media at media/cam01.mp4 — mount footage, or set ISMS_SOURCE=rtsp"

    step "services"
    if [ "$WIPE" = 1 ]; then
      bash scripts/run_services.sh reset || die "services failed to start"
    else
      bash scripts/run_services.sh start || die "services failed to start"
    fi

    if [ "${WAIT_VLM_S:-0}" -gt 0 ] 2>/dev/null; then
      step "watching for the VLM in the background (up to ${WAIT_VLM_S}s)"
      join_reasoning_when_ready &
    fi

    step "mediamtx (RTSP :8554 · WebRTC :8889 · HLS :8888)"
    bash scripts/serve_rtsp.sh start || echo "!! mediamtx failed — live view unavailable, everything else works"

    if [ "$AUTOSTART" = 1 ]; then
      step "pipeline: ${STREAMS} streams, source=${SOURCE}"
      for _ in $(seq 1 30); do
        curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
        sleep 1
      done
      curl -s -m 180 -X POST \
        "http://127.0.0.1:8080/pipeline/start?streams=${STREAMS}&source=${SOURCE}&rtsp_out=true" \
        | sed 's/^/    /'
      echo
    fi

    cat <<EOF

==============================================================
 Up.  Dashboard: http://<host>:8080/
==============================================================
EOF

    # The API owns the container's lifetime: it serves the dashboard, and it is what starts and
    # stops the pipeline. Logs are tailed so `docker logs` shows something useful; the loop exits
    # non-zero when the API dies, so the restart policy can act on it.
    tail -F logs/*.log 2>/dev/null &
    while :; do
      if ! ps -eo args | grep '[s]ervices/api.py' >/dev/null 2>&1; then
        echo "!! API exited — stopping container"
        tail -40 logs/api.log 2>/dev/null
        exit 1
      fi
      sleep 5
    done
    ;;

  *)
    exec "$@"
    ;;
esac
