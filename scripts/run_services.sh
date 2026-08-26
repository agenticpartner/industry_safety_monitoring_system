#!/usr/bin/env bash
# Runs ON THE JETSON. Start/stop the Phase 2 background services.
#
#   ./scripts/run_services.sh start [--from-start]   event service (+ redis check)
#   ./scripts/run_services.sh reasoning                start ONLY the reasoning service
#   ./scripts/run_services.sh stop
#   ./scripts/run_services.sh status
#   ./scripts/run_services.sh reset                  wipe stream + database, then start clean
#
# Development convenience only — Phase 2.8 replaces this with systemd units that restart on
# failure. The services are deliberately separate processes from the pipeline: killing one must
# never touch the other, which is verified by stopping redis under a running pipeline.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

VENV=build/venv-services/bin/python3
LOG=logs/event_service.log
CLIPLOG=logs/clip_service.log
REASONLOG=logs/reasoning_service.log
APILOG=logs/api.log
NOTIFYLOG=logs/notify_service.log
mkdir -p logs

# Kill by PID, never `pkill -f event_service.py`: over ssh the pattern also matches the invoking
# command line and takes the session down with it (exit 255). Bracket-matching does not help,
# because it only excludes the matcher's own process, not its parents.
service_pids() {
  ps -eo pid,args | grep '[e]vent_service.py' | grep -v 'bash -c' | awk '{print $1}'
}

clip_pids() {
  ps -eo pid,args | grep '[c]lip_service.py' | grep -v 'bash -c' | awk '{print $1}'
}

reason_pids() {
  ps -eo pid,args | grep '[r]easoning_service.py' | grep -v 'bash -c' | awk '{print $1}'
}

api_pids() {
  ps -eo pid,args | grep '[s]ervices/api.py' | grep -v 'bash -c' | awk '{print $1}'
}

notify_pids() {
  ps -eo pid,args | grep '[n]otify_service.py' | grep -v 'bash -c' | awk '{print $1}'
}

# Reasoning is optional and only useful with a VLM endpoint up. `nice` keeps it preemptible: it
# shares the GPU with 20 camera streams and the hot path must always win.
#
# Split out so it can be started ON ITS OWN, after the rest. The model server takes minutes to
# load — and on a container's first boot, longer still while it pulls ~12 GB of weights — and
# there is no reason for the dashboard and the pipeline to wait on that. `run_services.sh
# reasoning` is what joins the reasoning layer to a system that is already running.
start_reasoning() {
  if ! curl -sf --max-time 3 "$(python3 -c "import yaml;print(yaml.safe_load(open('configs/services.yml'))['reasoning']['endpoint'])")/models" >/dev/null 2>&1; then
    echo "!! no VLM endpoint — reasoning service NOT started."
    echo "!!   ./scripts/setup_reasoning.sh llamacpp --serve   (or bring up the vlm container)"
    return 1
  fi
  PIDS=$(reason_pids); [ -n "$PIDS" ] && { echo "$PIDS" | xargs -r kill -9; sleep 1; }
  nohup setsid nice -n 10 python3 services/reasoning_service.py > "$REASONLOG" 2>&1 < /dev/null &
  sleep 2
  if [ -n "$(reason_pids)" ]; then
    echo "==> reasoning service up (log: $REASONLOG)"
  else
    echo "!! reasoning service failed:"; tail -10 "$REASONLOG"; return 1
  fi
}

case "${1:-}" in
  start|reset)
    redis-cli ping >/dev/null 2>&1 || {
      echo "redis is not responding — sudo systemctl start redis-server"; exit 1; }

    EXTRA=""
    if [ "${1:-}" = reset ] || [ "${2:-}" = --from-start ]; then
      EXTRA="--from-start"
    fi
    if [ "${1:-}" = reset ]; then
      STREAM=$(python3 -c "import yaml;print((yaml.safe_load(open('configs/services.yml'))['events']['redis']).get('stream','safety:events'))")
      echo "==> wiping ${STREAM}, data/events.db and data/clips/"
      redis-cli DEL "$STREAM" >/dev/null
      rm -f data/events.db data/events.db-wal data/events.db-shm
      rm -rf data/clips
    fi

    PIDS=$(service_pids); [ -n "$PIDS" ] && { echo "$PIDS" | xargs -r kill -9; sleep 1; }
    nohup setsid "$VENV" services/event_service.py $EXTRA > "$LOG" 2>&1 < /dev/null &
    sleep 2
    if [ -n "$(service_pids)" ]; then
      echo "==> event service up (log: $LOG)"
      head -2 "$LOG"
    else
      echo "!! event service failed to start:"; tail -20 "$LOG"; exit 1
    fi

    # Clip service runs on SYSTEM python: it needs only stdlib + ffmpeg, and keeping it out of the
    # venv means one less thing to rebuild. It is a separate process from the event service so a
    # slow ffmpeg cannot stall incident ingestion.
    PIDS=$(clip_pids); [ -n "$PIDS" ] && { echo "$PIDS" | xargs -r kill -9; sleep 1; }
    nohup setsid python3 services/clip_service.py > "$CLIPLOG" 2>&1 < /dev/null &
    sleep 2
    if [ -n "$(clip_pids)" ]; then
      echo "==> clip service up (log: $CLIPLOG)"
    else
      echo "!! clip service failed to start:"; tail -20 "$CLIPLOG"
    fi

    start_reasoning

    # Outbound notifications. Started only when enabled in configs/services.yml AND credentials
    # are present: this is the one component that sends anything off the device, and an alert
    # pushed to a phone cannot be unsent, so it never starts by accident. Credentials come from
    # .env at the repo root, sourced via load_env() because Ubuntu's .bashrc returns early for
    # non-interactive shells and exports there are invisible to `ssh host 'cmd'`.
    load_env
    NOTIFY_ON=$(python3 -c "import yaml;print((yaml.safe_load(open('configs/services.yml')).get('notify') or {}).get('telegram',{}).get('enabled',False))" 2>/dev/null)
    if [ "$NOTIFY_ON" = "True" ]; then
      if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        PIDS=$(notify_pids); [ -n "$PIDS" ] && { echo "$PIDS" | xargs -r kill -9; sleep 1; }
        nohup setsid "$VENV" services/notify_service.py > "$NOTIFYLOG" 2>&1 < /dev/null &
        sleep 2
        [ -n "$(notify_pids)" ] && echo "==> notify service up (log: $NOTIFYLOG)" \
                               || { echo "!! notify service failed:"; tail -10 "$NOTIFYLOG"; }
      else
        echo "!! telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set"
        echo "!!   put them in .env at the repo root (see .env.example)"
      fi
    fi

    # REST + WebSocket control plane. Runs in the services venv (FastAPI/uvicorn live there, not
    # in system python). Default 8080; 8000 and 8001 are the two model servers. Override with
    # ISMS_API_PORT when the host already has something on 8080 (Label Studio does, on Spark).
    API_PORT="${ISMS_API_PORT:-8080}"
    PIDS=$(api_pids); [ -n "$PIDS" ] && { echo "$PIDS" | xargs -r kill -9; sleep 1; }
    nohup setsid "$VENV" services/api.py --port "$API_PORT" > "$APILOG" 2>&1 < /dev/null &
    sleep 3
    if [ -n "$(api_pids)" ]; then
      echo "==> API up on :${API_PORT}  (docs: http://$(hostname):${API_PORT}/docs)"
    else
      echo "!! API failed to start:"; tail -15 "$APILOG"
    fi
    ;;

  reasoning)
    start_reasoning || exit 1
    ;;

  stop)
    PIDS="$(service_pids) $(clip_pids) $(reason_pids) $(api_pids) $(notify_pids)"
    if [ -n "$(echo $PIDS)" ]; then echo "$PIDS" | xargs -r kill -9; echo "==> stopped"; \
    else echo "==> not running"; fi
    ;;

  status)
    redis-cli ping >/dev/null 2>&1 && echo "redis         : up" || echo "redis         : DOWN"
    PIDS=$(service_pids)
    [ -n "$PIDS" ] && echo "event service : up (pid $(echo $PIDS | tr '\n' ' '))" \
                   || echo "event service : down"
    CPIDS=$(clip_pids)
    [ -n "$CPIDS" ] && echo "clip service  : up (pid $(echo $CPIDS | tr '\n' ' '))" \
                    || echo "clip service  : down"
    RPIDS=$(reason_pids)
    [ -n "$RPIDS" ] && echo "reasoning svc : up (pid $(echo $RPIDS | tr '\n' ' '))" \
                    || echo "reasoning svc : down"
    APIDS=$(api_pids)
    _api_port="${ISMS_API_PORT:-8080}"
    [ -n "$APIDS" ] && echo "api :${_api_port}     : up (pid $(echo $APIDS | tr '\n' ' '))" \
                    || echo "api :${_api_port}     : down"
    if [ -d data/clips ]; then
      echo "clips         : $(ls data/clips/*.mp4 2>/dev/null | wc -l) files, $(du -sh data/clips 2>/dev/null | cut -f1)"
    fi
    STREAM=$(python3 -c "import yaml;print((yaml.safe_load(open('configs/services.yml'))['events']['redis']).get('stream','safety:events'))" 2>/dev/null || echo safety:events)
    echo "stream len    : $(redis-cli XLEN "$STREAM" 2>/dev/null || echo '-')"
    # Pending entries are the honest backlog signal: delivered to the consumer group but not yet
    # acked. A number that keeps climbing means the service is falling behind, not idling.
    echo "pending       : $(redis-cli XPENDING "$STREAM" event_service 2>/dev/null | head -1 || echo '-')"
    [ -f data/events.db ] && python3 tools/inspect_db.py --check-only || echo "db            : none"
    ;;

  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
