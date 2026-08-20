#!/usr/bin/env bash
# Runs ON THE JETSON. Finish the Telegram wiring: discover the chat id and store it.
#
#   ./scripts/telegram_setup.sh            wait for a message, save the chat id, send a probe
#   ./scripts/telegram_setup.sh --chat ID  skip discovery, use a chat id you already know
#   ./scripts/telegram_setup.sh --check     show current config and bot identity, change nothing
#
# A Telegram bot CANNOT open a conversation — the user has to speak first, which is what creates
# the chat and its id. So this waits for that message rather than guessing, and there is no way
# to automate around it.
#
# The token is read from .env at the repo root and never printed. Values are masked in all
# output here because this script's output routinely ends up pasted into a terminal log.
set -uo pipefail
cd "$(dirname "$0")/.."

ENVF="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$ENVF" ] || { echo "ERROR: $ENVF missing — put TELEGRAM_BOT_TOKEN in it first"; exit 1; }
set -a; . "$ENVF"; set +a
: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN not set in $ENVF}"

api() { curl -s -m 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/$1"; }

identity() {
  api getMe | python3 -c '
import json, sys
d = json.load(sys.stdin)
if not d.get("ok"):
    print("   token REJECTED:", d.get("description")); raise SystemExit(1)
r = d["result"]
print("   bot: @%s (%s)" % (r["username"], r["first_name"]))
'
}

save_chat() {
  grep -v "^TELEGRAM_CHAT_ID=" "$ENVF" > "${ENVF}.new" 2>/dev/null || true
  echo "TELEGRAM_CHAT_ID=$1" >> "${ENVF}.new"
  mv "${ENVF}.new" "$ENVF"; chmod 600 "$ENVF"
  echo "==> saved TELEGRAM_CHAT_ID to $ENVF"
  # Re-source with `set -a` so the value is EXPORTED into this shell's environment. Assigning it
  # as a plain shell variable is not enough: the probe below runs python3 as a child process,
  # which inherits only exported variables — the first version set it and the probe still
  # reported "TELEGRAM_CHAT_ID not set", one line after saving it. Re-sourcing the file also
  # means the probe sees exactly what the service will see, rather than what this script thinks
  # it wrote.
  set -a; . "$ENVF"; set +a
}

if [ "${1:-}" = "--check" ]; then
  identity || exit 1
  echo "   chat id: ${TELEGRAM_CHAT_ID:-(not set)}"
  exit 0
fi

if [ "${1:-}" = "--chat" ]; then
  [ -n "${2:-}" ] || { echo "usage: $0 --chat <id>"; exit 1; }
  identity || exit 1
  save_chat "$2"
  TELEGRAM_CHAT_ID="$2"
else
  identity || exit 1
  echo
  echo "==> Open Telegram, find the bot above, and send it any message (e.g. /start)."
  echo "    To alert a whole team instead, add the bot to a group and post there."
  echo "    Waiting up to 120s..."
  FOUND=""
  for _ in $(seq 1 40); do
    FOUND=$(api getUpdates | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
if not d.get("ok"):
    raise SystemExit
for u in d.get("result", []):
    m = u.get("message") or u.get("channel_post") or {}
    c = m.get("chat")
    if c:
        # Last one wins: if several chats messaged, the most recent is the one being set up.
        print("%s\t%s\t%s" % (c["id"], c.get("type"), c.get("title") or c.get("username") or c.get("first_name")))
' | tail -1)
    [ -n "$FOUND" ] && break
    sleep 3
  done
  if [ -z "$FOUND" ]; then
    echo "!! No message received. The bot cannot be configured until someone messages it."
    echo "!! Re-run this once you have."
    exit 1
  fi
  CID=$(echo "$FOUND" | cut -f1); CTYPE=$(echo "$FOUND" | cut -f2); CNAME=$(echo "$FOUND" | cut -f3)
  echo "==> found chat: ${CNAME} (${CTYPE})"
  save_chat "$CID"
  TELEGRAM_CHAT_ID="$CID"
fi

echo "==> sending a probe message"
if build/venv-services/bin/python3 services/notify_service.py --test; then
  echo
  echo "==> Telegram is wired up. To start sending real alerts:"
  echo "      set notify.telegram.enabled: true in configs/services.yml"
  echo "      ./scripts/demo_up.sh --down && ./scripts/demo_up.sh"
else
  echo "!! probe failed — see the error above"
  exit 1
fi
