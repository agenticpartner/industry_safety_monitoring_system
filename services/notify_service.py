#!/usr/bin/env python3
"""
Notification service — pushes incidents to Telegram with their evidence clip.

    python3 services/notify_service.py                 # run (honours configs/services.yml)
    python3 services/notify_service.py --dry-run       # decide and log, send nothing
    python3 services/notify_service.py --once          # one pass, for testing
    python3 services/notify_service.py --test          # send one probe message and exit

## Why a fourth polling service rather than a hook in the event service

Same reason clips and reasoning are separate (2.3, 2.4): the event service is the only writer on
the hot path and must never block on anything slow or remote. Telegram is a network call to
another continent — retries, rate limits, TLS. Polling SQLite makes this restartable, inspectable
with a SELECT, and naturally rate-limited, and a dead Telegram cannot touch incident capture.

## What is worth interrupting someone for

A phone notification costs far more than a row in a feed. Twenty cameras produce dozens of PPE
incidents a minute; a channel that buzzes that often gets muted, and a muted channel protects
nobody. So the policy (configs/services.yml) is:

* fire and VLM-raised hazards go **immediately** — waiting for adjudication on a fire is
  indefensible;
* PPE and overcrowding go **only once the VLM has confirmed them**, so a traffic cone mistaken
  for a worker never reaches anyone's phone;
* a per-camera cooldown stops one persistent situation dominating the channel, and a global cap
  stops a burst across many cameras doing the same. Fire is exempt from the cooldown.

## The clip has to be H.264

Evidence clips are H.265, because 20 concurrent 1080p streams do not fit the H.264 NVDEC budget
on this device. Telegram's clients will not preview HEVC — it arrives as a file attachment
nobody can play inline, which defeats the entire point of attaching it. The API already
transcodes lazily for the browser (`browser_playable`); this reuses that exact cache so a clip
watched in the dashboard and a clip pushed to Telegram are the same file, converted once.

## Sending is at-most-once, and the outcome is recorded

`notify_state` on the incident row goes pending -> sent | skipped | failed, with `notify_ts` and
`notify_error`. Claimed BEFORE the network call, so a crash mid-send cannot produce a duplicate
on restart: a duplicate alert is worse than a missing one here, because the whole value of the
channel is that a message in it means something new happened.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from media import browser_playable  # noqa: E402
from store import connect, migrate  # noqa: E402

SEV_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}
API = "https://api.telegram.org"


# ---------------------------------------------------------------------------------------------
# Telegram transport — urllib only, no new dependency
# ---------------------------------------------------------------------------------------------
def _multipart(fields: dict[str, str], file_field: str | None = None,
               file_path: Path | None = None) -> tuple[bytes, str]:
    """Build a multipart/form-data body.

    Hand-rolled because `requests` is not installed in build/venv-services and this service is not
    worth adding a dependency for — the whole protocol surface used here is two form posts.
    """
    boundary = f"----safetydemo{uuid.uuid4().hex}"
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        out += f"{v}\r\n".encode()
    if file_field and file_path is not None:
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n').encode()
        out += f"Content-Type: {ctype}\r\n\r\n".encode()
        out += file_path.read_bytes()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class Telegram:
    def __init__(self, token: str, chat_id: str, timeout: float = 60.0):
        self.token, self.chat_id, self.timeout = token, chat_id, timeout
        self.ctx = ssl.create_default_context()

    def _call(self, method: str, fields: dict, file_field=None, file_path=None) -> dict:
        body, ctype = _multipart({**fields, "chat_id": self.chat_id}, file_field, file_path)
        req = urllib.request.Request(f"{API}/bot{self.token}/{method}", data=body,
                                     headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as resp:
            out = json.load(resp)
        if not out.get("ok"):
            raise RuntimeError(f"telegram {method}: {out.get('description')}")
        return out

    def send_text(self, text: str) -> dict:
        return self._call("sendMessage", {"text": text, "parse_mode": "HTML",
                                          "disable_web_page_preview": "true"})

    def send_video(self, path: Path, caption: str) -> dict:
        # `supports_streaming` is what makes it play inline rather than download as a file.
        return self._call("sendVideo",
                          {"caption": caption, "parse_mode": "HTML",
                           "supports_streaming": "true"},
                          file_field="video", file_path=path)


# ---------------------------------------------------------------------------------------------
# Policy — decided here, in code, from the incident row
# ---------------------------------------------------------------------------------------------
def should_notify(row: dict, cfg: dict, now: float) -> tuple[bool, str]:
    """(send?, why). `why` is recorded either way, so a silent channel is explainable."""
    etype = row["type"]
    always = set(cfg.get("always_types") or [])
    confirmed_only = set(cfg.get("confirmed_types") or [])
    min_rank = SEV_RANK.get(cfg.get("min_severity", "high"), 2)

    if SEV_RANK.get(row["severity"], 0) < min_rank:
        return False, f"severity {row['severity']} below {cfg.get('min_severity')}"

    if etype in always:
        return True, "type is always-notify"

    if etype in confirmed_only:
        verdict = row.get("vlm_verdict")
        if verdict == "confirmed":
            return True, "VLM confirmed"
        if verdict in ("rejected", "uncertain"):
            return False, f"VLM {verdict}"
        # Still unadjudicated: not a no, just not yet. The caller leaves it pending.
        return False, "waiting for VLM verdict"

    return False, f"type {etype} not in notify policy"


def clip_ready_or_gave_up(row: dict, cfg: dict, now: float) -> tuple[bool, Path | None]:
    """(proceed?, clip path). The clip is the point, but it must not delay a fire indefinitely."""
    if row.get("clip_state") == "ready" and row.get("clip_uri"):
        return True, ROOT / row["clip_uri"]
    waited = now - float(row["ts"])
    if row.get("clip_state") in ("skipped", "failed", "expired"):
        return True, None            # never coming; send text-only
    if waited >= float(cfg.get("clip_wait_s", 20)):
        return True, None            # gave up waiting
    return False, None               # still cutting; try again next poll


def caption_for(row: dict) -> str:
    """A message someone can act on from a phone, without opening the dashboard."""
    icon = {"fire_alert": "🔥", "hazard_alert": "⚠️",
            "ppe_violation": "🦺", "overcrowding": "👥"}.get(row["type"], "❗")
    when = time.strftime("%H:%M:%S", time.localtime(row["ts"]))
    head = f"{icon} <b>{esc(row.get('label') or row['type'])}</b>"
    bits = [f"cam{row['camera_id']:02d}"]
    if row.get("zone"):
        bits.append(esc(row["zone"]))
    bits.append(row["severity"])
    if row.get("vlm_verdict") and row["vlm_verdict"] != "unverified":
        bits.append(f"VLM: {row['vlm_verdict']}")
    n = int(row.get("reminder_count") or 0)
    if n:
        mins = int((time.time() - float(row["ts"])) / 60)
        head = (f"{icon} <b>STILL OPEN · {mins} min</b> — "
                f"{esc(row.get('label') or row['type'])}")
    lines = [head, " · ".join(bits) + f" · {when}"]
    if row.get("vlm_reason"):
        lines.append("")
        lines.append(esc(row["vlm_reason"][:300]))
    return "\n".join(lines)


def esc(s: str) -> str:
    """Telegram HTML parse_mode. An un-escaped '<' in a VLM description kills the whole message."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------------------------
def h264_for(path: Path) -> Path:
    """The browser-playable proxy the API already builds, or the original if it is not HEVC.

    Deliberately reads the SAME cache rather than transcoding independently: two converters
    producing two files for one clip would double the disk cost and let them drift.
    """
    proxy = path.parent / "h264" / path.name
    if proxy.exists() and proxy.stat().st_size and proxy.stat().st_mtime >= path.stat().st_mtime:
        return proxy
    try:
        return browser_playable(path)
    except Exception:  # noqa: BLE001 — a failed transcode must not stop the alert
        return path


def load_cfg() -> dict:
    import yaml  # noqa: PLC0415
    cfg = yaml.safe_load((ROOT / "configs/services.yml").read_text()) or {}
    return cfg.get("notify") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="decide and log, send nothing")
    ap.add_argument("--test", action="store_true", help="send one probe message and exit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = load_cfg()
    tg_cfg = cfg.get("telegram") or {}
    token = os.environ.get(tg_cfg.get("token_env", "TELEGRAM_BOT_TOKEN"), "").strip()
    chat = os.environ.get(tg_cfg.get("chat_env", "TELEGRAM_CHAT_ID"), "").strip()

    # --dry-run deliberately ignores the enabled flag: it sends nothing, and being able to preview
    # exactly which incidents WOULD reach a phone is the main way to sanity-check the policy
    # before turning the channel on.
    if not tg_cfg.get("enabled") and not a.test and not a.dry_run:
        print("[notify] telegram disabled in configs/services.yml — nothing to do", flush=True)
        return 0
    if not (token and chat) and not a.dry_run:
        print(f"[notify] ERROR: {tg_cfg.get('token_env')} / {tg_cfg.get('chat_env')} not set. "
              f"Put them in .env at the repo root (see .env.example) and source it "
              f"(set -a; . ./.env; set +a).", flush=True)
        return 2

    tg = Telegram(token, chat) if (token and chat) else None

    if a.test:
        if tg is None:
            print("[notify] cannot test: credentials not set", flush=True)
            return 2
        tg.send_text("✅ <b>Safety Operations</b>\nTelegram notifications are wired up correctly.")
        print("[notify] test message sent", flush=True)
        return 0

    db_path = a.db or str(ROOT / "data/events.db")
    db = connect(db_path)
    migrate(db)

    # A row left in 'sending' means the process died between claiming and knowing the outcome —
    # the message may or may not have gone out. It is resolved to 'failed', NOT back to 'pending':
    # retrying could deliver a second copy of an alert somebody already acted on, and this channel
    # is only trustworthy if a message in it means something new happened. The row says exactly
    # what is unknown, so a human can check the chat and decide.
    stuck = db.execute(
        "UPDATE events SET notify_state='failed', "
        "  notify_error='interrupted mid-send; delivery unknown, not retried' "
        " WHERE notify_state='sending'").rowcount
    db.commit()
    if stuck:
        print(f"[notify] {stuck} incident(s) were interrupted mid-send — marked failed, "
              f"not retried (delivery unknown)", flush=True)

    print(f"[notify] db={db_path} chat={chat[:6]}… "
          f"always={cfg.get('always_types')} confirmed={cfg.get('confirmed_types')} "
          f"min_severity={cfg.get('min_severity')}{' DRY-RUN' if a.dry_run else ''}", flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sent_times: list[float] = []          # for the global per-minute cap
    last_by_camera: dict[int, float] = {}
    counts: dict[str, int] = {}
    done = 0
    exempt = set(cfg.get("cooldown_exempt_types") or [])

    # An incident that is merely "not decided yet" stays `pending` in the database — it is not a
    # skip, and it must stay eligible. But it must not be re-examined on every tick either: the
    # first version selected one pending row at a time, and a row awaiting a VLM verdict was
    # re-selected forever while every other pending incident starved behind it. So: work a BATCH,
    # and hold deferred ids in memory with a retry time.
    defer_until: dict[str, float] = {}

    while running:
        now = time.time()
        # A re-raised incident (still open past the threshold) is news again. Rather than build a
        # second notification path, its notify_state is returned to `pending` so it flows through
        # exactly the same policy, rate limits and clip handling as a first alert. `notified_at`
        # tracking lives in the incident row, so this survives a restart of this service.
        # COMMIT UNCONDITIONALLY. Python's sqlite3 opens a transaction on any DML and holds it
        # until commit or rollback — so committing only when rows matched left an empty write
        # transaction open for the life of the process, holding a RESERVED lock. Every other
        # writer then failed with "database is locked", and the event service, which acks entries
        # it cannot process, acked and DISCARDED live events.
        reraised = db.execute(
            "UPDATE events SET notify_state='pending' "
            " WHERE ended_ts IS NULL AND reminder_count > 0 AND notify_state IN ('sent','skipped') "
            "   AND (notify_ts IS NULL OR last_reminder_ts > notify_ts)").rowcount
        db.commit()
        batch = [dict(r) for r in db.execute(
            "SELECT * FROM events WHERE notify_state = 'pending' ORDER BY "
            "  CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, ts "
            " LIMIT 50")]
        ready = [r for r in batch if defer_until.get(r["event_id"], 0) <= now]
        if not ready:
            if a.once and not batch:
                break
            if a.once and all(defer_until.get(r["event_id"], 0) > now for r in batch):
                # One pass is done: everything left is explicitly waiting on something else.
                print(f"[notify] {len(batch)} incident(s) still awaiting a verdict or clip",
                      flush=True)
                break
            time.sleep(float(cfg.get("poll_interval_s", 2.0)))
            continue

        row = ready[0]
        eid = row["event_id"]

        send, why = should_notify(row, cfg, now)
        if not send:
            if why == "waiting for VLM verdict":
                # Genuinely not decided yet — leave pending and come back to it later. Everything
                # else is a decision, and gets recorded so the channel's silence is explainable.
                defer_until[eid] = now + 5.0
                continue
            db.execute("UPDATE events SET notify_state='skipped', notify_ts=?, notify_error=? "
                       " WHERE event_id=?", (now, why, eid))
            db.commit()
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue

        # Rate limits. Applied AFTER the policy decision so a suppressed alert is recorded as
        # rate-limited rather than silently indistinguishable from one that failed policy.
        sent_times = [t for t in sent_times if now - t < 60]
        if len(sent_times) >= int(cfg.get("max_per_minute", 6)):
            time.sleep(1.0)
            continue
        cam_last = last_by_camera.get(row["camera_id"], 0.0)
        cooldown = float(cfg.get("cooldown_per_camera_s", 120))
        if row["type"] not in exempt and now - cam_last < cooldown:
            db.execute("UPDATE events SET notify_state='skipped', notify_ts=?, notify_error=? "
                       " WHERE event_id=?",
                       (now, f"cooldown: cam{row['camera_id']:02d} notified "
                             f"{int(now - cam_last)}s ago", eid))
            db.commit()
            counts["cooldown"] = counts.get("cooldown", 0) + 1
            continue

        proceed, clip = clip_ready_or_gave_up(row, cfg, now)
        if not proceed:
            time.sleep(float(cfg.get("poll_interval_s", 2.0)))
            continue

        # Claim BEFORE the network call: a crash mid-send must not produce a duplicate on
        # restart. A duplicate is worse than a miss here — the channel's value is that a message
        # in it means something new happened.
        db.execute("UPDATE events SET notify_state='sending' WHERE event_id=?", (eid,))
        db.commit()

        caption = caption_for(row)
        if a.dry_run:
            print(f"[notify] DRY-RUN would send {eid[:8]} cam{row['camera_id']:02d} "
                  f"{row['type']} clip={'yes' if clip else 'no'} ({why})", flush=True)
            db.execute("UPDATE events SET notify_state='skipped', notify_ts=?, notify_error=? "
                       " WHERE event_id=?", (now, "dry-run", eid))
            db.commit()
            counts["dry-run"] = counts.get("dry-run", 0) + 1
            done += 1
            if a.limit and done >= a.limit:
                break
            continue

        try:
            if clip is not None and clip.exists():
                tg.send_video(h264_for(clip), caption)
                kind = "video"
            else:
                tg.send_text(caption)
                kind = "text"
            db.execute("UPDATE events SET notify_state='sent', notify_ts=?, notify_error=NULL "
                       " WHERE event_id=?", (time.time(), eid))
            db.commit()
            sent_times.append(time.time())
            last_by_camera[row["camera_id"]] = time.time()
            counts[kind] = counts.get(kind, 0) + 1
            print(f"[notify] sent {kind} {eid[:8]} cam{row['camera_id']:02d} {row['type']} "
                  f"({why})", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError,
                ValueError, json.JSONDecodeError) as e:
            msg = f"{type(e).__name__}: {str(e)[:150]}"
            db.execute("UPDATE events SET notify_state='failed', notify_ts=?, notify_error=? "
                       " WHERE event_id=?", (time.time(), msg, eid))
            db.commit()
            counts["failed"] = counts.get("failed", 0) + 1
            print(f"[notify] FAILED {eid[:8]}: {msg}", flush=True)
            time.sleep(2.0)

        done += 1
        if a.limit and done >= a.limit:
            break
        if a.once:
            continue

    print(f"[notify] {counts}", flush=True)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
