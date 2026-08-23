#!/usr/bin/env python3
"""
Event service — consume the Redis stream, fold transitions into incidents, persist to SQLite.

    build/venv-services/bin/python3 services/event_service.py [--from-start] [--once]

Runs in `build/venv-services` (real `redis` client), NOT on system Python. The no-dependency
constraint applies to `app/events.py`, which is imported by the pipeline; this side is free.

The incident model and all the SQL live in `services/store.py`, which has no broker dependency
and is unit-tested on its own. This file is only transport: read, apply, ack.

VSS counterpart: Behavior Analytics + incident records. Deliberately a separate process talking
over a bus rather than a thread inside the pipeline, so it can be killed, restarted, or fall
hours behind without the pipeline noticing — the property Phase 2.8 verifies directly by stalling
it on purpose.

## Consumer groups, and why

Reads via `XREADGROUP` with an explicit group, not a plain `XREAD $`. A plain read starts at "now"
and silently loses anything published while the service was down, which would make "restart the
service" a lossy operation and undermine the durability the bus exists to provide. With a group,
Redis tracks the last-delivered id server-side, so a restart resumes where it left off.

Redelivery (a crash between XREADGROUP and XACK) is harmless: `event_id` is the primary key, so a
re-applied open is `INSERT OR IGNORE`, and a re-applied close finds its track mapping already
gone and is counted as `close-unmatched`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
from pathlib import Path

import redis
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import EventStore, connect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GROUP = "event_service"
CONSUMER = "worker-1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/services.yml"))
    ap.add_argument("--from-start", action="store_true",
                    help="(re)create the consumer group at the beginning of the stream and "
                         "replay everything — useful after wiping the database")
    ap.add_argument("--once", action="store_true",
                    help="drain what is pending and exit (for tests)")
    ap.add_argument("--block-ms", type=int, default=5000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rc = (cfg.get("events") or {}).get("redis") or {}
    store_cfg = cfg.get("store") or {}
    stream = rc.get("stream", "safety:events")

    r = redis.Redis(host=rc.get("host", "127.0.0.1"), port=int(rc.get("port", 6379)),
                    decode_responses=True)
    r.ping()

    db_path = Path(store_cfg.get("path", "data/events.db"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    store = EventStore(connect(db_path), float(store_cfg.get("merge_window_s", 30)))

    if args.from_start:
        try:
            r.xgroup_destroy(stream, GROUP)
        except redis.ResponseError:
            pass
    try:
        # mkstream=True so the service can start BEFORE the pipeline has ever published;
        # otherwise first boot fails with "no such key" purely because of start order.
        r.xgroup_create(stream, GROUP, id="0" if args.from_start else "$", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    print(f"[event-service] stream={stream} group={GROUP} db={db_path} "
          f"merge_window={store.merge_window_s}s", flush=True)

    running = True

    def _stop(_sig, _frm):
        nonlocal running
        running = False
        print("[event-service] stopping", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    counts: dict[str, int] = {}
    last_report = time.monotonic()

    # Re-raise long-running incidents. This lives HERE, in the service that owns the incident
    # lifecycle, not in the notifier: "this has been unresolved for eight minutes" is a fact about
    # the incident, and the dashboard must reflect it whether or not Telegram is configured.
    realert_after_s = float((cfg.get("store") or {}).get("realert_after_s", 0) or 0)
    last_sweep = time.monotonic()

    while running:
        try:
            resp = r.xreadgroup(GROUP, CONSUMER, {stream: ">"}, count=200, block=args.block_ms)
        except redis.RedisError as e:
            # The broker going away must not kill the service: the pipeline keeps publishing and
            # we want to resume when it returns.
            print(f"[event-service] redis error: {e}; retrying", flush=True)
            time.sleep(2)
            continue

        # Swept on a timer, and BEFORE the empty-read continue below. In steady state almost
        # nothing transitions — everyone already violating produces merges, not new events — so
        # `xreadgroup` returns empty most of the time. Putting the sweep after that early
        # `continue` meant it ran only while events were arriving, i.e. never at the exact moment
        # a long-open incident most needs re-raising.
        mono = time.monotonic()
        if realert_after_s and mono - last_sweep >= 15:
            last_sweep = mono
            for row in store.raise_stale(realert_after_s):
                counts["re-raised"] = counts.get("re-raised", 0) + 1
                print(f"[event-service] re-raised {row['event_id'][:8]} "
                      f"cam{row['camera_id']:02d} {row['type']} "
                      f"open {(time.time() - row['ts']) / 60:.0f} min "
                      f"(reminder #{row['reminder_count']})", flush=True)

        if not resp:
            if args.once:
                break
            continue

        for _stream_name, entries in resp:
            for entry_id, fields in entries:
                try:
                    rec = json.loads(fields["data"])
                    # Records carrying a `kind` are out-of-band facts about an EXISTING incident
                    # rather than transitions to fold into one. Only smart record uses this so
                    # far, to report the evidence clip it just finished writing. Events have no
                    # `kind`, which is what keeps the common path untouched.
                    if rec.get("kind") == "crop_ready":
                        what = store.attach_crop(rec)
                    elif rec.get("kind") == "clip_ready":
                        what = store.attach_clip(rec["event_id"], rec["clip_uri"])
                    else:
                        what = store.apply(rec)
                    counts[what] = counts.get(what, 0) + 1
                except sqlite3.OperationalError as e:
                    # TRANSIENT — a lock or a busy database. This is NOT a malformed entry and
                    # must not be acked: acking discards a real event. Leave it pending so the
                    # next read redelivers it, and back off so a stuck writer does not become a
                    # hot loop. Observed for real: another service left an empty write
                    # transaction open, and every event arriving during that window was
                    # classified "bad" and thrown away.
                    print(f"[event-service] transient store error, will retry: {e}", flush=True)
                    counts["retry"] = counts.get("retry", 0) + 1
                    time.sleep(0.5)
                    continue
                except Exception as e:  # noqa: BLE001
                    # A malformed entry must not wedge the consumer. Ack it — leaving it pending
                    # would redeliver it forever and block everything behind it.
                    print(f"[event-service] bad entry {entry_id}: {type(e).__name__}: {e}",
                          flush=True)
                    counts["bad"] = counts.get("bad", 0) + 1
                r.xack(stream, GROUP, entry_id)

        now = time.monotonic()
        if now - last_report >= 5:
            last_report = now
            print(f"[event-service] {dict(sorted(counts.items()))} | "
                  f"rows={store.count()} open={len(store.open_incidents())}", flush=True)

        if args.once:
            break

    print(f"[event-service] final {dict(sorted(counts.items()))} | rows={store.count()} "
          f"open={len(store.open_incidents())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
