#!/usr/bin/env python3
"""How long does an alert take to reach an operator? Measured, not assumed.

    python3 scripts/measure_alert_latency.py --seconds 300

RUN THIS ON THE JETSON. It compares each incident's `ts` (set by the pipeline at the moment the
rule fired, on the Jetson's clock) against the wall-clock time the row became visible. Running it
from the laptop would fold Mac/Jetson clock skew into every number and the result would be
meaningless — the skew is unbounded and silent.

## What is being measured

An alert crosses four boundaries, and only the whole chain matters to an operator:

    rule fires (event.ts)
      -> emit to Redis          fire-and-forget, bounded queue, drop-oldest
      -> event service          consumer group, incident state machine
      -> SQLite row             the incident exists and /events will return it
      -> dashboard              WebSocket push, or the 8 s feed poll as backstop

`visible` below is the first three: emit -> queryable. That is the number the dashboard's own
latency is added to, and it is the one the system controls.

`verdict` is a separate, much longer clock: the VLM has to cut a crop and run inference before an
incident stops saying "unverified". An operator watching the feed sees the incident at `visible`
and its adjudication at `verdict`; reporting only one of them would misrepresent the system.

Percentiles, not means: a mean hides the tail, and the tail is what gets noticed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def summarise(name: str, vals: list[float]) -> str:
    if not vals:
        return f"  {name:<22} no samples"
    return (f"  {name:<22} n={len(vals):<4} "
            f"min={min(vals):6.2f}s  p50={statistics.median(vals):6.2f}s  "
            f"p90={pct(vals, 90):6.2f}s  max={max(vals):6.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--interval", type=float, default=0.25,
                    help="poll period; it bounds the resolution of every number here")
    a = ap.parse_args()

    seen: dict[str, float] = {}          # id -> event ts
    visible: list[float] = []            # emit -> queryable
    verdict: list[float] = []            # emit -> adjudicated
    clip: list[float] = []               # emit -> clip available
    pending_verdict: dict[str, float] = {}
    pending_clip: dict[str, float] = {}

    end = time.time() + a.seconds
    print(f"measuring for {a.seconds:.0f}s against {a.url} "
          f"(poll {a.interval}s)\n", flush=True)

    # PRIME: everything already in the database at startup is backlog, not a measurement. Counting
    # it produces "latency" equal to each incident's AGE — the first run of this script reported a
    # tidy 43.4s for eighteen incidents at once, which is the age of the pipeline, not its
    # response time. Only incidents that appear while we are watching can be timed.
    # (Same trap as the dashboard's `primed` flag, which stops history toasting on first load.)
    primed = 0
    try:
        with urllib.request.urlopen(f"{a.url}/events?limit=200", timeout=5) as fh:
            for r in json.load(fh)["events"]:
                seen[r["id"]] = r["ts"]
                primed += 1
    except Exception:  # noqa: BLE001
        pass
    print(f"  ignoring {primed} incident(s) already present\n", flush=True)

    while time.time() < end:
        now = time.time()
        try:
            with urllib.request.urlopen(f"{a.url}/events?limit=60", timeout=5) as fh:
                rows = json.load(fh)["events"]
        except Exception as e:  # noqa: BLE001 — a blip must not end the measurement
            time.sleep(a.interval)
            continue

        for r in rows:
            rid, ts = r["id"], r["ts"]
            if rid not in seen:
                seen[rid] = ts
                visible.append(now - ts)
                # An incident arrives unverified and with no clip; both are filled in later.
                if (r.get("vlm_verdict") or "unverified") == "unverified":
                    pending_verdict[rid] = ts
                else:
                    verdict.append(now - ts)
                if not r.get("clip_url"):
                    pending_clip[rid] = ts
                else:
                    clip.append(now - ts)
                print(f"  + {r['sensor']:<6} {r['type']:<14} "
                      f"visible in {now - ts:5.2f}s   {r.get('label') or ''}", flush=True)

            if rid in pending_verdict and (r.get("vlm_verdict") or "unverified") != "unverified":
                verdict.append(now - pending_verdict.pop(rid))
            if rid in pending_clip and r.get("clip_url"):
                clip.append(now - pending_clip.pop(rid))

        time.sleep(a.interval)

    print("\n" + "=" * 78)
    print(f"ALERT LATENCY  ({len(seen)} incidents seen)")
    print(summarise("emit -> visible", visible))
    print(summarise("emit -> VLM verdict", verdict))
    print(summarise("emit -> clip ready", clip))
    print(f"\n  still unadjudicated at end: {len(pending_verdict)}")
    print(f"  still without a clip:       {len(pending_clip)}")
    print("\n  'visible' is emit -> queryable via the API. The dashboard adds its own delivery")
    print("  time on top: a WebSocket push (immediate) with an 8 s feed poll as the backstop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
