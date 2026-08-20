#!/usr/bin/env python3
"""
Inspect the incident store: list incidents and run integrity checks.

    python3 tools/inspect_db.py [--db data/events.db] [--limit 40] [--check-only]

Stdlib only (sqlite3), so it runs under system Python on the Jetson without the services venv.

The `--check` assertions are the point. Row counts alone cannot tell you the incident state
machine is sound — a store that closes incidents early, or leaks open ones, looks perfectly
healthy from a `SELECT COUNT(*)`. These are the invariants that must hold:

  * a closed incident has no contributing tracks left
  * an open incident has at least one
  * `open_tracks` across all incidents equals the number of live track mappings
  * at most one open incident per (camera, type) — otherwise merging is not working
  * no negative durations, no track mappings pointing at incidents that do not exist

Exit code is non-zero if any invariant fails, so this can gate a test run.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS = [
    ("closed incidents with tracks still open",
     "SELECT COUNT(*) FROM events WHERE ended_ts IS NOT NULL AND open_tracks > 0"),
    ("open incidents with no tracks",
     "SELECT COUNT(*) FROM events WHERE ended_ts IS NULL AND open_tracks = 0"),
    ("negative durations",
     "SELECT COUNT(*) FROM events WHERE duration_s < 0"),
    ("more than one open incident per (camera, type)",
     "SELECT COUNT(*) FROM (SELECT camera_id, type FROM events WHERE ended_ts IS NULL "
     " GROUP BY camera_id, type HAVING COUNT(*) > 1)"),
    ("track mappings pointing at missing incidents",
     "SELECT COUNT(*) FROM incident_tracks t LEFT JOIN events e USING(event_id) "
     " WHERE e.event_id IS NULL"),
    ("track mappings for already-closed incidents",
     "SELECT COUNT(*) FROM incident_tracks t JOIN events e USING(event_id) "
     " WHERE e.ended_ts IS NOT NULL"),
    ("clips marked ready with no file path",
     "SELECT COUNT(*) FROM events WHERE clip_state = 'ready' AND clip_uri IS NULL"),
    ("clips with a path but not marked ready",
     "SELECT COUNT(*) FROM events WHERE clip_uri IS NOT NULL AND clip_state != 'ready'"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(ROOT / "data/events.db"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    path = Path(args.db)
    if not path.exists():
        print(f"no database at {path}")
        return 2
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    open_n = db.execute("SELECT COUNT(*) FROM events WHERE ended_ts IS NULL").fetchone()[0]

    if not args.check_only:
        print(f"=== incidents ({total} total, {open_n} open) ===")
        rows = db.execute(
            "SELECT camera_id, type, severity, label, hits, open_tracks, duration_s, ended_ts "
            "  FROM events ORDER BY camera_id, ts LIMIT ?", (args.limit,)).fetchall()
        for r in rows:
            state = "OPEN " if r["ended_ts"] is None else "ended"
            dur = f"{r['duration_s']:.1f}s" if r["duration_s"] is not None else "-"
            label = (r["label"] or "")[:30]
            print(f"  cam{r['camera_id']:02d} {state} {r['severity']:8s} "
                  f"hits={r['hits']:<4d} open={r['open_tracks']:<3d} dur={dur:>8s}  {label}")

        print("\n=== by camera ===")
        for r in db.execute(
                "SELECT camera_id, COUNT(*) n, SUM(hits) hits, "
                "       SUM(CASE WHEN severity='high' THEN 1 ELSE 0 END) high "
                "  FROM events GROUP BY camera_id ORDER BY camera_id"):
            print(f"  cam{r['camera_id']:02d}  incidents={r['n']:<4d} "
                  f"contributing_tracks={r['hits']:<5d} high_severity={r['high']}")

        print("\n=== by type/severity ===")
        for r in db.execute("SELECT type, severity, COUNT(*) n FROM events "
                            "GROUP BY type, severity ORDER BY n DESC"):
            print(f"  {r['type']:16s} {r['severity']:9s} {r['n']}")

        print("\n=== clips ===")
        for r in db.execute("SELECT clip_state, COUNT(*) n FROM events "
                            "GROUP BY clip_state ORDER BY n DESC"):
            print(f"  {r['clip_state']:10s} {r['n']}")
        for r in db.execute("SELECT clip_error, COUNT(*) n FROM events "
                            " WHERE clip_error IS NOT NULL GROUP BY clip_error "
                            " ORDER BY n DESC LIMIT 3"):
            print(f"    failure: {r['clip_error'][:70]} (x{r['n']})")

    print("\n=== integrity ===")
    failed = 0
    for name, sql in CHECKS:
        n = db.execute(sql).fetchone()[0]
        ok = "ok  " if n == 0 else "FAIL"
        if n:
            failed += 1
        print(f"  [{ok}] {name}: {n}")

    sum_open = db.execute("SELECT COALESCE(SUM(open_tracks), 0) FROM events").fetchone()[0]
    n_maps = db.execute("SELECT COUNT(*) FROM incident_tracks").fetchone()[0]
    ok = "ok  " if sum_open == n_maps else "FAIL"
    if sum_open != n_maps:
        failed += 1
    print(f"  [{ok}] sum(open_tracks)={sum_open} equals live track mappings={n_maps}")

    print(f"\n{'ALL CHECKS PASSED' if failed == 0 else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
