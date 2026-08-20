#!/usr/bin/env python3
"""
Incident search — structured filters + SQLite FTS5 full-text over incident records.

    python3 services/search_service.py --text "forklift" --verdict confirmed
    python3 services/search_service.py --sync          # rebuild the index

Stdlib only (sqlite3 with FTS5, verified present on this build). Importable as a library; the
agent calls it rather than touching SQL.

## Why FTS5 and not embeddings

The Phase 2 plan called for "text-embedding search over incident records… sqlite-vec or FAISS,
embeddings from a small local model". FTS5 is used instead, deliberately:

* **An incident is mostly structured.** camera, zone, type, severity, verdict, time, duration.
  "When did someone last enter zone 3 without a helmet" is a WHERE clause, not a nearest-neighbour
  problem, and a WHERE clause answers it exactly rather than approximately.
* **The free text is short and literal.** `vlm_reason` is one sentence of concrete nouns —
  "traffic cone", "hi-vis vest", "fire extinguisher". Lexical matching handles that well.
* **An embedding model is another always-on GPU tenant.** Every model loaded competes with 20
  camera streams for the same iGPU, and Phase 2.0 measured how tight that budget is. Spending it
  on approximate matching over a few hundred short strings is a bad trade at this scale.

The honest limit: FTS5 will not match "near a vehicle" to "forklift" — there is no synonymy. If
that turns out to matter, add `sqlite-vec` over `vlm_reason` alongside this, not instead of it;
the structured filters stay useful either way. VSS uses Cosmos Embed1 for genuine VIDEO search,
which is a different and much larger capability than what this replaces.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import connect  # noqa: E402

FTS_SCHEMA = """
-- Contentless-adjacent FTS index over the human-readable parts of an incident. `event_id` is
-- UNINDEXED: it is carried so a hit can be joined back, never matched against.
CREATE VIRTUAL TABLE IF NOT EXISTS incident_fts USING fts5(
    event_id UNINDEXED,
    body,
    tokenize = 'porter unicode61'
);
"""


def _body(r: sqlite3.Row) -> str:
    """The searchable text for one incident.

    Deliberately includes the structured fields as words too (camera name, zone, severity, type),
    so a lexical query like "SpillZone" or "cam02" hits without the caller having to know whether
    a term is structured or free text.
    """
    parts = [
        f"cam{r['camera_id']:02d}",
        (r["type"] or "").replace("_", " "),
        r["severity"] or "",
        r["zone"] or "",
        r["label"] or "",
        r["vlm_verdict"] or "",
        r["vlm_reason"] or "",
    ]
    return " ".join(p for p in parts if p)


def sync(db: sqlite3.Connection) -> int:
    """Rebuild the FTS index from `events`. Returns rows indexed.

    A full rebuild rather than incremental upkeep: at this scale (hundreds of incidents) it is
    milliseconds, and it cannot drift out of sync with the source table — which incremental
    triggers can, silently, exactly when a schema changes underneath them.
    """
    try:
        db.executescript(FTS_SCHEMA)
        db.execute("DELETE FROM incident_fts")
        rows = db.execute("SELECT * FROM events").fetchall()
        db.executemany("INSERT INTO incident_fts (event_id, body) VALUES (?,?)",
                       [(r["event_id"], _body(r)) for r in rows])
        db.commit()
        return len(rows)
    except sqlite3.OperationalError as e:
        # SQLite allows one writer at a time even in WAL mode, and the event service holds that
        # slot while incidents are landing. A slightly stale search index is far better than a
        # failed query, so a locked database degrades to "use the index as it stands".
        if "locked" not in str(e) and "busy" not in str(e):
            raise
        db.rollback()
        return -1


def _fts_query(text: str) -> str:
    """Turn free text into an FTS5 OR-query, quoting each term.

    Quoting matters: FTS5 treats bare `-`, `"` and `*` as operators, so an unquoted user string
    like "hi-vis" is a syntax error rather than a search. OR rather than AND because a question
    ("someone without a helmet near the forklift") carries more words than any single incident
    line will contain, and AND would return nothing.
    """
    terms = [t.strip('"').replace('"', '') for t in text.split()]
    terms = [t for t in terms if len(t) > 2]
    return " OR ".join(f'"{t}"' for t in terms) if terms else ""


def _filters(camera_id=None, event_type=None, severity=None, zone=None, vlm_verdict=None,
             since_ts=None, until_ts=None, open_only=False) -> tuple[list[str], list]:
    """The WHERE clause shared by `search` and `count_matching`.

    Shared deliberately: a count that is computed from a different predicate than the rows it
    describes is worse than no count at all, because it looks authoritative while disagreeing
    with what the caller can see.
    """
    where, args = ["1=1"], []
    if camera_id is not None:
        where.append("e.camera_id = ?"); args.append(camera_id)
    if event_type and event_type != "any":
        where.append("e.type = ?"); args.append(event_type)
    if severity and severity != "any":
        where.append("e.severity = ?"); args.append(severity)
    if zone and zone != "any":
        where.append("e.zone = ?"); args.append(zone)
    if vlm_verdict and vlm_verdict != "any":
        if vlm_verdict == "unverified":
            # Matches the VSS meaning: nobody has looked yet.
            where.append("e.vlm_verdict IS NULL")
        else:
            where.append("e.vlm_verdict = ?"); args.append(vlm_verdict)
    if since_ts is not None:
        where.append("e.ts >= ?"); args.append(since_ts)
    if until_ts is not None:
        where.append("e.ts <= ?"); args.append(until_ts)
    if open_only:
        where.append("e.ended_ts IS NULL")
    return where, args


def count_matching(db: sqlite3.Connection, camera_id=None, event_type=None, severity=None,
                   zone=None, vlm_verdict=None, since_ts=None, until_ts=None,
                   open_only=False) -> int:
    """How many rows the structured filters match, ignoring any row limit.

    This is what makes truncation visible. Without it a caller receiving 10 rows cannot tell
    whether that is the whole answer or the first page of ninety — and an LLM handed ten rows
    will cheerfully count them and report ten.

    Text relevance is deliberately excluded: `text` only reorders within the filters, so the
    honest denominator is the structured match count.
    """
    where, args = _filters(camera_id, event_type, severity, zone, vlm_verdict,
                           since_ts, until_ts, open_only)
    sql = f"SELECT COUNT(*) FROM events e WHERE {' AND '.join(where)}"
    return int(db.execute(sql, args).fetchone()[0])


def search(db: sqlite3.Connection,
           text: str | None = None,
           camera_id: int | None = None,
           event_type: str | None = None,
           severity: str | None = None,
           zone: str | None = None,
           vlm_verdict: str | None = None,
           since_ts: float | None = None,
           until_ts: float | None = None,
           open_only: bool = False,
           limit: int = 20) -> list[dict]:
    """Structured filters, optionally narrowed by full-text relevance.

    Filters are ANDed and applied in SQL; `text` only reorders and narrows within them. That way a
    caller asking for "confirmed helmet violations on camera 3" gets exactly those, and the text
    is a refinement rather than a competing signal.
    """
    where, args = _filters(camera_id, event_type, severity, zone, vlm_verdict,
                           since_ts, until_ts, open_only)

    q = _fts_query(text) if text else ""
    if q:
        sql = ("SELECT e.*, bm25(incident_fts) AS rank FROM incident_fts "
               " JOIN events e ON e.event_id = incident_fts.event_id "
               f" WHERE incident_fts MATCH ? AND {' AND '.join(where)} "
               " ORDER BY rank LIMIT ?")
        try:
            rows = db.execute(sql, (q, *args, limit)).fetchall()
        except sqlite3.OperationalError:
            # A malformed FTS expression must degrade to the structured query, not to an error —
            # the filters are the part the caller actually depends on.
            rows = []
        if rows:
            return [dict(r) for r in rows]

    sql = (f"SELECT e.* FROM events e WHERE {' AND '.join(where)} "
           " ORDER BY e.ts DESC LIMIT ?")
    return [dict(r) for r in db.execute(sql, (*args, limit)).fetchall()]


def summarise(db: sqlite3.Connection, hours: float | None = None,
              vlm_verdict: str | None = None, event_type: str | None = None,
              zone: str | None = None, camera_id: int | None = None,
              severity: str | None = None, open_only: bool = False,
              since_ts: float | None = None) -> dict:
    """Counts an operator or an agent would want before drilling in.

    Every count here comes from SQL over the whole matching set. That matters more than it
    sounds: the alternative — letting the LLM count the rows it was shown — produced a fluent,
    confidently wrong answer (5 cameras named out of 9, two counts wrong) because the rows were
    a truncated top-N page. Arithmetic is not something to ask a language model for when the
    database can do it exactly.

    It shares `_filters` with `search`, so a summary and a retrieval asked the same question
    describe the same set.
    """
    if hours and since_ts is None:
        since_ts = time.time() - hours * 3600
    where, args = _filters(camera_id, event_type, severity, zone, vlm_verdict,
                           since_ts, None, open_only)
    w = " AND ".join(where)
    one = lambda sql: db.execute(sql, args).fetchone()  # noqa: E731
    grp = lambda col: db.execute(  # noqa: E731
        f"SELECT {col}, COUNT(*) FROM events e WHERE {w} GROUP BY {col}", args)
    out = {
        "incidents": one(f"SELECT COUNT(*) FROM events e WHERE {w}")[0],
        "open": one(f"SELECT COUNT(*) FROM events e WHERE {w} AND e.ended_ts IS NULL")[0],
        "by_type": {r[0]: r[1] for r in grp("e.type")},
        "by_severity": {r[0]: r[1] for r in grp("e.severity")},
        "by_verdict": {(r[0] or "unverified"): r[1] for r in grp("e.vlm_verdict")},
        "by_zone": {(r[0] or "no zone"): r[1] for r in grp("e.zone")},
        # The rule that fired, verbatim ("no vest?", "NO HELMET + no vest?", "OVERCROWDED
        # MainFloorOC"). `type` says ppe_violation; only the label says WHICH protective
        # equipment was missing, and "how many were helmet vs vest" is an obvious question that
        # nothing else in this summary could answer.
        "by_label": {(r[0] or "(none)"): r[1] for r in grp("e.label")},
        "cameras_affected": one(
            f"SELECT COUNT(DISTINCT e.camera_id) FROM events e WHERE {w}")[0],
        "with_clips": one(
            f"SELECT COUNT(*) FROM events e WHERE {w} AND e.clip_uri IS NOT NULL")[0],
        # Keyed by sensor name, not camera_id, so the agent can quote it verbatim without having
        # to reformat an integer into "cam07" — a step it sometimes got wrong.
        "by_camera": {f"cam{r[0]:02d}": r[1] for r in db.execute(
            f"SELECT e.camera_id, COUNT(*) FROM events e WHERE {w} "
            f"GROUP BY e.camera_id ORDER BY e.camera_id", args)},
        "by_camera_type": {f"cam{r[0]:02d}/{r[1]}": r[2] for r in db.execute(
            f"SELECT e.camera_id, e.type, COUNT(*) FROM events e WHERE {w} "
            f"GROUP BY e.camera_id, e.type ORDER BY e.camera_id", args)},
        "by_camera_verdict": {f"cam{r[0]:02d}/{r[1] or 'unverified'}": r[2] for r in db.execute(
            f"SELECT e.camera_id, e.vlm_verdict, COUNT(*) FROM events e WHERE {w} "
            f"GROUP BY e.camera_id, e.vlm_verdict ORDER BY e.camera_id", args)},
        # The filters are reported back so a reader can tell WHAT was counted. A bare total is
        # ambiguous — "23" meant "all incidents" while being read as "23 confirmed".
        "filters": {k: v for k, v in
                    (("hours", hours), ("vlm_verdict", vlm_verdict), ("event_type", event_type),
                     ("zone", zone), ("sensor", f"cam{camera_id:02d}" if camera_id else None),
                     ("severity", severity), ("open_only", open_only or None)) if v},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--text")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--type", dest="etype")
    ap.add_argument("--severity")
    ap.add_argument("--zone")
    ap.add_argument("--verdict")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    path = Path(args.db) if args.db else ROOT / "data/events.db"
    db = connect(path)
    db.row_factory = sqlite3.Row

    if args.sync or args.text:
        n = sync(db)
        if args.sync:
            print(f"indexed {n} incidents")
            if not args.text and not args.summary:
                return 0

    if args.summary:
        print(json.dumps(summarise(db, args.hours), indent=2))
        return 0

    rows = search(db, text=args.text, camera_id=args.camera, event_type=args.etype,
                  severity=args.severity, zone=args.zone, vlm_verdict=args.verdict,
                  since_ts=(time.time() - args.hours * 3600) if args.hours else None,
                  limit=args.limit)
    print(f"{len(rows)} incident(s)")
    for r in rows:
        v = r.get("vlm_verdict") or "unverified"
        print(f"  cam{r['camera_id']:02d} {r['type']:14s} {r['severity']:8s} "
              f"{str(r['zone']):14s} {v:10s} {(r['label'] or '')[:22]:22s} "
              f"{'clip' if r.get('clip_uri') else '----'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
