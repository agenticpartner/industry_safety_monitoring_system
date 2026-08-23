"""
SQLite incident store. No Redis, no DeepStream — pure logic over a database, so it can be
unit-tested on a laptop the way `app/rules.py` can.

## The model: an incident is not a detection

The pipeline emits per-TRACK transitions ("track 7 on camera 3 started violating"). Operators do
not think in tracks — they think "camera 3 has a helmet problem right now". So the store folds
tracks into **incidents**: one open incident per (camera, type), with the contributing tracks
counted.

That folding is what makes the merge window necessary. ~26% of tracks at 15 fps analytics are too
short to be adjudicated, so one person walking through a scene can churn through several track
ids; without merging, each id becomes its own incident and the feed fills with duplicates of the
same person.

## Why incidents are reference-counted

The first implementation merged N tracks into one row but closed that row on the FIRST track that
cleared — so an incident ended while other people were still violating, and the next transition
opened a fresh row. Measured on a 6-camera run: 531 unmatched closes against 306 real ones, and
310 incident rows in 100 seconds.

So `incident_tracks` maps each contributing track to its incident, and an incident closes only
when its LAST track clears. The mapping lives in the database rather than in memory because the
service is expected to be restartable without turning every in-flight incident into a leak.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    camera_id    INTEGER NOT NULL,
    type         TEXT NOT NULL,
    severity     TEXT NOT NULL,
    track_id     INTEGER,
    label        TEXT,
    bbox         TEXT,
    confidence   REAL,
    zone         TEXT,
    clip_uri     TEXT,
    vlm_verdict  TEXT,
    vlm_reason   TEXT,
    state        TEXT NOT NULL DEFAULT 'new',
    duration_s   REAL,
    ended_ts     REAL,
    hits         INTEGER NOT NULL DEFAULT 1,
    open_tracks  INTEGER NOT NULL DEFAULT 0,
    -- Phase 2.3: where in the SOURCE video this incident began, and the clip cut from it.
    source_pts_ns INTEGER,
    clip_state   TEXT NOT NULL DEFAULT 'pending',   -- pending | ready | failed | skipped | recording
    clip_error   TEXT,
    -- Which backend produced the evidence: NULL/'ffmpeg' (cut from media/camNN.mp4 at
    -- source_pts_ns) or 'smart_record' (dumped from the live RTSP ring buffer by the pipeline).
    -- Recorded per incident rather than read from config because it decides how the clip may be
    -- USED later: only the ffmpeg path has a local source that source_pts_ns actually indexes
    -- into, and the reasoning service must not seek into a file the timestamps do not belong to.
    clip_mode    TEXT,
    -- Crop of the flagged subject, captured by the pipeline at the instant of detection.
    crop_uri     TEXT,
    -- Phase 2.9: outbound notification (Telegram).
    notify_state TEXT NOT NULL DEFAULT 'pending',   -- pending | sent | skipped | failed
    notify_ts    REAL,
    notify_error TEXT,
    -- How many times this still-open incident has been re-raised, and when last.
    reminder_count INTEGER NOT NULL DEFAULT 0,
    last_reminder_ts REAL
);

-- Which tracks are currently contributing to which open incident. Rows are deleted as tracks
-- clear; an empty set is what closes the incident.
CREATE TABLE IF NOT EXISTS incident_tracks (
    camera_id  INTEGER NOT NULL,
    type       TEXT    NOT NULL,
    track_id   INTEGER NOT NULL,
    event_id   TEXT    NOT NULL,
    PRIMARY KEY (camera_id, type, track_id)
);

"""

# Indices are created SEPARATELY, after migrations. An index references columns, so running these
# against a database predating a column fails with "no such column" — and `CREATE TABLE IF NOT
# EXISTS` silently skips the table, so the new column would never arrive by that route either.
# Order is: create tables -> migrate columns -> create indices.
INDICES = """
-- The two access patterns the dashboard and agent actually have: "what happened on camera 7
-- recently" and "show me every fire alert". Both are (key, time) range scans.
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events (camera_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_ts   ON events (type, ts DESC);
-- Phase 2.4 polls "incidents still needing reasoning" constantly; it gets its own index.
CREATE INDEX IF NOT EXISTS idx_events_state     ON events (state, ts DESC);
-- Finding the open incident to merge into is the hottest query in this file.
CREATE INDEX IF NOT EXISTS idx_events_open      ON events (camera_id, type, ended_ts);
-- The clip service's work queue: "incidents still needing a clip", polled continuously.
CREATE INDEX IF NOT EXISTS idx_events_clip      ON events (clip_state, ts);
"""

# Every optional column `events` should have, with the DDL to add it. SQLite has no
# "ADD COLUMN IF NOT EXISTS" and `CREATE TABLE IF NOT EXISTS` silently skips an existing table, so
# a database created by an older version keeps its old shape forever unless something adds the
# columns explicitly. A demo box may already hold incidents someone is about to look at, so
# dropping and recreating is not an option.
#
# This is the FULL optional-column set rather than only the newest phase's additions: migrating
# just the latest columns leaves a sufficiently old database still missing earlier ones, and the
# failure surfaces as a confusing "no such column" while building an index.
COLUMNS = {
    "track_id": "INTEGER",
    "label": "TEXT",
    "bbox": "TEXT",
    "confidence": "REAL",
    "zone": "TEXT",
    "clip_uri": "TEXT",
    "vlm_verdict": "TEXT",
    "vlm_reason": "TEXT",
    "state": "TEXT NOT NULL DEFAULT 'new'",
    "duration_s": "REAL",
    "ended_ts": "REAL",
    "hits": "INTEGER NOT NULL DEFAULT 1",
    "open_tracks": "INTEGER NOT NULL DEFAULT 0",
    "source_pts_ns": "INTEGER",
    "clip_state": "TEXT NOT NULL DEFAULT 'pending'",
    "clip_error": "TEXT",
    "clip_mode": "TEXT",
    "crop_uri": "TEXT",
    # Outbound notification (Phase 2.9). Recorded in the incident row rather than in the notifier's
    # own state, so "was this alert sent, and when" survives a restart of the notifier and is
    # answerable with a SELECT — the same reason clip and VLM state live here.
    "notify_state": "TEXT NOT NULL DEFAULT 'pending'",   # pending|sent|skipped|failed
    "notify_ts": "REAL",
    "notify_error": "TEXT",
    "reminder_count": "INTEGER NOT NULL DEFAULT 0",
    "last_reminder_ts": "REAL",
}


def migrate(db: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an existing database. Returns what was added."""
    have = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    applied = []
    for col, ddl in COLUMNS.items():
        if col not in have:
            db.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")
            applied.append(col)
    if applied:
        db.commit()
    return applied


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=10.0)
    # Row factory set HERE, not by each caller. Every consumer wants column access, and a caller
    # that forgets turns `dict(row)` into "dictionary update sequence element #0 has length 32"
    # — a confusing error a long way from its cause. sqlite3.Row still supports integer indexing,
    # so positional access in existing code keeps working.
    db.row_factory = sqlite3.Row
    # WAL: the API (2.6) and reasoning service (2.4) read while this process writes. Under the
    # default rollback journal a writer blocks all readers, so the dashboard would stutter every
    # time an event landed.
    db.execute("PRAGMA journal_mode=WAL")
    # NORMAL, not FULL: one fsync per checkpoint rather than per commit. Events are already
    # durable in the Redis stream, so a power cut costs a replay of unacked entries, not data.
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    added = migrate(db)
    if added:
        print(f"[store] migrated: added columns {added}", flush=True)
    db.executescript(INDICES)
    db.commit()
    return db


def _initial_clip_state(ev: dict) -> str:
    """Which clip backend, if any, owns this incident's evidence video.

    * `recording` — the PIPELINE is cutting it, right now, out of nvurisrcbin's ring buffer
      (RTSP sources). `clip_service` must not touch these: there is no local source file it
      could cut from, so it would only mark them failed. The row moves to `ready` when the
      smart-record callback reports the filename via `attach_clip`.
    * `pending`   — `clip_service` will cut it from `media/camNN.mp4` (file sources).
    * `skipped`   — no source timestamp, so the moment cannot be located in the video at all.
      Recorded as skipped rather than left in the queue for a clip that can never be produced.
    """
    # Already carries its evidence. VLM escalations inherit the clip the hazard was seen in,
    # which is both the most accurate footage available and the only one obtainable — nothing
    # downstream can produce a clip for a synthetic event the pipeline never saw.
    if ev.get("clip_uri"):
        return "ready"
    if ev.get("clip_mode") == "smart_record":
        return "recording"
    return "pending" if ev.get("source_pts_ns") is not None else "skipped"


class EventStore:
    """Applies transition events to the incident tables. Idempotent on `event_id`."""

    def __init__(self, db: sqlite3.Connection, merge_window_s: float = 30.0):
        self.db = db
        self.merge_window_s = merge_window_s

    def raise_stale(self, after_s: float, now: float | None = None) -> list[sqlite3.Row]:
        """Re-raise incidents that have been open longer than `after_s`. Returns what was raised.

        A continuously-violating situation merges rather than reopening, which is correct — one
        person violating for an hour is one incident, not thousands of alerts. But the
        consequence is silence: an incident opens, absorbs tens of thousands of subsequent
        detections, and is never mentioned again. Measured on a 20-camera run: 19 PPE incidents
        open for 82 minutes having absorbed ~53,000 detections between them, with no alert after
        the first minute.

        Silence is the wrong answer to "still unresolved". This re-raises such an incident on a
        fixed period so it resurfaces in the feed and, if notifications are on, on someone's
        phone — without fragmenting it into separate incidents, which would destroy the duration
        and the reference counting that make it one situation.

        The clock is `last_reminder_ts` when set and `ts` otherwise, so the FIRST reminder lands
        `after_s` after the incident opened and subsequent ones `after_s` apart.
        """
        # 0 (or negative) means DISABLED, matching the config. Without this a zero threshold
        # means "open longer than zero seconds", which re-raises every open incident on every
        # sweep — the loudest possible reading of a value that was meant to switch it off.
        if after_s <= 0:
            return []
        now = time.time() if now is None else now
        cutoff = now - after_s
        rows = self.db.execute(
            "SELECT * FROM events "
            " WHERE ended_ts IS NULL AND COALESCE(last_reminder_ts, ts) <= ? "
            " ORDER BY ts", (cutoff,)).fetchall()
        if not rows:
            return []
        self.db.executemany(
            "UPDATE events SET reminder_count = reminder_count + 1, last_reminder_ts = ? "
            " WHERE event_id = ?", [(now, r["event_id"]) for r in rows])
        self.db.commit()
        # Re-read so callers see the incremented count rather than the pre-update snapshot.
        ids = tuple(r["event_id"] for r in rows)
        q = f"SELECT * FROM events WHERE event_id IN ({','.join('?' * len(ids))})"
        return self.db.execute(q, ids).fetchall()

    # -- queries used by later phases ------------------------------------------------------------
    def open_incidents(self, camera_id: int | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM events WHERE ended_ts IS NULL"
        args: tuple = ()
        if camera_id is not None:
            q += " AND camera_id = ?"
            args = (camera_id,)
        return self.db.execute(q + " ORDER BY ts DESC", args).fetchall()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def attach_crop(self, rec: dict) -> str:
        """Record a subject crop against the incident its transition belongs to.

        Resolved through `incident_tracks`, NOT by event_id. The pipeline names a crop for the
        TRANSITION that triggered it, and the store folds transitions into incidents — measured
        on a live run, only 1 of 138 crops carried an id that was itself an incident row. Every
        other one belonged to a real incident and would have been thrown away. The track mapping
        is the thing that actually knows which incident a transition joined.
        """
        row = self.db.execute(
            "SELECT event_id FROM incident_tracks "
            " WHERE camera_id = ? AND type = ? AND track_id = ?",
            (rec.get("camera_id"), rec.get("type"), rec.get("track_id"))).fetchone()
        target = row[0] if row else rec.get("event_id")
        if not target:
            return "crop-unmatched"
        # First crop wins: it is the one closest to when the incident opened, and a later
        # transition's subject may be a different person entirely.
        cur = self.db.execute(
            "UPDATE events SET crop_uri = ? WHERE event_id = ? AND crop_uri IS NULL",
            (rec.get("crop_uri"), target))
        self.db.commit()
        return "crop-attached" if cur.rowcount else "crop-unmatched"

    def attach_clip(self, event_id: str, clip_uri: str) -> str:
        """Record a finished smart-record clip against its incident.

        Returns a short tag for logging, mirroring `apply()`.

        A miss is NORMAL and is not an error. The pipeline emits one transition per track, but a
        recording covers a MOMENT ON A CAMERA — several tracks starting together share one file,
        and every one of them reports it. Meanwhile the store folds those transitions into a
        single incident, so only the first has a row of its own; the rest legitimately match
        nothing. The incident they merged into already carries the same file.

        The guard on `clip_state` matters: without it a late callback would overwrite a clip an
        operator may already have opened, and re-point an `expired` row at a file that retention
        has since deleted.
        """
        cur = self.db.execute(
            "UPDATE events SET clip_uri = ?, clip_state = 'ready', clip_error = NULL "
            " WHERE event_id = ? AND clip_state IN ('recording', 'pending')",
            (clip_uri, event_id))
        self.db.commit()
        return "clip-attached" if cur.rowcount else "clip-unmatched"

    # -- the state machine ------------------------------------------------------------------------
    def apply(self, ev: dict) -> str:
        """Apply one transition. Returns a short tag describing what happened, for logging."""
        return self._close(ev) if ev.get("ended") else self._open(ev)

    def _track_key(self, ev: dict) -> tuple:
        return (ev["camera_id"], ev["type"], ev.get("track_id", -1))

    def _open(self, ev: dict) -> str:
        key = self._track_key(ev)
        row = self.db.execute(
            "SELECT event_id FROM incident_tracks "
            " WHERE camera_id = ? AND type = ? AND track_id = ?", key).fetchone()

        if row is not None:
            # This track is already counted against an incident. Reaching here means severity
            # escalated (e.g. "no vest?" became "NO HELMET + no vest?"), which must NOT increment
            # open_tracks — doing so would leave the incident permanently un-closable, because
            # only one close will ever arrive for this track.
            self.db.execute(
                "UPDATE events SET severity = ? "
                " WHERE event_id = ? AND severity = 'medium' AND ? = 'high'",
                (ev["severity"], row[0], ev["severity"]))
            self.db.commit()
            return "escalated"

        found = self._find_mergeable(ev)
        if found is not None:
            merge_id, was_closed = found
            self.db.execute(
                "INSERT OR REPLACE INTO incident_tracks VALUES (?,?,?,?)", (*key, merge_id))
            self.db.execute(
                "UPDATE events SET hits = hits + 1, open_tracks = open_tracks + 1, "
                "  severity = CASE WHEN ? = 'high' THEN 'high' ELSE severity END "
                " WHERE event_id = ?", (ev["severity"], merge_id))
            if was_closed:
                # Reopen. `duration_s` is cleared rather than kept, because it will be recomputed
                # from the original start when the incident finally closes for good — leaving the
                # stale value would under-report the incident if it were read mid-reopen.
                self.db.execute(
                    "UPDATE events SET ended_ts = NULL, duration_s = NULL WHERE event_id = ?",
                    (merge_id,))
            self.db.commit()
            return "reopened" if was_closed else "merged"

        self.db.execute(
            "INSERT OR IGNORE INTO events "
            "(event_id, ts, camera_id, type, severity, track_id, label, bbox, confidence, "
            " zone, clip_uri, vlm_verdict, vlm_reason, state, duration_s, open_tracks, "
            " source_pts_ns, clip_state, clip_mode) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)",
            (ev["event_id"], ev["ts"], ev["camera_id"], ev["type"], ev["severity"],
             ev.get("track_id"), ev.get("label"),
             json.dumps(ev["bbox"]) if ev.get("bbox") else None,
             ev.get("confidence"), ev.get("zone"), ev.get("clip_uri"),
             ev.get("vlm_verdict"), ev.get("vlm_reason"), ev.get("state", "new"),
             ev.get("duration_s"), ev.get("source_pts_ns"),
             _initial_clip_state(ev), ev.get("clip_mode")))
        self.db.execute(
            "INSERT OR REPLACE INTO incident_tracks VALUES (?,?,?,?)", (*key, ev["event_id"]))
        self.db.commit()
        return "inserted"

    def _find_mergeable(self, ev: dict) -> tuple[str, bool] | None:
        """The incident this track should join, and whether it needs reopening.

        Two candidates, and getting the distinction right matters:

        * **Still open** — at least one person is currently violating on this camera. A new track
          joining is the same situation, ALWAYS, regardless of how long it has been running.
          Bounding this by the incident's start time was a bug: incidents routinely run longer
          than the merge window (37 s observed against a 30 s window), after which the incident
          stopped accepting tracks and a second open incident appeared alongside the first.

        * **Closed within `merge_window_s`** — the linger period. This is what absorbs track
          churn: a person whose track id changes produces a close immediately followed by an
          open, which would otherwise be two incidents for one person. Reopening the recent
          incident is the whole reason the window exists.
        """
        if self.merge_window_s <= 0:
            # Merging disabled: every track is its own incident.
            return None
        row = self.db.execute(
            "SELECT event_id, ended_ts FROM events "
            " WHERE camera_id = ? AND type = ? "
            "   AND (ended_ts IS NULL OR ended_ts >= ?) "
            " ORDER BY ts DESC LIMIT 1",
            (ev["camera_id"], ev["type"], ev["ts"] - self.merge_window_s)).fetchone()
        if row is None:
            return None
        return row[0], row[1] is not None

    def _close(self, ev: dict) -> str:
        key = self._track_key(ev)
        row = self.db.execute(
            "SELECT event_id FROM incident_tracks "
            " WHERE camera_id = ? AND type = ? AND track_id = ?", key).fetchone()
        if row is None:
            # A close with no known track: the service started mid-violation, or the opening
            # event was dropped under backpressure. Inventing an incident here would report
            # something that was never observed, so it is counted and discarded.
            return "close-unmatched"

        event_id = row[0]
        self.db.execute(
            "DELETE FROM incident_tracks WHERE camera_id = ? AND type = ? AND track_id = ?", key)
        self.db.execute(
            "UPDATE events SET open_tracks = MAX(0, open_tracks - 1) WHERE event_id = ?",
            (event_id,))

        remaining = self.db.execute(
            "SELECT open_tracks, ts FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if remaining is None:
            self.db.commit()
            return "close-unmatched"

        if remaining[0] > 0:
            # Other people are still violating on this camera. The incident is still happening.
            self.db.commit()
            return "close-partial"

        # Duration is recomputed from the stored start rather than trusting the publisher's
        # figure, which is only right if one process observed both ends.
        self.db.execute(
            "UPDATE events SET ended_ts = ?, duration_s = ? WHERE event_id = ?",
            (ev["ts"], max(0.0, ev["ts"] - remaining[1]), event_id))
        self.db.commit()
        return "closed"
