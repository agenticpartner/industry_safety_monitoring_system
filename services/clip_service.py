#!/usr/bin/env python3
"""
Clip service — cut an evidence clip for each incident and index it against the record.

    python3 services/clip_service.py [--once] [--interval 2] [--gc]

Stdlib + ffmpeg only, so it runs under system Python (no venv needed) — but it is a COLD PATH
worker: it polls SQLite, never touches the pipeline, and can be killed or fall behind without
anything upstream noticing. Same shape the reasoning service takes in 2.4.

## Why polling the store rather than consuming the bus

The bus carries raw per-track TRANSITIONS; the store folds those into incidents. A clip belongs to
an incident, and only the store knows which transitions merged into which incident. A bus consumer
would have to duplicate that logic and would still race with the writer. Polling
`clip_state='pending'` is simpler, restartable, and naturally rate-limited — and it makes the work
queue inspectable with a SELECT.

## Two capture backends, because smart-record cannot do file sources

Measured on this build, not assumed (see `bench/clip_capture.md`):

* **RTSP sources** — `nvurisrcbin` smart-record works. It caches the *encoded* stream ahead of the
  decoder, so a clip is full source rate no matter how slowly analytics runs, and costs no extra
  decode. Verified: a clip was written and `sr-done` returned a populated `RecordingInfo`.
* **file:// sources** — smart-record writes **nothing**. The property help says so three times
  ("Sources must be of type source-type-rtsp") and a direct test confirmed it: `start_recording`
  returns a session id, no file ever appears.

The demo and every benchmark run in file mode, so this service implements the file path by cutting
the window straight out of the source with `ffmpeg -c copy`. That is stream-copy, not re-encode:
no GPU, no quality loss, and the clip is inherently at the source's full frame rate — which is the
property Phase 2 claims for evidence clips.

## Finding the moment in the file

Incidents carry `source_pts_ns` (`frame_meta.buffer_pts`), the frame's position in SOURCE time.
Wall-clock would not do: in file mode the pipeline runs faster or slower than realtime depending
on load and `drop-frame-interval`, so elapsed seconds say nothing about where in the video the
incident is.

Sources loop (`file-loop=1`), so the offset is `pts modulo duration`. That is exact rather than
approximate **because every loop replays identical content** — landing in the wrong loop still
lands on the right frame.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import connect  # noqa: E402


def load_cfg() -> dict:
    """Read services.yml without requiring PyYAML — this runs on system Python."""
    try:
        import yaml
        return yaml.safe_load((ROOT / "configs/services.yml").read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def window_for(pts_ns: int, duration_s: float, pre_s: float,
               post_s: float) -> tuple[float, float] | None:
    """Map a source PTS to an ffmpeg (start, length) window inside the source file.

    Pure arithmetic, separated from the subprocess call so it can be tested without media.

    Sources loop, so the position is `pts modulo duration`. That is exact rather than approximate
    because every loop replays identical content — landing in the wrong loop still lands on the
    right frame.

    Returns None when the window would be degenerate (source shorter than the pre-roll, or the
    incident within a fraction of a second of the end).
    """
    if duration_s <= 0:
        return None
    offset = (pts_ns / 1e9) % duration_s
    start = max(0.0, offset - pre_s)
    # Never run past the end: a clip that stops early is fine, one that silently wraps into
    # unrelated footage is a lie about what was recorded.
    length = min(pre_s + post_s, duration_s - start)
    if length <= 0.5:
        return None
    return start, length


class ClipCutter:
    def __init__(self, media_dir: Path, clips_dir: Path, pre_s: float, post_s: float):
        self.media_dir = media_dir
        self.clips_dir = clips_dir
        self.pre_s = pre_s
        self.post_s = post_s
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._durations: dict[Path, float] = {}

    def source_for(self, camera_id: int) -> Path:
        # camera_id is 1-based and matches the media filenames by construction.
        return self.media_dir / f"cam{camera_id:02d}.mp4"

    def duration_of(self, src: Path) -> float | None:
        if src not in self._durations:
            d = probe_duration(src)
            if d is None:
                return None
            self._durations[src] = d
        return self._durations[src]

    def cut(self, event_id: str, camera_id: int, pts_ns: int) -> tuple[Path | None, str | None]:
        src = self.source_for(camera_id)
        if not src.exists():
            return None, f"source {src.name} not found"
        dur = self.duration_of(src)
        if not dur or dur <= 0:
            return None, f"could not probe duration of {src.name}"

        win = window_for(pts_ns, dur, self.pre_s, self.post_s)
        if win is None:
            return None, (f"degenerate window (pts {pts_ns/1e9:.1f}s in a {dur:.1f}s source)")
        start, length = win

        out = self.clips_dir / f"{event_id}.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            # -ss BEFORE -i is the fast input seek; with -c copy it snaps to the nearest
            # keyframe. The media is encoded with a 1s closed GOP (see scripts/make_streams.sh),
            # so the snap is at most 1s — well inside the pre-roll.
            "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{length:.3f}",
            "-c", "copy",             # stream copy: no re-encode, no GPU, no quality loss
            "-movflags", "+faststart",
            str(out),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None, "ffmpeg timed out"
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            return None, f"ffmpeg rc={r.returncode}: {r.stderr.strip()[:200]}"
        return out, None


def gc(clips_dir: Path, db: sqlite3.Connection, budget_mb: int, retention_days: int) -> int:
    """Enforce the disk budget. Oldest clips go first; the DB row keeps the incident.

    A demo that fills the NVMe is a demo that stops working, so this runs on every pass rather
    than being a separate cron. Deleting the FILE but keeping the ROW is deliberate: the incident
    still happened, and the record should say the evidence has aged out rather than vanish.
    """
    clips = sorted(clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not clips:
        return 0
    total = sum(p.stat().st_size for p in clips)
    cutoff = time.time() - retention_days * 86400
    removed = 0

    for p in clips:
        too_old = p.stat().st_mtime < cutoff
        over_budget = total > budget_mb * 1e6
        if not (too_old or over_budget):
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)
        db.execute("UPDATE events SET clip_uri = NULL, clip_state = 'expired' "
                   " WHERE clip_uri LIKE ?", (f"%{p.name}",))
        removed += 1
    if removed:
        db.commit()
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--clips-dir", default=None)
    ap.add_argument("--media-dir", default=str(ROOT / "media"))
    ap.add_argument("--pre", type=float, default=None, help="seconds of pre-roll")
    ap.add_argument("--post", type=float, default=None, help="seconds after the incident")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4,
                    help="clips per pass — bounded so a backlog cannot monopolise the CPU")
    ap.add_argument("--budget-mb", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--gc", action="store_true", help="run retention only, then exit")
    args = ap.parse_args()

    cfg = load_cfg()
    store_cfg = cfg.get("store") or {}
    clip_cfg = cfg.get("clips") or {}

    db_path = Path(args.db or store_cfg.get("path", "data/events.db"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    clips_dir = Path(args.clips_dir or clip_cfg.get("dir", "data/clips"))
    if not clips_dir.is_absolute():
        clips_dir = ROOT / clips_dir

    pre = args.pre if args.pre is not None else float(clip_cfg.get("pre_roll_s", 6))
    post = args.post if args.post is not None else float(clip_cfg.get("post_roll_s", 6))
    budget = args.budget_mb or int(clip_cfg.get("budget_mb", 4000))
    retention = int(store_cfg.get("retention_days", 30))

    if not shutil.which("ffmpeg"):
        print("[clip-service] ERROR: ffmpeg not on PATH", flush=True)
        return 2

    db = connect(db_path)
    cutter = ClipCutter(Path(args.media_dir), clips_dir, pre, post)
    print(f"[clip-service] db={db_path} clips={clips_dir} pre={pre}s post={post}s "
          f"budget={budget}MB", flush=True)

    if args.gc:
        n = gc(clips_dir, db, budget, retention)
        print(f"[clip-service] gc removed {n} clip(s)", flush=True)
        return 0

    made = failed = 0
    while True:
        rows = db.execute(
            "SELECT event_id, camera_id, source_pts_ns FROM events "
            " WHERE clip_state = 'pending' AND source_pts_ns IS NOT NULL "
            " ORDER BY ts LIMIT ?", (args.batch,)).fetchall()

        for event_id, camera_id, pts_ns in rows:
            path, err = cutter.cut(event_id, camera_id, pts_ns)
            if path is None:
                db.execute("UPDATE events SET clip_state='failed', clip_error=? "
                           " WHERE event_id=?", (err, event_id))
                failed += 1
                print(f"[clip-service] FAILED {event_id[:8]} cam{camera_id:02d}: {err}",
                      flush=True)
            else:
                # Stored relative to the project root so the database stays portable between the
                # device and a laptop copy.
                rel = os.path.relpath(path, ROOT)
                db.execute("UPDATE events SET clip_uri=?, clip_state='ready', clip_error=NULL "
                           " WHERE event_id=?", (rel, event_id))
                made += 1
                print(f"[clip-service] cut {rel} ({path.stat().st_size/1e6:.1f} MB) "
                      f"cam{camera_id:02d}", flush=True)
        if rows:
            db.commit()

        removed = gc(clips_dir, db, budget, retention)
        if removed:
            print(f"[clip-service] gc removed {removed} clip(s) over budget/retention", flush=True)

        if args.once:
            break
        if not rows:
            time.sleep(args.interval)

    pending = db.execute(
        "SELECT COUNT(*) FROM events WHERE clip_state='pending'").fetchone()[0]
    print(f"[clip-service] made={made} failed={failed} pending={pending}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
