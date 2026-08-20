"""
Unit tests for clip window arithmetic and retention.

No ffmpeg, no media, no Jetson — the subprocess call is deliberately separated from the maths so
the maths can be checked here. The failure these guard against is a clip that plays perfectly and
shows the wrong moment, which no amount of "it produced a file" testing would catch.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from clip_service import gc, window_for  # noqa: E402
from store import connect  # noqa: E402

NS = 1_000_000_000


class TestWindow(unittest.TestCase):
    def test_centres_the_window_on_the_incident(self):
        start, length = window_for(20 * NS, duration_s=40, pre_s=6, post_s=6)
        self.assertAlmostEqual(start, 14.0)
        self.assertAlmostEqual(length, 12.0)

    def test_clamps_at_the_start_of_the_file(self):
        """An incident 2s in cannot have 6s of pre-roll; it must not seek negative."""
        start, length = window_for(2 * NS, duration_s=40, pre_s=6, post_s=6)
        self.assertEqual(start, 0.0)
        self.assertAlmostEqual(length, 12.0)

    def test_truncates_rather_than_running_past_the_end(self):
        """Better a short clip than one that wraps into unrelated footage."""
        start, length = window_for(39 * NS, duration_s=40, pre_s=6, post_s=6)
        self.assertAlmostEqual(start, 33.0)
        self.assertAlmostEqual(length, 7.0)          # 40 - 33, not 12

    def test_looped_source_maps_back_into_the_file(self):
        """PTS keeps climbing across loops; the offset must fold back.

        Exact rather than approximate because every loop is identical content.
        """
        for loop in range(5):
            start, length = window_for(int((loop * 40 + 20) * NS), 40, 6, 6)
            self.assertAlmostEqual(start, 14.0, msg=f"loop {loop}")
            self.assertAlmostEqual(length, 12.0)

    def test_fractional_pts_survives_the_modulo(self):
        """85.7s of a 40s source folds to 5.7s in; 6s of pre-roll then clamps to the file start."""
        start, length = window_for(int(85.7 * NS), duration_s=40, pre_s=6, post_s=6)
        self.assertAlmostEqual(start, 0.0, places=3)
        self.assertAlmostEqual(length, 12.0, places=3)

    def test_fractional_pts_mid_file_is_not_clamped(self):
        start, _ = window_for(int(95.7 * NS), duration_s=40, pre_s=6, post_s=6)
        self.assertAlmostEqual(start, 95.7 % 40 - 6, places=3)

    def test_zero_duration_source_is_rejected(self):
        self.assertIsNone(window_for(NS, duration_s=0, pre_s=6, post_s=6))

    def test_degenerate_window_is_rejected(self):
        """A source shorter than the requested window yields nothing usable."""
        self.assertIsNone(window_for(0, duration_s=0.4, pre_s=6, post_s=6))

    def test_incident_at_the_very_end_is_rejected_not_truncated_to_nothing(self):
        self.assertIsNone(window_for(int(39.9 * NS), duration_s=40.0, pre_s=0.0, post_s=6))

    def test_asymmetric_roll(self):
        start, length = window_for(20 * NS, duration_s=40, pre_s=2, post_s=10)
        self.assertAlmostEqual(start, 18.0)
        self.assertAlmostEqual(length, 12.0)


class TestRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clips = Path(self.tmp.name) / "clips"
        self.clips.mkdir()
        self.db = connect(":memory:")
        self.addCleanup(self.db.close)

    def make_clip(self, name: str, size: int, age_days: float = 0) -> Path:
        p = self.clips / name
        p.write_bytes(b"\0" * size)
        t = time.time() - age_days * 86400
        import os
        os.utime(p, (t, t))
        self.db.execute(
            "INSERT INTO events (event_id, ts, camera_id, type, severity, clip_uri, clip_state) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, time.time(), 1, "ppe_violation", "high", f"data/clips/{name}", "ready"))
        self.db.commit()
        return p

    def test_under_budget_keeps_everything(self):
        for i in range(3):
            self.make_clip(f"c{i}.mp4", 1_000_000)
        self.assertEqual(gc(self.clips, self.db, budget_mb=100, retention_days=30), 0)
        self.assertEqual(len(list(self.clips.glob("*.mp4"))), 3)

    def test_over_budget_deletes_oldest_first(self):
        import os
        for i in range(5):
            p = self.make_clip(f"c{i}.mp4", 1_000_000)
            t = time.time() - (5 - i) * 100      # c0 oldest
            os.utime(p, (t, t))
        gc(self.clips, self.db, budget_mb=3, retention_days=30)
        left = sorted(p.name for p in self.clips.glob("*.mp4"))
        self.assertNotIn("c0.mp4", left, "oldest clip should go first")
        self.assertIn("c4.mp4", left, "newest clip must survive")

    def test_expired_clip_keeps_its_incident_row(self):
        """The incident still happened; only the evidence aged out."""
        self.make_clip("old.mp4", 1_000_000, age_days=99)
        gc(self.clips, self.db, budget_mb=10_000, retention_days=30)
        row = self.db.execute(
            "SELECT clip_uri, clip_state FROM events WHERE event_id='old.mp4'").fetchone()
        self.assertIsNotNone(row, "row must survive clip deletion")
        self.assertIsNone(row[0])
        self.assertEqual(row[1], "expired")

    def test_retention_deletes_old_even_when_under_budget(self):
        self.make_clip("old.mp4", 1000, age_days=99)
        self.make_clip("new.mp4", 1000, age_days=0)
        gc(self.clips, self.db, budget_mb=10_000, retention_days=30)
        left = [p.name for p in self.clips.glob("*.mp4")]
        self.assertEqual(left, ["new.mp4"])

    def test_empty_dir_is_a_noop(self):
        self.assertEqual(gc(self.clips, self.db, budget_mb=1, retention_days=1), 0)


class TestSchema(unittest.TestCase):
    def test_migration_adds_columns_to_an_existing_database(self):
        """A demo box may already hold a populated events.db — it must not need wiping."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "old.db"
        old = sqlite3.connect(str(path))
        old.execute("CREATE TABLE events (event_id TEXT PRIMARY KEY, ts REAL, camera_id INTEGER,"
                    " type TEXT, severity TEXT, clip_uri TEXT)")
        old.execute("INSERT INTO events VALUES ('a', 1.0, 1, 'ppe_violation', 'high', NULL)")
        old.commit()
        old.close()

        db = connect(path)
        self.addCleanup(db.close)
        cols = {r[1] for r in db.execute("PRAGMA table_info(events)")}
        for c in ("source_pts_ns", "clip_state", "clip_error"):
            self.assertIn(c, cols)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1,
                         "existing rows must survive the migration")


if __name__ == "__main__":
    unittest.main(verbosity=2)
