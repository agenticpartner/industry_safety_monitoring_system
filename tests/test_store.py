"""
Unit tests for the incident store.

In-memory SQLite, no Redis, no Jetson. The regression these exist to prevent is specific and was
observed in a real run: merging N tracks into one incident and then closing that incident on the
FIRST track to clear, which produced 531 unmatched closes against 306 real ones and turned a
handful of ongoing situations into 310 short-lived rows.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from store import EventStore, connect  # noqa: E402

CAM, TYPE = 3, "ppe_violation"


def ev(track_id: int, *, ts: float, ended: bool = False, severity: str = "high",
       event_id: str | None = None, camera_id: int = CAM, etype: str = TYPE) -> dict:
    return {
        "event_id": event_id or f"e{camera_id}-{etype}-{track_id}-{ts}-{int(ended)}",
        "ts": ts, "camera_id": camera_id, "type": etype, "severity": severity,
        "track_id": track_id, "label": "NO HELMET", "bbox": [1, 2, 3, 4],
        "confidence": 0.9, "state": "new", "ended": ended,
    }


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = EventStore(connect(":memory:"), merge_window_s=30.0)
        self.addCleanup(self.store.db.close)

    def rows(self):
        return self.store.db.execute(
            "SELECT event_id, hits, open_tracks, ended_ts, severity FROM events "
            "ORDER BY ts").fetchall()


class TestSingleTrack(StoreTest):
    def test_open_then_close_is_one_closed_incident(self):
        self.assertEqual(self.store.apply(ev(1, ts=100.0)), "inserted")
        self.assertEqual(self.store.apply(ev(1, ts=110.0, ended=True)), "closed")
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 0)             # open_tracks
        self.assertIsNotNone(rows[0][3])            # ended_ts
        self.assertAlmostEqual(
            self.store.db.execute("SELECT duration_s FROM events").fetchone()[0], 10.0)

    def test_duplicate_open_is_idempotent(self):
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(1, ts=100.0))           # redelivery
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0][2], 1, "redelivery must not inflate open_tracks")

    def test_close_without_open_is_unmatched_not_invented(self):
        self.assertEqual(self.store.apply(ev(9, ts=100.0, ended=True)), "close-unmatched")
        self.assertEqual(self.store.count(), 0)

    def test_double_close_is_unmatched_second_time(self):
        self.store.apply(ev(1, ts=100.0))
        self.assertEqual(self.store.apply(ev(1, ts=110.0, ended=True)), "closed")
        self.assertEqual(self.store.apply(ev(1, ts=111.0, ended=True)), "close-unmatched")
        self.assertEqual(len(self.rows()), 1)


class TestMergeRefCounting(StoreTest):
    """The regression suite for the close-on-first-track bug."""

    def test_two_tracks_merge_into_one_incident(self):
        self.assertEqual(self.store.apply(ev(1, ts=100.0)), "inserted")
        self.assertEqual(self.store.apply(ev(2, ts=101.0)), "merged")
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 2, "hits should count contributing tracks")
        self.assertEqual(rows[0][2], 2, "open_tracks should count contributing tracks")

    def test_incident_stays_open_until_last_track_clears(self):
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(2, ts=101.0))
        self.assertEqual(self.store.apply(ev(1, ts=110.0, ended=True)), "close-partial")
        self.assertIsNone(self.rows()[0][3], "incident closed while someone was still violating")
        self.assertEqual(self.store.apply(ev(2, ts=120.0, ended=True)), "closed")
        self.assertIsNotNone(self.rows()[0][3])

    def test_no_unmatched_closes_for_merged_tracks(self):
        """The exact failure mode: every open must have a close that finds its incident."""
        for t in range(1, 6):
            self.store.apply(ev(t, ts=100.0 + t))
        results = [self.store.apply(ev(t, ts=200.0 + t, ended=True)) for t in range(1, 6)]
        self.assertNotIn("close-unmatched", results)
        self.assertEqual(results.count("closed"), 1)
        self.assertEqual(results.count("close-partial"), 4)
        self.assertEqual(len(self.rows()), 1, "five tracks, one camera, one incident")

    def test_duration_spans_the_whole_incident(self):
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(2, ts=105.0))
        self.store.apply(ev(1, ts=110.0, ended=True))
        self.store.apply(ev(2, ts=160.0, ended=True))
        dur = self.store.db.execute("SELECT duration_s FROM events").fetchone()[0]
        self.assertAlmostEqual(dur, 60.0, msg="duration must span first open to last close")

    def test_new_track_shortly_after_close_reopens_not_duplicates(self):
        """Within the linger window a close/open pair is one continuing situation.

        This asserts the OPPOSITE of what it did before the linger period existed. The change is
        deliberate: at 15fps analytics ~26% of tracks are too short to adjudicate, so a single
        person routinely produces close-then-open, and treating that as two incidents is what
        floods the feed. The genuine-gap case is covered by
        TestLingerAndChurn.test_after_linger_expires_a_new_incident_starts.
        """
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(1, ts=110.0, ended=True))
        self.assertEqual(self.store.apply(ev(2, ts=120.0)), "reopened")
        self.assertEqual(len(self.rows()), 1)

    def test_long_running_incident_keeps_accepting_tracks(self):
        """Regression: an incident outliving the merge window must not spawn a second one.

        Observed live — incidents ran 37s against a 30s window, after which new tracks stopped
        merging and a second OPEN incident appeared on the same camera.
        """
        self.store.apply(ev(1, ts=100.0))
        self.assertEqual(self.store.apply(ev(2, ts=100.0 + 500)), "merged")
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.store.open_incidents()), 1)

    def test_never_two_open_incidents_for_one_camera_and_type(self):
        for t in range(1, 30):
            self.store.apply(ev(t, ts=100.0 + t * 20))     # far apart, all overlapping
        openn = self.store.db.execute(
            "SELECT camera_id, type, COUNT(*) c FROM events WHERE ended_ts IS NULL "
            "GROUP BY camera_id, type HAVING c > 1").fetchall()
        self.assertEqual(openn, [], "more than one open incident per (camera, type)")


class TestLingerAndChurn(StoreTest):
    """The merge window's real job: absorbing track-id churn, not bounding incident length."""

    def test_churn_reopens_rather_than_duplicating(self):
        """One person, new track id right after the old one cleared, is ONE incident."""
        self.store.apply(ev(1, ts=100.0))
        self.assertEqual(self.store.apply(ev(1, ts=101.0, ended=True)), "closed")
        self.assertEqual(self.store.apply(ev(2, ts=101.5)), "reopened")
        self.assertEqual(len(self.rows()), 1)
        self.assertIsNone(self.rows()[0][3], "reopened incident must not stay ended")

    def test_reopened_incident_duration_spans_the_whole_thing(self):
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(1, ts=110.0, ended=True))
        self.store.apply(ev(2, ts=112.0))              # reopen
        self.store.apply(ev(2, ts=150.0, ended=True))
        dur = self.store.db.execute("SELECT duration_s FROM events").fetchone()[0]
        self.assertAlmostEqual(dur, 50.0)

    def test_after_linger_expires_a_new_incident_starts(self):
        """A genuine gap is a genuine new incident."""
        self.store.apply(ev(1, ts=100.0))
        self.store.apply(ev(1, ts=110.0, ended=True))
        self.assertEqual(self.store.apply(ev(2, ts=110.0 + 31)), "inserted")
        self.assertEqual(len(self.rows()), 2)

    def test_merge_disabled_gives_one_incident_per_track(self):
        s = EventStore(connect(":memory:"), merge_window_s=0)
        self.addCleanup(s.db.close)
        s.apply(ev(1, ts=100.0))
        s.apply(ev(2, ts=101.0))
        self.assertEqual(s.count(), 2)


class TestIsolation(StoreTest):
    def test_cameras_do_not_merge_into_each_other(self):
        self.store.apply(ev(1, ts=100.0, camera_id=1))
        self.store.apply(ev(1, ts=101.0, camera_id=2))
        self.assertEqual(len(self.rows()), 2)

    def test_types_do_not_merge_into_each_other(self):
        self.store.apply(ev(1, ts=100.0, etype="ppe_violation"))
        self.store.apply(ev(1, ts=101.0, etype="fire_alert"))
        self.assertEqual(len(self.rows()), 2)

    def test_close_matches_only_its_own_camera(self):
        self.store.apply(ev(1, ts=100.0, camera_id=1))
        self.assertEqual(
            self.store.apply(ev(1, ts=110.0, ended=True, camera_id=2)), "close-unmatched")


class TestSeverity(StoreTest):
    def test_escalation_does_not_inflate_open_tracks(self):
        """The subtle one: a re-emit for an existing track must not add a reference.

        If it did, the incident would need two closes for one track and could never close.
        """
        self.store.apply(ev(1, ts=100.0, severity="medium"))
        self.assertEqual(self.store.apply(ev(1, ts=101.0, severity="high")), "escalated")
        self.assertEqual(self.rows()[0][2], 1)
        self.assertEqual(self.rows()[0][4], "high")
        self.assertEqual(self.store.apply(ev(1, ts=110.0, ended=True)), "closed")

    def test_escalation_never_downgrades(self):
        self.store.apply(ev(1, ts=100.0, severity="high"))
        self.store.apply(ev(1, ts=101.0, severity="medium"))
        self.assertEqual(self.rows()[0][4], "high")

    def test_merged_high_raises_incident_severity(self):
        self.store.apply(ev(1, ts=100.0, severity="medium"))
        self.store.apply(ev(2, ts=101.0, severity="high"))
        self.assertEqual(self.rows()[0][4], "high")


class TestQueries(StoreTest):
    def test_open_incidents_excludes_closed(self):
        self.store.apply(ev(1, ts=100.0, camera_id=1))
        self.store.apply(ev(1, ts=100.0, camera_id=2))
        self.store.apply(ev(1, ts=110.0, ended=True, camera_id=1))
        self.assertEqual(len(self.store.open_incidents()), 1)
        self.assertEqual(len(self.store.open_incidents(camera_id=2)), 1)
        self.assertEqual(len(self.store.open_incidents(camera_id=1)), 0)

    def test_schema_has_fields_later_phases_fill(self):
        cols = {r[1] for r in self.store.db.execute("PRAGMA table_info(events)")}
        for c in ("zone", "clip_uri", "vlm_verdict", "vlm_reason", "state"):
            self.assertIn(c, cols)


class TestReRaise(StoreTest):
    """Re-raising incidents that stay open.

    A continuously-violating situation merges rather than reopening, which is right — but the
    consequence is silence: measured on a 20-camera run, 19 PPE incidents sat open for 82 minutes
    having absorbed ~53,000 detections between them, with nothing raised after the first minute.
    Silence is the wrong answer to "still unresolved".
    """

    THRESH = 480.0

    def test_fresh_incident_is_not_re_raised(self):
        now = time.time()
        self.store.apply(ev(1, ts=now))
        self.assertEqual(self.store.raise_stale(self.THRESH, now=now), [])

    def test_incident_open_past_the_threshold_is_re_raised(self):
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        out = self.store.raise_stale(self.THRESH, now=now)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["reminder_count"], 1,
                         "the caller must see the incremented count, not the pre-update row")

    def test_it_does_not_re_raise_again_immediately(self):
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        self.assertEqual(len(self.store.raise_stale(self.THRESH, now=now)), 1)
        self.assertEqual(self.store.raise_stale(self.THRESH, now=now + 10), [],
                         "the clock restarts from the reminder, not from the incident start")

    def test_it_re_raises_again_after_another_interval(self):
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        self.store.raise_stale(self.THRESH, now=now)
        out = self.store.raise_stale(self.THRESH, now=now + self.THRESH + 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["reminder_count"], 2)

    def test_closed_incidents_are_never_re_raised(self):
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        self.store.apply(ev(1, ts=now - 590, ended=True))
        self.assertEqual(self.store.raise_stale(self.THRESH, now=now), [],
                         "a resolved situation must not keep nagging")

    def test_re_raising_does_not_fragment_the_incident(self):
        # Duration and reference counting are what make it ONE situation; a reminder must not
        # insert a row, reopen anything, or touch the track mapping.
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        before = self.rows()
        self.store.raise_stale(self.THRESH, now=now)
        after = self.rows()
        self.assertEqual(len(before), len(after))
        self.assertEqual(before[0][1], after[0][1], "hits unchanged")
        self.assertEqual(before[0][2], after[0][2], "open_tracks unchanged")
        self.assertIsNone(after[0][3], "still open")

    def test_disabled_when_threshold_is_zero(self):
        # 0 means "off" in config; a 0-second threshold would otherwise re-raise everything on
        # every sweep.
        now = time.time()
        self.store.apply(ev(1, ts=now - 600))
        self.assertEqual(self.store.raise_stale(0, now=now), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
