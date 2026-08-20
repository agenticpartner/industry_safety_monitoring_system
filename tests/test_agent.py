"""
Unit tests for the agent's deterministic parts: vocabulary, search, tools, confidence, and the
guardrails that stop an LLM inventing things.

No LLM, no network, no Jetson. The planner and the answer writer are the only components that
need a model; everything tested here is what stands between a wrong plan and a wrong answer.

The property that matters: **an LLM can be wrong, but it cannot make this layer report something
that is not in the database.**
"""

from __future__ import annotations

import sqlite3
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from agent import (  # noqa: E402
    ANSWER_SCHEMA, TOOLS, _cam, _iso, _row_public, _salvage_json, confidence, execute,
    plan_schema, video_analytics__get_incident, video_analytics__get_incidents,
    video_analytics__get_sensor_ids, vocabulary,
)
from search_service import search, summarise, sync  # noqa: E402
from store import connect  # noqa: E402

NOW = time.time()


def add(db, event_id, camera_id=1, etype="ppe_violation", severity="high", zone="AisleLeft",
        verdict=None, label="no vest?", ts=None, clip=None, hits=1, reason=None):
    db.execute(
        "INSERT INTO events (event_id, ts, camera_id, type, severity, zone, label, "
        " vlm_verdict, vlm_reason, clip_uri, clip_state, hits, state) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (event_id, ts if ts is not None else NOW, camera_id, etype, severity, zone, label,
         verdict, reason, clip, "ready" if clip else "pending", hits, "new"))
    db.commit()


class Base(unittest.TestCase):
    def setUp(self):
        self.db = connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)


class TestVocabulary(Base):
    def test_vocabulary_comes_from_the_data_not_from_code(self):
        """The whole point: add a zone to the DB and it becomes selectable, with no code edit."""
        add(self.db, "a", camera_id=3, zone="ForkliftAisle", etype="ppe_violation")
        add(self.db, "b", camera_id=7, zone="SpillZone", etype="overcrowding", severity="medium")
        v = vocabulary(self.db)
        self.assertEqual(v["sensors"], ["cam03", "cam07"])
        self.assertEqual(sorted(v["zones"]), ["ForkliftAisle", "SpillZone"])
        self.assertEqual(sorted(v["event_types"]), ["overcrowding", "ppe_violation"])
        self.assertIn("medium", v["severities"])

    def test_empty_database_yields_empty_enums_not_a_crash(self):
        v = vocabulary(self.db)
        self.assertEqual(v["sensors"], [])
        self.assertEqual(v["zones"], [])

    def test_plan_schema_enums_track_the_vocabulary(self):
        """A zone that does not exist cannot even be expressed in the plan."""
        add(self.db, "a", zone="SpillZone")
        props = plan_schema(vocabulary(self.db))["schema"]["properties"]
        self.assertIn("SpillZone", props["zone"]["enum"])
        self.assertIn("any", props["zone"]["enum"])
        self.assertNotIn("ForkliftAisle", props["zone"]["enum"])
        self.assertEqual(set(props["tool"]["enum"]), set(TOOLS))


class TestConfidence(Base):
    def test_confirmed_outranks_unverified_outranks_rejected(self):
        base = {"severity": "high", "hits": 5}
        c = confidence({**base, "vlm_verdict": "confirmed"})
        u = confidence({**base, "vlm_verdict": "unverified"})
        r = confidence({**base, "vlm_verdict": "rejected"})
        self.assertGreater(c, u)
        self.assertGreater(u, r)

    def test_severity_breaks_ties(self):
        a = confidence({"vlm_verdict": "confirmed", "severity": "critical", "hits": 1})
        b = confidence({"vlm_verdict": "confirmed", "severity": "medium", "hits": 1})
        self.assertGreater(a, b)

    def test_more_sightings_raise_confidence_but_saturate(self):
        low = confidence({"vlm_verdict": "confirmed", "severity": "high", "hits": 1})
        mid = confidence({"vlm_verdict": "confirmed", "severity": "high", "hits": 20})
        high = confidence({"vlm_verdict": "confirmed", "severity": "high", "hits": 500})
        self.assertGreater(mid, low)
        self.assertEqual(mid, high, "hits must saturate, not dominate")

    def test_bounded_zero_to_one(self):
        for v in ("confirmed", "rejected", "unverified", None):
            for s in ("critical", "high", "medium", None):
                c = confidence({"vlm_verdict": v, "severity": s, "hits": 999})
                self.assertGreaterEqual(c, 0.0)
                self.assertLessEqual(c, 1.0)


class TestSearchFilters(Base):
    def setUp(self):
        super().setUp()
        add(self.db, "aa", camera_id=1, zone="SpillZone", verdict="confirmed", severity="high")
        add(self.db, "bb", camera_id=2, zone="AisleLeft", verdict="rejected", severity="medium")
        add(self.db, "cc", camera_id=2, zone="SpillZone", verdict=None, severity="high",
            etype="overcrowding")
        sync(self.db)

    def test_filters_are_anded(self):
        rows = search(self.db, camera_id=2, zone="SpillZone")
        self.assertEqual([r["event_id"] for r in rows], ["cc"])

    def test_unverified_means_null_verdict_matching_vss(self):
        rows = search(self.db, vlm_verdict="unverified")
        self.assertEqual([r["event_id"] for r in rows], ["cc"])

    def test_any_is_treated_as_no_filter(self):
        self.assertEqual(len(search(self.db, zone="any", severity="any")), 3)

    def test_text_search_finds_by_zone_word(self):
        rows = search(self.db, text="SpillZone")
        self.assertEqual({r["event_id"] for r in rows}, {"aa", "cc"})

    def test_text_with_punctuation_does_not_raise(self):
        """FTS5 treats bare -, \" and * as operators; unquoted user text is a syntax error."""
        for q in ["hi-vis", 'a "quoted" thing', "wildcard*", "-leading", "a AND"]:
            with self.subTest(q=q):
                search(self.db, text=q)   # must not raise

    def test_unmatched_text_falls_back_to_structured_results(self):
        """A word nobody used must not silently drop the filters too."""
        rows = search(self.db, text="zzzznotpresent", camera_id=2)
        self.assertEqual({r["event_id"] for r in rows}, {"bb", "cc"})


class TestTools(Base):
    def setUp(self):
        super().setUp()
        add(self.db, "aa" * 16, camera_id=2, verdict="confirmed", clip="data/clips/x.mp4",
            hits=10, reason="not wearing a vest")
        add(self.db, "bb" * 16, camera_id=5, verdict="rejected", clip="data/clips/y.mp4")
        sync(self.db)

    def test_get_incidents_filters_by_sensor_string(self):
        out = video_analytics__get_incidents(self.db, source="cam05")
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["incidents"][0]["sensor"], "cam05")

    def test_get_incidents_vss_verdict_filter(self):
        self.assertEqual(
            video_analytics__get_incidents(self.db, vlm_verdict="rejected")["count"], 1)

    def test_get_incident_accepts_short_id_prefix(self):
        out = video_analytics__get_incident(self.db, id="aa" * 4)
        self.assertIn("incident", out)
        self.assertEqual(out["incident"]["sensor"], "cam02")

    def test_get_incident_missing_is_an_error_not_an_invention(self):
        self.assertIn("error", video_analytics__get_incident(self.db, id="deadbeef"))

    def test_sensor_ids_lists_only_cameras_with_incidents(self):
        ids = [s["id"] for s in video_analytics__get_sensor_ids(self.db)["sensors"]]
        self.assertEqual(ids, ["cam02", "cam05"])

    def test_get_clips_ranks_by_confidence(self):
        out = TOOLS["get_clips"](self.db, max_count=5)
        self.assertEqual(out["clips"][0]["id"], ("aa" * 16)[:8],
                         "confirmed+high+many hits should rank first")
        self.assertGreaterEqual(out["clips"][0]["confidence"], out["clips"][-1]["confidence"])

    def test_get_clips_only_returns_incidents_that_have_one(self):
        add(self.db, "cc" * 16, camera_id=9, clip=None)
        out = TOOLS["get_clips"](self.db, max_count=10)
        self.assertNotIn(("cc" * 16)[:8], [c["id"] for c in out["clips"]])

    def test_summary_counts(self):
        s = summarise(self.db)
        self.assertEqual(s["incidents"], 2)
        self.assertEqual(s["by_verdict"], {"confirmed": 1, "rejected": 1})
        self.assertEqual(s["with_clips"], 2)


class TestExecuteGuardrails(Base):
    """`execute()` is what stands between a bad plan and a bad query."""

    def setUp(self):
        super().setUp()
        add(self.db, "aa", camera_id=2, zone="SpillZone", verdict="confirmed")
        sync(self.db)

    def test_unknown_tool_falls_back_rather_than_raising(self):
        tool, _ = execute(self.db, {"tool": "definitely_not_a_tool"})
        self.assertEqual(tool, "search_incidents")

    def test_any_values_are_stripped_to_no_filter(self):
        tool, res = execute(self.db, {"tool": "search_incidents", "sensor": "any",
                                      "zone": "any", "severity": "any", "vlm_verdict": "any",
                                      "event_type": "any", "text": "any", "hours": 0})
        self.assertEqual(res["count"], 1)

    def test_max_count_is_clamped(self):
        """A model asking for 10000 rows must not get them."""
        for bad, _ in ((0, 1), (99999, 50), (-5, 1)):
            tool, res = execute(self.db, {"tool": "search_incidents", "max_count": bad})
            self.assertLessEqual(res["count"], 50)

    def test_summary_tool_ignores_row_filters(self):
        tool, res = execute(self.db, {"tool": "get_summary", "sensor": "cam02"})
        self.assertEqual(tool, "get_summary")
        self.assertIn("incidents", res)


class TestCoercion(unittest.TestCase):
    def test_cam_parses_sensor_ids_and_bare_numbers(self):
        self.assertEqual(_cam("cam07"), 7)
        self.assertEqual(_cam("7"), 7)
        self.assertEqual(_cam(7), 7)
        self.assertIsNone(_cam("any"))
        self.assertIsNone(_cam(None))
        self.assertIsNone(_cam("all cameras"))

    def test_iso_accepts_several_shapes(self):
        self.assertIsNone(_iso(None))
        self.assertIsNone(_iso("any"))
        self.assertEqual(_iso(1000.0), 1000.0)
        self.assertIsNotNone(_iso("2026-08-17"))
        self.assertIsNotNone(_iso("2026-08-17T10:00:00Z"))
        self.assertIsNone(_iso("last tuesday"), "unparseable time must not become a bogus filter")

    def test_row_public_hides_internals_unless_asked(self):
        r = {"event_id": "x" * 32, "ts": NOW, "camera_id": 3, "type": "ppe_violation",
             "severity": "high", "zone": "Z", "label": "no vest?", "vlm_verdict": None,
             "vlm_reason": "secret", "clip_uri": "data/clips/z.mp4", "state": "new", "hits": 4}
        plain = _row_public(r)
        self.assertNotIn("vlm_reason", plain)
        self.assertNotIn("clip_uri", plain)
        self.assertTrue(plain["has_clip"])
        self.assertEqual(plain["vlm_verdict"], "unverified", "NULL verdict reads as unverified")
        full = _row_public(r, ["info"])
        self.assertEqual(full["vlm_reason"], "secret")


class TestSchemaTermination(Base):
    """Guards the bug that cost 99 of the agent's 105 seconds.

    Under grammar-constrained decoding an unbounded string has no reason to stop. `text` was
    unbounded, so the planner wrote a correct plan and then padded that one field until it hit
    max_tokens — every single time. The call raised "truncated", `ask()` swallowed it, and the
    agent silently fell back to a raw full-text search. Nothing failed loudly; it was only
    visible as latency.

    These assertions are cheap and they fail the moment someone adds a free string.
    """

    def test_every_string_in_the_plan_schema_is_bounded(self):
        props = plan_schema(vocabulary(self.db))["schema"]["properties"]
        for name, spec in props.items():
            if spec.get("type") == "string" and "enum" not in spec:
                self.assertIn("maxLength", spec,
                              f"{name} is an unbounded string: under a grammar it can generate "
                              f"until max_tokens and never terminate")

    def test_every_string_in_the_answer_schema_is_bounded(self):
        props = ANSWER_SCHEMA["schema"]["properties"]
        self.assertIn("maxLength", props["answer"])
        self.assertIn("maxLength", props["cited_ids"]["items"])

    def test_schemas_are_closed(self):
        # Without this the grammar lets the model invent extra keys once the required ones are
        # written — the same runaway by a different route.
        self.assertFalse(plan_schema(vocabulary(self.db))["schema"]["additionalProperties"])
        self.assertFalse(ANSWER_SCHEMA["schema"]["additionalProperties"])

    def test_text_is_optional_so_it_need_not_be_padded(self):
        # Bounded but mandatory made the model fill it with repetition, which poisons FTS
        # relevance on a search.
        self.assertNotIn("text", plan_schema(vocabulary(self.db))["schema"]["required"])


class TestSalvage(unittest.TestCase):
    """A truncated completion has already been paid for; recover what is in it."""

    def test_recovers_fields_written_before_the_cut(self):
        out = _salvage_json('{"tool": "get_summary", "vlm_verdict": "confirmed", "text": "abc')
        self.assertEqual(out["tool"], "get_summary")
        self.assertEqual(out["vlm_verdict"], "confirmed")

    def test_recovers_when_cut_on_a_trailing_comma(self):
        out = _salvage_json('{"tool": "get_clips", "zone": "SpillZone",')
        self.assertEqual(out["zone"], "SpillZone")

    def test_returns_none_when_there_is_no_object(self):
        self.assertIsNone(_salvage_json("I'm sorry, I cannot help with that."))

    def test_handles_a_reasoning_wrapper(self):
        out = _salvage_json('<think>hmm</think>{"tool": "get_summary", "hours": 0')
        self.assertEqual(out["tool"], "get_summary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
