"""
Unit tests for verdict decision logic.

No VLM, no network, no Jetson. `decide()` is where ALL the policy lives — the model only reports
what it sees — so this is the file that decides whether a violation stands.

The regression these exist to prevent is the one that actually happened: the first version asked
the model for the verdict directly and got 100% rejections, including an overcrowding incident
where it counted 4 people against a limit of 2. Moving the judgement into code made it testable;
these are the tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from reasoning_service import (  # noqa: E402
    CANNOT_TELL, CONFIRMED, REJECTED, SCHEMAS, UNCERTAIN, CircuitBreaker, decide,
    escalation_for,
)


def ppe(person="yes", vest="no", hat="no", people=1, desc="a worker"):
    return {"subject_is_person": person, "wearing_high_vis_vest": vest,
            "wearing_hard_hat": hat, "people_visible": people, "description": desc}


class TestPPE(unittest.TestCase):
    def test_missing_vest_on_a_real_person_is_confirmed(self):
        v, r = decide("ppe_violation", "no vest?", ppe(vest="no"), None)
        self.assertEqual(v, CONFIRMED)
        self.assertIn("vest", r)

    def test_person_actually_wearing_the_vest_is_rejected(self):
        """The detector reports absence of evidence; the VLM can see the vest."""
        v, r = decide("ppe_violation", "no vest?", ppe(vest="yes"), None)
        self.assertEqual(v, REJECTED)
        self.assertIn("IS wearing", r)

    def test_not_a_person_is_rejected_whatever_the_ppe_answers(self):
        """The headline case: a traffic cone tracked as a worker.

        `subject_is_person=no` must short-circuit — a cone cannot be wearing or not wearing a
        vest, and letting the vest answer decide would confirm the violation.
        """
        v, r = decide("ppe_violation", "no vest?",
                      ppe(person="no", vest="no", desc="an orange traffic cone"), None)
        self.assertEqual(v, REJECTED)
        self.assertIn("not a person", r)

    def test_cannot_tell_if_person_is_uncertain_not_rejected(self):
        """Ambiguity must not be laundered into a confident answer either way."""
        v, _ = decide("ppe_violation", "no vest?", ppe(person=CANNOT_TELL), None)
        self.assertEqual(v, UNCERTAIN)

    def test_missing_helmet_is_confirmed(self):
        v, r = decide("ppe_violation", "NO HELMET", ppe(hat="no"), None)
        self.assertEqual(v, CONFIRMED)
        self.assertIn("hard hat", r)

    def test_helmet_present_is_rejected(self):
        v, _ = decide("ppe_violation", "NO HELMET", ppe(hat="yes"), None)
        self.assertEqual(v, REJECTED)

    def test_combined_label_confirms_if_either_item_is_missing(self):
        """'NO HELMET + no vest?' stands if the person is missing either one."""
        v, _ = decide("ppe_violation", "NO HELMET + no vest?", ppe(hat="yes", vest="no"), None)
        self.assertEqual(v, CONFIRMED)
        v, _ = decide("ppe_violation", "NO HELMET + no vest?", ppe(hat="no", vest="yes"), None)
        self.assertEqual(v, CONFIRMED)

    def test_combined_label_rejected_only_when_both_present(self):
        v, _ = decide("ppe_violation", "NO HELMET + no vest?", ppe(hat="yes", vest="yes"), None)
        self.assertEqual(v, REJECTED)

    def test_only_the_reported_item_is_judged(self):
        """A missing hard hat must not confirm an incident that was only about the vest."""
        v, _ = decide("ppe_violation", "no vest?", ppe(vest="yes", hat="no"), None)
        self.assertEqual(v, REJECTED)

    def test_unsure_about_the_reported_item_is_uncertain(self):
        v, _ = decide("ppe_violation", "no vest?", ppe(vest=CANNOT_TELL), None)
        self.assertEqual(v, UNCERTAIN)

    def test_unrecognised_label_is_uncertain_not_confirmed(self):
        v, _ = decide("ppe_violation", "something new", ppe(), None)
        self.assertEqual(v, UNCERTAIN)


class TestOvercrowding(unittest.TestCase):
    def test_over_the_limit_is_confirmed(self):
        v, r = decide("overcrowding", "OVERCROWDED Z", {"people_visible": 4, "description": ""}, 2)
        self.assertEqual(v, CONFIRMED)
        self.assertIn("limit of 2", r)

    def test_exactly_at_the_limit_is_rejected(self):
        """'more than the threshold' — equal is not over. This exact case appeared live."""
        v, r = decide("overcrowding", "OVERCROWDED Z", {"people_visible": 3, "description": ""}, 3)
        self.assertEqual(v, REJECTED)
        self.assertIn("within the limit", r)

    def test_under_the_limit_is_rejected(self):
        v, _ = decide("overcrowding", "OVERCROWDED Z", {"people_visible": 1, "description": ""}, 5)
        self.assertEqual(v, REJECTED)

    def test_missing_threshold_is_uncertain_not_a_guess(self):
        """Without the configured limit there is no policy to apply, so do not invent one."""
        v, r = decide("overcrowding", "OVERCROWDED Z",
                      {"people_visible": 9, "description": ""}, None)
        self.assertEqual(v, UNCERTAIN)
        self.assertIn("no occupancy limit", r)

    def test_non_integer_count_is_uncertain(self):
        v, _ = decide("overcrowding", "OVERCROWDED Z",
                      {"people_visible": None, "description": ""}, 3)
        self.assertEqual(v, UNCERTAIN)

    def test_zero_people_rejects(self):
        v, _ = decide("overcrowding", "OVERCROWDED Z", {"people_visible": 0, "description": ""}, 2)
        self.assertEqual(v, REJECTED)


class TestFire(unittest.TestCase):
    def test_visible_fire_is_confirmed(self):
        v, _ = decide("fire_alert", "FIRE",
                      {"fire_or_smoke_visible": "yes", "people_visible": 0,
                       "description": "flames"}, None)
        self.assertEqual(v, CONFIRMED)

    def test_no_fire_is_rejected(self):
        v, r = decide("fire_alert", "FIRE",
                      {"fire_or_smoke_visible": "no", "people_visible": 0,
                       "description": "clear"}, None)
        self.assertEqual(v, REJECTED)
        self.assertIn("no flames", r)

    def test_ambiguous_fire_is_uncertain(self):
        v, _ = decide("fire_alert", "SMOKE",
                      {"fire_or_smoke_visible": CANNOT_TELL, "people_visible": 0,
                       "description": "hazy"}, None)
        self.assertEqual(v, UNCERTAIN)


class TestCircuitBreaker(unittest.TestCase):
    def test_stays_closed_below_threshold(self):
        b = CircuitBreaker(threshold=3, cooldown_s=60)
        for _ in range(2):
            b.record(False)
        self.assertFalse(b.is_open)

    def test_opens_at_threshold(self):
        b = CircuitBreaker(threshold=3, cooldown_s=60)
        for _ in range(3):
            b.record(False)
        self.assertTrue(b.is_open)

    def test_success_resets_the_count(self):
        b = CircuitBreaker(threshold=3, cooldown_s=60)
        b.record(False); b.record(False)
        b.record(True)
        b.record(False)
        self.assertFalse(b.is_open)

    def test_half_opens_after_cooldown(self):
        """After the cooldown one request is let through to test the endpoint."""
        b = CircuitBreaker(threshold=2, cooldown_s=0.0)
        b.record(False); b.record(False)
        self.assertFalse(b.is_open, "cooldown of 0 should immediately half-open")

    def test_repeated_failure_after_half_open_reopens(self):
        b = CircuitBreaker(threshold=2, cooldown_s=60)
        b.record(False); b.record(False)
        self.assertTrue(b.is_open)
        b.record(False)
        self.assertTrue(b.is_open)


class TestEscalation(unittest.TestCase):
    """The VLM is the only component that can report a hazard nobody trained a class for."""

    FIRE = {"fire_or_smoke_visible": "yes", "hazard_visible": "yes",
            "hazard_description": "fire on the floor"}
    SPILL = {"fire_or_smoke_visible": "no", "hazard_visible": "yes",
             "hazard_description": "liquid spill across the aisle"}
    CLEAR = {"fire_or_smoke_visible": "no", "hazard_visible": "no"}

    def test_fire_seen_during_a_ppe_check_raises_a_fire_alert(self):
        out = escalation_for("ppe_violation", self.FIRE)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], "fire_alert")
        self.assertEqual(out[1], "critical")

    def test_other_hazard_raises_a_hazard_alert_naming_it(self):
        out = escalation_for("overcrowding", self.SPILL)
        self.assertEqual(out[0], "hazard_alert")
        self.assertIn("spill", out[2].lower())

    def test_ordinary_scene_escalates_nothing(self):
        self.assertIsNone(escalation_for("ppe_violation", self.CLEAR))

    def test_an_unnamed_hazard_is_not_actionable(self):
        # A bare "yes" with no description would flood the operator with notifications that
        # cannot be triaged.
        self.assertIsNone(escalation_for(
            "ppe_violation", {"fire_or_smoke_visible": "no", "hazard_visible": "yes",
                              "hazard_description": ""}))

    def test_escalations_never_escalate(self):
        """Guards an observed infinite loop.

        A fire_alert escalated to a hazard_alert (a fire IS a hazard), that hazard_alert was
        adjudicated, saw fire, and escalated to a new fire_alert -- one new incident per cycle,
        each with its own clip and VLM call, forever.
        """
        for etype in ("fire_alert", "hazard_alert"):
            for obs in (self.FIRE, self.SPILL):
                self.assertIsNone(escalation_for(etype, obs),
                                  f"{etype} must not escalate: alerts breeding alerts is a loop")

    def test_every_free_string_in_the_vlm_schemas_is_bounded(self):
        # Same defect class as the agent's plan schema: an unbounded string under grammar-
        # constrained decoding runs to max_tokens instead of terminating.
        for name, spec in SCHEMAS.items():
            for prop, ps in spec["schema"]["properties"].items():
                if ps.get("type") == "string" and "enum" not in ps:
                    self.assertIn("maxLength", ps, f"{name}.{prop} is unbounded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
