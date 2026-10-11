"""
Tests: the harness that measures whether the trust detector
detects.

The metrics are the deliverable, so they are tested against cases with a known
answer rather than against themselves. All offline.

Run from repo root:
    pytest backend/tests/test_trust_eval.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.eval import trust_eval as TE


# ── AUROC ─────────────────────────────────────────────────────────────────────

class TestAuroc:

    def test_a_perfect_ranking_scores_one(self):
        assert TE.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0

    def test_an_inverted_ranking_scores_zero(self):
        assert TE.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0

    def test_all_scores_tied_is_exactly_a_coin_flip(self):
        """The case the naive formulation gets wrong: counting every tie as a
        win would report 1.0 for a signal carrying no information at all."""
        assert TE.auroc([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0]) == 0.5

    def test_partial_ties_use_average_ranks(self):
        # One correct and one incorrect share the top score; the remaining
        # incorrect ranks last. Half credit on the tie → 0.75.
        assert TE.auroc([1.0, 1.0, 0.0], [1, 0, 0]) == 0.75

    def test_a_single_class_has_no_auroc(self):
        assert TE.auroc([1.0, 0.5], [1, 1]) is None
        assert TE.auroc([1.0, 0.5], [0, 0]) is None
        assert TE.auroc([], []) is None

    def test_it_matches_the_published_definition_on_a_worked_case(self):
        # 2 correct (scores 1.0, 0.5), 2 incorrect (0.5, 0.0). Pairs won:
        # (1.0>0.5), (1.0>0.0), (0.5 tie 0.5)=½, (0.5>0.0) → 3.5/4.
        assert TE.auroc([1.0, 0.5, 0.5, 0.0], [1, 1, 0, 0]) == 0.875


# ── per-check rates ───────────────────────────────────────────────────────────

class TestFlagRates:

    def test_a_check_that_only_ever_flags_wrong_answers(self):
        r = TE.flag_rates([True, True, False], [0, 0, 1])
        assert r == {"fired": 2, "precision_pct": 100.0,
                     "recall_pct": 100.0, "false_alarm_pct": 0.0}

    def test_false_alarm_is_measured_against_correct_answers_only(self):
        """It is the number that decides whether a user keeps reading the
        warnings, so it must not be diluted by the wrong answers."""
        r = TE.flag_rates([True, False, False, False], [1, 1, 1, 0])
        assert r["false_alarm_pct"] == pytest.approx(33.3, abs=0.1)
        assert r["precision_pct"] == 0.0

    def test_a_check_that_never_fires_reports_no_precision(self):
        r = TE.flag_rates([False, False], [1, 0])
        assert r["fired"] == 0 and r["precision_pct"] is None
        assert r["recall_pct"] == 0.0


# ── risk–coverage ─────────────────────────────────────────────────────────────

class TestRiskCoverage:

    def test_full_coverage_equals_plain_accuracy(self):
        curve = TE.risk_coverage([1.0, 0.5, 0.0], [1, 1, 0])
        assert curve[-1]["coverage_pct"] == 100.0
        assert curve[-1]["selective_accuracy_pct"] == pytest.approx(66.7, abs=0.1)

    def test_a_useful_detector_buys_accuracy_by_declining(self):
        curve = TE.risk_coverage([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
        assert curve[0]["coverage_pct"] == 50.0
        assert curve[0]["selective_accuracy_pct"] == 100.0
        assert curve[-1]["selective_accuracy_pct"] == 50.0

    def test_one_point_per_threshold_not_per_record(self):
        curve = TE.risk_coverage([1.0, 1.0, 1.0], [1, 1, 0])
        assert len(curve) == 1


# ── replay ────────────────────────────────────────────────────────────────────

class TestVerificationFromRecord:

    REC = {"exec_vote": {"candidates": 5, "executed": 5, "groups": 1,
                         "winner_votes": 5},
           "valid": True, "pred_sql": "SELECT a FROM t", "pred_row_count": 3,
           "question": "list a", "ex": 1.0}

    def test_it_produces_the_shape_grade_consumes(self):
        from backend.agents import trust as T
        graded = T.grade(TE.verification_from_record(self.REC))
        assert graded["level"] in T.VERDICTS

    def test_groups_is_a_count_in_stored_records_not_a_list(self):
        """It is a list in the live block and an int on disk; reading it as a
        list crashed the first run of this harness."""
        ev = TE.verification_from_record(self.REC)
        assert ev["distinct_result_sets"] == 1

    def test_a_record_with_no_sql_runs_no_checks(self):
        ev = TE.verification_from_record({**self.REC, "pred_sql": None})
        assert ev["risks"] == []

    def test_the_unmeasurable_gate_is_left_empty_not_faked(self):
        assert TE.verification_from_record(self.REC)["back_translation"] is None

    def test_zero_rows_is_detected_from_the_stored_row_count(self):
        ev = TE.verification_from_record({**self.REC, "pred_row_count": 0})
        assert any(r["code"] == "zero_rows" for r in ev["risks"])


# ── the harness cannot silently miss a check ──────────────────────────────────

def test_every_risk_code_is_classified():
    """
    risk.py is the source of truth. A check added there without being listed
    as replayable or unmeasurable would simply never appear in the report —
    the failure this test exists to prevent.
    """
    src = (Path(__file__).resolve().parent.parent / "agents" / "risk.py").read_text()
    codes = set(re.findall(r'_risk\(\s*"([a-z_]+)"', src))
    classified = set(TE.REPLAYABLE_CODES) | set(TE.UNMEASURED_CODES)
    assert codes - classified == set(), f"unclassified risk codes: {codes - classified}"
    assert classified - codes == set(), f"listed but never emitted: {classified - codes}"


def test_evaluate_reports_codes_it_does_not_know_about():
    res = TE.evaluate([{"exec_vote": {"executed": 1, "winner_votes": 1},
                        "valid": True, "pred_sql": "SELECT 1",
                        "question": "q", "pred_row_count": 1, "ex": 1.0}])
    assert res["unclassified_codes"] == []
    assert res["total"] == 1


def test_evaluate_names_what_it_could_not_measure():
    res = TE.evaluate([{"exec_vote": {"executed": 2, "winner_votes": 2},
                        "valid": True, "pred_sql": "SELECT a FROM t",
                        "question": "q", "pred_row_count": 2, "ex": 0.0}])
    assert res["not_measured"]["gates"] == ["back_translation"]
    assert not res["gates"]["back_translation"]["measured"]


# ── Phase B: the block bird.py now stores ────────────────────────────

class TestStoredVerificationIsPreferred:

    REC = {"exec_vote": {"executed": 5, "winner_votes": 3, "groups": 2},
           "valid": True, "pred_sql": "SELECT a FROM t", "pred_row_count": 2,
           "question": "list a", "ex": 0.0}

    def test_a_stored_block_is_used_verbatim(self):
        """It holds evidence — the executed rows' checks, the judge verdict —
        that cannot be recovered from disk afterwards."""
        stored = {"executed": 5, "agreement": 5, "validated": True,
                  "risks": [{"code": "cross_join", "severity": "medium",
                             "message": "m"}],
                  "back_translation": {"verdict": "mismatch", "reason": "r"}}
        assert TE.verification_from_record({**self.REC, "verification": stored}) \
            is stored

    def test_an_empty_or_absent_block_falls_back_to_reconstruction(self):
        for value in (None, {}, "not a dict"):
            ev = TE.verification_from_record({**self.REC, "verification": value})
            assert ev["agreement"] == 3        # rebuilt from exec_vote

    def test_the_judge_gate_becomes_measured_when_the_data_has_it(self):
        recs = [{**self.REC, "ex": e, "verification": {
                    "executed": 2, "agreement": 2, "validated": True,
                    "validation_checks": {"semantic_ok": True}, "risks": [],
                    "back_translation": {"verdict": v, "description": "d"}}}
                for e, v in ((1.0, "match"), (0.0, "mismatch"))]
        res = TE.evaluate(recs)
        assert res["gates"]["back_translation"]["measured"] is True
        assert "back_translation" not in res["not_measured"]["gates"]
        assert res["runs_with_stored_verification"] == 2

    def test_a_gate_skipped_everywhere_is_reported_unmeasured(self):
        """Its 0.5 AUROC must read as 'never ran', not 'measured and
        worthless' — the difference matters in a report."""
        res = TE.evaluate([{**self.REC, "ex": 1.0}, {**self.REC, "ex": 0.0}])
        assert res["gates"]["back_translation"]["measured"] is False
        assert "back_translation" in res["not_measured"]["gates"]

    def test_a_rows_based_code_that_fired_is_no_longer_listed_unmeasured(self):
        recs = [{**self.REC, "ex": 0.0, "verification": {
                    "executed": 1, "agreement": 1, "validated": True,
                    "risks": [{"code": "cross_join", "severity": "medium",
                               "message": "m"}]}}]
        res = TE.evaluate(recs)
        assert "cross_join" not in res["not_measured"]["checks"]
        assert res["checks"]["cross_join"]["fired"] == 1


class TestBirdStoresTheBlock:

    def _validation(self):
        return {"valid": True, "syntax_ok": True, "safety_ok": True,
                "semantic_ok": True}

    def test_deterministic_checks_run_without_a_judge(self):
        from backend.eval.bird import _verification_for
        ev = _verification_for(
            sql="SELECT a FROM t", rows=[], question="list every a",
            evidence="", schema_context="Table: t", query_plan=None,
            validation=self._validation(),
            vote_info={"candidates": 3, "executed": 3, "winner_votes": 3,
                       "groups": 1},
            refined=False, judge_llm=None)
        assert ev["agreement"] == 3 and ev["validated"] is True
        assert any(r["code"] == "zero_rows" for r in ev["risks"])
        assert ev["back_translation"] is None      # not run, not faked

    def test_the_judge_runs_only_when_given_one(self):
        from unittest.mock import MagicMock, patch
        import backend.eval.bird as B
        with patch("backend.agents.back_translation.round_trip",
                   return_value={"verdict": "match", "description": "d",
                                 "reason": None}) as rt:
            ev = B._verification_for(
                sql="SELECT 1", rows=[{"x": 1}], question="q", evidence="",
                schema_context="", query_plan=None,
                validation=self._validation(), vote_info={}, refined=False,
                judge_llm=MagicMock())
        rt.assert_called_once()
        assert ev["back_translation"]["verdict"] == "match"

    def test_a_judge_failure_costs_the_flag_not_the_run(self):
        from unittest.mock import MagicMock, patch
        import backend.eval.bird as B
        with patch("backend.agents.back_translation.round_trip",
                   side_effect=RuntimeError("provider down")):
            ev = B._verification_for(
                sql="SELECT 1", rows=[{"x": 1}], question="q", evidence="",
                schema_context="", query_plan=None,
                validation=self._validation(), vote_info={}, refined=False,
                judge_llm=MagicMock())
        assert ev["back_translation"] is None
        assert ev["validated"] is True             # the rest survived

    def test_no_sql_produces_an_empty_but_well_formed_block(self):
        from backend.eval.bird import _verification_for
        ev = _verification_for(
            sql="", rows=[], question="q", evidence="", schema_context="",
            query_plan=None, validation={"valid": False}, vote_info=None,
            refined=False, judge_llm=None)
        assert ev["risks"] == [] and ev["validated"] is False

    def test_the_block_is_attached_to_every_record(self):
        import inspect
        import backend.eval.bird as B
        src = inspect.getsource(B.run_bird_eval)
        assert '"verification":   verification,' in src
