"""
Tests for GH#91 — a verdict the user can act on, plus the two new gates
(plan conformance, back-translation).

The design constraint is as important as the behaviour: nothing here may read
a benchmark file or present a percentage as a property of an individual
answer. Several tests exist purely to hold that line.

All offline: no LLM, no database.

Run from repo root:
    pytest backend/tests/test_trust_verdict.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agents import back_translation as BT
from backend.agents import trust as T
from backend.agents.risk import check_plan_conformance

CLEAN = {
    "executed": 5, "agreement": 5, "candidates": 5, "validated": True,
    "validation_checks": {"syntax_ok": True, "safety_ok": True, "semantic_ok": True},
    "risks": [], "decided_by": "plurality",
    "back_translation": {"verdict": "match", "description": "d"},
}


def _with(**over):
    return {**CLEAN, **over}


# ── the verdict ───────────────────────────────────────────────────────────────

class TestVerdict:

    def test_all_gates_green_passes(self):
        g = T.grade(CLEAN)
        assert g["level"] == T.PASSED
        assert g["failed"] == []

    def test_the_top_verdict_never_claims_correctness(self):
        """'Checks passed' is defensible; 'this is correct' is not — the checks
        are necessary conditions, not sufficient ones."""
        text = (T.HEADLINE[T.PASSED] + T.STATEMENT[T.PASSED]).lower()
        assert "not a guarantee" in text
        assert "correct" not in T.HEADLINE[T.PASSED].lower()

    def test_a_high_severity_flag_is_untrusted(self):
        g = T.grade(_with(risks=[{"code": "unknown_value", "severity": "high",
                                  "message": "'SME' does not exist."}]))
        assert g["level"] == T.UNTRUSTED
        assert "Matches your question" in g["failed"]

    def test_a_medium_flag_asks_for_review_not_rejection(self):
        g = T.grade(_with(risks=[{"code": "zero_rows", "severity": "medium",
                                  "message": "No rows."}]))
        assert g["level"] == T.REVIEW

    def test_split_candidates_ask_for_review(self):
        assert T.grade(_with(agreement=3, executed=5))["level"] == T.REVIEW

    def test_nothing_executed_is_untrusted(self):
        assert T.grade(_with(executed=0, agreement=0))["level"] == T.UNTRUSTED

    def test_failed_validation_is_untrusted(self):
        g = T.grade(_with(validated=False,
                          validation_checks={"syntax_ok": True, "safety_ok": True,
                                             "semantic_ok": False}))
        assert g["level"] == T.UNTRUSTED
        assert "semantic" in g["gates"][0]["detail"]

    def test_a_back_translation_mismatch_is_untrusted(self):
        g = T.grade(_with(back_translation={
            "verdict": "mismatch", "description": "the smallest single reading",
            "reason": "You asked for a total; this returns one row's value."}))
        assert g["level"] == T.UNTRUSTED
        assert "total" in [x for x in
                           (gate["detail"] for gate in g["gates"])
                           if "total" in x][0]

    def test_the_worst_gate_decides(self):
        g = T.grade(_with(agreement=3,
                          risks=[{"code": "out_of_schema", "severity": "high",
                                  "message": "unknown table"}]))
        assert g["level"] == T.UNTRUSTED

    def test_a_turn_with_no_query_is_not_graded_as_a_failure(self):
        for reason in ("not_generated", "generation_failed"):
            g = T.grade({"decided_by": reason})
            assert g["level"] == T.NOT_GRADED
            assert g["gates"] == []

    def test_missing_or_junk_input_does_not_raise(self):
        for v in (None, {}, {"risks": None}):
            assert T.grade(v)["level"] in T.VERDICTS


class TestGatesNeverPunishAbsence:
    """'Not checked' must never read as 'found wrong' — otherwise turning a
    check off would make every answer look untrustworthy."""

    def test_back_translation_off_does_not_lower_the_verdict(self):
        g = T.grade(_with(back_translation=None))
        assert g["level"] == T.PASSED
        bt = [x for x in g["gates"] if x["id"] == "back_translation"][0]
        assert bt["status"] == T.SKIPPED

    def test_an_unknown_round_trip_verdict_is_skipped(self):
        g = T.grade(_with(back_translation={"verdict": "unknown",
                                            "description": "d"}))
        assert g["level"] == T.PASSED

    def test_a_cache_hit_is_not_penalised_for_having_no_candidates(self):
        g = T.grade(_with(decided_by="cache_hit", executed=0, agreement=0))
        assert g["level"] == T.PASSED

    def test_every_failing_gate_offers_a_remedy(self):
        g = T.grade(_with(agreement=1, executed=3,
                          risks=[{"code": "unknown_value", "severity": "high",
                                  "message": "x"}]))
        for gate in g["gates"]:
            if gate["status"] in (T.WARN, T.FAIL):
                assert gate["remedy"], gate["id"]


class TestNoBenchmarkDependency:
    """The constraint that killed the first design: benchmark base rates
    describe the DETECTOR, not an individual answer over the user's data."""

    def test_the_module_reads_no_files(self):
        import inspect
        src = inspect.getsource(T)
        for forbidden in ("open(", "json.load", "outputs/", "Path("):
            assert forbidden not in src, forbidden

    def test_no_percentage_is_presented_as_a_property_of_an_answer(self):
        g = T.grade(CLEAN)
        blob = g["statement"] + " ".join(x["detail"] for x in g["gates"])
        assert "%" not in blob


# ── gate 3: plan conformance ──────────────────────────────────────────────────

class TestPlanConformance:

    def test_the_reported_min_vs_sum_failure_is_caught(self):
        out = check_plan_conformance(
            "SELECT customerid, MIN(consumption) FROM yearmonth GROUP BY customerid",
            {"aggregation": "SUM(consumption)", "grain": "one row per customer"})
        assert out and out[0]["code"] == "plan_aggregate_mismatch"
        assert out[0]["severity"] == "high"

    def test_a_conforming_query_is_silent(self):
        assert check_plan_conformance(
            "SELECT customerid, SUM(consumption) FROM yearmonth GROUP BY customerid",
            {"aggregation": "SUM(consumption)", "group_by": "customerid"}) == []

    def test_a_dropped_group_by_is_reported(self):
        out = check_plan_conformance("SELECT SUM(x) FROM t",
                                     {"aggregation": "SUM(x)", "group_by": "t.id"})
        assert out and out[0]["code"] == "plan_group_by_dropped"

    def test_synonyms_do_not_produce_a_false_mismatch(self):
        assert check_plan_conformance(
            "SELECT AVG(x) FROM t", {"aggregation": "average of x"}) == []

    def test_a_plain_list_is_not_a_grain_error(self):
        """GH#93: `SELECT name FROM player WHERE ...` already returns one row
        per player — no GROUP BY is needed, because each source row IS a
        player. Flagging it fired on every "list the X" question."""
        assert check_plan_conformance(
            "SELECT player_name FROM player WHERE height > 180",
            {"grain": "one row per player"}) == []

    def test_distinct_counts_as_grouping(self):
        assert check_plan_conformance(
            "SELECT DISTINCT p.name FROM player p JOIN team t ON 1=1",
            {"grain": "one row per player"}) == []

    def test_an_aggregate_with_no_group_by_is_a_grain_error(self):
        """The failure the check exists for: many rows collapsed into one."""
        out = check_plan_conformance("SELECT SUM(consumption) FROM yearmonth",
                                     {"grain": "one row per customer"})
        assert out and out[0]["code"] == "plan_grain_mismatch"

    def test_a_grouped_aggregate_is_fine(self):
        assert check_plan_conformance(
            "SELECT customerid, SUM(c) FROM yearmonth GROUP BY customerid",
            {"grain": "one row per customer"}) == []

    def test_no_plan_means_no_opinion(self):
        assert check_plan_conformance("SELECT 1", None) == []
        assert check_plan_conformance("", {"aggregation": "SUM(x)"}) == []

    def test_it_is_wired_into_assess(self):
        import inspect
        from backend.agents import risk
        assert "check_plan_conformance" in inspect.getsource(risk.assess)


# ── gate 5: back-translation ──────────────────────────────────────────────────

class TestBackTranslation:

    def _llm(self, reply):
        m = MagicMock()
        m.content = reply
        return m

    def test_the_description_prompt_hides_the_question(self):
        """Shown the question, the model paraphrases it back and the check
        becomes a mirror."""
        assert "NOT the question" in BT._DESCRIBE_SYSTEM

    def test_describe_returns_one_trimmed_sentence(self):
        with patch.object(BT, "invoke_tracked",
                          return_value=self._llm("  the total  spend\n\n")):
            assert BT.describe_sql(MagicMock(), "SELECT 1") == "the total spend"

    def test_describe_degrades_to_none(self):
        with patch.object(BT, "invoke_tracked", side_effect=RuntimeError("x")):
            assert BT.describe_sql(MagicMock(), "SELECT 1") is None
        assert BT.describe_sql(MagicMock(), "") is None

    def test_compare_parses_a_mismatch_with_its_reason(self):
        with patch.object(BT, "invoke_tracked", return_value=self._llm(
                "VERDICT: MISMATCH\nREASON: You asked for a total.")):
            out = BT.compare(MagicMock(), "q", "d")
        assert out == {"verdict": "mismatch", "reason": "You asked for a total."}

    def test_compare_parses_a_match(self):
        with patch.object(BT, "invoke_tracked",
                          return_value=self._llm("VERDICT: MATCH\nREASON: same")):
            assert BT.compare(MagicMock(), "q", "d")["verdict"] == "match"

    @pytest.mark.parametrize("reply", ["not a verdict", "", "VERDICT: maybe"])
    def test_an_unparseable_reply_is_unknown_not_a_fault(self, reply):
        with patch.object(BT, "invoke_tracked", return_value=self._llm(reply)):
            assert BT.compare(MagicMock(), "q", "d")["verdict"] == "unknown"

    def test_a_comparison_failure_is_unknown(self):
        with patch.object(BT, "invoke_tracked", side_effect=RuntimeError("x")):
            assert BT.compare(MagicMock(), "q", "d")["verdict"] == "unknown"

    def test_round_trip_keeps_the_description_when_comparison_fails(self):
        """The description alone is often enough for the user to spot the
        error, so it must survive an inconclusive comparison."""
        with patch.object(BT, "describe_sql", return_value="the lowest reading"), \
             patch.object(BT, "compare",
                          return_value={"verdict": "unknown", "reason": None}):
            out = BT.round_trip(MagicMock(), MagicMock(), "SELECT 1", "q")
        assert out["description"] == "the lowest reading"
        assert out["verdict"] == "unknown"

    def test_round_trip_skips_the_comparison_with_no_description(self):
        with patch.object(BT, "describe_sql", return_value=None), \
             patch.object(BT, "compare") as cmp_:
            out = BT.round_trip(MagicMock(), MagicMock(), "SELECT 1", "q")
        cmp_.assert_not_called()
        assert out["verdict"] == "unknown"


# ── wiring ────────────────────────────────────────────────────────────────────

class TestWiredIntoTheGraph:

    def test_every_turn_carries_a_verdict(self):
        from backend.agents.db_agent import OrchestrationAgent
        ev = OrchestrationAgent._verification_block({
            "verification": {"candidates": 5, "executed": 5, "agreement": 5,
                             "decided_by": "plurality", "risks": []},
            "validation": {"valid": True, "syntax_ok": True, "safety_ok": True,
                           "semantic_ok": True},
        })
        assert ev["trust"]["level"] == T.PASSED

    def test_a_cache_hit_still_gets_a_verdict(self):
        from backend.agents.db_agent import OrchestrationAgent
        ev = OrchestrationAgent._verification_block(
            {"cache_hit": True, "validation": {"valid": True}})
        assert ev["trust"]["level"] in T.VERDICTS

    def test_the_verdict_is_derived_last(self):
        """It must be computed from the FINISHED block, so it can never
        contradict the evidence rendered beside it."""
        import inspect
        from backend.agents.db_agent import OrchestrationAgent
        src = inspect.getsource(OrchestrationAgent._verification_block)
        assert src.index("grade(ev)") > src.index('ev["validation_checks"]')
        assert src.rstrip().endswith("return ev")

    def test_back_translation_can_be_switched_off(self):
        import inspect
        from backend.agents.db_agent import OrchestrationAgent
        src = inspect.getsource(OrchestrationAgent.assess_risk)
        assert '_back_translation' in src

    def test_the_round_trip_never_breaks_a_turn(self):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a._back_translation = True
        with patch("backend.agents.back_translation.round_trip",
                   side_effect=RuntimeError("provider down")):
            out = a._round_trip({"user_query": "q"}, "SELECT 1")
        assert out == {"description": None, "verdict": "unknown", "reason": None}
