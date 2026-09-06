"""
Tests for plan diversity and plan rationale (GH#76 Fixes 1 & 2).

Both target the finding that 74% of failures had the RIGHT tables and joins
and differed only in projection, aggregation or grain — 52% of those picking
the wrong aggregate. Every candidate today descends from ONE plan, and
_plan_penalty ranks candidates by conformance to it, so voting confirms that
plan rather than testing it. That is why 41 of 57 errors had unanimous
agreement: candidates cannot disagree about a decision none of them made.

Both are DEFAULT OFF. They change what the model is asked to produce, so their
effect on accuracy is a hypothesis until the BIRD ablation runs — these tests
pin the plumbing, not the benefit.

Run from repo root:
    pytest backend/tests/test_plan_diversity.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.schema_linking_agent import (
    PLAN_KEYS, SchemaLinkingAgent, _same_shape, render_query_plan,
)

FIRST = {"target": "consumption", "tables": ["yearmonth"],
         "aggregation": "MAX", "grain": "a single overall maximum"}


def _linker(reply: str, rationale: bool = False) -> SchemaLinkingAgent:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply)
    a = SchemaLinkingAgent({"db": {"dialect": "sqlite"}}, llm, rationale=rationale)
    return a


# ── Fix 2: grain + rationale ──────────────────────────────────────────────────

class TestPlanRationale:

    def test_the_new_keys_are_recognised_as_a_usable_plan(self):
        assert "grain" in PLAN_KEYS and "aggregation_rationale" in PLAN_KEYS

    def test_off_by_default(self):
        a = SchemaLinkingAgent({"db": {}}, MagicMock())
        assert a.rationale is False
        assert "grain" not in a._system_prompt()

    def test_enabling_adds_the_instruction(self):
        a = SchemaLinkingAgent({"db": {}}, MagicMock(), rationale=True)
        p = a._system_prompt()
        assert "grain" in p and "aggregation_rationale" in p

    def test_grain_renders_before_the_aggregate(self):
        """Stating the row unit first is the point: "the highest monthly X"
        means one row per month, summed, then the largest."""
        out = render_query_plan({"target": "c", "tables": ["t"],
                                 "grain": "one row per month", "aggregation": "SUM"})
        assert out.index("grain:") < out.index("aggregation:")

    def test_rationale_renders(self):
        out = render_query_plan({"target": "c", "tables": ["t"],
                                 "aggregation_rationale": "evidence says 5th-6th char"})
        assert "aggregation_rationale: evidence says 5th-6th char" in out

    def test_a_plan_without_the_new_fields_renders_as_before(self):
        out = render_query_plan({"target": "c", "tables": ["t"], "aggregation": "SUM"})
        assert "grain" not in out and "aggregation_rationale" not in out


# ── Fix 1: an alternative plan ────────────────────────────────────────────────

class TestSameShape:

    def test_identical_plans_are_the_same_shape(self):
        assert _same_shape(dict(FIRST), dict(FIRST))

    def test_a_different_aggregate_is_a_real_disagreement(self):
        assert not _same_shape({**FIRST, "aggregation": "SUM"}, FIRST)

    def test_a_different_grain_is_a_real_disagreement(self):
        assert not _same_shape({**FIRST, "grain": "one row per month"}, FIRST)

    def test_cosmetic_differences_are_not(self):
        """Aliases, column order and wording are not disagreements — spending
        candidates on them would just duplicate the first plan."""
        assert _same_shape({**FIRST, "tables": ["yearmonth AS ym"],
                            "join_path": ["a = b"]}, FIRST)

    def test_case_and_whitespace_are_ignored(self):
        assert _same_shape({**FIRST, "aggregation": "  max "}, FIRST)


class TestAnalyzeAlternative:

    ALT = ('{"target":"consumption","tables":["yearmonth"],'
           '"aggregation":"SUM","group_by":"month","grain":"one row per month"}')

    def test_returns_a_plan_that_disagrees(self):
        out = _linker(self.ALT).analyze_alternative("q", "schema", FIRST)
        assert out is not None and out["aggregation"] == "SUM"

    def test_an_identical_alternative_is_discarded(self):
        """No disagreement means nothing to diversify — returning it would
        double the candidate cost for a duplicate."""
        same = ('{"target":"consumption","tables":["yearmonth"],'
                '"aggregation":"MAX","grain":"a single overall maximum"}')
        assert _linker(same).analyze_alternative("q", "schema", FIRST) is None

    def test_the_first_plan_is_shown_to_the_model(self):
        a = _linker(self.ALT)
        a.analyze_alternative("q", "schema", FIRST)
        system_msg = a.llm.invoke.call_args[0][0][0].content
        assert "DIFFERENT" in system_msg and "MAX" in system_msg

    def test_an_llm_failure_returns_none(self):
        a = _linker(self.ALT)
        a.llm.invoke.side_effect = RuntimeError("down")
        assert a.analyze_alternative("q", "schema", FIRST) is None

    def test_unparseable_output_returns_none(self):
        assert _linker("no json here").analyze_alternative("q", "s", FIRST) is None

    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_empty_question_returns_none(self, bad):
        assert _linker(self.ALT).analyze_alternative(bad, "s", FIRST) is None

    def test_a_non_dict_first_plan_returns_none(self):
        assert _linker(self.ALT).analyze_alternative("q", "s", None) is None


# ── wiring ────────────────────────────────────────────────────────────────────

class TestWiring:

    def test_both_flags_default_off(self):
        import backend.agents.db_agent as dbm
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent.__init__)
        assert '"plan_diversity", False' in src
        assert '"plan_rationale", False' in src

    def test_the_alternative_is_only_sought_for_hard_questions(self):
        import backend.agents.db_agent as dbm
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent.link_schema)
        assert "_question_looks_hard(question)" in src

    def test_generate_sql_uses_both_plans(self):
        import backend.agents.db_agent as dbm
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent.generate_sql)
        assert 'state.get("query_plan_alt")' in src

    def test_the_bird_flag_is_not_a_no_op(self):
        """--plan-diversity must actually reach the generation loop; a flag
        recorded in run_config but wired to nothing would silently invalidate
        the ablation it exists to enable."""
        import inspect
        from backend.eval import bird
        assert "plan_diversity: bool = False" in inspect.getsource(bird.run_bird_eval)
        assert "analyze_alternative" in inspect.getsource(bird.run_bird_eval)
