"""
Tests for the gold-free risk checks.

Phase 1's agreement signal has a hard ceiling: 41 of gpt-5-mini's 57 BIRD
errors happened on turns where every candidate agreed — wrong together. These
checks look at the query and its rows instead, needing no ground truth.

Every check gets both a firing case AND a false-positive guard. Precision
matters more than recall here: a warning that cries wolf gets ignored, and an
ignored warning is worse than none, so the guards are the point.

Run from repo root:
    pytest backend/tests/test_risk_checks.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents import risk as R


def codes(risks):
    return {r["code"] for r in risks}


# ── 1. zero rows ──────────────────────────────────────────────────────────────

class TestZeroRows:

    def test_fires_on_empty_result(self):
        assert codes(R.check_zero_rows([])) == {"zero_rows"}

    def test_silent_on_rows(self):
        assert R.check_zero_rows([{"x": 1}]) == []

    def test_silent_when_execution_never_produced_a_list(self):
        """None means "didn't run", which is an error path, not a zero-row
        answer — flagging it would double-report."""
        assert R.check_zero_rows(None) == []


# ── 2. ungrounded literal ─────────────────────────────────────────────────────

class TestUngroundedLiterals:

    def _index(self, hints):
        idx = MagicMock()
        idx.ground.return_value = hints
        return idx

    def test_fires_when_the_value_is_not_in_the_data(self):
        idx = self._index(["'Brazil' → use 'BRA' (players.country)"])
        risks = R.check_ungrounded_literals(
            "SELECT n FROM players WHERE country = 'Brazil'", ["players"], idx)
        assert codes(risks) == {"ungrounded_literal"}
        assert "BRA" in risks[0]["message"]

    def test_silent_when_the_value_grounds_cleanly(self):
        risks = R.check_ungrounded_literals(
            "SELECT n FROM players WHERE country = 'BRA'", ["players"],
            self._index([]))
        assert risks == []

    @pytest.mark.parametrize("lit", ["2013-01-01", "12.5", "%abc%", "  "])
    def test_skips_literals_the_index_cannot_adjudicate(self, lit):
        """Dates, numbers and wildcards aren't column values to look up —
        flagging them would be pure noise."""
        idx = self._index(["should not be consulted"])
        assert R.check_ungrounded_literals(
            f"SELECT a FROM t WHERE c = '{lit}'", ["t"], idx) == []

    def test_silent_without_an_index_or_tables(self):
        assert R.check_ungrounded_literals("SELECT a FROM t WHERE c='x'", ["t"], None) == []
        assert R.check_ungrounded_literals("SELECT a FROM t WHERE c='x'", [], self._index(["h"])) == []

    def test_index_failure_is_swallowed(self):
        idx = MagicMock()
        idx.ground.side_effect = RuntimeError("index down")
        assert R.check_ungrounded_literals(
            "SELECT a FROM t WHERE c = 'Brazil'", ["t"], idx) == []

    def test_caps_the_number_of_flags(self):
        idx = self._index(["'x' → use 'y' (t.c)"])
        sql = "SELECT a FROM t WHERE " + " OR ".join(
            f"c = 'value{i}'" for i in range(10))
        assert len(R.check_ungrounded_literals(sql, ["t"], idx)) <= 3


# ── 3. join fan-out ───────────────────────────────────────────────────────────

class TestJoinFanout:

    def test_fires_on_comma_join_without_where(self):
        risks = R.check_join_fanout("SELECT a FROM t1, t2", None)
        assert codes(risks) == {"cross_join"}

    def test_fires_on_explicit_cross_join(self):
        assert codes(R.check_join_fanout(
            "SELECT a FROM t1 CROSS JOIN t2", None)) == {"cross_join"}

    def test_silent_on_a_comma_join_that_is_constrained_in_where(self):
        """The old-style join syntax is fine when WHERE links the tables."""
        assert R.check_join_fanout(
            "SELECT a FROM t1, t2 WHERE t1.id = t2.id", None) == []

    def test_silent_on_a_proper_join_on(self):
        assert R.check_join_fanout(
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id", None) == []

    def test_silent_on_a_single_table(self):
        assert R.check_join_fanout("SELECT a FROM t1 WHERE x = 1", None) == []


# ── 4. aggregate grain ────────────────────────────────────────────────────────

class TestAggregateGrain:

    def test_fires_on_per_group_question_without_group_by(self):
        assert codes(R.check_aggregate_grain(
            "SELECT COUNT(*) FROM orders", "How many orders for each customer?"
        )) == {"missing_group_by"}

    def test_silent_when_group_by_is_present(self):
        assert R.check_aggregate_grain(
            "SELECT c, COUNT(*) FROM orders GROUP BY c",
            "How many orders for each customer?") == []

    def test_silent_on_a_whole_table_question(self):
        assert R.check_aggregate_grain(
            "SELECT COUNT(*) FROM orders", "How many orders are there?") == []

    @pytest.mark.parametrize("question", [
        "What is the amount spent by customer 38508 at the gas stations?",
        "Find the average number of long-shots done by Ahmed Samir Farag.",
        "Calculate the total amount spent by the Student_Club in 2019.",
    ])
    def test_the_bare_preposition_by_does_not_fire(self, question):
        """Regression: an earlier `by \\w+` pattern matched the ordinary
        preposition and produced 3 false alarms against 1 real catch on the
        labelled data. These are all correct answers."""
        assert R.check_aggregate_grain("SELECT SUM(x) FROM t", question) == []


# ── 5. LIMIT 1 on a plural question ───────────────────────────────────────────

class TestLimitOneOnPlural:

    def test_fires_when_a_list_is_answered_with_one_row(self):
        assert codes(R.check_limit_one_on_plural(
            "SELECT name FROM players LIMIT 1", "List all players from Brazil"
        )) == {"limit_one_on_plural"}

    def test_silent_on_a_superlative(self):
        """"The highest-paid player" legitimately wants exactly one row — this
        exclusion is what stops the check firing constantly."""
        assert R.check_limit_one_on_plural(
            "SELECT name FROM players ORDER BY pay DESC LIMIT 1",
            "Which player has the highest pay?") == []

    def test_silent_without_limit_one(self):
        assert R.check_limit_one_on_plural(
            "SELECT name FROM players LIMIT 100", "List all players") == []

    def test_silent_on_a_singular_question(self):
        assert R.check_limit_one_on_plural(
            "SELECT name FROM players WHERE id = 3 LIMIT 1",
            "What is the name of player 3?") == []


# ── 6. arity ──────────────────────────────────────────────────────────────────

class TestArity:

    def test_fires_when_column_count_differs_from_the_plan(self):
        risks = R.check_arity([{"a": 1}], {"target": "name, phone_number"})
        assert codes(risks) == {"arity_mismatch"}
        assert risks[0]["detail"] == {"wanted": 2, "actual": 1}

    def test_silent_when_they_match(self):
        assert R.check_arity([{"a": 1, "b": 2}], {"target": "name, phone"}) == []

    def test_silent_on_a_vague_plan(self):
        """A single-item target says nothing about how many columns to expect."""
        assert R.check_arity([{"a": 1, "b": 2}], {"target": "the player name"}) == []
        assert R.check_arity([{"a": 1}], None) == []

    def test_silent_without_rows(self):
        assert R.check_arity([], {"target": "a, b"}) == []


# ── 7. out-of-schema identifiers ──────────────────────────────────────────────

class TestOutOfSchema:

    SCHEMA = "Table players: id, name\nTable teams: id, city"

    def test_fires_on_a_table_never_shown_to_the_model(self):
        risks = R.check_out_of_schema_tables(["salaries"], self.SCHEMA)
        assert codes(risks) == {"out_of_schema"}
        assert "salaries" in risks[0]["message"]

    def test_silent_when_all_tables_are_present(self):
        assert R.check_out_of_schema_tables(["players", "teams"], self.SCHEMA) == []

    def test_case_insensitive(self):
        assert R.check_out_of_schema_tables(["PLAYERS"], self.SCHEMA) == []

    def test_silent_when_the_context_is_unavailable(self):
        """No context means "cannot tell", which must not read as "invented"."""
        assert R.check_out_of_schema_tables(["players"], "") == []


# ── combiner ──────────────────────────────────────────────────────────────────

class TestAssess:

    def test_a_clean_query_produces_no_flags(self):
        """The single most important property: an unremarkable correct answer
        must be silent, or users learn to ignore the warnings."""
        out = R.assess(
            sql="SELECT name FROM players WHERE country = 'BRA' LIMIT 100",
            rows=[{"name": "A"}, {"name": "B"}],
            question="Which players are from BRA?",
            schema_context="Table players: name, country",
            tables=["players"],
        )
        assert out["risks"] == []
        assert out["risk_score"] == 0

    def test_score_is_severity_weighted(self):
        out = R.assess(sql="SELECT a FROM t1, t2", rows=[],
                       question="q", tables=[])
        # cross_join (high=3) + zero_rows (medium=2)
        assert codes(out["risks"]) == {"cross_join", "zero_rows"}
        assert out["risk_score"] == 5

    def test_never_raises_on_junk_input(self):
        for bad in (None, "", "not sql at all", 12345):
            out = R.assess(sql=bad, rows=None, question=None)  # type: ignore[arg-type]
            assert isinstance(out["risks"], list)

    def test_one_broken_check_does_not_lose_the_others(self):
        idx = MagicMock()
        idx.ground.side_effect = RuntimeError("boom")
        out = R.assess(sql="SELECT a FROM t WHERE c = 'Brazil'", rows=[],
                       question="q", tables=["t"], value_index=idx)
        assert "zero_rows" in codes(out["risks"])   # survived the failure


# ── node wiring ───────────────────────────────────────────────────────────────

class TestAssessRiskNode:

    def _agent(self, **attrs):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        for k, v in dict(_risk_checks=True, _dialect="postgres",
                         value_index=None).items():
            setattr(a, k, v)
        for k, v in attrs.items():
            setattr(a, k, v)
        return a

    def _state(self, **over):
        s = {"best_sql": "SELECT a FROM t1, t2", "execution_result": [],
             "user_query": "q", "schema_context": "", "query_plan": None,
             "verification": {"decided_by": "plurality", "agreement": 3}}
        s.update(over)
        return s

    def test_merges_risks_into_the_existing_verification_block(self):
        out = self._agent().assess_risk(self._state())
        v = out["verification"]
        assert v["decided_by"] == "plurality"      # Phase 1 data preserved
        assert v["agreement"] == 3
        assert "zero_rows" in {r["code"] for r in v["risks"]}
        assert v["risk_score"] > 0

    def test_config_gate_skips_it_entirely(self):
        out = self._agent(_risk_checks=False).assess_risk(self._state())
        assert "risks" not in (out.get("verification") or {})

    def test_clean_result_records_an_empty_list_not_a_missing_key(self):
        out = self._agent().assess_risk(self._state(
            best_sql="SELECT a FROM t WHERE b = 1", execution_result=[{"a": 1}]))
        assert out["verification"]["risks"] == []

    def test_never_breaks_the_turn(self):
        a = self._agent()
        out = a.assess_risk(self._state(best_sql=None, execution_result="junk"))
        assert "verification" in out

    def test_uses_the_condensed_question_when_present(self):
        """A follow-up's raw text ("what about 2014?") carries none of the
        phrasing the question-based checks read."""
        out = self._agent().assess_risk(self._state(
            best_sql="SELECT COUNT(*) FROM t",
            execution_result=[{"c": 1}],
            user_query="what about 2014?",
            standalone_query="How many orders for each customer in 2014?"))
        assert "missing_group_by" in {r["code"] for r in out["verification"]["risks"]}


# ── 8/9. question ↔ SQL coverage ──────────────────────────────────────

class _FakeIndex:
    """Stands in for ValueIndex.resolve_spans: {span: matched value | None}."""

    def __init__(self, mapping):
        self.mapping = mapping

    def resolve_spans(self, text, tables):
        return self.mapping


class TestQuestionCoverage:
    """
    Every other check reads the SQL and asks whether it is well-formed. This
    one reads the QUESTION and asks whether the SQL used it — the gap that let
    the reported failure through: asked for "customers from the country
    Wakanda" with no country column, the model dropped the constraint and
    substituted `segment='LAM' AND currency='EUR'`. Every literal in that SQL
    is real, so nothing SQL-side could object.
    """

    WAKANDA_Q = "Show me customers from the country Wakanda"
    WAKANDA_SQL = ("SELECT DISTINCT c.customerid FROM customers AS c "
                   "WHERE c.segment = 'LAM' AND c.currency = 'EUR' LIMIT 100")

    def test_the_reported_failure_is_caught(self):
        risks = R.check_question_coverage(
            self.WAKANDA_SQL, self.WAKANDA_Q, ["customers"],
            _FakeIndex({"Wakanda": None}))
        assert codes(risks) == {"unknown_value"}
        assert "Wakanda" in risks[0]["message"]

    def test_a_dropped_constraint_on_a_real_value(self):
        risks = R.check_question_coverage(
            "SELECT n FROM players WHERE segment = 'LAM'",
            "Show me customers from Brazil", ["players"],
            _FakeIndex({"Brazil": "BRA"}))
        assert codes(risks) == {"dropped_constraint"}
        assert risks[0]["detail"]["stored_as"] == "BRA"

    def test_a_correctly_grounded_value_is_silent(self):
        """Value grounding rewrites 'Brazil' to 'BRA' on purpose. A naive
        span-not-in-SQL test would fire on a CORRECT query."""
        assert R.check_question_coverage(
            "SELECT n FROM players WHERE country = 'BRA'",
            "Show me customers from Brazil", ["players"],
            _FakeIndex({"Brazil": "BRA"})) == []

    def test_a_value_used_verbatim_is_silent(self):
        assert R.check_question_coverage(
            "SELECT n FROM c WHERE segment = 'LAM'",
            "customers in the LAM segment", ["c"],
            _FakeIndex({"LAM": "LAM"})) == []

    def test_silent_without_a_value_index(self):
        """Measured: guessing here drove false alarms to 54% and precision to
        42% on the labelled data. Absent the index, "not in the SQL" cannot be
        told from "expressed another way", so silence is the only defensible
        answer."""
        assert R.check_question_coverage(
            self.WAKANDA_SQL, self.WAKANDA_Q, ["customers"], None) == []

    def test_silent_when_the_index_cannot_answer(self):
        assert R.check_question_coverage(
            self.WAKANDA_SQL, self.WAKANDA_Q, ["customers"],
            _FakeIndex({})) == []

    def test_index_failure_is_swallowed(self):
        class Boom:
            def resolve_spans(self, text, tables):
                raise RuntimeError("index down")
        assert R.check_question_coverage(
            self.WAKANDA_SQL, self.WAKANDA_Q, ["customers"], Boom()) == []

    @pytest.mark.parametrize("span,question", [
        ("August", "How much did customer 6 consume between August and November 2013?"),
        ("2012/8/24", "What currency did the customer pay at 16:25:00 in 2012/8/24?"),
        ("Is", "Is it true that more SMEs pay in Czech koruna than in euros?"),
    ])
    def test_untestable_spans_never_fire(self, span, question):
        """Dates are rendered as '201308'-style literals and sentence openers
        are not values at all. Every one of these produced a false alarm on
        the labelled data before they were excluded."""
        assert R._span_is_untestable(span)
        assert R.check_question_coverage(
            "SELECT SUM(x) FROM t WHERE Date = '201308'", question, ["t"],
            _FakeIndex({span: None})) == []

    def test_a_real_entity_span_is_still_testable(self):
        assert not R._span_is_untestable("Wakanda")
        assert not R._span_is_untestable("Brazil")

    def test_no_spans_means_no_flags(self):
        assert R.check_question_coverage(
            "SELECT COUNT(*) FROM customers", "how many customers are there?",
            ["customers"], _FakeIndex({})) == []

    def test_reachable_through_assess(self):
        out = R.assess(sql=self.WAKANDA_SQL, rows=[{"customerid": 1}],
                       question=self.WAKANDA_Q,
                       schema_context="Table customers: CustomerID, Segment, Currency",
                       tables=["customers"], value_index=_FakeIndex({"Wakanda": None}))
        assert "unknown_value" in codes(out["risks"])
        assert out["risk_score"] > 0
