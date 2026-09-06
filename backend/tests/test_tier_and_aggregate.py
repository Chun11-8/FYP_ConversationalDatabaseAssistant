"""
Tests for the effort-budget floor and the aggregate-intent checks (GH#76).

Attributing all 57 gpt-5-mini failures to a pipeline stage showed the received
wisdom was wrong: joins are fine. 74% of failures had the RIGHT tables and
joins and differed only in projection, aggregation or grain; only 4% were the
generator diverging from its plan.

Worse, the effort budget is read FROM the plan — the artefact that already
failed. A linker that collapses a hard question to one table yields "simple",
which buys 2 candidates and no path diversity, so the question is
under-resourced precisely because the reasoning about it went wrong.

Run from repo root:
    pytest backend/tests/test_tier_and_aggregate.py -v
"""

from __future__ import annotations

import pytest

from backend.agents import risk as R
from backend.agents.db_agent import OrchestrationAgent as A, _question_looks_hard

THIN = {"target": "consumption", "tables": ["yearmonth"]}
RICH = {"target": "total", "tables": ["customers", "transactions_1k", "gasstations"],
        "join_path": ["customers.CustomerID = transactions_1k.CustomerID"]}


class TestQuestionLooksHard:

    @pytest.mark.parametrize("q", [
        "What is the highest monthly consumption in the year 2012?",
        "What is the percentage of female patients born after 1930?",
        "What is the ratio of outpatient to inpatient treatment?",
        "What is the average consumption per customer?",
        "How much higher is Ariel's rating than Paulin's?",
        'What is the amount spent by customer "38508"? How much in January 2012?',
    ])
    def test_real_failures_are_recognised_as_hard(self, q):
        """Every one of these was mis-tiered 'simple' and answered wrongly."""
        assert _question_looks_hard(q)

    def test_a_compound_question_is_hard_regardless_of_wording(self):
        """Two questions in one: the linker has no sub-question field, so both
        collapse into a single plan that looks deceptively simple."""
        assert _question_looks_hard("Who is the manager? Where are they based?")

    @pytest.mark.parametrize("q", [
        "How many customers are there?",
        "List all events.",
        "What is the name of player 3?",
        "Which customers pay in EUR?",
    ])
    def test_genuinely_simple_questions_stay_simple(self, q):
        """The floor may only ever RAISE effort; if it fired on everything the
        simple tier's cost saving would be gone."""
        assert not _question_looks_hard(q)

    def test_empty_input_is_safe(self):
        assert not _question_looks_hard(None)
        assert not _question_looks_hard("")


class TestTierIsNoLongerInferred:
    """
    GH#85 removed the plan-derived tier outright rather than patching the
    floor: the budget was read from the artefact that fails, so no floor could
    make it trustworthy. Live now runs one uniform tier, the BIRD harness
    still gets the dataset's real label, and `_question_looks_hard` survives
    because plan diversity (GH#76) still gates on it.
    """

    def test_the_plan_derived_tier_is_gone(self):
        assert not hasattr(A, "_infer_tier")

    def test_generate_sql_uses_the_fixed_tier_not_a_guess(self):
        import inspect
        src = inspect.getsource(A.generate_sql)
        assert '_live_tier' in src
        assert '_infer_tier' not in src

    def test_question_hardness_still_gates_plan_diversity(self):
        """The signal was not deleted with the tier — link_schema still uses
        it to decide whether a second, disagreeing plan is worth an LLM call."""
        import inspect
        assert "_question_looks_hard(question)" in inspect.getsource(A.link_schema)

class TestAggregateIntent:

    def test_extremum_over_rows_where_per_group_was_asked(self):
        """The measured catch: gold sums per month then takes the largest
        month; MAX over raw rows returns the largest single row — a different,
        smaller, entirely plausible number."""
        assert "extremum_without_grouping" in {
            r["code"] for r in R.check_aggregate_intent(
                "SELECT MAX(Consumption) FROM yearmonth WHERE Date LIKE '2012%'",
                "What is the highest monthly consumption in the year 2012?")}

    def test_silent_once_grouping_is_present(self):
        assert R.check_aggregate_intent(
            "SELECT SUM(c) FROM t GROUP BY m ORDER BY SUM(c) DESC LIMIT 1",
            "What is the highest monthly consumption in 2012?") == []

    def test_total_answered_with_an_extremum(self):
        assert "aggregate_mismatch" in {
            r["code"] for r in R.check_aggregate_intent(
                "SELECT MAX(Price) FROM transactions WHERE CustomerID = 1",
                "What is the total amount spent by customer 1?")}

    def test_a_superlative_total_is_not_a_mismatch(self):
        """"the highest total" legitimately combines both words."""
        assert not any(r["code"] == "aggregate_mismatch" for r in
                       R.check_aggregate_intent(
                           "SELECT MAX(x) FROM t", "What is the highest total spend?"))

    def test_percentage_without_a_ratio(self):
        assert "percentage_without_division" in {
            r["code"] for r in R.check_aggregate_intent(
                "SELECT COUNT(*) FROM customers WHERE Currency = 'EUR'",
                "What percentage of customers used EUR?")}

    def test_percentage_with_a_ratio_is_silent(self):
        assert R.check_aggregate_intent(
            "SELECT 100.0 * COUNT(a) / NULLIF(COUNT(b), 0) FROM t",
            "What percentage of customers used EUR?") == []

    @pytest.mark.parametrize("q,sql", [
        ("How many customers are there?", "SELECT COUNT(*) FROM customers"),
        ("List all events.", "SELECT event_name FROM event"),
        ("What is the total spend?", "SELECT SUM(x) FROM t"),
    ])
    def test_ordinary_correct_queries_stay_silent(self, q, sql):
        assert R.check_aggregate_intent(sql, q) == []

    def test_junk_input_is_safe(self):
        assert R.check_aggregate_intent("", "q") == []
        assert R.check_aggregate_intent("SELECT 1", "") == []


class TestCustomerIdIsNoLongerSkipped:
    """`\\d{4}` unanchored matched "3850" inside 38508, so a 5-digit customer
    id was skipped as date-shaped and its dropped constraint went unflagged."""

    def test_a_five_digit_id_is_testable(self):
        assert not R._span_is_untestable("38508")

    def test_a_bare_year_is_still_skipped(self):
        assert R._span_is_untestable("2012")

    def test_a_slashed_date_is_still_skipped(self):
        assert R._span_is_untestable("2012/8/24")
