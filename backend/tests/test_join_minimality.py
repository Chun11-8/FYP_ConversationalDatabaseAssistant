"""
Tests for SQLCodeAgent._join_penalty — a deterministic candidate
demotion for spurious joins: a joined table contributing no column anywhere
outside its own ON clause is pure risk (an INNER JOIN can silently drop rows
via referential gaps, corrupting aggregates for no logical reason; BIRD
q1189). This is the inverse of _plan_penalty's missing-join check.

All offline (no LLM calls needed — pure sqlglot parsing).

Run from repo root:
    pytest backend/tests/test_join_minimality.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.agents.sql_code_agent import SQLCodeAgent


def _agent() -> SQLCodeAgent:
    return SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())


# The Examination-alone query is correct; joining Patient for no reason
# mirrors BIRD q1189 (COUNT dropped valid rows via a referential gap). Only
# tables introduced via an explicit JOIN clause are assessed — the base FROM
# table is exempt (you can't "not join" your only table).
NO_JOIN = "SELECT COUNT(*) FROM Examination WHERE Thrombosis = 2"

SPURIOUS_JOIN = (
    "SELECT COUNT(*) FROM Examination "
    "INNER JOIN Patient ON Examination.ID = Patient.ID "
    "WHERE Examination.Thrombosis = 2"
)

NEEDED_JOIN = (
    "SELECT Patient.Birthday FROM Examination "
    "INNER JOIN Patient ON Examination.ID = Patient.ID "
    "WHERE Examination.Thrombosis = 2"
)

TWO_JOINS_ONE_SPURIOUS = (
    "SELECT a.x FROM a "
    "JOIN b ON a.id = b.id "
    "JOIN c ON b.id = c.id "
    "WHERE b.y = 1"
)


class TestJoinPenalty:

    def test_no_join_not_penalized(self):
        assert _agent()._join_penalty(NO_JOIN) == 0

    def test_spurious_join_penalized(self):
        # Patient contributes no column outside its own ON clause.
        assert _agent()._join_penalty(SPURIOUS_JOIN) == 1

    def test_needed_join_not_penalized(self):
        # Patient.Birthday is selected — the join is justified.
        assert _agent()._join_penalty(NEEDED_JOIN) == 0

    def test_join_used_only_in_where_not_penalized(self):
        sql = ("SELECT a.x FROM a JOIN b ON a.id = b.id "
               "WHERE b.status = 'active'")
        assert _agent()._join_penalty(sql) == 0

    def test_second_join_spurious_first_needed(self):
        # b is used in WHERE (justified); c contributes nothing beyond its
        # own ON clause (spurious) — exactly one penalty point.
        assert _agent()._join_penalty(TWO_JOINS_ONE_SPURIOUS) == 1

    def test_both_joins_spurious(self):
        sql = "SELECT a.x FROM a JOIN b ON a.id = b.id JOIN c ON a.id = c.id"
        assert _agent()._join_penalty(sql) == 2

    def test_unparseable_sql_not_penalized(self):
        assert _agent()._join_penalty("SELECT broken FROM (") == 0

    def test_empty_sql_not_penalized(self):
        assert _agent()._join_penalty("") == 0

    def test_join_column_used_via_group_by(self):
        sql = ("SELECT b.name, COUNT(*) FROM a JOIN b ON a.id = b.id "
               "GROUP BY b.name")
        assert _agent()._join_penalty(sql) == 0

    def test_join_column_used_via_order_by(self):
        sql = "SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY b.name"
        assert _agent()._join_penalty(sql) == 0


class TestJoinPenaltyOrdering:

    def test_order_candidates_prefers_minimal_join(self):
        agent = _agent()
        ordered = agent._order_candidates([SPURIOUS_JOIN, NO_JOIN], schema_context="")
        assert ordered == [NO_JOIN, SPURIOUS_JOIN]

    def test_order_candidates_keeps_needed_join_first(self):
        agent = _agent()
        # Both are equally "minimal" (0 penalty) — order preserved (stable sort).
        ordered = agent._order_candidates([NEEDED_JOIN, NO_JOIN], schema_context="")
        assert ordered == [NEEDED_JOIN, NO_JOIN]
