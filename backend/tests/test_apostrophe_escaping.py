"""
Tests for apostrophe escaping in string literals.

A grounded value like "Women's Soccer" must appear as 'Women''s Soccer' or the SQL
is invalid (sqlglot TokenError). Covers:
  - escape_sql_literal helper
  - parse-gated literal repair in SQLCodeAgent (generate + regenerate)
  - validator resilience (TokenError → valid:False, not a crash)

All LLM calls are mocked, so these run offline.

Run from repo root:
    pytest backend/tests/test_apostrophe_escaping.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import sqlglot

from backend.agents.sql_code_agent import (
    SQLCodeAgent,
    escape_sql_literal,
    _double_interior_apostrophes,
)
from backend.agents.sql_validator_agent import SQLValidatorAgent


def _parses(sql: str, dialect: str = "sqlite") -> bool:
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True
    except Exception:
        return False


# ── escape_sql_literal ────────────────────────────────────────────────────────

class TestEscapeSqlLiteral:

    def test_single_apostrophe(self):
        assert escape_sql_literal("Women's Soccer") == "Women''s Soccer"

    def test_multiple_apostrophes(self):
        assert escape_sql_literal("O'Brien's Team") == "O''Brien''s Team"

    def test_no_apostrophe_unchanged(self):
        assert escape_sql_literal("Medium") == "Medium"

    def test_used_in_literal_parses(self):
        v = escape_sql_literal("O'Brien's Team")
        assert _parses(f"SELECT x FROM t WHERE name = '{v}'")


# ── repair heuristic ──────────────────────────────────────────────────────────

class TestRepairHeuristic:

    def test_repairs_womens_soccer(self):
        broken = "SELECT event_name FROM event WHERE event_name = 'Women's Soccer'"
        fixed = _double_interior_apostrophes(broken)
        assert "'Women''s Soccer'" in fixed
        assert _parses(fixed)

    def test_repairs_edge_multiple_apostrophes(self):
        broken = "SELECT x FROM t WHERE name = 'O'Brien's Team'"
        fixed = _double_interior_apostrophes(broken)
        assert "'O''Brien''s Team'" in fixed
        assert _parses(fixed)

    def test_valid_sql_unchanged_when_repaired(self):
        # Already-valid SQL with quoted values must not be corrupted.
        for sql in [
            "SELECT x FROM t WHERE a = 'Medium' AND b = 'Small'",
            "SELECT x FROM t WHERE a IN ('Fundraiser', 'Meeting')",
            "SELECT AVG(c) / 12 FROM t WHERE date = '201208'",
            "SELECT x FROM t WHERE name = 'Women''s Soccer'",  # already escaped
        ]:
            assert _parses(_double_interior_apostrophes(sql))


# ── SQLCodeAgent._repair_literals (parse-gated) ───────────────────────────────

class TestRepairLiterals:

    def _agent(self, dialect="sqlite"):
        return SQLCodeAgent({"db": {"dialect": dialect}}, MagicMock())

    def test_repairs_broken_literal(self):
        agent = self._agent()
        out = agent._repair_literals("SELECT n FROM t WHERE n = 'Women's Soccer'")
        assert _parses(out)
        assert "''" in out

    def test_noop_on_valid_sql(self):
        agent = self._agent()
        valid = "SELECT n FROM t WHERE n = 'Medium'"
        assert agent._repair_literals(valid) == valid

    def test_empty_returns_empty(self):
        agent = self._agent()
        assert agent._repair_literals("") == ""


# ── generate() escapes on the first attempt ───────────────────────────────────

class TestGenerateEscapes:

    def test_generate_returns_valid_sql_for_apostrophe_value(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT event_name FROM event "
                    "WHERE event_name = 'Women's Soccer'\n```"
        )
        sqls = agent.generate(query="q", schema_context="event(event_name)",
                              few_shots=[], history=[])
        assert len(sqls) == 1
        # The single generation attempt yields parseable SQL — no fallback needed.
        assert _parses(sqls[0])
        assert "Women" in sqls[0]

    def test_regenerate_returns_valid_sql_for_apostrophe_value(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT x FROM t WHERE name = 'O'Brien's Team'\n```"
        )
        sqls = agent.regenerate(query="q", schema_context="t(name)",
                                original_sql="bad", error="boom", history=[])
        assert len(sqls) == 1
        assert _parses(sqls[0])


# ── validator resilience ──────────────────────────────────────────

class TestValidatorResilience:

    def _validator(self):
        return SQLValidatorAgent({"db": {"dialect": "sqlite"}})

    def test_check_syntax_no_crash_on_token_error(self):
        v = self._validator()
        ok, err = v._check_syntax("SELECT n FROM t WHERE n = 'Women's Soccer'")
        assert ok is False
        assert err  # a message, not an exception

    def test_validate_returns_invalid_not_raises(self):
        v = self._validator()
        result = v.validate("SELECT n FROM t WHERE n = 'Women's Soccer'", "t(n)")
        assert result["valid"] is False
        assert result["syntax_ok"] is False

    def test_valid_escaped_sql_passes_syntax(self):
        v = self._validator()
        ok, _ = v._check_syntax("SELECT n FROM t WHERE n = 'Women''s Soccer'")
        assert ok is True
