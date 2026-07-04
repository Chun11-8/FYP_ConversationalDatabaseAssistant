"""
Tests for execution-guided candidate selection (Tier 3, item 8) and the
config-gated safety LIMIT.

select_by_execution() runs candidates against SQLite, groups them by an
order-insensitive result fingerprint, and plurality-votes; the caller falls back
to the text reranker when voting is inconclusive. Non-SELECT / invalid
candidates must never be executed (read-only safety).

Run from repo root:
    pytest backend/tests/test_execution_voting.py -v
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent
from backend.eval.bird import select_by_execution


SCHEMA_CTX = (
    "Table: event\nColumns:\n  - event_id: INTEGER [PK]\n"
    "  - event_name: TEXT\n  - type: TEXT"
)


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "vote.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE event (event_id INTEGER PRIMARY KEY, event_name TEXT, type TEXT);
        INSERT INTO event VALUES (1, 'Open Day', 'Meeting');
        INSERT INTO event VALUES (2, 'Gala', 'Fundraiser');
        INSERT INTO event VALUES (3, 'Soccer', 'Game');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture()
def validator():
    return SQLValidatorAgent({"db": {"dialect": "sqlite"}})


class TestSelectByExecution:

    def test_plurality_wins(self, db_path, validator):
        # Two candidates return the same 2-row set; one returns a different set.
        candidates = [
            "SELECT event_name FROM event WHERE type <> 'Fundraiser'",
            "SELECT event_name FROM event WHERE type IN ('Meeting', 'Game')",
            "SELECT event_name FROM event",  # 3 rows — minority
        ]
        winner, info = select_by_execution(candidates, db_path, validator, SCHEMA_CTX)
        assert winner in candidates[:2]
        assert info["winner_votes"] == 2
        assert info["executed"] == 3 and info["groups"] == 2

    def test_all_distinct_falls_back(self, db_path, validator):
        candidates = [
            "SELECT event_name FROM event WHERE type = 'Meeting'",
            "SELECT event_name FROM event WHERE type = 'Game'",
            "SELECT event_name FROM event WHERE type = 'Fundraiser'",
        ]
        winner, info = select_by_execution(candidates, db_path, validator, SCHEMA_CTX)
        assert winner is None
        assert info["fallback_reason"] == "tie between distinct result sets"

    def test_sole_executable_candidate_wins(self, db_path, validator):
        candidates = [
            "SELECT event_name FROM event WHERE type = 'Meeting'",
            "SELECT nonexistent_col FROM event",          # execution error
            "SELECT event_name FROM event WHERE name = 'Women's Soccer'",  # invalid
        ]
        winner, info = select_by_execution(candidates, db_path, validator, SCHEMA_CTX)
        assert winner == candidates[0]
        assert info["executed"] == 1

    def test_tie_prefers_non_empty_result(self, db_path, validator):
        candidates = [
            "SELECT event_name FROM event WHERE type = 'Meeting'",   # 1 row
            "SELECT event_name FROM event WHERE type = 'NoSuch'",    # 0 rows
        ]
        winner, info = select_by_execution(candidates, db_path, validator, SCHEMA_CTX)
        assert winner == candidates[0]

    def test_non_select_never_executed(self, db_path, validator):
        # A DML candidate must be blocked by the validator, not run.
        candidates = [
            "DELETE FROM event",
            "SELECT event_name FROM event WHERE type = 'Meeting'",
        ]
        winner, _ = select_by_execution(candidates, db_path, validator, SCHEMA_CTX)
        assert winner == candidates[1]
        # Table contents untouched — the DELETE was never executed.
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 3
        conn.close()

    def test_no_candidates(self, db_path, validator):
        winner, info = select_by_execution([], db_path, validator, SCHEMA_CTX)
        assert winner is None and info["fallback_reason"] == "no candidates"

    def test_nothing_executable_falls_back(self, db_path, validator):
        winner, info = select_by_execution(
            ["SELECT broken FROM nowhere("], db_path, validator, SCHEMA_CTX
        )
        assert winner is None
        assert info["fallback_reason"] == "no candidate executed successfully"


# ── n_candidates config (feeds the voter more diverse candidates) ─────────────

class TestNCandidatesConfig:

    def test_default_top_k(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        assert agent.top_k == SQLCodeAgent.TOP_K

    def test_config_override(self):
        agent = SQLCodeAgent(
            {"db": {"dialect": "sqlite"}, "sql": {"n_candidates": 5}}, MagicMock()
        )
        assert agent.top_k == 5

    def test_prompt_asks_for_configured_count(self):
        agent = SQLCodeAgent(
            {"db": {"dialect": "sqlite"}, "sql": {"n_candidates": 5}}, MagicMock()
        )
        prompt = agent._build_prompt("q", "s", ["x"], [])
        assert "Generate 5 candidate queries" in prompt

    def test_bad_value_falls_back_to_default(self):
        agent = SQLCodeAgent(
            {"db": {"dialect": "sqlite"}, "sql": {"n_candidates": "lots"}}, MagicMock()
        )
        assert agent.top_k == SQLCodeAgent.TOP_K


# ── config-gated safety LIMIT in the live pipeline ────────────────────────────

class TestEnforceLimitGate:

    def _agent(self, enforce: bool):
        from backend.agents.db_agent import OrchestrationAgent
        agent = OrchestrationAgent.__new__(OrchestrationAgent)
        agent._enforce_limit = enforce
        return agent

    def _run_execute(self, agent, captured):
        import backend.agents.db_agent as m
        original = m.execute_query
        m.execute_query = lambda sql: captured.append(sql) or [{"x": 1}]
        try:
            state = {"best_sql": "SELECT event_name FROM event"}
            return agent.execute_sql(state)
        finally:
            m.execute_query = original

    def test_limit_appended_when_enabled(self):
        captured: list = []
        out = self._run_execute(self._agent(True), captured)
        assert "LIMIT 100" in captured[0]
        assert "LIMIT 100" in out["best_sql"]

    def test_limit_not_appended_when_disabled(self):
        captured: list = []
        out = self._run_execute(self._agent(False), captured)
        assert "LIMIT" not in captured[0]
        assert out["best_sql"] == "SELECT event_name FROM event"
