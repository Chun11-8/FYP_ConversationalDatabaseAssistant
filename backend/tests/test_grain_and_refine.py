"""
Tests for the GROUP BY-grain demotion and the BIRD-path ReAct refine round.

- SQLCodeAgent._grain_penalty / _order_by_grain: candidates that GROUP BY a
  name-only column while the schema exposes a PK are demoted (names recur —
  grouping by name merges distinct entities; BIRD q1323).
- run_bird_eval(refine=True): one ReAct round on an observable failure
  (invalid SQL / execution error / 0 rows); refined SQL kept only when
  strictly better.

All LLM calls are mocked; SQLite DBs are built in tmp dirs. Runs offline.

Run from repo root:
    pytest backend/tests/test_grain_and_refine.py -v
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent
from backend.agents.utils import PromptHarness
from backend.eval.bird import run_bird_eval


# ── Grain penalty / demotion ──────────────────────────────────────────────────

BIRD_SCHEMA = (
    "Table: event\n"
    "Columns:\n"
    "  - event_id: INTEGER [PK]\n"
    "  - event_name: TEXT\n"
    "  - type: TEXT"
)

CHUNK_SCHEMA = (
    "Table: event\n"
    "Columns:\n"
    "  event_id (INTEGER [PK])\n"
    "  event_name (TEXT)"
)

GROUP_BY_NAME = (
    "SELECT event_name FROM event GROUP BY event_name HAVING COUNT(*) > 20"
)
GROUP_BY_PK = (
    "SELECT event_name FROM event GROUP BY event_id HAVING COUNT(*) > 20"
)


def _agent() -> SQLCodeAgent:
    return SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())


class TestGrainPenalty:

    def test_group_by_name_with_pk_penalized(self):
        assert _agent()._grain_penalty(GROUP_BY_NAME, BIRD_SCHEMA) == 1

    def test_group_by_pk_not_penalized(self):
        assert _agent()._grain_penalty(GROUP_BY_PK, BIRD_SCHEMA) == 0

    def test_no_pk_in_schema_not_penalized(self):
        schema = "Table: event\nColumns:\n  - event_name: TEXT"
        assert _agent()._grain_penalty(GROUP_BY_NAME, schema) == 0

    def test_no_group_by_not_penalized(self):
        assert _agent()._grain_penalty("SELECT event_name FROM event", BIRD_SCHEMA) == 0

    def test_mixed_group_with_pk_not_penalized(self):
        sql = "SELECT event_name FROM event GROUP BY event_id, event_name"
        assert _agent()._grain_penalty(sql, BIRD_SCHEMA) == 0

    def test_chunk_format_pk_detected(self):
        assert _agent()._grain_penalty(GROUP_BY_NAME, CHUNK_SCHEMA) == 1

    def test_unparseable_sql_not_penalized(self):
        assert _agent()._grain_penalty("SELECT broken FROM (", BIRD_SCHEMA) == 0

    def test_order_by_grain_demotes_name_grouping(self):
        ordered = _agent()._order_by_grain([GROUP_BY_NAME, GROUP_BY_PK], BIRD_SCHEMA)
        assert ordered == [GROUP_BY_PK, GROUP_BY_NAME]

    def test_generate_demotes_wrong_grain_candidate(self):
        agent = _agent()
        agent.llm.invoke.return_value = MagicMock(
            content=f"```sql\n{GROUP_BY_NAME}\n```\n```sql\n{GROUP_BY_PK}\n```"
        )
        sqls = agent.generate(query="q", schema_context=BIRD_SCHEMA,
                              few_shots=["x"], history=[])
        assert sqls[0] == GROUP_BY_PK

    def test_prompts_state_pk_grouping_rule(self):
        assert "PRIMARY KEY" in PromptHarness.get("SQL_CODE_AGENT")
        assert "PRIMARY KEY" in PromptHarness.get_file("sql_reranker.md")


# ── BIRD-path ReAct refine ────────────────────────────────────────────────────

GOLD_SQL = "SELECT name FROM t WHERE id = 1"
GOOD_SQL = "SELECT name FROM t WHERE id = 1"
EMPTY_SQL = "SELECT name FROM t WHERE id = 99"
BROKEN_SQL = "SELECT broken FROM ("


@pytest.fixture()
def bird_env(tmp_path):
    """A tmp dev.json + {db_id}/{db_id}.sqlite pair for run_bird_eval."""
    db_dir = tmp_path / "dbs"
    (db_dir / "db1").mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "db1" / "db1.sqlite")
    conn.executescript(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);"
        "INSERT INTO t VALUES (1, 'A'); INSERT INTO t VALUES (2, 'B');"
    )
    conn.commit()
    conn.close()

    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps([{
        "question_id": 1, "db_id": "db1", "difficulty": "moderate",
        "question": "name of id 1?", "evidence": "", "SQL": GOLD_SQL,
    }]))
    return str(dev), str(db_dir)


def _mock_agent(generate_sqls, regenerate_sqls):
    agent = MagicMock()
    agent.generate.return_value = list(generate_sqls)
    agent.regenerate.return_value = list(regenerate_sqls)
    agent.rerank.side_effect = lambda sqls, q, s: sqls[0]
    return agent


@pytest.fixture()
def validator():
    return SQLValidatorAgent({"db": {"dialect": "sqlite"}})


class TestBirdRefine:

    def test_refine_recovers_from_invalid_sql(self, bird_env, validator):
        dev, dbs = bird_env
        agent = _mock_agent([BROKEN_SQL], [GOOD_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=True)
        r = results[0]
        assert r["refined"] is True
        assert r["ex"] == 1.0
        agent.regenerate.assert_called_once()

    def test_refine_recovers_from_zero_rows(self, bird_env, validator):
        dev, dbs = bird_env
        agent = _mock_agent([EMPTY_SQL], [GOOD_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=True)
        r = results[0]
        assert r["refined"] is True and r["ex"] == 1.0

    def test_refine_disabled_no_retry(self, bird_env, validator):
        dev, dbs = bird_env
        agent = _mock_agent([BROKEN_SQL], [GOOD_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=False)
        r = results[0]
        assert r["refined"] is False and r["ex"] == 0.0
        agent.regenerate.assert_not_called()

    def test_worse_refinement_keeps_original(self, bird_env, validator):
        # Original: valid but 0 rows. Refinement: broken → must keep original.
        dev, dbs = bird_env
        agent = _mock_agent([EMPTY_SQL], [BROKEN_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=True)
        r = results[0]
        assert r["refined"] is False
        assert r["pred_sql"] == EMPTY_SQL
        assert r["valid"] is True  # original stayed

    def test_no_refine_when_result_nonempty(self, bird_env, validator):
        # A valid, non-empty (even if wrong) result must NOT trigger a retry.
        dev, dbs = bird_env
        wrong_but_nonempty = "SELECT name FROM t WHERE id = 2"
        agent = _mock_agent([wrong_but_nonempty], [GOOD_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=True)
        assert results[0]["refined"] is False
        agent.regenerate.assert_not_called()

    def test_genuinely_empty_refine_returns_same_query_kept(self, bird_env, validator):
        # Refiner returns the SAME empty-result query (per its "return unchanged
        # if correct" instruction) → not accepted as "better", no flip-flop.
        dev, dbs = bird_env
        agent = _mock_agent([EMPTY_SQL], [EMPTY_SQL])
        results = run_bird_eval(dev, dbs, agent, validator, refine=True)
        r = results[0]
        assert r["refined"] is False and r["pred_sql"] == EMPTY_SQL
