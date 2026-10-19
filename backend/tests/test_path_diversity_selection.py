"""
Tests: reasoning-path-diverse candidate generation and
unit-test-screen + pairwise selection on inconclusive votes.
All offline: LLM and DB boundaries are mocked.

Run from repo root:
    pytest backend/tests/test_path_diversity_selection.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent
from backend.agents.sql_code_agent import SQLCodeAgent, PATH_STRATEGIES
from backend.agents.sql_validator_agent import SQLValidatorAgent


SCHEMA = "Table: t\nColumns:\n  - id: INTEGER [PK]\n  - name: TEXT\n  - grp: TEXT"


# ── path-diverse generation ───────────────────────────────────────────────────

def _sql_agent(**cfg) -> SQLCodeAgent:
    config = {"db": {"dialect": "postgres"}, "sql": cfg}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="```sql\nSELECT name FROM t LIMIT 5\n```")
    return SQLCodeAgent(config, llm)


class TestPathDiversity:

    def test_moderate_tier_fans_out_three_calls(self):
        agent = _sql_agent(path_diversity=True)
        sqls = agent.generate("q", SCHEMA, [], [], tier="moderate")
        # one direct call + one call per PATH_STRATEGIES entry
        assert agent.llm.invoke.call_count == 1 + len(PATH_STRATEGIES)
        assert sqls  # candidates extracted from every call

    def test_prompts_carry_distinct_strategy_blocks(self):
        agent = _sql_agent(path_diversity=True)
        agent.generate("q", SCHEMA, [], [], tier="challenging")
        prompts = [c.args[0][1].content for c in agent.llm.invoke.call_args_list]
        assert any("DIVIDE & CONQUER" in p for p in prompts)
        assert any("QUERY EXECUTION PLAN" in p for p in prompts)
        # direct prompt keeps the tier block instead
        assert any("CHALLENGING tier" in p for p in prompts)

    def test_simple_tier_stays_single_call(self):
        agent = _sql_agent(path_diversity=True)
        agent.generate("q", SCHEMA, [], [], tier="simple")
        assert agent.llm.invoke.call_count == 1

    def test_config_off_restores_single_call(self):
        agent = _sql_agent(path_diversity=False)
        agent.generate("q", SCHEMA, [], [], tier="moderate")
        assert agent.llm.invoke.call_count == 1

    def test_no_tier_stays_single_call(self):
        agent = _sql_agent(path_diversity=True)
        agent.generate("q", SCHEMA, [], [])
        assert agent.llm.invoke.call_count == 1

    def test_path_call_failure_degrades_gracefully(self):
        agent = _sql_agent(path_diversity=True)
        good = MagicMock(content="```sql\nSELECT name FROM t\n```")
        calls = {"n": 0}

        def flaky(messages):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("path call down")
            return good

        agent.llm.invoke.side_effect = flaky
        sqls = agent.generate("q", SCHEMA, [], [], tier="moderate")
        assert sqls == ["SELECT name FROM t"]   # direct candidates survive


# ── unit-test screen + pairwise selection ─────────────────────────────────────

def _bare_agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_exec_voting=True, _enforce_limit=False, _vote_max=5,
                    _stmt_timeout_ms=15000, _selection="pairwise",
                    _unit_tests=True)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    a.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
    return a


def _run_vote(agent, sqls, results_by_sql, query=None):
    original = dbm.execute_query

    def fake(sql, **kw):
        for k, rows in results_by_sql.items():
            if k in sql:
                return rows
        raise RuntimeError("boom")

    dbm.execute_query = fake
    try:
        return agent._vote_candidates(sqls, SCHEMA, query=query)
    finally:
        dbm.execute_query = original


# three candidates, three distinct non-empty result sets → inconclusive tie
TIE_SQLS = ["SELECT name FROM t WHERE grp = 'a'",
            "SELECT name FROM t WHERE grp = 'b'",
            "SELECT name FROM t WHERE grp = 'c'"]
TIE_RES = {"'a'": [{"name": "A"}], "'b'": [{"name": "B"}], "'c'": [{"name": "C"}]}


class TestPairwiseSelection:

    def test_no_query_keeps_reranker_fallback(self):
        winner, rows, _ = _run_vote(_bare_agent(), TIE_SQLS, TIE_RES)
        assert winner == "" and rows is None

    def test_selection_reranker_mode_skips_pairwise(self):
        a = _bare_agent(_selection="reranker")
        a.llm = MagicMock()          # must never be consulted
        winner, _, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert winner == "" and a.llm.invoke.call_count == 0

    def test_unit_test_screen_single_survivor_short_circuits(self):
        a = _bare_agent()
        a.llm = MagicMock()
        a.llm.invoke.return_value = MagicMock(content="PASS: 2")
        winner, rows, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert rows == [{"name": "B"}]
        assert a.llm.invoke.call_count == 1      # screen only, no pairwise

    def test_pairwise_round_robin_picks_majority(self):
        a = _bare_agent(_unit_tests=False)
        a.llm = MagicMock()
        # 3 groups → 3 pairwise calls; candidate shown as "1" always wins,
        # so the first group accumulates 2 wins.
        a.llm.invoke.return_value = MagicMock(content="1")
        winner, rows, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert rows == [{"name": "A"}]
        assert a.llm.invoke.call_count == 3

    def test_screen_then_pairwise_between_survivors(self):
        a = _bare_agent()
        a.llm = MagicMock()
        a.llm.invoke.side_effect = [
            MagicMock(content="PASS: 1, 3"),     # screen keeps A and C
            MagicMock(content="2"),              # pairwise: second shown wins → C
        ]
        winner, rows, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert rows == [{"name": "C"}]

    def test_screen_garbage_output_keeps_all_groups(self):
        a = _bare_agent()
        a.llm = MagicMock()
        a.llm.invoke.side_effect = [
            MagicMock(content="I cannot decide"),          # screen unusable
            MagicMock(content="1"), MagicMock(content="1"), MagicMock(content="1"),
        ]
        winner, rows, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert rows == [{"name": "A"}]           # pairwise still resolves

    def test_llm_failure_falls_back_to_reranker(self):
        a = _bare_agent(_unit_tests=False)
        a.llm = MagicMock()
        a.llm.invoke.side_effect = RuntimeError("LLM down")
        winner, rows, _ = _run_vote(a, TIE_SQLS, TIE_RES, query="which name?")
        assert winner == "" and rows is None

    def test_conclusive_plurality_never_consults_llm(self):
        a = _bare_agent()
        a.llm = MagicMock()
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp IN ('a')"]
        res = {"grp = 'a'": [{"name": "X"}], "grp IN ('a')": [{"name": "X"}]}
        winner, rows, _ = _run_vote(a, sqls, res, query="which name?")
        assert rows == [{"name": "X"}]
        assert a.llm.invoke.call_count == 0

    def test_no_reexecution_during_selection(self):
        a = _bare_agent(_unit_tests=False)
        a.llm = MagicMock()
        a.llm.invoke.return_value = MagicMock(content="1")
        original = dbm.execute_query
        calls: list = []

        def fake(sql, **kw):
            calls.append(sql)
            for k, rows in TIE_RES.items():
                if k in sql:
                    return rows
            raise RuntimeError("boom")

        dbm.execute_query = fake
        try:
            a._vote_candidates(TIE_SQLS, SCHEMA, query="which name?")
        finally:
            dbm.execute_query = original
        assert len(calls) == 3       # one execution per candidate, none extra
