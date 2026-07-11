"""
Tests for the remaining architecture build:
  1. Plan-conformance demotion (SQLCodeAgent._plan_penalty / _order_candidates)
  2. Live-pipeline execution voting (OrchestrationAgent._vote_candidates)
  3. Multi-hop FK expansion with cap + token budget (indexing.expand_fk_tables)
  4. Strict semantic validation with CTE/alias tolerance (SQLValidatorAgent)

All LLM/DB calls are mocked. Runs offline.

Run from repo root:
    pytest backend/tests/test_power_features.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent


def _sql_agent() -> SQLCodeAgent:
    return SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())


# ── 1. Plan-conformance demotion ──────────────────────────────────────────────

PLAN = {
    "tables": ["event", "attendance", "member"],
    "join_path": [
        "event.event_id = attendance.link_to_event",
        "attendance.link_to_member = member.member_id",
    ],
}

CONFORMING = (
    "SELECT COUNT(m.member_id) FROM event e "
    "JOIN attendance a ON e.event_id = a.link_to_event "
    "JOIN member m ON a.link_to_member = m.member_id "
    "WHERE m.t_shirt_size = 'Medium'"
)
MISSING_TABLE = (
    "SELECT COUNT(*) FROM attendance a "
    "JOIN member m ON a.link_to_member = m.member_id"
)
WRONG_JOIN = (
    "SELECT COUNT(m.member_id) FROM event e "
    "JOIN attendance a ON e.event_id = a.link_to_member "  # wrong FK column
    "JOIN member m ON a.link_to_member = m.member_id"
)


class TestPlanPenalty:

    def test_conforming_candidate_zero_penalty(self):
        assert _sql_agent()._plan_penalty(CONFORMING, PLAN) == 0

    def test_alias_resolution_conforms(self):
        # Aliases (e/a/m) must resolve to real table names before comparison.
        assert _sql_agent()._plan_penalty(CONFORMING, PLAN) == 0

    def test_missing_table_penalized(self):
        assert _sql_agent()._plan_penalty(MISSING_TABLE, PLAN) >= 1

    def test_missing_plan_join_penalized(self):
        assert _sql_agent()._plan_penalty(WRONG_JOIN, PLAN) >= 1

    def test_no_plan_no_penalty(self):
        assert _sql_agent()._plan_penalty(MISSING_TABLE, None) == 0
        assert _sql_agent()._plan_penalty(MISSING_TABLE, {}) == 0

    def test_unparseable_sql_no_penalty(self):
        assert _sql_agent()._plan_penalty("SELECT broken FROM (", PLAN) == 0

    def test_order_candidates_puts_conforming_first(self):
        agent = _sql_agent()
        ordered = agent._order_candidates([WRONG_JOIN, MISSING_TABLE, CONFORMING], "", PLAN)
        assert ordered[0] == CONFORMING

    def test_generate_orders_by_plan(self):
        agent = _sql_agent()
        agent.llm.invoke.return_value = MagicMock(
            content=f"```sql\n{MISSING_TABLE}\n```\n```sql\n{CONFORMING}\n```"
        )
        sqls = agent.generate(query="q", schema_context="s", few_shots=["x"],
                              history=[], query_plan=PLAN)
        # normalize_dialect may reformat the SQL; assert by content: the
        # conforming candidate (uses `event` + the t_shirt_size filter) is first.
        assert "event" in sqls[0] and "t_shirt_size" in sqls[0]


# ── 2. Live-pipeline execution voting ─────────────────────────────────────────

class TestLiveVoting:

    SCHEMA = "Table: t\nColumns:\n  - id: INTEGER [PK]\n  - name: TEXT\n  - grp: TEXT"

    def _agent(self, results_by_sql, enforce_limit=False):
        """OrchestrationAgent with mocked execute_query keyed by SQL text."""
        import backend.agents.db_agent as m
        agent = m.OrchestrationAgent.__new__(m.OrchestrationAgent)
        agent._exec_voting = True
        agent._enforce_limit = enforce_limit
        agent.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
        return agent, m

    def _run(self, agent, module, sqls, results_by_sql):
        original = module.execute_query

        def fake_execute(sql, **kw):
            for key, rows in results_by_sql.items():
                if key in sql:
                    return rows
            raise RuntimeError(f"no result stubbed for: {sql}")

        module.execute_query = fake_execute
        try:
            winner, _rows, _rsql = agent._vote_candidates(sqls, self.SCHEMA)
            return winner
        finally:
            module.execute_query = original

    def test_plurality_wins(self):
        sqls = [
            "SELECT name FROM t WHERE grp = 'a'",
            "SELECT name FROM t WHERE grp IN ('a')",
            "SELECT name FROM t",
        ]
        results = {
            "grp = 'a'": [{"name": "X"}],
            "grp IN ('a')": [{"name": "X"}],       # same result → same group
            "SELECT name FROM t": [{"name": "X"}, {"name": "Y"}],
        }
        agent, m = self._agent(results)
        winner = self._run(agent, m, sqls, results)
        assert winner in sqls[:2]

    def test_all_error_returns_empty(self):
        sqls = ["SELECT name FROM t WHERE grp = 'a'"]
        agent, m = self._agent({})
        assert self._run(agent, m, sqls, {}) == ""

    def test_invalid_candidate_never_executed(self):
        executed = []
        import backend.agents.db_agent as m
        agent = m.OrchestrationAgent.__new__(m.OrchestrationAgent)
        agent._exec_voting = True
        agent._enforce_limit = False
        agent.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
        original = m.execute_query
        m.execute_query = lambda sql, **kw: executed.append(sql) or [{"name": "X"}]
        try:
            winner, _r, _s = agent._vote_candidates(
                ["DELETE FROM t", "SELECT name FROM t"], self.SCHEMA
            )
        finally:
            m.execute_query = original
        assert winner == "SELECT name FROM t"
        assert all("DELETE" not in s for s in executed)

    def test_voting_applies_safety_limit(self):
        executed = []
        import backend.agents.db_agent as m
        agent = m.OrchestrationAgent.__new__(m.OrchestrationAgent)
        agent._exec_voting = True
        agent._enforce_limit = True
        agent.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
        original = m.execute_query
        m.execute_query = lambda sql, **kw: executed.append(sql) or [{"name": "X"}]
        try:
            agent._vote_candidates(["SELECT name FROM t"], self.SCHEMA)
        finally:
            m.execute_query = original
        assert "LIMIT 100" in executed[0]


# ── 3. Multi-hop FK expansion: 2 hops, cap, token budget ─────────────────────

class TestFkExpansionUpgrades:

    def _patch(self, monkeypatch, edges, chunks):
        import backend.retrieval.pgvector_store as pgstore
        graph: dict = {}
        for e in edges:
            graph.setdefault(e["from_table"], set()).add(e["to_table"])
            graph.setdefault(e["to_table"], set()).add(e["from_table"])
        monkeypatch.setattr(pgstore, "get_store", lambda *a, **kw: object())
        monkeypatch.setattr(pgstore, "get_fk_graph", lambda conn: (graph, edges))
        monkeypatch.setattr(
            pgstore, "get_chunks_by_tables",
            lambda conn, names, db: [c for c in chunks if c["table_name"] in names],
        )

    def _result_in(self, names):
        tables = [{"table_name": n, "text": f"Table: {n}"} for n in names]
        return {"tables": tables, "knowledge_base": [],
                "prompt_context": "\n".join(t["text"] for t in tables)}

    def test_two_hops_reaches_grandchild(self, monkeypatch):
        # players → player_stats → teams: retrieving only 'players' must now
        # pull in 'teams' via the 2-hop default.
        edges = [
            {"from_table": "player_stats", "from_col": "player_id",
             "to_table": "players", "to_col": "player_id"},
            {"from_table": "player_stats", "from_col": "team_id",
             "to_table": "teams", "to_col": "team_id"},
        ]
        chunks = [
            {"table_name": "player_stats", "content": "Table: player_stats"},
            {"table_name": "teams", "content": "Table: teams"},
        ]
        self._patch(monkeypatch, edges, chunks)
        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._result_in(["players"]), "FYP")
        names = {c["table_name"] for c in result["fk_tables"]}
        assert names == {"player_stats", "teams"}

    def test_expansion_cap_respected(self, monkeypatch):
        # Star schema: hub connects to 8 spokes → only max_expansion added.
        edges = [
            {"from_table": "hub", "from_col": f"s{i}_id",
             "to_table": f"spoke{i}", "to_col": "id"}
            for i in range(8)
        ]
        chunks = [{"table_name": f"spoke{i}", "content": f"Table: spoke{i}"}
                  for i in range(8)]
        self._patch(monkeypatch, edges, chunks)
        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._result_in(["hub"]), "FYP", max_expansion=5)
        assert len(result["fk_tables"]) == 5

    def test_one_hop_regression_guard(self, monkeypatch):
        edges = [
            {"from_table": "player_stats", "from_col": "player_id",
             "to_table": "players", "to_col": "player_id"},
            {"from_table": "player_stats", "from_col": "team_id",
             "to_table": "teams", "to_col": "team_id"},
        ]
        chunks = [
            {"table_name": "player_stats", "content": "Table: player_stats"},
            {"table_name": "teams", "content": "Table: teams"},
        ]
        self._patch(monkeypatch, edges, chunks)
        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._result_in(["players"]), "FYP", max_hops=1)
        names = {c["table_name"] for c in result["fk_tables"]}
        assert names == {"player_stats"}  # 2-hop 'teams' NOT included

    def test_token_budget_trims_fk_first_keeps_primary(self, monkeypatch):
        big = "x" * 8000  # ~2000 tokens each
        edges = [
            {"from_table": "players", "from_col": "a_id", "to_table": "a", "to_col": "id"},
            {"from_table": "players", "from_col": "b_id", "to_table": "b", "to_col": "id"},
        ]
        chunks = [
            {"table_name": "a", "content": f"Table: a\n{big}"},
            {"table_name": "b", "content": f"Table: b\n{big}"},
        ]
        self._patch(monkeypatch, edges, chunks)
        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(
            self._result_in(["players"]), "FYP", max_schema_tokens=2500
        )
        # At least one FK table dropped, primary always present.
        assert len(result["fk_tables"]) < 2
        assert "players" in result["prompt_context"]

    def test_under_budget_unchanged(self, monkeypatch):
        edges = [{"from_table": "players", "from_col": "a_id",
                  "to_table": "a", "to_col": "id"}]
        chunks = [{"table_name": "a", "content": "Table: a\nsmall"}]
        self._patch(monkeypatch, edges, chunks)
        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._result_in(["players"]), "FYP")
        assert len(result["fk_tables"]) == 1


# ── 4. Strict semantic validation: CTE/alias tolerance ───────────────────────

class TestStrictSemantic:

    @pytest.fixture()
    def validator(self):
        return SQLValidatorAgent({"db": {"dialect": "postgres"}})

    SCHEMA = "Table: players\nColumns:\n  - player_id: INTEGER [PK]\n  - rating: REAL"

    def test_unknown_column_blocked(self, validator):
        result = validator.validate(
            "SELECT player_api_id FROM players LIMIT 5", self.SCHEMA
        )
        assert result["valid"] is False
        assert "player_api_id" in result["error"]

    def test_cte_and_alias_not_flagged(self, validator):
        sql = (
            "WITH avg_rating AS ("
            "  SELECT player_id, AVG(rating) AS mean_r FROM players GROUP BY player_id"
            ") SELECT player_id, mean_r FROM avg_rating ORDER BY mean_r DESC LIMIT 5"
        )
        result = validator.validate(sql, self.SCHEMA)
        assert result["semantic_ok"] is True
        assert result["valid"] is True

    def test_table_alias_qualifier_not_flagged(self, validator):
        sql = "SELECT p.player_id, p.rating FROM players p LIMIT 5"
        result = validator.validate(sql, self.SCHEMA)
        assert result["valid"] is True

    def test_valid_columns_pass(self, validator):
        result = validator.validate(
            "SELECT player_id, rating FROM players LIMIT 5", self.SCHEMA
        )
        assert result["valid"] is True
