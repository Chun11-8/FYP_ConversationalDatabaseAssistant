"""
Tests for GH#41 (execution hardening) + GH#38 (latency/cost) + GH#45 step 4.

Covers: statement timeout, read-only query connection, vote cap + early
plurality + winner-row reuse, cache-hit shortcut routing, model routing,
embed prefetch, config DB name, DISTINCT post-pass. All offline.

Run from repo root:
    pytest backend/tests/test_hardening_latency.py -v
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent
from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent


SCHEMA = "Table: t\nColumns:\n  - id: INTEGER [PK]\n  - name: TEXT\n  - grp: TEXT"


def _bare_agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_exec_voting=True, _enforce_limit=False, _vote_max=5,
                    _stmt_timeout_ms=15000, _cache_shortcut=True,
                    _cache_hit_threshold=0.03)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    a.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
    return a


# ── GH#41: statement timeout + read-only connection ───────────────────────────

class TestRunQueryTimeout:

    def test_set_local_issued(self):
        import backend.retrieval.pgvector_store as pg
        conn, cur = MagicMock(), MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []
        pg.run_query(conn, "SELECT 1", timeout_ms=5000)
        first_call = cur.execute.call_args_list[0]
        assert "statement_timeout" in first_call[0][0]
        assert first_call[0][1] == (5000,)
        conn.rollback.assert_called()   # transaction reset after fetch

    def test_no_timeout_no_set_local(self):
        import backend.retrieval.pgvector_store as pg
        conn, cur = MagicMock(), MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []
        pg.run_query(conn, "SELECT 1")
        assert all("statement_timeout" not in c[0][0]
                   for c in cur.execute.call_args_list)


class TestReadOnlyConn:

    def test_app_credentials_used_when_configured(self, monkeypatch):
        import backend.retrieval.pgvector_store as pg
        monkeypatch.setenv("PG_APP_USER", "assistant_ro")
        monkeypatch.setenv("PG_APP_PASSWORD", "ro-pw")
        with patch.object(pg.psycopg2, "connect", return_value=MagicMock()) as mc:
            pg.get_query_conn()
        kwargs = mc.call_args[1]
        assert kwargs["user"] == "assistant_ro" and kwargs["password"] == "ro-pw"

    def test_fallback_to_default_user(self, monkeypatch):
        import backend.retrieval.pgvector_store as pg
        monkeypatch.delenv("PG_APP_USER", raising=False)
        monkeypatch.delenv("PG_APP_PASSWORD", raising=False)
        with patch.object(pg.psycopg2, "connect", return_value=MagicMock()) as mc:
            pg.get_query_conn()
        assert mc.call_args[1]["user"] == pg.DB_CONFIG["user"]


# ── GH#41: vote cap + early plurality; GH#38: winner-row reuse ────────────────

class TestVoteHardening:

    def _run(self, agent, sqls, results_by_sql, executed):
        original = dbm.execute_query

        def fake(sql, **kw):
            executed.append(sql)
            for k, rows in results_by_sql.items():
                if k in sql:
                    return rows
            raise RuntimeError("boom")

        dbm.execute_query = fake
        try:
            return agent._vote_candidates(sqls, SCHEMA)
        finally:
            dbm.execute_query = original

    def test_early_plurality_skips_remaining(self):
        executed: list = []
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp IN ('a')",
                "SELECT name FROM t WHERE grp = 'z'"]
        res = {"grp = 'a'": [{"name": "X"}], "grp IN ('a')": [{"name": "X"}],
               "grp = 'z'": [{"name": "Z"}]}
        winner, rows, run_sql = self._run(_bare_agent(), sqls, res, executed)
        assert winner in sqls[:2]
        assert len(executed) == 2          # third candidate never ran
        assert rows == [{"name": "X"}] and run_sql == winner

    def test_no_early_break_when_lead_is_beatable(self):
        # Ordering [A,B,A,B,B]: a naive "break at 2 votes" returns A (2 votes),
        # but B is the true plurality (3 votes). The lead is only unassailable
        # once no remaining candidate can catch up, so voting must run to the end.
        executed: list = []
        sqls = ["SELECT name FROM t WHERE grp = 'A1'",   # → resA
                "SELECT name FROM t WHERE grp = 'B1'",   # → resB
                "SELECT name FROM t WHERE grp = 'A2'",   # → resA
                "SELECT name FROM t WHERE grp = 'B2'",   # → resB
                "SELECT name FROM t WHERE grp = 'B3'"]   # → resB
        res = {"'A1'": [{"name": "a"}], "'A2'": [{"name": "a"}],
               "'B1'": [{"name": "b"}], "'B2'": [{"name": "b"}], "'B3'": [{"name": "b"}]}
        winner, rows, _ = self._run(_bare_agent(), sqls, res, executed)
        assert rows == [{"name": "b"}]           # majority (3 votes) wins, not A (2)
        assert winner in sqls[1::2]              # one of the B candidates
        assert len(executed) == 5               # ran all — lead was never unassailable early

    def test_vote_max_caps_candidates(self):
        executed: list = []
        sqls = [f"SELECT name FROM t WHERE grp = 'g{i}'" for i in range(5)]
        res = {f"g{i}": [{"name": f"N{i}"}] for i in range(5)}
        _bare = _bare_agent(_vote_max=2)
        self._run(_bare, sqls, res, executed)
        assert len(executed) == 2

    def test_timeout_forwarded(self):
        seen = {}
        original = dbm.execute_query

        def fake(sql, **kw):
            seen.update(kw)
            return [{"name": "X"}]

        dbm.execute_query = fake
        try:
            _bare_agent(_stmt_timeout_ms=7000)._vote_candidates(
                ["SELECT name FROM t"], SCHEMA)
        finally:
            dbm.execute_query = original
        assert seen.get("timeout_ms") == 7000

    def test_execute_sql_reuses_vote_rows(self):
        calls: list = []
        original = dbm.execute_query
        dbm.execute_query = lambda sql, **kw: calls.append(sql) or [{"x": 1}]
        try:
            agent = _bare_agent()
            state = {"best_sql": "SELECT name FROM t",
                     "vote_rows": [{"name": "cached"}],
                     "vote_rows_sql": "SELECT name FROM t"}
            out = agent.execute_sql(state)
        finally:
            dbm.execute_query = original
        assert calls == []                       # no re-execution
        assert out["execution_result"] == [{"name": "cached"}]

    def test_execute_sql_runs_on_sql_mismatch(self):
        calls: list = []
        original = dbm.execute_query
        dbm.execute_query = lambda sql, **kw: calls.append(sql) or [{"x": 1}]
        try:
            agent = _bare_agent()
            state = {"best_sql": "SELECT name FROM t",
                     "vote_rows": [{"name": "stale"}],
                     "vote_rows_sql": "SELECT other FROM t"}
            out = agent.execute_sql(state)
        finally:
            dbm.execute_query = original
        assert len(calls) == 1
        assert out["execution_result"] == [{"x": 1}]


# ── GH#38: cache-hit shortcut routing ─────────────────────────────────────────

class TestCacheShortcut:

    def test_route_after_retrieval_hit(self):
        a = _bare_agent()
        assert a.route_after_retrieval(
            {"cache_hit": True, "best_sql": "SELECT 1"}) == "validate_sql"

    def test_route_after_validation_stale_hit_regenerates(self):
        a = _bare_agent()
        state = {"validation": {"valid": False}, "cache_hit": True,
                 "fallback_level": 0}
        assert a.route_after_validation(state) == "link_schema"

    def test_link_schema_clears_flag(self):
        a = _bare_agent(_schema_linking=False)
        out = a.link_schema({"cache_hit": True, "best_sql": "stale",
                             "user_query": "q", "schema_context": "s",
                             "conversation_history": []})
        assert out["cache_hit"] is False

    def test_best_match_none_when_disabled(self):
        from backend.memory.semantic_cache import SemanticCache
        cache = SemanticCache(db_config=None)
        assert cache.best_match([0.1, 0.2]) is None
        assert cache.best_match(None) is None

    def test_update_cache_skips_resave_on_hit(self):
        a = _bare_agent()
        a.semantic_cache = MagicMock()
        final = {"validation": {"valid": True}, "best_sql": "SELECT 1",
                 "execution_result": [{"x": 1}], "cache_hit": True,
                 "intent": "READ", "response": "ok"}
        a._update_cache("s", "q", final, 1, 10)
        a.semantic_cache.save_pair.assert_not_called()
        a.semantic_cache.log_interaction.assert_called_once()


# ── GH#38: model routing + prefetch + db name ─────────────────────────────────

class TestModelRoutingAndPrefetch:

    def _mk(self, llm_cfg):
        with patch.object(dbm, "ChatOpenAI") as mc, \
             patch.object(dbm, "SQLCodeAgent", MagicMock()), \
             patch.object(dbm, "SQLValidatorAgent", MagicMock()), \
             patch.object(dbm, "SchemaLinkingAgent", MagicMock()), \
             patch.object(dbm, "SessionMemory", MagicMock()), \
             patch.object(dbm, "SemanticCache", MagicMock()), \
             patch.object(dbm, "ResponseOptimizer", MagicMock()), \
             patch.object(dbm, "ToolRegistry", MagicMock()):
            mc.side_effect = lambda **kw: MagicMock(model=kw.get("model"))
            agent = dbm.OrchestrationAgent(
                {"llm": llm_cfg, "db": {"name": "mydb"}})
        return agent

    def test_intent_model_routed(self):
        a = self._mk({"model": "big", "api_key": "k", "intent_model": "cheap"})
        assert a.intent_llm is not a.llm
        assert a.intent_llm.model == "cheap"

    def test_absent_intent_model_reuses_main_llm(self):
        a = self._mk({"model": "big", "api_key": "k"})
        assert a.intent_llm is a.llm
        assert a._format_llm is a.llm

    def test_db_name_from_config(self):
        a = self._mk({"model": "big", "api_key": "k"})
        assert a._db_name == "mydb"

    def test_prefetched_embedding_consumed(self):
        a = _bare_agent(_fk_max_hops=2, _fk_max_expansion=5,
                        _max_schema_tokens=4000, _db_name="d")
        a.semantic_cache = MagicMock()
        a.semantic_cache.retrieve_few_shots.return_value = []
        fut: Future = Future()
        fut.set_result([0.5] * 3)
        dbm._tls.embed_future = fut
        with patch.object(dbm, "retrieve", side_effect=RuntimeError("stop")):
            out = a.retrieve_schema({"user_query": "q",
                                     "conversation_history": [],
                                     "fallback_level": 0})
        assert out["query_embedding"] == [0.5] * 3
        a.semantic_cache.embed.assert_not_called()   # future used, not recomputed


# ── GH#45 step 4: DISTINCT post-pass ──────────────────────────────────────────

class TestDistinctPostpass:

    def _agent(self):
        return SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())

    JOIN_SQL = ("SELECT m.first_name, m.last_name FROM attendance a "
                "JOIN member m ON a.link_to_member = m.member_id")

    def test_injected_for_list_question_over_join(self):
        out = self._agent()._inject_distinct(self.JOIN_SQL, "List the names of students")
        assert "DISTINCT" in out.upper()

    def test_not_injected_without_list(self):
        out = self._agent()._inject_distinct(self.JOIN_SQL, "How many students attended?")
        assert "DISTINCT" not in out.upper()

    def test_not_injected_for_aggregate(self):
        sql = "SELECT COUNT(m.member_id) FROM attendance a JOIN member m ON a.link_to_member = m.member_id"
        out = self._agent()._inject_distinct(sql, "List how many attended")
        assert "DISTINCT" not in out.upper()

    def test_not_injected_with_group_by(self):
        sql = self.JOIN_SQL + " GROUP BY m.member_id"
        assert "DISTINCT" not in self._agent()._inject_distinct(sql, "List names").upper()

    def test_not_injected_without_join(self):
        sql = "SELECT name FROM member"
        assert "DISTINCT" not in self._agent()._inject_distinct(sql, "List names").upper()

    def test_already_distinct_unchanged(self):
        sql = self.JOIN_SQL.replace("SELECT", "SELECT DISTINCT", 1)
        assert self._agent()._inject_distinct(sql, "List names").upper().count("DISTINCT") == 1
