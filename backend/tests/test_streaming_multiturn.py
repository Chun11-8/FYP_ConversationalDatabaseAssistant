"""
Tests for GH#39 (SSE streaming) and GH#40 (multi-turn effectiveness).

Covers: stage-event wrapper + run(stage_callback), the /chat/stream endpoint
contract, anaphoric-follow-up detection + condensation, standalone-query
consumption in retrieval, previous-SQL prompt context, and proactive
clarification for AMBIGUOUS intent. All offline.

Run from repo root:
    pytest backend/tests/test_streaming_multiturn.py -v
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import backend.agents.db_agent as dbm
import backend.api.main as api_main
from backend.agents.db_agent import OrchestrationAgent, _staged
from backend.agents.sql_code_agent import SQLCodeAgent


def _bare(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


# ── GH#39: stage events ───────────────────────────────────────────────────────

class TestStageEvents:

    def test_wrapper_emits_stage_and_sql(self):
        events = []
        dbm._tls.stage_cb = lambda t, p: events.append((t, p))
        try:
            node = _staged("generate_sql", lambda s: {"best_sql": "SELECT 1"})
            node({})
        finally:
            dbm._tls.stage_cb = None
        assert ("stage", "generate_sql") in events
        assert ("sql", "SELECT 1") in events

    def test_wrapper_silent_without_callback(self):
        dbm._tls.stage_cb = None
        node = _staged("validate_sql", lambda s: {"ok": True})
        out = node({})
        assert out["ok"] is True and "validate_sql" in out["timings"]   # no exception

    def test_callback_errors_never_break_pipeline(self):
        def boom(t, p):
            raise RuntimeError("subscriber died")
        dbm._tls.stage_cb = boom
        try:
            out = _staged("execute_sql", lambda s: {"r": 1})({})
        finally:
            dbm._tls.stage_cb = None
        assert out["r"] == 1   # pipeline unbroken; timings added is fine

    def test_run_accepts_stage_callback(self):
        assert "stage_callback" in inspect.signature(OrchestrationAgent.run).parameters


class TestStreamEndpoint:

    @pytest.fixture()
    def client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("PG_PASSWORD", "pw")
        mock_agent = MagicMock()

        def fake_run(session_id, user_query, api_key=None, model=None, stage_callback=None):
            stage_callback("stage", "classify_intent")
            stage_callback("stage", "generate_sql")
            stage_callback("sql", "SELECT 1")
            return {"response": "ok", "sql": "SELECT 1", "intent": "READ",
                    "validation": {}, "row_count": 1, "rows": [{"x": 1}],
                    "csv_path": None, "turn_id": 7, "cache_hit": False}

        mock_agent.run.side_effect = fake_run
        with patch.object(api_main, "_agent", mock_agent):
            yield TestClient(api_main.app)

    def test_stream_event_sequence_and_done(self, client):
        resp = client.post("/chat/stream", json={"session_id": "s", "message": "q"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [json.loads(line[6:]) for line in resp.text.split("\n")
                  if line.startswith("data: ")]
        types = [e["type"] for e in events]
        assert types[:3] == ["stage", "stage", "sql"]
        assert types[-1] == "done"
        done = events[-1]
        assert done["response"] == "ok" and done["turn_id"] == 7

    def test_stream_error_event_terminal(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("kaput")
        with patch.object(api_main, "_agent", mock_agent):
            resp = TestClient(api_main.app).post(
                "/chat/stream", json={"session_id": "s", "message": "q"})
        events = [json.loads(l[6:]) for l in resp.text.split("\n") if l.startswith("data: ")]
        assert events[-1]["type"] == "error"
        assert "kaput" in events[-1]["message"]


# ── GH#40: follow-up condensation ─────────────────────────────────────────────

class TestCondense:

    HIST = [{"role": "user", "content": "show 5 players"},
            {"role": "assistant", "content": "here"}]

    def test_anaphoric_detection(self):
        f = OrchestrationAgent._looks_anaphoric
        assert f("what about 2013?", self.HIST)
        assert f("only the top 3", self.HIST)
        assert f("and for SME customers", self.HIST)
        assert not f("what about 2013?", [])               # no history → no condense
        assert not f("List every transaction made by customer 38508 in Prague "
                     "during 2013 grouped by month", self.HIST)  # standalone

    def _agent(self, condense=True):
        a = _bare(_condense=condense)
        a.intent_llm = MagicMock()
        a.intent_llm.invoke.return_value = MagicMock(
            content="Show only the top 3 players by rating")
        dbm._tls.__dict__.clear()
        return a

    def test_condense_sets_standalone(self):
        a = self._agent()
        out = a.condense_query({"user_query": "only the top 3",
                                "conversation_history": self.HIST,
                                "previous_sql": "SELECT name FROM players LIMIT 5"})
        assert out["standalone_query"] == "Show only the top 3 players by rating"
        # previous SQL is offered as context to the condenser
        sent = a.intent_llm.invoke.call_args[0][0][1].content
        assert "SELECT name FROM players LIMIT 5" in sent

    def test_standalone_question_skips_llm(self):
        a = self._agent()
        state = {"user_query": "List every transaction made by customer 38508 "
                               "in Prague during 2013 grouped by month",
                 "conversation_history": self.HIST}
        out = a.condense_query(state)
        assert "standalone_query" not in out or out.get("standalone_query") is None
        a.intent_llm.invoke.assert_not_called()

    def test_config_off_skips(self):
        a = self._agent(condense=False)
        out = a.condense_query({"user_query": "only the top 3",
                                "conversation_history": self.HIST})
        a.intent_llm.invoke.assert_not_called()
        assert out.get("standalone_query") is None

    def test_llm_failure_degrades(self):
        a = self._agent()
        a.intent_llm.invoke.side_effect = RuntimeError("down")
        out = a.condense_query({"user_query": "only the top 3",
                                "conversation_history": self.HIST})
        assert out.get("standalone_query") is None

    def test_retrieval_uses_standalone_query(self):
        a = _bare(_fk_max_hops=2, _fk_max_expansion=5, _max_schema_tokens=4000,
                  _db_name="d", _cache_shortcut=False, _cache_hit_threshold=0.03)
        a.semantic_cache = MagicMock()
        a.semantic_cache.embed.return_value = [0.1]
        a.semantic_cache.retrieve_few_shots.return_value = []
        dbm._tls.__dict__.clear()
        captured = {}
        def fake_retrieve(q, *args, **kw):
            captured["q"] = q
            raise RuntimeError("stop here")
        with patch.object(dbm, "retrieve", side_effect=fake_retrieve):
            a.retrieve_schema({"user_query": "only the top 3",
                               "standalone_query": "Show the top 3 players by rating",
                               "conversation_history": [], "fallback_level": 0})
        assert captured["q"] == "Show the top 3 players by rating"
        # embedding computed on the standalone text too
        a.semantic_cache.embed.assert_called_with("Show the top 3 players by rating")


# ── GH#40: previous-SQL context + clarification ───────────────────────────────

class TestPreviousSqlAndClarify:

    def test_previous_sql_block_in_generation_prompt(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="only top 3", schema_context="s", few_shots=["x"],
                       history=[], previous_sql="SELECT name FROM players LIMIT 5")
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "<previous_sql>" in sent
        assert "SELECT name FROM players LIMIT 5" in sent

    def test_no_previous_sql_no_block(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="q", schema_context="s", few_shots=["x"], history=[])
        assert "<previous_sql>" not in agent.llm.invoke.call_args[0][0][1].content

    def test_route_ambiguous_to_clarification(self):
        a = _bare(_clarify=True)
        assert a.route_by_intent({"intent": "AMBIGUOUS"}) == "ask_clarification"

    def test_route_ambiguous_config_off_keeps_old_path(self):
        a = _bare(_clarify=False)
        assert a.route_by_intent({"intent": "AMBIGUOUS"}) == "retrieve_schema"

    def test_clarification_node_names_candidate_tables(self):
        a = _bare(_db_name="d")
        a.memory = MagicMock()
        a.memory.add.return_value = 42
        with patch.object(dbm, "retrieve", return_value={
                "tables": [{"table_name": "players"}, {"table_name": "teams"}]}):
            out = a.ask_clarification({"session_id": "s", "user_query": "show stuff"})
        assert "?" in out["response"]
        assert "players" in out["response"]
        assert out["turn_id"] == 42

    def test_clarification_survives_retrieval_failure(self):
        a = _bare(_db_name="d")
        a.memory = MagicMock()
        a.memory.add.return_value = 1
        with patch.object(dbm, "retrieve", side_effect=RuntimeError("no store")):
            out = a.ask_clarification({"session_id": "s", "user_query": "??"})
        assert out["response"]   # still asks a question

    def test_graph_contains_new_nodes(self):
        with patch.object(dbm, "ChatOpenAI", MagicMock()), \
             patch.object(dbm, "SQLCodeAgent", MagicMock()), \
             patch.object(dbm, "SQLValidatorAgent", MagicMock()), \
             patch.object(dbm, "SchemaLinkingAgent", MagicMock()), \
             patch.object(dbm, "SessionMemory", MagicMock()), \
             patch.object(dbm, "SemanticCache", MagicMock()), \
             patch.object(dbm, "ResponseOptimizer", MagicMock()), \
             patch.object(dbm, "ToolRegistry", MagicMock()):
            agent = dbm.OrchestrationAgent({"llm": {"model": "x", "api_key": "k"}, "db": {}})
        nodes = agent.graph.get_graph().nodes
        assert "condense_query" in nodes and "ask_clarification" in nodes
