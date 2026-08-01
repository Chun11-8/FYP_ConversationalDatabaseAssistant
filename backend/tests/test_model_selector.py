"""
Tests for GH#52 — per-request model override + the /models allow-list.

Covers:
  - OrchestrationAgent.__init__: `_selectable_models` construction
  - OrchestrationAgent.run(): the `has_override` branch (renamed/extended from
    `use_custom_key`) constructs a per-request ChatOpenAI only when a
    non-default model or an api_key is actually supplied, and always restores
    the singleton defaults afterward.
  - GET /models and model validation on /chat + /chat/stream.

All LLM/DB boundaries are mocked. Runs offline.

Run from repo root:
    pytest backend/tests/test_model_selector.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import backend.agents.db_agent as dbm
import backend.api.main as api_main


# ── OrchestrationAgent construction ────────────────────────────────────────────

@contextmanager
def _mk_agent(llm_cfg: dict, retrieval_cfg: dict | None = None):
    """Construct a real OrchestrationAgent with every dependency mocked,
    mirroring TestModelRoutingAndPrefetch._mk in test_hardening_latency.py.

    A context manager (not a plain helper) because the ChatOpenAI patch must
    stay active across BOTH __init__ (the default self.llm) AND the later
    run() call under test (the per-request override) — a helper that exits
    its `with` block before returning would un-patch ChatOpenAI before run()
    ever executes, silently constructing the REAL client instead.
    """
    with patch.object(dbm, "ChatOpenAI") as mc, \
         patch.object(dbm, "SQLCodeAgent", MagicMock()), \
         patch.object(dbm, "SQLValidatorAgent", MagicMock()), \
         patch.object(dbm, "SchemaLinkingAgent", MagicMock()), \
         patch.object(dbm, "SessionMemory", MagicMock()), \
         patch.object(dbm, "SemanticCache", MagicMock()), \
         patch.object(dbm, "ResponseOptimizer", MagicMock()), \
         patch.object(dbm, "ToolRegistry", MagicMock()):
        mc.side_effect = lambda **kw: MagicMock(model=kw.get("model"), api_key=kw.get("api_key"))
        agent = dbm.OrchestrationAgent({
            "llm": llm_cfg, "db": {"name": "mydb"},
            "retrieval": retrieval_cfg or {},
        })
        yield agent, mc


_TLS_ATTRS = ("llm", "intent_llm", "sql_agent", "linker", "optimizer",
              "stage_cb", "embed_future")


@pytest.fixture(autouse=True)
def _clean_tls():
    """run() is the only thing in this file that populates dbm._tls (every
    other test module only exercises individual node methods on a bare
    agent). Without cleanup, an override test leaves _tls.llm/sql_agent/etc.
    pointing at THIS test's throwaway mocks, which later tests in OTHER
    modules would silently pick up via _get_llm()/_get_sql_agent()'s
    getattr(_tls, ..., self.attr) fallback."""
    yield
    for attr in _TLS_ATTRS:
        if hasattr(dbm._tls, attr):
            delattr(dbm._tls, attr)


_FINAL_STATE = {
    "response": "ok", "best_sql": "", "intent": "READ", "validation": {},
    "execution_result": None, "csv_path": None, "turn_id": None,
    "cache_hit": False, "timings": {}, "error": None,
}


def _run_with_mocked_graph(agent, **run_kwargs):
    """Bypass the real LangGraph execution (irrelevant to the override logic
    under test) while still exercising run()'s own setup/teardown code."""
    agent.graph = MagicMock(invoke=MagicMock(return_value=dict(_FINAL_STATE)))
    return agent.run(session_id="s1", user_query="how many rows?", **run_kwargs)


class TestSelectableModels:

    def test_default_model_always_included(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, _):
            assert agent.selectable_models == ["gpt-4o-mini"]
            assert agent.default_model == "gpt-4o-mini"

    def test_configured_list_used_verbatim(self):
        with _mk_agent({
            "model": "gpt-4o-mini", "api_key": "k",
            "selectable_models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        }) as (agent, _):
            assert agent.selectable_models == ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

    def test_default_prepended_if_missing_from_list(self):
        with _mk_agent({
            "model": "gpt-4o-mini", "api_key": "k",
            "selectable_models": ["gpt-4o", "gpt-4.1-mini"],
        }) as (agent, _):
            assert agent.selectable_models[0] == "gpt-4o-mini"
            assert set(agent.selectable_models) == {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"}

    def test_selectable_models_property_returns_a_copy(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, _):
            agent.selectable_models.append("mutated")
            assert agent.selectable_models == ["gpt-4o-mini"]   # internal list untouched


class TestRunModelOverride:

    def test_no_override_when_model_and_key_absent(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            calls_before = mc.call_count
            _run_with_mocked_graph(agent)
            assert mc.call_count == calls_before   # no extra ChatOpenAI construction

    def test_no_override_when_model_equals_default(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            calls_before = mc.call_count
            _run_with_mocked_graph(agent, model="gpt-4o-mini")
            assert mc.call_count == calls_before

    def test_override_fires_for_non_default_model(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            calls_before = mc.call_count
            _run_with_mocked_graph(agent, model="gpt-4o")
            assert mc.call_count == calls_before + 1
            kwargs = mc.call_args_list[-1].kwargs
            assert kwargs["model"] == "gpt-4o"
            assert kwargs["api_key"] == "k"   # falls back to the default key

    def test_override_fires_for_api_key_alone(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            calls_before = mc.call_count
            _run_with_mocked_graph(agent, api_key="custom-key")
            assert mc.call_count == calls_before + 1
            kwargs = mc.call_args_list[-1].kwargs
            assert kwargs["model"] == "gpt-4o-mini"   # falls back to the default model
            assert kwargs["api_key"] == "custom-key"

    def test_override_combines_model_and_key(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            _run_with_mocked_graph(agent, api_key="custom-key", model="gpt-4o")
            kwargs = mc.call_args_list[-1].kwargs
            assert kwargs["model"] == "gpt-4o"
            assert kwargs["api_key"] == "custom-key"

    def test_teardown_restores_singleton_llm(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, _):
            _run_with_mocked_graph(agent, model="gpt-4o")
            assert dbm._tls.llm is agent.llm
            assert dbm._tls.intent_llm is agent.intent_llm
            assert dbm._tls.sql_agent is agent.sql_agent
            assert dbm._tls.linker is agent.linker
            assert dbm._tls.optimizer is agent.optimizer

    def test_blank_model_string_is_no_override(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, mc):
            calls_before = mc.call_count
            _run_with_mocked_graph(agent, model="   ")
            assert mc.call_count == calls_before


# ── API layer ───────────────────────────────────────────────────────────────────

class TestModelsEndpoint:

    def _client(self) -> TestClient:
        return TestClient(api_main.app)

    def test_returns_configured_list_and_default(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
        fake.default_model = "gpt-4o-mini"
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().get("/models")
        assert r.status_code == 200
        body = r.json()
        assert body["models"] == ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
        assert body["default"] == "gpt-4o-mini"


class TestValidateModel:

    def test_none_passthrough(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        assert api_main._validate_model(None, agent) is None

    def test_empty_string_passthrough(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        assert api_main._validate_model("", agent) is None

    def test_listed_model_returned(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini", "gpt-4o"])
        assert api_main._validate_model("gpt-4o", agent) == "gpt-4o"

    def test_unlisted_model_raises_400(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        with pytest.raises(api_main.HTTPException) as exc_info:
            api_main._validate_model("not-a-real-model", agent)
        assert exc_info.value.status_code == 400


class TestChatEndpointModelValidation:

    def _client(self) -> TestClient:
        return TestClient(api_main.app)

    def test_chat_rejects_unlisted_model(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini"]
        fake.default_model = "gpt-4o-mini"
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat", json={
                "session_id": "s1", "message": "hi", "model": "bogus-model",
            })
        assert r.status_code == 400
        fake.run.assert_not_called()

    def test_chat_forwards_valid_model(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini", "gpt-4o"]
        fake.default_model = "gpt-4o-mini"
        fake.run.return_value = {
            "response": "ok", "sql": None, "intent": "READ", "validation": {},
            "row_count": None, "rows": None, "csv_path": None, "turn_id": None,
            "cache_hit": False, "latency_ms": 1, "timings": {},
        }
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat", json={
                "session_id": "s1", "message": "hi", "model": "gpt-4o",
            })
        assert r.status_code == 200
        assert fake.run.call_args.kwargs["model"] == "gpt-4o"

    def test_chat_stream_rejects_unlisted_model_before_thread_starts(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini"]
        fake.default_model = "gpt-4o-mini"
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat/stream", json={
                "session_id": "s1", "message": "hi", "model": "bogus-model",
            })
        assert r.status_code == 400
        fake.run.assert_not_called()
