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
import backend.agents.llm_factory as llm_factory
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

    Patched on llm_factory, not db_agent: since GH#65 every query-stage client
    is built by llm_factory.build_chat_llm(), so that module is the real (and
    only) construction point. db_agent still imports ChatOpenAI, but purely as
    a type annotation.
    """
    with patch.object(llm_factory, "ChatOpenAI") as mc, \
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

    def test_audit_log_records_the_overridden_model_not_the_default(self):
        """Self-review finding: query_interactions.model must match whatever
        model this request's tokens/cost were actually billed against."""
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, _):
            _run_with_mocked_graph(agent, model="gpt-4o")
            kwargs = agent.semantic_cache.log_interaction.call_args.kwargs
            assert kwargs["model"] == "gpt-4o"

    def test_audit_log_records_the_default_model_when_no_override(self):
        with _mk_agent({"model": "gpt-4o-mini", "api_key": "k"}) as (agent, _):
            _run_with_mocked_graph(agent)
            kwargs = agent.semantic_cache.log_interaction.call_args.kwargs
            assert kwargs["model"] == "gpt-4o-mini"


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

    # GH#58: a well-formed model id outside `selectable_models` is now
    # accepted — the frontend's picker offers free-text "Custom…" entry, and
    # this is a shape check (reject garbage), not an allow-list (restrict
    # models). Matches backend/eval/bird.py::resolve_model's warn-not-block
    # posture for the exact same reason.
    def test_unlisted_but_well_formed_model_is_accepted(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        assert api_main._validate_model("gpt-4.1", agent) == "gpt-4.1"

    def test_unlisted_fine_tuned_model_id_is_accepted(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        ft_id = "ft:gpt-4o-mini-2024-07-18:my-org::abc123"
        assert api_main._validate_model(ft_id, agent) == ft_id

    def test_whitespace_only_model_raises_400(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        with pytest.raises(api_main.HTTPException) as exc_info:
            api_main._validate_model("   ", agent)
        assert exc_info.value.status_code == 400

    def test_malformed_model_raises_400(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        with pytest.raises(api_main.HTTPException) as exc_info:
            api_main._validate_model("gpt-4o; DROP TABLE users;--", agent)
        assert exc_info.value.status_code == 400

    def test_single_char_model_raises_400(self):
        # Below _MODEL_ID_RE's minimum length — also guards against a
        # one-character typo silently reaching OpenAI as a "real" model id.
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        with pytest.raises(api_main.HTTPException) as exc_info:
            api_main._validate_model("x", agent)
        assert exc_info.value.status_code == 400

    def test_overlong_model_raises_400(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        with pytest.raises(api_main.HTTPException) as exc_info:
            api_main._validate_model("g" * 100, agent)
        assert exc_info.value.status_code == 400

    def test_model_is_stripped(self):
        agent = MagicMock(selectable_models=["gpt-4o-mini"])
        assert api_main._validate_model("  gpt-4o-mini  ", agent) == "gpt-4o-mini"


class TestChatEndpointModelValidation:

    def _client(self) -> TestClient:
        return TestClient(api_main.app)

    def test_chat_accepts_unlisted_well_formed_model(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini"]
        fake.default_model = "gpt-4o-mini"
        fake.run.return_value = {
            "response": "ok", "sql": None, "intent": "GENERAL", "validation": {},
            "row_count": None, "rows": None, "csv_path": None, "turn_id": 1,
            "cache_hit": False, "latency_ms": 1, "timings": {}, "usage": {},
        }
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat", json={
                "session_id": "s1", "message": "hi", "model": "gpt-4.1",
            })
        assert r.status_code == 200
        fake.run.assert_called_once()

    def test_chat_rejects_malformed_model(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini"]
        fake.default_model = "gpt-4o-mini"
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat", json={
                "session_id": "s1", "message": "hi", "model": "bogus model!!",
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

    def test_chat_stream_rejects_malformed_model_before_thread_starts(self):
        fake = MagicMock()
        fake.selectable_models = ["gpt-4o-mini"]
        fake.default_model = "gpt-4o-mini"
        with patch.object(api_main, "_get_agent", return_value=fake):
            r = self._client().post("/chat/stream", json={
                "session_id": "s1", "message": "hi", "model": "bogus model!!",
            })
        assert r.status_code == 400
        fake.run.assert_not_called()
