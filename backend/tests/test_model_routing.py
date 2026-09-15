"""
Tests for GH#108 — per-model provider routing.

The governing constraint: adding OpenRouter models must leave the OpenAI path
BYTE-IDENTICAL. Several tests exist only to hold that line, because the whole
point of routing per model rather than per process is that the existing
architecture keeps working untouched.

All offline: no network, no provider, no key.

Run from repo root:
    pytest backend/tests/test_model_routing.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import backend.agents.llm_factory as lf
from backend.agents.llm_factory import (
    build_for_target, resolve_model_target,
)

OPENROUTER = "https://openrouter.ai/api/v1"

CONFIG = {
    "llm": {
        "api_key": "sk-openai",
        "selectable_models": [
            "gpt-4o-mini",                                    # bare id
            {"id": "google/gemini-3.7-flash", "provider": "openrouter"},
            {"id": "openai/gpt-5.6-luna", "provider": "openrouter"},
            {"id": "qwen/qwen3.5-122b-a10b", "provider": "openrouter"},
            {"id": "meta-llama/llama-4-maverick", "provider": "openrouter"},
            {"id": "openai/gpt-5.4-mini", "provider": "openrouter"},
            {"id": "x/reasoner", "provider": "openrouter", "temperature": None},
            {"id": "y/warm", "provider": "openrouter", "temperature": 0.7},
            {"id": "z/unknown-provider", "provider": "nope"},
        ],
        "providers": {
            "openrouter": {"base_url": OPENROUTER,
                           "api_key_env": "OPENROUTER_API_KEY"},
        },
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")


# ── the OpenAI path must not move ─────────────────────────────────────────────

class TestOpenAIPathUnchanged:

    def test_a_bare_id_reaches_openai_directly(self):
        t = resolve_model_target("gpt-4o-mini", CONFIG)
        assert t.base_url is None
        assert t.api_key == "sk-openai"
        assert t.provider == "openai"

    def test_the_client_is_byte_identical_to_the_old_construction(self):
        """No base_url kwarg at all — not base_url=None, which some clients
        treat differently from an omitted parameter."""
        with patch.object(lf, "ChatOpenAI") as mc:
            build_for_target(resolve_model_target("gpt-4o-mini", CONFIG))
        assert mc.call_args.kwargs == {"model": "gpt-4o-mini",
                                       "temperature": 0, "api_key": "sk-openai"}

    def test_an_unconfigured_model_is_not_routed_anywhere(self):
        """A custom id typed into the picker must not inherit a provider."""
        t = resolve_model_target("some-new-openai-model", CONFIG)
        assert t.base_url is None and t.provider == "openai"

    def test_a_config_with_no_providers_block_still_works(self):
        t = resolve_model_target("gpt-4o-mini", {"llm": {"api_key": "k"}})
        assert t.base_url is None and t.api_key == "k"

    def test_no_config_at_all_is_safe(self):
        assert resolve_model_target("gpt-4o-mini").base_url is None


# ── routing ───────────────────────────────────────────────────────────────────

class TestProviderRouting:

    @pytest.mark.parametrize("model", [
        "google/gemini-3.7-flash", "openai/gpt-5.6-luna",
        "qwen/qwen3.5-122b-a10b", "meta-llama/llama-4-maverick",
        "openai/gpt-5.4-mini",
    ])
    def test_each_tagged_model_reaches_openrouter_with_its_own_key(self, model):
        t = resolve_model_target(model, CONFIG)
        assert t.base_url == OPENROUTER
        assert t.api_key == "sk-or-v1-test"
        assert t.provider == "openrouter"

    def test_an_openai_branded_model_via_openrouter_is_not_sent_to_openai(self):
        """`openai/gpt-5.6-luna` is an OpenRouter catalogue id — sending it to
        api.openai.com with an OpenAI key would 404 on the model name."""
        t = resolve_model_target("openai/gpt-5.6-luna", CONFIG)
        assert t.base_url == OPENROUTER and t.api_key.startswith("sk-or-")

    def test_a_missing_provider_key_falls_back_and_warns(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with caplog.at_level("WARNING"):
            t = resolve_model_target("qwen/qwen3.5-122b-a10b", CONFIG)
        assert t.api_key == "sk-openai"          # still usable, will be rejected
        assert "OPENROUTER_API_KEY is not set" in caplog.text

    def test_an_unknown_provider_name_is_treated_as_unrouted(self, caplog):
        with caplog.at_level("WARNING"):
            t = resolve_model_target("z/unknown-provider", CONFIG)
        assert t.base_url is None
        assert "not in llm.providers" in caplog.text

    def test_a_bring_your_own_key_overrides_the_provider_key(self):
        t = resolve_model_target("qwen/qwen3.5-122b-a10b", CONFIG, api_key="sk-user")
        assert t.api_key == "sk-user"

    def test_an_untagged_model_uses_the_callers_process_wide_endpoint(self):
        """GH#65's behaviour: an agent that cached a base_url keeps applying it
        to models that name no provider."""
        t = resolve_model_target("gpt-4o-mini", CONFIG,
                                 default_base_url="http://localhost:11434/v1")
        assert t.base_url == "http://localhost:11434/v1"

    def test_a_provider_tag_beats_the_process_wide_endpoint(self):
        t = resolve_model_target("qwen/qwen3.5-122b-a10b", CONFIG,
                                 default_base_url="http://localhost:11434/v1")
        assert t.base_url == OPENROUTER


# ── per-model temperature ─────────────────────────────────────────────────────

class TestTemperature:

    def test_temperature_is_zero_by_default_for_everything(self):
        with patch.object(lf, "ChatOpenAI") as mc:
            build_for_target(resolve_model_target("google/gemini-3.7-flash", CONFIG))
        assert mc.call_args.kwargs["temperature"] == 0

    def test_null_omits_the_parameter_entirely(self):
        """Some reasoning models reject ANY explicit temperature, so sending
        the default 0 fails the call — the parameter has to be absent."""
        with patch.object(lf, "ChatOpenAI") as mc:
            build_for_target(resolve_model_target("x/reasoner", CONFIG))
        assert "temperature" not in mc.call_args.kwargs

    def test_an_explicit_value_is_passed_through(self):
        with patch.object(lf, "ChatOpenAI") as mc:
            build_for_target(resolve_model_target("y/warm", CONFIG))
        assert mc.call_args.kwargs["temperature"] == 0.7

    def test_caller_kwargs_still_win(self):
        with patch.object(lf, "ChatOpenAI") as mc:
            build_for_target(resolve_model_target("gpt-4o-mini", CONFIG),
                             max_retries=1)
        assert mc.call_args.kwargs["max_retries"] == 1


# ── the agent ─────────────────────────────────────────────────────────────────

class TestAgentRouting:

    def _agent(self):
        from backend.agents.db_agent import OrchestrationAgent
        cfg = {**CONFIG, "db": {"name": "FYP", "dialect": "postgres"},
               "llm": {**CONFIG["llm"], "model": "gpt-4o-mini"}}
        return OrchestrationAgent(cfg)

    def test_selectable_models_is_still_a_list_of_ids(self):
        """Mapping entries must not leak dicts into /models, the picker, or
        _validate_model — all of which have always seen strings."""
        models = self._agent().selectable_models
        assert all(isinstance(m, str) for m in models)
        assert "qwen/qwen3.5-122b-a10b" in models

    def test_the_provider_map_is_exposed_for_the_picker(self):
        providers = self._agent().model_providers
        assert providers["gpt-4o-mini"] == "openai"
        assert providers["meta-llama/llama-4-maverick"] == "openrouter"

    def test_per_request_model_switching_changes_provider(self):
        """The sidebar picker's whole purpose: one deployment, two endpoints."""
        agent = self._agent()
        with patch.object(lf, "ChatOpenAI") as mc:
            agent._chat_llm("gpt-4o-mini", None)
            assert "base_url" not in mc.call_args.kwargs
            agent._chat_llm("qwen/qwen3.5-122b-a10b", None)
            assert mc.call_args.kwargs["base_url"] == OPENROUTER


# ── the verification command ──────────────────────────────────────────────────

class TestCheckModels:

    def test_a_served_id_passes(self):
        from backend.config import check_models as C
        with patch.object(C, "_catalogue",
                          return_value={"google/gemini-3.7-flash"}):
            rows, fails = C.check({"llm": {
                "api_key": "sk-openai",
                "selectable_models": [{"id": "google/gemini-3.7-flash",
                                       "provider": "openrouter"}],
                "providers": CONFIG["llm"]["providers"]}})
        assert fails == 0 and rows[0]["status"] == C.OK

    def test_an_unserved_id_fails_before_anything_is_spent(self):
        from backend.config import check_models as C
        with patch.object(C, "_catalogue", return_value={"something/else"}):
            rows, fails = C.check({"llm": {
                "api_key": "sk-openai",
                "selectable_models": [{"id": "bogus/not-real",
                                       "provider": "openrouter"}],
                "providers": CONFIG["llm"]["providers"]}})
        assert fails == 1
        assert "not served" in rows[0]["note"]

    def test_an_unreachable_catalogue_warns_rather_than_accuses(self):
        """'Could not check' is not 'does not exist' — the same distinction the
        value index has to make about absent data."""
        from backend.config import check_models as C
        with patch.object(C, "_catalogue", return_value=None):
            rows, fails = C.check({"llm": {
                "api_key": "sk-openai",
                "selectable_models": [{"id": "anything/at-all",
                                       "provider": "openrouter"}],
                "providers": CONFIG["llm"]["providers"]}})
        assert fails == 0 and rows[0]["status"] == C.WARN

    def test_a_missing_key_is_a_failure(self, monkeypatch):
        from backend.config import check_models as C
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        rows, fails = C.check({"llm": {
            "selectable_models": [{"id": "a/b", "provider": "openrouter"}],
            "providers": CONFIG["llm"]["providers"]}})
        assert fails == 1 and "no API key" in rows[0]["note"]

    def test_direct_openai_models_are_not_catalogue_checked(self):
        """OpenAI's /models lists only what the key can reach, so absence is
        not proof of a bad id."""
        from backend.config import check_models as C
        with patch.object(C, "_catalogue") as cat:
            rows, fails = C.check({"llm": {"api_key": "sk-openai",
                                           "selectable_models": ["gpt-4o-mini"]}})
        cat.assert_not_called()
        assert fails == 0
