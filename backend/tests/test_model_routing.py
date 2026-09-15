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


# ── GH#111: the picker list is not the whole pipeline ─────────────────────────

class TestPipelineModelsCoverage:
    """
    A 401 on intent classification got past a green `check_models` because the
    checker only walked `selectable_models`. `llm.model` is the default for
    every stage and `intent_model` / `format_model` route the cheap stages
    elsewhere (GH#38) — none of them has to be in the picker list.
    """

    def test_the_default_model_is_checked_even_if_not_in_the_picker(self):
        from backend.config.check_models import pipeline_models
        roles = dict((m, r) for r, m in pipeline_models(
            {"llm": {"model": "gpt-4o-mini", "selectable_models": ["gpt-4o"]}}))
        assert roles["gpt-4o-mini"] == "default"

    def test_the_intent_and_format_models_are_checked(self):
        from backend.config.check_models import pipeline_models
        roles = dict((m, r) for r, m in pipeline_models({"llm": {
            "model": "gpt-4o", "intent_model": "openai/gpt-5.4-mini",
            "format_model": "meta-llama/llama-4-maverick"}}))
        assert roles["openai/gpt-5.4-mini"] == "intent"
        assert roles["meta-llama/llama-4-maverick"] == "format"

    def test_a_model_serving_several_roles_appears_once(self):
        from backend.config.check_models import pipeline_models
        out = pipeline_models({"llm": {
            "model": "gpt-4o-mini", "intent_model": "gpt-4o-mini",
            "selectable_models": ["gpt-4o-mini"]}})
        assert out == [("default", "gpt-4o-mini")]

    def test_mapping_entries_are_flattened_to_ids(self):
        from backend.config.check_models import pipeline_models
        out = pipeline_models({"llm": {"selectable_models": [
            {"id": "qwen/q", "provider": "openrouter"}]}})
        assert out == [("picker", "qwen/q")]

    def test_an_empty_config_yields_nothing_rather_than_crashing(self):
        from backend.config.check_models import pipeline_models
        assert pipeline_models({}) == []
        assert pipeline_models({"llm": {}}) == []

    def test_the_reported_failure_is_now_surfaced(self, monkeypatch):
        """A process-wide base_url sends the DEFAULT model — and therefore the
        intent call — to that provider. That is what produced
        'Missing Authentication header' from OpenRouter on the first question
        while the picker list looked fine."""
        from unittest.mock import patch as _patch
        from backend.config import check_models as C
        monkeypatch.setenv("LLM_BASE_URL", OPENROUTER)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with _patch.object(C, "_catalogue", return_value={"qwen/q"}):
            rows, fails = C.check({"llm": {
                "api_key": "sk-openai", "model": "gpt-4o-mini",
                "selectable_models": [{"id": "qwen/q", "provider": "openrouter"}],
                "providers": CONFIG["llm"]["providers"]}})
        default_row = next(r for r in rows if r["role"] == "default")
        assert default_row["status"] == C.FAIL
        assert default_row["provider"] == "via base_url"

    def test_keys_are_fingerprinted_never_printed(self):
        from backend.config.check_models import _mask
        assert _mask("sk-proj-abcdefghijklmnop") == "sk-proj…24ch"
        assert "abcdefghijklmnop" not in _mask("sk-proj-abcdefghijklmnop")
        assert _mask(None) == "-"
        assert _mask("short") == "…5ch"


# ── GH#112: the process default must never override a provider key ────────────

class TestDefaultKeyIsAFallbackNotAnOverride:
    """
    The reported failure: `check_models` reported every model usable, then the
    chat UI 401'd with OpenRouter's "Missing Authentication header" the moment
    a routed model was picked.

    `resolve_model_target`'s `api_key` means "the USER brought their own key
    for this request". Every construction site in db_agent was passing the
    process default there instead, so a provider-tagged model was called at
    the right endpoint with the OpenAI credential. check_models passes no key,
    which is precisely why it resolved correctly and the app did not.
    """

    def _agent(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-provider")
        from backend.agents.db_agent import OrchestrationAgent
        return OrchestrationAgent({
            "llm": {"model": "gpt-4o-mini", "api_key": "sk-openai-default",
                    "selectable_models": ["gpt-4o-mini",
                        {"id": "qwen/qwen3.5-122b-a10b", "provider": "openrouter"}],
                    "providers": CONFIG["llm"]["providers"]},
            "db": {"name": "FYP", "dialect": "postgres"}})

    def _built(self, agent, model, user_key=None):
        with patch.object(lf, "ChatOpenAI") as mc:
            agent._chat_llm(model, user_key)
            return mc.call_args.kwargs

    def test_a_routed_model_gets_the_provider_key_not_the_default(self, monkeypatch):
        kw = self._built(self._agent(monkeypatch), "qwen/qwen3.5-122b-a10b")
        assert kw["base_url"] == OPENROUTER
        assert kw["api_key"] == "sk-or-v1-provider"

    def test_an_untagged_model_still_gets_the_default_key(self, monkeypatch):
        kw = self._built(self._agent(monkeypatch), "gpt-4o-mini")
        assert "base_url" not in kw
        assert kw["api_key"] == "sk-openai-default"

    def test_a_user_supplied_key_still_wins(self, monkeypatch):
        """Bring-your-own-key is a real feature (GH#52) and must survive: a
        user handing over their own credential means it."""
        kw = self._built(self._agent(monkeypatch),
                         "qwen/qwen3.5-122b-a10b", "sk-user-byo")
        assert kw["api_key"] == "sk-user-byo"

    def test_the_startup_clients_do_not_pass_the_default_as_an_override(self):
        """A provider-tagged DEFAULT model would otherwise be mis-keyed before
        any request is made — the same bug one stage earlier."""
        import inspect
        from backend.agents.db_agent import OrchestrationAgent
        src = inspect.getsource(OrchestrationAgent.__init__)
        assert "_chat_llm(self._model, None)" in src
        assert "_chat_llm(self._model, self._default_api_key)" not in src

    def test_a_provider_tagged_default_model_is_keyed_correctly(self, monkeypatch):
        """Someone whose llm.model IS an OpenRouter id — the whole pipeline,
        not just the picker, has to route."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-provider")
        from backend.agents.db_agent import OrchestrationAgent
        with patch.object(lf, "ChatOpenAI") as mc:
            OrchestrationAgent({
                "llm": {"model": "qwen/qwen3.5-122b-a10b",
                        "api_key": "sk-openai-default",
                        "selectable_models": [
                            {"id": "qwen/qwen3.5-122b-a10b",
                             "provider": "openrouter"}],
                        "providers": CONFIG["llm"]["providers"]},
                "db": {"name": "FYP", "dialect": "postgres"}})
        first = mc.call_args_list[0].kwargs
        assert first["base_url"] == OPENROUTER
        assert first["api_key"] == "sk-or-v1-provider"

    def test_a_bare_agent_still_falls_back_to_its_cached_key(self):
        """Agents built via __new__ in other tests hold _default_api_key and no
        _config; the fallback keeps them working."""
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a._default_api_key = "sk-cached"
        with patch.object(lf, "ChatOpenAI") as mc:
            a._chat_llm("gpt-4o-mini", None)
        assert mc.call_args.kwargs["api_key"] == "sk-cached"


# ── GH#114: show which model answered, and never fake a cost ──────────────────

class TestUsageReportsModelAndCostHonestly:
    """
    A cost of $0.0000 beside an OpenRouter model reads as "this was free" when
    it means "nobody told us what this model costs". The two have to be
    distinguishable, or the UI quietly misreports spend.
    """

    def _tracker(self, model):
        from backend.agents import usage as U
        t = U.UsageTracker()
        t.record("generate_sql", model, 1000, 200)
        return t.as_dict()

    def test_the_model_that_was_billed_is_reported(self):
        assert self._tracker("gpt-4o-mini")["models"] == ["gpt-4o-mini"]

    def test_a_priced_model_reports_a_real_cost(self):
        d = self._tracker("gpt-4o-mini")
        assert d["cost_known"] is True and d["est_cost_usd"] > 0

    def test_an_unpriced_model_is_flagged_rather_than_costed_at_zero(self):
        d = self._tracker("qwen/qwen3.5-122b-a10b")
        assert d["cost_known"] is False
        assert d["unpriced_models"] == ["qwen/qwen3.5-122b-a10b"]
        assert d["est_cost_usd"] == 0.0        # present, but not to be shown

    def test_several_models_in_one_turn_are_all_listed(self):
        """GH#38 routes intent/format elsewhere, so a turn can bill two."""
        from backend.agents import usage as U
        t = U.UsageTracker()
        t.record("classify_intent", "gpt-4o-mini", 50, 5)
        t.record("generate_sql", "qwen/qwen3.5-122b-a10b", 900, 180)
        d = t.as_dict()
        assert d["models"] == ["gpt-4o-mini", "qwen/qwen3.5-122b-a10b"]
        assert d["cost_known"] is False        # one unpriced poisons the total

    def test_a_repeated_model_is_listed_once(self):
        from backend.agents import usage as U
        t = U.UsageTracker()
        for _ in range(3):
            t.record("generate_sql", "gpt-4o-mini", 10, 2)
        assert t.as_dict()["models"] == ["gpt-4o-mini"]

    def test_is_priced_distinguishes_unknown_from_free(self):
        from backend.agents.usage import is_priced
        assert is_priced("gpt-4o-mini") is True
        assert is_priced("qwen/qwen3.5-122b-a10b") is False
        assert is_priced(None) is False


class TestPriceSync:

    def test_per_token_strings_become_per_million_floats(self):
        """OpenRouter quotes per-token; config.yaml and usage.py are per
        million. Getting this wrong by 10^6 would be invisible in the UI."""
        from unittest.mock import patch as _patch
        from backend.config import check_models as C
        catalogue = [{"id": "qwen/q", "pricing": {"prompt": "0.00000015",
                                                  "completion": "0.0000006"}}]
        with _patch.object(C, "_catalogue_raw", return_value=catalogue):
            prices = C._prices("https://openrouter.ai/api/v1", "k")
        assert prices == {"qwen/q": {"input": 0.15, "output": 0.6}}

    def test_malformed_or_missing_pricing_is_skipped_not_guessed(self):
        from unittest.mock import patch as _patch
        from backend.config import check_models as C
        catalogue = [{"id": "a", "pricing": {"prompt": "free"}},
                     {"id": "b"}, {"id": "c", "pricing": {}}, "junk"]
        with _patch.object(C, "_catalogue_raw", return_value=catalogue):
            assert C._prices("https://x/v1", "k") == {}

    def test_an_unreachable_catalogue_yields_nothing(self):
        from unittest.mock import patch as _patch
        from backend.config import check_models as C
        with _patch.object(C, "_catalogue_raw", return_value=None):
            assert C._prices("https://x/v1", "k") == {}
