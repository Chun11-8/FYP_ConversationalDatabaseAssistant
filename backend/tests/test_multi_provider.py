"""
Tests for multi-provider support.

Covers the three things that let an open-weight model from another vendor
(Llama, DeepSeek, Qwen, Mistral, Gemma, GPT-OSS) run through this pipeline:

  1. llm_factory  — base_url resolution + client construction
  2. strip_reasoning — the four free-text parsers surviving a chain-of-thought
  3. _MODEL_ID_RE — vendor-namespaced ids passing validation

All offline: ChatOpenAI is patched, no network, no API key.

Run from repo root:
    pytest backend/tests/test_multi_provider.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import backend.agents.llm_factory as llm_factory
from backend.agents.llm_factory import (
    build_chat_llm, describe_target, resolve_base_url,
)
from backend.agents.utils import strip_reasoning
from backend.api.main import _MODEL_ID_RE

OPENROUTER = "https://openrouter.ai/api/v1"


# ── base_url resolution ───────────────────────────────────────────────────────

class TestResolveBaseUrl:

    def test_none_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert resolve_base_url({}, None) is None
        assert resolve_base_url(None, None) is None

    def test_reads_config(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert resolve_base_url({"llm": {"base_url": OPENROUTER}}) == OPENROUTER

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
        assert resolve_base_url({"llm": {"base_url": OPENROUTER}}) \
            == "http://localhost:11434/v1"

    def test_cli_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
        assert resolve_base_url({"llm": {"base_url": OPENROUTER}},
                                "https://api.groq.com/openai/v1") \
            == "https://api.groq.com/openai/v1"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_is_treated_as_unset(self, monkeypatch, blank):
        """An exported-but-empty env var must mean OpenAI, not an invalid client."""
        monkeypatch.setenv("LLM_BASE_URL", blank)
        assert resolve_base_url({"llm": {"base_url": blank}}, blank) is None

    def test_whitespace_is_trimmed(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert resolve_base_url({}, f"  {OPENROUTER}  ") == OPENROUTER


# ── client construction ───────────────────────────────────────────────────────

class TestBuildChatLlm:

    def test_no_base_url_kwarg_when_unset(self):
        """The default path must stay byte-identical to a plain ChatOpenAI —
        passing base_url=None is NOT the same as omitting it."""
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            build_chat_llm("gpt-4o-mini", "k")
        assert "base_url" not in mc.call_args.kwargs

    def test_base_url_passed_when_set(self):
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            build_chat_llm("deepseek-chat", "k", "https://api.deepseek.com")
        assert mc.call_args.kwargs["base_url"] == "https://api.deepseek.com"

    def test_model_key_and_temperature(self):
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            build_chat_llm("qwen/qwen3-32b", "sk-x")
        kw = mc.call_args.kwargs
        assert kw["model"] == "qwen/qwen3-32b"
        assert kw["api_key"] == "sk-x"
        assert kw["temperature"] == 0

    def test_extra_kwargs_pass_through(self):
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            build_chat_llm("m", "k", None, max_retries=1)
        assert mc.call_args.kwargs["max_retries"] == 1

    def test_describe_target_names_the_endpoint(self):
        assert "api.openai.com" in describe_target("gpt-4o-mini", None)
        assert OPENROUTER in describe_target("meta-llama/Llama-3.3-70B", OPENROUTER)


# ── the agent threads base_url through EVERY chat call site ───────────────────

class TestAgentUsesBaseUrl:
    """The per-request override is the site most easily missed — if it built a
    bare client, a bring-your-own-key request would silently hit OpenAI while
    the rest of the deployment talked to the configured provider."""

    def _agent(self, base_url):
        import backend.agents.db_agent as dbm
        a = dbm.OrchestrationAgent.__new__(dbm.OrchestrationAgent)
        a._base_url = base_url
        return a

    def test_chat_llm_forwards_the_configured_base_url(self):
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            self._agent(OPENROUTER)._chat_llm("deepseek-chat", "k")
        assert mc.call_args.kwargs["base_url"] == OPENROUTER

    def test_chat_llm_omits_it_when_unset(self):
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            self._agent(None)._chat_llm("gpt-4o-mini", "k")
        assert "base_url" not in mc.call_args.kwargs

    def test_missing_attribute_degrades_to_openai(self):
        """Agents built via __new__ in other tests have no _base_url at all."""
        import backend.agents.db_agent as dbm
        a = dbm.OrchestrationAgent.__new__(dbm.OrchestrationAgent)
        with patch.object(llm_factory, "ChatOpenAI") as mc:
            a._chat_llm("gpt-4o-mini", "k")
        assert "base_url" not in mc.call_args.kwargs


# ── embeddings must NOT follow base_url ───────────────────────────────────────

class TestEmbeddingsStayOnOpenAI:

    def test_get_openai_client_ignores_base_url(self, monkeypatch):
        """Scope guard: the pgvector columns are 1536-dim for
        text-embedding-3-small, so the embedder must not be retargeted."""
        monkeypatch.setenv("LLM_BASE_URL", OPENROUTER)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import backend.retrieval.embedding as emb
        with patch.object(emb, "OpenAI") as mock_openai:
            emb.get_openai_client()
        assert "base_url" not in mock_openai.call_args.kwargs


# ── reasoning-block stripping ─────────────────────────────────────────────────

class TestStripReasoning:

    def test_plain_reply_is_untouched(self):
        assert strip_reasoning("READ") == "READ"
        assert strip_reasoning("SELECT 1") == "SELECT 1"

    def test_none_and_empty_are_safe(self):
        assert strip_reasoning("") == ""
        assert strip_reasoning(None) is None

    def test_removes_a_think_block(self):
        raw = "<think>Hmm, could be GENERAL or AMBIGUOUS...</think>\nREAD"
        assert strip_reasoning(raw) == "READ"

    @pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "thought"])
    def test_handles_each_tag_spelling(self, tag):
        assert strip_reasoning(f"<{tag}>noise</{tag}>ANSWER") == "ANSWER"

    def test_is_case_insensitive(self):
        assert strip_reasoning("<THINK>x</THINK>ANSWER") == "ANSWER"

    def test_unclosed_block_drops_the_remainder(self):
        """The model hit its token cap mid-thought — everything after the
        opening tag is reasoning, so there is no answer to salvage."""
        assert strip_reasoning("ANSWER\n<think>ran out of tok") == "ANSWER"

    def test_returns_the_original_rather_than_an_empty_string(self):
        """A reply that is ONLY reasoning must not become "" — a messy full
        response gives the parser something to work with; nothing does not."""
        raw = "<think>all of it was reasoning</think>"
        assert strip_reasoning(raw) == raw

    def test_gpt_oss_harmony_final_channel(self):
        raw = ("<|channel|>analysis<|message|>weighing options<|end|>"
               "<|channel|>final<|message|>READ<|end|>")
        assert strip_reasoning(raw) == "READ"


# ── the four parsers survive a reasoning model ────────────────────────────────

class TestParsersSurviveReasoning:

    def test_intent_normalization(self):
        from backend.agents.db_agent import _normalize_intent
        raw = ("<think>This might be AMBIGUOUS, or maybe GENERAL. "
               "Actually they want data.</think>\nREAD")
        assert _normalize_intent(raw) == "READ"

    def test_query_plan_json_extraction(self):
        """Braces inside the reasoning used to make the outermost-{...}
        fallback span both regions and fail to parse — losing the plan."""
        from backend.agents.schema_linking_agent import _extract_json
        raw = ('<think>Maybe {"tables": ["wrong"]} is right? No.</think>\n'
               '{"tables": ["players"]}')
        assert _extract_json(raw) == {"tables": ["players"]}

    def test_sql_extraction_ignores_sql_inside_the_think_block(self):
        from backend.agents.sql_code_agent import SQLCodeAgent
        agent = SQLCodeAgent.__new__(SQLCodeAgent)
        raw = ("<think>Maybe ```sql SELECT wrong FROM t``` ?</think>\n"
               "```sql\nSELECT right FROM t\n```")
        sqls = agent._extract_sqls(raw)
        assert len(sqls) == 1
        assert "right" in sqls[0]

    def test_condense_query_stays_under_the_length_guard(self):
        """The regression that would re-break multi-turn: a think block pushes
        the rewrite past 500 chars, so it is silently discarded."""
        raw = "<think>" + ("deliberating " * 60) + "</think>\nrevenue in 2014"
        assert len(raw) > 500
        assert len(strip_reasoning(raw)) < 500


# ── unfenced SQL fallback ─────────────────────────────────────────────────────

class TestUnfencedSqlFallback:

    def _agent(self):
        from backend.agents.sql_code_agent import SQLCodeAgent
        return SQLCodeAgent.__new__(SQLCodeAgent)

    def test_bare_select_is_extracted(self):
        """Open-weight models honour the fencing instruction far less
        reliably; returning [] wasted two fallback rounds on good SQL."""
        assert self._agent()._extract_sqls("SELECT a FROM t") == ["SELECT a FROM t"]

    def test_fenced_still_wins(self):
        sqls = self._agent()._extract_sqls("```sql\nSELECT a FROM t\n```")
        assert sqls == ["SELECT a FROM t"]

    def test_leading_prose_is_not_swallowed(self):
        sqls = self._agent()._extract_sqls("Here is the query:\nSELECT a FROM t;")
        assert sqls == ["SELECT a FROM t"]

    def test_non_sql_reply_yields_nothing(self):
        assert self._agent()._extract_sqls("I cannot answer that.") == []


# ── namespaced model ids ──────────────────────────────────────────────────────

class TestNamespacedModelIds:

    @pytest.mark.parametrize("model", [
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-120b",
        "deepseek-chat",
        "gpt-4o-mini",
        "ft:gpt-4o-mini:acme::abc123",
    ])
    def test_real_provider_ids_are_accepted(self, model):
        assert _MODEL_ID_RE.match(model)

    @pytest.mark.parametrize("model", [
        "/leading-slash", "-leading-dash", "", "x", "a" * 81,
        "has space", "semi;colon", "quote'd",
    ])
    def test_malformed_ids_are_still_rejected(self, model):
        assert not _MODEL_ID_RE.match(model)


# ── the key must travel with the endpoint ─────────────────────────────────────

class TestProviderApiKey:
    """
    base_url and the API key are a pair. Previously the key was always
    OpenAI's, so pointing base_url at OpenRouter/Groq/DeepSeek sent an OpenAI
    key to a provider that would reject it — the documented setup did not work.
    """

    CFG = {"llm": {"api_key": "sk-openai"}}

    def test_without_an_env_var_nothing_changes(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        assert llm_factory.resolve_api_key(self.CFG) == "sk-openai"

    def test_llm_api_key_overrides_the_openai_key(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-abc")
        assert llm_factory.resolve_api_key(self.CFG) == "sk-or-v1-abc"

    def test_an_explicit_value_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-env")
        assert llm_factory.resolve_api_key(self.CFG, "sk-explicit") == "sk-explicit"

    def test_a_blank_env_var_means_unset(self, monkeypatch):
        """An exported-but-empty var must not produce a client with no key."""
        monkeypatch.setenv("LLM_API_KEY", "   ")
        assert llm_factory.resolve_api_key(self.CFG) == "sk-openai"

    def test_no_key_anywhere_is_none_not_a_crash(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        assert llm_factory.resolve_api_key({}) is None

    def test_the_judge_inherits_the_query_key_by_default(self, monkeypatch):
        """One gateway (OpenRouter) serving both needs only one key."""
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-abc")
        assert llm_factory.resolve_judge_api_key(self.CFG) == "sk-or-v1-abc"

    def test_the_judge_can_have_its_own_key(self, monkeypatch):
        """Generator and judge on DIFFERENT providers — the cross-provider
        configuration is the whole reason the judge is configurable."""
        monkeypatch.setenv("LLM_API_KEY", "sk-openai-proj")
        monkeypatch.setenv("JUDGE_API_KEY", "sk-deepseek")
        assert llm_factory.resolve_judge_api_key(self.CFG) == "sk-deepseek"

    def test_embeddings_are_untouched(self):
        """They stay on OpenAI: the pgvector columns are sized for
        text-embedding-3-small, so a different embedder means a re-index."""
        import inspect
        from backend.retrieval import embedding
        src = inspect.getsource(embedding)
        assert "LLM_API_KEY" not in src
        assert "resolve_api_key" not in src


class TestEveryCallSiteUsesIt:
    """One stage left on the old key would silently talk to OpenAI while the
    rest of the run targets the configured provider — which looks like a flaky
    model rather than a config bug. That is why llm_factory is centralised."""

    def test_no_call_site_still_reads_the_raw_openai_key(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for path in (root / "agents" / "db_agent.py", root / "eval" / "bird.py",
                     root / "eval" / "db_backends.py",
                     root / "eval" / "trust_eval.py"):
            text = path.read_text()
            if 'config["llm"]["api_key"]' in text:
                offenders.append(path.name)
        assert offenders == [], f"still passing the OpenAI key directly: {offenders}"
