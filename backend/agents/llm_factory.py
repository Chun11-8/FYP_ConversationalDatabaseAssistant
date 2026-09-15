"""
llm_factory.py — one place that builds a query-stage chat client (GH#65).

Every chat call in this project goes through ``langchain_openai.ChatOpenAI``,
which means any provider exposing an **OpenAI-compatible** endpoint works by
setting a base URL — OpenRouter, Groq, DeepSeek, Together, Cerebras, Mistral,
Google's compat layer, or a local Ollama / vLLM server. That in turn makes
open-weight models (Llama, DeepSeek, Qwen, Mistral, Gemma, GPT-OSS) usable
without touching the agent wiring.

Centralised deliberately. The serving path builds chat clients at four sites
and the eval CLIs at three more; resolving the base URL separately at each one
invites the failure where a single stage silently keeps talking to OpenAI
while the rest of the run targets the configured provider — which would look
like a flaky model rather than a config bug.

SCOPE: the query stage only. Embeddings stay on OpenAI
(``backend/retrieval/embedding.py``) because the pgvector columns are sized
for text-embedding-3-small at 1536 dimensions; swapping the embedder would
require a full re-index.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Env var wins over config so a benchmark run can retarget a provider without
# editing config.yaml — mirrors how the eval CLIs already treat --model.
_BASE_URL_ENV = "LLM_BASE_URL"
# GH#106: the key for the QUERY-STAGE provider. Env-only — config.yaml must
# never hold a secret. Unset falls back to OPENAI_API_KEY, so the default
# OpenAI path is unchanged.
_API_KEY_ENV = "LLM_API_KEY"
# The judge's own key (eval only), for a generator and judge on DIFFERENT
# providers. Unset falls back to the query-stage key, which is right when both
# run through one gateway such as OpenRouter.
_JUDGE_KEY_ENV = "JUDGE_API_KEY"


def resolve_base_url(config: Optional[dict] = None,
                     cli_value: Optional[str] = None) -> Optional[str]:
    """
    The OpenAI-compatible endpoint to target, or None for OpenAI itself.

    Precedence: explicit CLI value → ``LLM_BASE_URL`` env → config
    ``llm.base_url``. Blank strings are treated as unset throughout, so an
    empty config key or an exported-but-empty env var means "use OpenAI"
    rather than producing an invalid client.
    """
    for candidate in (
        cli_value,
        os.getenv(_BASE_URL_ENV),
        ((config or {}).get("llm") or {}).get("base_url"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def resolve_api_key(config: Optional[dict] = None,
                    cli_value: Optional[str] = None,
                    env_var: str = _API_KEY_ENV) -> Optional[str]:
    """
    The API key for the query-stage provider (GH#106).

    Precedence: explicit value → ``$LLM_API_KEY`` → ``llm.api_key`` (which
    config_loader fills from ``OPENAI_API_KEY``).

    This exists because base_url and the key travel together and were not: the
    key was always OpenAI's, so pointing base_url at OpenRouter, Groq or
    DeepSeek sent an OpenAI key to a provider that would reject it. The
    fallback keeps the default path byte-identical — with no env var set, this
    returns exactly what the callers passed before.

    Embeddings deliberately do NOT use this: they stay on OpenAI and keep
    reading ``OPENAI_API_KEY`` (see the module docstring).
    """
    for candidate in (cli_value, os.getenv(env_var),
                      ((config or {}).get("llm") or {}).get("api_key")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def resolve_judge_api_key(config: Optional[dict] = None,
                          cli_value: Optional[str] = None) -> Optional[str]:
    """
    The key for the back-translation judge (GH#106), so the judge can run on a
    different provider from the generator — the configuration the literature
    reports at 0.82 AUROC against 0.72–0.78 for a single judge.

    Precedence: explicit → ``$JUDGE_API_KEY`` → the query-stage key.
    """
    explicit = cli_value or os.getenv(_JUDGE_KEY_ENV)
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    return resolve_api_key(config)


def build_chat_llm(model: str, api_key: Optional[str],
                   base_url: Optional[str] = None, **extra: Any) -> ChatOpenAI:
    """
    A chat client for ``model``, optionally against an OpenAI-compatible
    ``base_url``. ``extra`` passes through provider/caller-specific kwargs
    (e.g. ``max_retries``).

    When no base_url is given the kwarg is omitted entirely rather than passed
    as None, so the default path stays byte-identical to a plain
    ``ChatOpenAI(model=..., temperature=0, api_key=...)`` construction.
    """
    kwargs: Dict[str, Any] = {"model": model, "temperature": 0, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs.update(extra)
    return ChatOpenAI(**kwargs)


def describe_target(model: str, base_url: Optional[str]) -> str:
    """One-line 'what am I actually calling' string for logs and eval headers.

    Benchmark output that doesn't record the endpoint is unreproducible — the
    same model id can be served by several providers with different weights,
    quantisation, or sampling defaults.
    """
    return f"{model} @ {base_url or 'api.openai.com (default)'}"
