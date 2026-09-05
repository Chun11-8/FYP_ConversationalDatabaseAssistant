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
