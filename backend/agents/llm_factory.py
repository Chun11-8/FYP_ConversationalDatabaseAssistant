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
from typing import Any, Dict, NamedTuple, Optional

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

# GH#118: how long one chat call may take, and how many times the SDK retries
# it. Both are capped deliberately. Neither the agent graph nor the BIRD loop
# set a timeout today, so the OpenAI SDK default applies — 600s per attempt
# with 2 retries, i.e. a single stuck call can occupy the run for half an hour
# with no output. Gateways make that reachable in normal use: a non-streaming
# request to OpenRouter returns its 200 as soon as the headers are written and
# then holds the body open for the whole generation, so a slow reasoning model
# looks exactly like a hang in the logs.
_DEFAULT_REQUEST_TIMEOUT_S = 120.0
_DEFAULT_MAX_RETRIES = 2


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


def resolve_call_limits(config: Optional[dict] = None,
                        entry: Optional[dict] = None) -> Dict[str, Any]:
    """
    ``{"timeout": seconds, "max_retries": n}`` for one chat call (GH#118).

    Precedence: the model's own entry in ``llm.selectable_models`` → the
    ``llm.*`` defaults → the constants above. A per-model override exists
    because the right ceiling is a property of the model: a reasoning model
    that thinks for two minutes is working, while the same two minutes from a
    small instruct model means the connection is dead.

    ``request_timeout_s: null`` restores the SDK default (no cap) for a model
    that genuinely needs it, rather than forcing a number.
    """
    llm_cfg = (config or {}).get("llm") or {}
    entry = entry or {}

    if "request_timeout_s" in entry:
        timeout = entry["request_timeout_s"]
    elif "request_timeout_s" in llm_cfg:
        timeout = llm_cfg["request_timeout_s"]
    else:
        timeout = _DEFAULT_REQUEST_TIMEOUT_S

    if "max_retries" in entry:
        retries = entry["max_retries"]
    elif "max_retries" in llm_cfg:
        retries = llm_cfg["max_retries"]
    else:
        retries = _DEFAULT_MAX_RETRIES

    limits: Dict[str, Any] = {"max_retries": int(retries)}
    # None means "no cap" — omit the kwarg so the SDK default stands.
    if timeout is not None:
        limits["timeout"] = float(timeout)
    return limits


# ── per-model routing (GH#108) ───────────────────────────────────────────────

class ModelTarget(NamedTuple):
    """Where one model id is served, and how to call it."""
    model: str
    base_url: Optional[str]          # None = OpenAI itself
    api_key: Optional[str]
    provider: str                    # label, for logs and the UI
    extra: Dict[str, Any]            # per-model kwargs, e.g. temperature


def _model_entries(config: Optional[dict]) -> Dict[str, dict]:
    """
    `llm.selectable_models` normalised to {id: entry}.

    Entries may be a plain string (the original format — provider unset, so it
    goes wherever the process-wide base_url points) or a mapping with `id` and
    optionally `provider`, `label`, `temperature`. Both forms are supported
    deliberately: the string form is what every existing config uses.
    """
    out: Dict[str, dict] = {}
    for item in ((config or {}).get("llm") or {}).get("selectable_models") or []:
        if isinstance(item, str):
            out[item] = {"id": item}
        elif isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = dict(item)
    return out


def resolve_model_target(model: str, config: Optional[dict] = None,
                         api_key: Optional[str] = None,
                         default_base_url: Optional[str] = None,
                         default_api_key: Optional[str] = None) -> ModelTarget:
    """
    The endpoint, key and call kwargs for ONE model id (GH#108).

    This is what lets a single deployment serve OpenAI and OpenRouter models
    side by side: the sidebar picker changes the provider implicitly, because
    the provider is a property of the model rather than of the process.

    Resolution:
      * the model's `provider` (from `llm.selectable_models`) is looked up in
        `llm.providers` for a base_url and a key env var;
      * an untagged model keeps the OLD behaviour exactly — the process-wide
        base_url (config/`$LLM_BASE_URL`) and the query-stage key. That is what
        makes this change invisible to an existing OpenAI-only install.

    *api_key* is a BRING-YOUR-OWN key for this one request, and only that. It
    overrides the provider's key, because a user supplying their own
    credential means it. Passing the process default here instead (GH#112)
    defeated routing entirely: every construction site handed over the OpenAI
    key, so a provider-tagged model was called at the right endpoint with the
    wrong credential and 401'd — while check_models, which passes nothing,
    resolved correctly and reported it usable.

    *default_base_url* / *default_api_key* are the caller's already-resolved
    process-wide values, used only as a LAST resort for an untagged model.
    They are fallbacks, never overrides — that distinction is the fix.
    """
    entry = _model_entries(config).get(model, {})
    provider_name = str(entry.get("provider") or "").strip()
    providers = ((config or {}).get("llm") or {}).get("providers") or {}
    provider = providers.get(provider_name) if provider_name else None

    if provider:
        base_url = str(provider.get("base_url") or "").strip() or None
        key_env = str(provider.get("api_key_env") or "").strip()
        key = (api_key or (os.getenv(key_env) if key_env else None)
               or resolve_api_key(config) or default_api_key)
        if key_env and not os.getenv(key_env):
            logger.warning(
                "[LLM] model %r is routed to provider %r but %s is not set — "
                "falling back to the query-stage key, which that provider will "
                "probably reject", model, provider_name, key_env)
        label = provider_name
    else:
        if provider_name:
            logger.warning(
                "[LLM] model %r names provider %r, which is not in "
                "llm.providers — treating it as unrouted", model, provider_name)
        base_url = default_base_url or resolve_base_url(config)
        key = api_key or resolve_api_key(config) or default_api_key
        label = "openai" if base_url is None else "via base_url"

    # Per-model call kwargs. `temperature: null` OMITS the parameter, which is
    # how a reasoning model that rejects a non-default temperature is handled;
    # absent means the caller's default (0) is kept, so existing models are
    # untouched.
    extra: Dict[str, Any] = dict(resolve_call_limits(config, entry))
    if "temperature" in entry:
        temp = entry["temperature"]
        extra["temperature"] = None if temp is None else temp
    return ModelTarget(model=model, base_url=base_url, api_key=key,
                       provider=label, extra=extra)


def build_for_target(target: ModelTarget, **extra: Any) -> ChatOpenAI:
    """
    A client for a resolved :class:`ModelTarget`.

    ``temperature=None`` in the target's extras means OMIT the parameter
    entirely — some reasoning models reject any explicit temperature, and
    sending the default 0 fails the call outright.
    """
    kwargs: Dict[str, Any] = {"model": target.model, "temperature": 0,
                              "api_key": target.api_key}
    if target.base_url:
        kwargs["base_url"] = target.base_url
    kwargs.update(target.extra)
    kwargs.update(extra)
    if kwargs.get("temperature", 0) is None:
        kwargs.pop("temperature")
    return ChatOpenAI(**kwargs)


def build_chat_llm(model: str, api_key: Optional[str],
                   base_url: Optional[str] = None,
                   config: Optional[dict] = None, **extra: Any) -> ChatOpenAI:
    """
    A chat client for ``model``, optionally against an OpenAI-compatible
    ``base_url``. ``extra`` passes through provider/caller-specific kwargs
    (e.g. ``max_retries``).

    When no base_url is given the kwarg is omitted entirely rather than passed
    as None, so the default path differs from a plain
    ``ChatOpenAI(model=..., temperature=0, api_key=...)`` construction only by
    the GH#118 call limits, which ``extra`` can override.
    """
    kwargs: Dict[str, Any] = {"model": model, "temperature": 0, "api_key": api_key}
    kwargs.update(resolve_call_limits(config))
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
