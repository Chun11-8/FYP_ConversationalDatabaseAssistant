"""
usage.py — per-request LLM token & cost accounting.

Every LLM call in the pipeline used to do ``llm.invoke(...).content`` and throw
away ``usage_metadata``, so the system could not report what a turn cost — only
how long it took. ``invoke_tracked()`` is a drop-in replacement for
``llm.invoke()`` that records tokens against the current request before handing
back the same response object.

Accounting is per-request via thread-local storage, mirroring db_agent's ``_tls``
override pattern. The tracker itself is lock-guarded and can be passed
explicitly (``tracker=``) so the generator's ThreadPoolExecutor fan-out attributes
its worker-thread calls to the request that spawned them.

Never fatal: a model whose price is unknown, or a provider response without usage
metadata, contributes zero cost rather than raising.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_tls: threading.local = threading.local()

# USD per 1M tokens. Overridden/extended by config `llm.pricing`; a model absent
# from the table still has its tokens counted, it just contributes no cost.
_DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-mini":   {"input": 0.25, "output": 2.00},
}

_pricing: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in _DEFAULT_PRICING.items()}
_unpriced_warned: set[str] = set()


def set_pricing(table: Optional[Dict[str, Any]]) -> None:
    """Merge a config `llm.pricing` block over the built-in defaults."""
    if not isinstance(table, dict):
        return
    for model, rates in table.items():
        if not isinstance(rates, dict):
            continue
        try:
            _pricing[str(model)] = {
                "input": float(rates.get("input", 0.0)),
                "output": float(rates.get("output", 0.0)),
            }
        except (TypeError, ValueError):
            logger.warning("[USAGE] ignoring malformed pricing entry for %s", model)


def is_priced(model: Optional[str]) -> bool:
    """Is a rate known for *model*?

    Without this the UI cannot tell "this turn cost nothing" from "nobody told
    us what this model costs" — both arrive as 0.0, and $0.0000 next to an
    OpenRouter model reads as free when it is merely unknown.
    """
    return _pricing.get(model or "") is not None


def cost_usd(model: Optional[str], input_tokens: int, output_tokens: int) -> float:
    rates = _pricing.get(model or "")
    if rates is None:
        if model and model not in _unpriced_warned:
            _unpriced_warned.add(model)
            logger.info("[USAGE] no pricing for %s — tokens counted, cost omitted", model)
        return 0.0
    return (input_tokens / 1e6) * rates["input"] + (output_tokens / 1e6) * rates["output"]


class UsageTracker:
    """Accumulates token/cost totals for one request, in total and per stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.est_cost_usd = 0.0
        self.by_stage: Dict[str, Dict[str, Any]] = {}
        # Which model(s) this turn actually billed against, in call
        # order. Usually one; two when a stage is routed elsewhere or
        # the judge runs on another provider.
        self.models: list[str] = []
        self._unpriced: set[str] = set()

    def record(self, stage: str, model: Optional[str],
               input_tokens: int, output_tokens: int) -> None:
        c = cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            if model and model not in self.models:
                self.models.append(model)
            if model and not is_priced(model):
                self._unpriced.add(model)
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.est_cost_usd += c
            s = self.by_stage.setdefault(
                stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                        "est_cost_usd": 0.0})
            s["calls"] += 1
            s["input_tokens"] += input_tokens
            s["output_tokens"] += output_tokens
            s["est_cost_usd"] += c

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "est_cost_usd": round(self.est_cost_usd, 6),
                # What was actually billed, and whether the figure
                # above means anything. `cost_known` False = a rate is missing
                # for at least one model, so est_cost_usd UNDERSTATES the turn
                # and must not be rendered as a dollar amount.
                "models": list(self.models),
                "cost_known": not self._unpriced,
                "unpriced_models": sorted(self._unpriced),
                "by_stage": {
                    k: {**v, "est_cost_usd": round(v["est_cost_usd"], 6)}
                    for k, v in self.by_stage.items()
                },
            }


def start_request() -> UsageTracker:
    """Begin accounting for a new turn on this thread, replacing any previous one."""
    tracker = UsageTracker()
    _tls.usage = tracker
    return tracker


def current() -> Optional[UsageTracker]:
    return getattr(_tls, "usage", None)


def bind(tracker: Optional[UsageTracker]) -> None:
    """Attach an existing tracker to this thread (for pool workers)."""
    _tls.usage = tracker


def clear() -> None:
    _tls.usage = None


def _model_of(llm: Any) -> Optional[str]:
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    return str(name) if name else None


def _tokens_of(response: Any) -> tuple[int, int]:
    """(input, output) tokens, tolerating providers that report neither."""
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    rmeta = getattr(response, "response_metadata", None)
    if isinstance(rmeta, dict):
        tu = rmeta.get("token_usage") or {}
        if isinstance(tu, dict):
            return int(tu.get("prompt_tokens") or 0), int(tu.get("completion_tokens") or 0)
    return 0, 0


def invoke_tracked(llm: Any, messages: List[Any], stage: str,
                   tracker: Optional[UsageTracker] = None) -> Any:
    """
    ``llm.invoke(messages)`` with the token usage recorded against ``stage``.
    Returns the response object unchanged, so call sites keep reading ``.content``.
    """
    response = llm.invoke(messages)
    tracker = tracker if tracker is not None else current()
    if tracker is not None:
        try:
            tracker.record(stage, _model_of(llm), *_tokens_of(response))
        except Exception as exc:   # accounting must never break a turn
            logger.warning("[USAGE] record failed at %s: %s", stage, exc)
    return response
