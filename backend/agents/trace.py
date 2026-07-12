"""
trace.py — gated console tracing for the pipeline (GH#42).

The rich console banners are the right tool for interactive tuning but noise in
production. `trace()` replaces bare print() in the pipeline modules; the flag is
set once from config `logging.pipeline_trace` at agent startup (default: on, so
the tuning workflow is unchanged — flip to false for structured-logs-only).
"""

from __future__ import annotations

_enabled: bool = True


def set_pipeline_trace(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)


def pipeline_trace_enabled() -> bool:
    return _enabled


def trace(*args, **kwargs) -> None:
    if _enabled:
        print(*args, **kwargs)
