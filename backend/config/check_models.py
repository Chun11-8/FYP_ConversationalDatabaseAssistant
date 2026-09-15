"""
check_models.py — verify every configured model id before spending on a run.

A model slug is the one part of this configuration that cannot be validated by
reading the repo: providers retire ids (OpenRouter's `deepseek/deepseek-chat`
is gone), and a wrong id fails at the first call — mid-benchmark, after paying
for everything before it.

This asks each provider what it actually serves and reports, per model, where
it routes, whether the key is present, and whether the id exists.

    python -m backend.config.check_models          # id + key + routing checks
    python -m backend.config.check_models --call   # also make one tiny call

Exit code is non-zero when anything failed, so it can gate a run:

    python -m backend.config.check_models && python -m backend.eval.bird …
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

from backend.agents.llm_factory import (
    resolve_base_url, resolve_model_target,
)
from backend.config.config_loader import load_config

_TIMEOUT = 15
OK, WARN, FAIL = "✓", "!", "✗"


def _catalogue_raw(base_url: str, api_key: Optional[str]) -> Optional[list]:
    """The provider's raw /models payload, or None if it could not be read."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            TimeoutError, OSError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, list) else None


def _prices(base_url: str, api_key: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    {model id: {input, output}} in USD per MILLION tokens, from the provider's
    catalogue (GH#114).

    OpenRouter quotes per-token strings ("0.00000015"); config.yaml and
    usage.py both work in per-million, so convert once here rather than
    leaving a unit mismatch for whoever pastes the output.
    """
    out: Dict[str, Dict[str, float]] = {}
    for entry in _catalogue_raw(base_url, api_key) or []:
        if not isinstance(entry, dict):
            continue
        pricing = entry.get("pricing")
        mid = entry.get("id")
        if not mid or not isinstance(pricing, dict):
            continue
        try:
            inp = float(pricing.get("prompt", "nan")) * 1e6
            outp = float(pricing.get("completion", "nan")) * 1e6
        except (TypeError, ValueError):
            continue
        if inp == inp and outp == outp:            # both non-NaN
            out[str(mid)] = {"input": round(inp, 4), "output": round(outp, 4)}
    return out


def _catalogue(base_url: str, api_key: Optional[str]) -> Optional[Set[str]]:
    """
    The set of ids a provider serves, from its OpenAI-compatible /models
    endpoint. None when the catalogue could not be read — which must be
    reported as "could not check", never as "the model does not exist".
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            TimeoutError, OSError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    return {str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")}


def _mask(key: Optional[str]) -> str:
    """A key fingerprint that identifies it without disclosing it."""
    if not key:
        return "-"
    return f"{key[:7]}…{len(key)}ch" if len(key) > 10 else f"…{len(key)}ch"


def pipeline_models(config: dict) -> List[Tuple[str, str]]:
    """
    Every model the app will actually construct a client for, as
    (role, model id) — deduped, first role wins.

    GH#111: the picker list is NOT the whole story. `llm.model` is the default
    for every stage, and `llm.intent_model` / `llm.format_model` route the
    cheap stages elsewhere (GH#38); none of them has to appear in
    `selectable_models`. Checking only the picker list let a broken default or
    intent model pass here and fail on the first question — which is exactly
    how a 401 on intent classification got past a green check.
    """
    llm = config.get("llm") or {}
    out: List[Tuple[str, str]] = []
    seen = set()

    def add(role: str, model) -> None:
        if not model:
            return
        mid = model if isinstance(model, str) else str(model.get("id"))
        if mid and mid not in seen:
            seen.add(mid)
            out.append((role, mid))

    add("default", llm.get("model"))
    add("intent", llm.get("intent_model"))
    add("format", llm.get("format_model"))
    for entry in llm.get("selectable_models") or []:
        add("picker", entry)
    return out


def check(config: dict, do_call: bool = False) -> Tuple[List[dict], int]:
    """(rows, failures) — one row per model the pipeline can construct."""
    roles = pipeline_models(config)
    models: List[str] = [m for _role, m in roles]
    role_of = {m: r for r, m in roles}

    catalogues: Dict[str, Optional[Set[str]]] = {}
    rows: List[dict] = []
    failures = 0

    for model in models:
        target = resolve_model_target(model, config)
        row = {"model": model, "role": role_of.get(model, "picker"),
               "provider": target.provider,
               "base_url": target.base_url or "api.openai.com (default)",
               "key": bool(target.api_key), "key_hint": _mask(target.api_key),
               "status": OK, "note": ""}

        if not target.api_key:
            row["status"], row["note"] = FAIL, "no API key resolved"
            failures += 1
        elif target.base_url:
            if target.base_url not in catalogues:
                catalogues[target.base_url] = _catalogue(target.base_url,
                                                         target.api_key)
            known = catalogues[target.base_url]
            if known is None:
                row["status"] = WARN
                row["note"] = "catalogue unreachable — id not checked"
            elif model not in known:
                row["status"], row["note"] = FAIL, "id not served by provider"
                failures += 1
        else:
            # OpenAI's catalogue lists only what the key can reach, so absence
            # is not proof of a bad id — verified by --call instead.
            row["note"] = "direct OpenAI — id not catalogue-checked"

        if do_call and row["status"] != FAIL:
            try:
                from backend.agents.llm_factory import build_for_target
                from langchain_core.messages import HumanMessage
                build_for_target(target, max_retries=0).invoke(
                    [HumanMessage(content="Reply with the single word: ok")])
                row["note"] = (row["note"] + "; live call ok").lstrip("; ")
            except Exception as exc:
                row["status"] = FAIL
                row["note"] = f"live call failed: {str(exc)[:90]}"
                failures += 1
        rows.append(row)
    return rows, failures


def sync_pricing(config: dict) -> None:
    """Print an `llm.pricing` block with each routed model's real rates."""
    wanted = [m for _role, m in pipeline_models(config)]
    by_endpoint: Dict[str, Dict[str, Dict[str, float]]] = {}
    found: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []

    for model in wanted:
        target = resolve_model_target(model, config)
        if not target.base_url:
            # Direct OpenAI: no price list is exposed on /models, so the
            # built-in table in usage.py stays the source for those.
            continue
        if target.base_url not in by_endpoint:
            by_endpoint[target.base_url] = _prices(target.base_url,
                                                   target.api_key)
        price = by_endpoint[target.base_url].get(model)
        if price:
            found[model] = price
        else:
            missing.append(model)

    if not found and not missing:
        print("No routed models configured — nothing to price.")
        return

    print("\nPaste this under `llm:` in backend/config/config.yaml\n")
    print("  pricing:")
    for model, rates in sorted(found.items()):
        print(f'    "{model}":')
        print(f"      input: {rates['input']}")
        print(f"      output: {rates['output']}")
    if missing:
        print("\n  # no price returned by the provider for:")
        for model in missing:
            print(f"  #   {model}")
    print("\nUSD per MILLION tokens. Rates change — re-run before quoting a "
          "cost figure in a report.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-pricing", action="store_true",
                    help="Fetch the real per-token prices for the configured "
                         "models from their provider and print a ready-to-paste "
                         "llm.pricing block. Until rates are configured the UI "
                         "reports 'cost n/a' rather than a misleading $0.0000.")
    ap.add_argument("--call", action="store_true",
                    help="Also make one tiny real call per model (costs a few "
                         "tokens each, but proves the whole path end to end)")
    args = ap.parse_args()

    config = load_config()

    if args.sync_pricing:
        sync_pricing(config)
        return

    rows, failures = check(config, do_call=args.call)
    if not rows:
        print("No models configured under llm.selectable_models.")
        return

    width = max(len(r["model"]) for r in rows)
    pw = max(len(r["provider"]) for r in rows) + 2
    kw = max(len(r["key_hint"]) for r in rows) + 2
    print(f"\n{'':2}{'model':<{width}}  {'role':<9}{'provider':<{pw}}"
          f"{'key':<{kw}}note")
    for r in rows:
        print(f"{r['status']:<2}{r['model']:<{width}}  {r['role']:<9}"
              f"{r['provider']:<{pw}}{r['key_hint']:<{kw}}{r['note']}")

    # GH#111: a process-wide endpoint sends EVERY untagged model — including
    # the default and the intent model — to that provider. It is the usual
    # cause of "the picker looks right but the first question 401s".
    process_wide = resolve_base_url(config)
    if process_wide:
        print(f"\n  NOTE: llm.base_url / $LLM_BASE_URL = {process_wide}")
        print("  Every model WITHOUT its own `provider:` tag is sent there, "
              "using the\n  query-stage key ($LLM_API_KEY, else OPENAI_API_KEY). "
              "Unset it if you\n  only meant to route the tagged models.")

    print(f"\n{len(rows) - failures}/{len(rows)} usable.")
    if failures:
        print("Fix the ✗ rows before running a benchmark — a bad id fails at "
              "the first call, after everything before it has been paid for.")
    if not args.call:
        print("Re-run with --call to prove the full path, not just the id.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
