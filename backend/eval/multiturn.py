"""
multiturn.py — multi-turn conversation evaluation (GH#71).

BIRD is single-turn, so the conversational half of this system has never been
measured: follow-up condensation, the clarification loop, cross-turn cache
reuse and previous-SQL context all shipped without a regression test. The GH#63
bug — a follow-up differing only in a literal silently replaying the previous
turn's answer — reached production and was found by hand.

This harness replays scripted conversations through the FULL OrchestrationAgent
graph (not SQLCodeAgent directly, as bird.py does) because the mechanisms under
test live in the graph, not the generator.

Self-contained by design
------------------------
It runs against the reconstructed SQLite databases from smoke_examples.py, so
it needs **no BIRD download and no PostgreSQL** — only a key for SQL
generation. Two seams are redirected for the run:

* ``execute_query``  → the case's SQLite file instead of Postgres.
* ``retrieve``       → the case DB's FULL schema instead of a pgvector search.

Injecting the whole schema is deliberate, not a shortcut: these fixtures have
three tables each, so retrieval would be trivially correct anyway, and removing
it isolates what is actually being measured — whether the follow-up was
understood — from RAG noise that would otherwise confound every result.

The semantic cache still needs Postgres. Without it, cases asserting a cache
hit are reported as SKIPPED rather than failed (see `legitimate_repeat`).

Usage (from repo root)::

    python -m backend.eval.multiturn --output out/multiturn.json
    python -m backend.eval.multiturn --model deepseek-chat \\
        --base-url https://api.deepseek.com
    python -m backend.eval.multiturn --case literal-swap-month   # one case
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "multiturn_cases.json"

# Verdicts. The first is success; `stale_repeat` is the GH#63 signature and is
# kept distinct from `disjoint` on purpose — "returned the PREVIOUS answer" is a
# specific, diagnosable conversational failure, and burying it in generic
# wrongness is what let it ship.
CORRECT = "correct"
STALE_REPEAT = "stale_repeat"
WRONG = "wrong"
ZERO_ROWS = "zero_rows"
EXEC_ERROR = "exec_error"
NO_SQL = "no_sql"
SKIPPED = "skipped"


# ── scoring ───────────────────────────────────────────────────────────────────

def _row_set(rows: Optional[List[tuple]]) -> Optional[frozenset]:
    if rows is None:
        return None
    return frozenset(tuple(r) for r in rows)


def score_turn(
    pred_rows: Optional[List[tuple]],
    gold_rows: Optional[List[tuple]],
    prev_pred_rows: Optional[List[tuple]] = None,
    expect_differs_from_previous: bool = False,
    pred_error: Optional[str] = None,
    had_sql: bool = True,
) -> str:
    """
    One turn's verdict.

    ``stale_repeat`` outranks a plain wrong answer: when a turn was supposed to
    move on from the previous one and instead returned byte-identical rows,
    naming that specifically is the whole point of this harness.
    """
    if not had_sql:
        return NO_SQL
    if pred_error:
        return EXEC_ERROR
    pred, gold = _row_set(pred_rows), _row_set(gold_rows)
    if pred is None:
        return EXEC_ERROR
    if gold is not None and pred == gold:
        return CORRECT
    # Wrong — but is it wrong in the specific way that means "the conversation
    # was ignored"?
    if expect_differs_from_previous and prev_pred_rows is not None:
        if pred == _row_set(prev_pred_rows):
            return STALE_REPEAT
    if not pred_rows:
        return ZERO_ROWS
    return WRONG


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-turn records.

    Reports first-turn and follow-up accuracy SEPARATELY: a system can look
    fine on an aggregate dominated by opening questions while being useless at
    the thing it is named for. The gap between the two is the measurement.
    """
    turns = [r for r in results if r["verdict"] != SKIPPED]
    firsts = [r for r in turns if r["turn_index"] == 0]
    follows = [r for r in turns if r["turn_index"] > 0]

    def acc(rs: List[Dict[str, Any]]) -> Optional[float]:
        return round(100.0 * sum(r["verdict"] == CORRECT for r in rs) / len(rs), 1) \
            if rs else None

    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat, group in _group(follows, "category").items():
        by_cat[cat] = {"n": len(group), "accuracy": acc(group),
                       "verdicts": dict(Counter(r["verdict"] for r in group))}

    return {
        "cases": len({r["case_id"] for r in results}),
        "turns_scored": len(turns),
        "turns_skipped": sum(1 for r in results if r["verdict"] == SKIPPED),
        "accuracy_overall": acc(turns),
        "accuracy_first_turn": acc(firsts),
        "accuracy_followup": acc(follows),
        "stale_repeats": sum(1 for r in turns if r["verdict"] == STALE_REPEAT),
        "cache_hits": sum(1 for r in turns if r.get("cache_hit")),
        # Tracked apart from accuracy: a regenerated-but-correct answer is not
        # a wrong answer, it is a cost regression. If these go to zero the
        # cache has been over-corrected into uselessness — the failure mode
        # the `legitimate_repeat` cases exist to catch.
        "cache_expectations_met": sum(
            1 for r in turns if r.get("expect_cache_hit") and r.get("cache_hit")),
        "cache_expectations_missed": sum(
            1 for r in turns if r.get("expect_cache_hit") and not r.get("cache_hit")),
        "verdicts": dict(Counter(r["verdict"] for r in turns)),
        "by_category": by_cat,
    }


def _group(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r.get(key) or "uncategorised"].append(r)
    return dict(out)


# ── database plumbing ─────────────────────────────────────────────────────────

def prepare_databases(root: Path) -> Path:
    """Build the reconstructed SQLite fixtures (no download required)."""
    from backend.eval.smoke_examples import _prepare_reconstructed_dbs
    root.mkdir(parents=True, exist_ok=True)
    _prepare_reconstructed_dbs(root)
    return root


def db_path_for(root: Path, db_id: str) -> str:
    p = root / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(f"No database for {db_id} under {root}")
    return str(p)


def run_sqlite(db_path: str, sql: str, timeout: float = 15.0
               ) -> Tuple[Optional[List[tuple]], Optional[str]]:
    """(rows, error). Read-only: the connection is opened in immutable mode."""
    if not sql or not sql.strip():
        return None, "empty sql"
    try:
        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True,
                               timeout=timeout)
        try:
            return conn.execute(sql).fetchall(), None
        finally:
            conn.close()
    except Exception as exc:
        return None, str(exc)


# ── the run ───────────────────────────────────────────────────────────────────

def load_cases(path: Path = CASES_PATH, only: Optional[str] = None
               ) -> List[Dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] == only]
        if not cases:
            raise SystemExit(f"No case with id {only!r} in {path}")
    return cases


def run_case(agent: Any, case: Dict[str, Any], db_path: str,
             cache_enabled: bool) -> List[Dict[str, Any]]:
    """Replay one conversation in a single session and score every turn."""
    session_id = f"mt-{case['id']}-{uuid.uuid4().hex[:8]}"
    out: List[Dict[str, Any]] = []
    prev_pred_rows: Optional[List[tuple]] = None

    for i, turn in enumerate(case["turns"]):
        t0 = time.perf_counter()
        try:
            result = agent.run(session_id=session_id, user_query=turn["question"])
        except Exception as exc:
            logger.warning("[MT] %s turn %s failed: %s", case["id"], i, exc)
            result = {"sql": None, "response": f"ERROR: {exc}"}
        elapsed = int((time.perf_counter() - t0) * 1000)

        pred_sql = result.get("sql")
        gold_sql = turn.get("gold_sql")

        # A turn with no gold (an opener we allow to be answered with a
        # clarifying question) is recorded but never scored against rows.
        if not gold_sql:
            out.append(_record(case, i, turn, result, SKIPPED, None, None, elapsed,
                               note="no gold — clarification permitted"))
            prev_pred_rows = None
            continue

        # A cache-hit assertion is unverifiable without the cache; report that
        # honestly rather than failing the case for a missing dependency.
        if turn.get("expect_cache_hit") and not cache_enabled:
            out.append(_record(case, i, turn, result, SKIPPED, None, None, elapsed,
                               note="expect_cache_hit needs the semantic cache "
                                    "(PostgreSQL) — not available"))
            prev_pred_rows = None
            continue

        pred_rows, pred_err = (run_sqlite(db_path, pred_sql) if pred_sql
                               else (None, None))
        gold_rows, gold_err = run_sqlite(db_path, gold_sql)
        if gold_err:      # a broken fixture must not be scored as a model error
            out.append(_record(case, i, turn, result, SKIPPED, pred_rows, gold_rows,
                               elapsed, note=f"gold SQL failed: {gold_err}"))
            continue

        verdict = score_turn(
            pred_rows, gold_rows, prev_pred_rows,
            expect_differs_from_previous=bool(turn.get("expect_differs_from_previous")),
            pred_error=pred_err, had_sql=bool(pred_sql),
        )
        # A missed cache hit does NOT make the answer wrong — the rows can be
        # perfectly correct, just regenerated. It is tracked separately
        # (cache_expectations_* in the summary) because the thing it measures
        # is cost and over-correction, not accuracy.
        out.append(_record(case, i, turn, result, verdict, pred_rows, gold_rows,
                           elapsed))
        prev_pred_rows = pred_rows

    return out


def _record(case, i, turn, result, verdict, pred_rows, gold_rows, elapsed_ms,
            note: Optional[str] = None) -> Dict[str, Any]:
    v = result.get("verification") or {}
    return {
        "case_id": case["id"],
        "category": case.get("category"),
        "turn_index": i,
        "question": turn["question"],
        "verdict": verdict,
        "pred_sql": result.get("sql"),
        "gold_sql": turn.get("gold_sql"),
        "pred_row_count": len(pred_rows) if pred_rows is not None else None,
        "gold_row_count": len(gold_rows) if gold_rows is not None else None,
        # The two fields that say WHY a follow-up failed: what it was rewritten
        # into, and whether the answer came from cache rather than generation.
        "standalone_query": result.get("standalone_query"),
        "cache_hit": bool(result.get("cache_hit")),
        "expect_cache_hit": bool(turn.get("expect_cache_hit")),
        "agreement": v.get("agreement"),
        "decided_by": v.get("decided_by"),
        "elapsed_ms": elapsed_ms,
        "note": note,
    }


def build_agent(model: Optional[str], base_url: Optional[str], db_root: Path):
    """
    An OrchestrationAgent wired to the SQLite fixtures.

    ``execute_query`` and ``retrieve`` are redirected at module level in
    db_agent — the graph calls them as free functions, so this is the only
    seam — and the dialect is forced to sqlite so generation targets it.
    Returns (agent, set_active_db, cache_enabled).
    """
    import backend.agents.db_agent as dbm
    from backend.config.config_loader import load_config
    from backend.eval.bird import extract_sqlite_schema

    config = load_config()
    if model:
        config = {**config, "llm": {**config["llm"], "model": model}}
    if base_url:
        config = {**config, "llm": {**config["llm"], "base_url": base_url}}
    config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}

    active: Dict[str, str] = {"db_path": ""}
    schema_cache: Dict[str, str] = {}

    def _schema_for(db_path: str) -> str:
        if db_path not in schema_cache:
            schema_cache[db_path] = extract_sqlite_schema(db_path)
        return schema_cache[db_path]

    def fake_retrieve(query, db_name=None, top_k_tables=3, **kw) -> dict:
        ctx = _schema_for(active["db_path"])
        return {"prompt_context": ctx, "tables": [], "fk_tables": []}

    def fake_execute(sql, **kw):
        rows, err = run_sqlite(active["db_path"], sql)
        if err:
            raise RuntimeError(err)
        # The pipeline expects mapping-like rows; sqlite3 tuples would break
        # the formatter and the fingerprinting.
        conn = sqlite3.connect(f"file:{active['db_path']}?immutable=1", uri=True)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    dbm.retrieve = fake_retrieve
    dbm.execute_query = fake_execute
    # expand_fk_tables would try to widen a schema we already injected whole.
    dbm.expand_fk_tables = lambda chunks, *a, **kw: chunks

    agent = dbm.OrchestrationAgent(config)
    cache_enabled = bool(getattr(agent.semantic_cache, "is_enabled", False))

    def set_active_db(db_id: str) -> str:
        active["db_path"] = db_path_for(db_root, db_id)
        return active["db_path"]

    return agent, set_active_db, cache_enabled


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(CASES_PATH),
                        help="Case file (default: backend/eval/multiturn_cases.json)")
    parser.add_argument("--case", default=None, help="Run a single case by id")
    parser.add_argument("--output", default=None, help="Write the full result JSON here")
    parser.add_argument("--db-root", default=None,
                        help="Where to build the SQLite fixtures (default: a temp dir)")
    parser.add_argument("--model", default=None,
                        help="Override config.yaml's llm.model for this run")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cases = load_cases(Path(args.cases), args.case)
    db_root = Path(args.db_root) if args.db_root \
        else Path(__file__).parent / "_multiturn_dbs"
    prepare_databases(db_root)

    agent, set_active_db, cache_enabled = build_agent(
        args.model, args.base_url, db_root)
    if not cache_enabled:
        print("  [note] semantic cache unavailable — cache-hit cases will be skipped")

    results: List[Dict[str, Any]] = []
    for case in cases:
        set_active_db(case["db_id"])
        print(f"  • {case['id']} ({case['category']}) …", flush=True)
        results.extend(run_case(agent, case, db_path_for(db_root, case["db_id"]),
                                cache_enabled))

    metrics = summarize(results)
    _print_report(metrics, results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"metrics": metrics, "turns": results,
             "run_config": {"model": args.model or "(config default)",
                            "base_url": args.base_url or "openai-default"}},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Written to {out}")
    return 0


def _print_report(m: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("\n── Multi-turn Evaluation ─────────────────────────────────────────")
    print(f"  Cases              : {m['cases']}")
    print(f"  Turns scored       : {m['turns_scored']}"
          + (f"  ({m['turns_skipped']} skipped)" if m["turns_skipped"] else ""))
    print(f"  Accuracy (overall) : {m['accuracy_overall']}%")
    print(f"  Accuracy (1st turn): {m['accuracy_first_turn']}%")
    print(f"  Accuracy (follow-up): {m['accuracy_followup']}%   ← the real measure")
    print(f"  Stale repeats      : {m['stale_repeats']}"
          + ("   ⚠️  a follow-up replayed the previous answer"
             if m["stale_repeats"] else ""))
    print(f"  Cache hits         : {m['cache_hits']}"
          + (f"   ({m['cache_expectations_met']} of "
             f"{m['cache_expectations_met'] + m['cache_expectations_missed']} "
             f"expected reuses)"
             if (m["cache_expectations_met"] + m["cache_expectations_missed"])
             else ""))
    print("\n  Follow-ups by category")
    for cat, s in sorted(m["by_category"].items()):
        print(f"    {cat:<20} n={s['n']:<3} {str(s['accuracy']) + '%':>7}   {s['verdicts']}")
    bad = [r for r in results if r["verdict"] not in (CORRECT, SKIPPED)]
    if bad:
        print("\n  Failures")
        for r in bad:
            print(f"    [{r['verdict']:<13}] {r['case_id']} T{r['turn_index']}: "
                  f"{r['question'][:56]}")
            if r.get("standalone_query"):
                print(f"                     rewritten as: {r['standalone_query'][:70]}")
    print()


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
