"""
backend/eval/run_eval.py — Ground-truth evaluation harness.

Metrics
-------
EX  Execution Accuracy  — predicted result set == gold result set (order-insensitive)
HR  Hallucination Rate  — agent produced SQL that executed but returned wrong results
RA  Refusal Accuracy    — agent correctly abstained on unanswerable questions

Outcome taxonomy per question
-----------------------------
CORRECT   SQL executed; result set matches gold
WRONG     SQL executed; result set differs from gold
ABSTAINED No SQL generated / agent gave a graceful refusal / intent was not READ

Ground-truth format (JSON array or JSONL)
-----------------------------------------
[
  {
    "question_id": "q1",
    "question": "How many players have an overall rating above 80?",
    "gold_sql": "SELECT COUNT(*) FROM players WHERE overall > 80",
    "db_id": "FYP",
    "difficulty": "simple"    // optional: simple|moderate|challenging
  },
  ...
]

Unanswerable format (JSONL — one JSON object per line)
------------------------------------------------------
{"question_id": "u1", "question": "What is the weather in London?"}
{"question_id": "u2", "question": "Delete all players from Brazil"}

Usage
-----
    python -m backend.eval.run_eval \\
        --ground-truth backend/eval/ground_truth.json \\
        --unanswerable backend/eval/unanswerable.jsonl \\
        --output outputs/eval_results.csv \\
        [--session-prefix eval-2026] \\
        [--db FYP]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Result-set normalisation ────────────────────────────────────────────────────

def _normalise(rows: Any) -> frozenset:
    """Convert a result set to an order-insensitive frozenset of hashable rows."""
    if rows is None:
        return frozenset()
    if isinstance(rows, (int, float, str)):
        return frozenset({str(rows).strip()})
    if isinstance(rows, dict):
        # Single row returned as dict
        return frozenset({tuple(sorted((k, str(v)) for k, v in rows.items()))})
    if isinstance(rows, list):
        normalised = []
        for row in rows:
            if isinstance(row, dict):
                normalised.append(tuple(sorted((k, str(v)) for k, v in row.items())))
            elif isinstance(row, (list, tuple)):
                normalised.append(tuple(str(c) for c in row))
            else:
                normalised.append(str(row))
        return frozenset(normalised)
    return frozenset({str(rows)})


def results_match(gold_rows: Any, pred_rows: Any) -> bool:
    """Return True when the two result sets are semantically equal."""
    return _normalise(gold_rows) == _normalise(pred_rows)


# ── Ground-truth loading ────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> List[Dict]:
    """Load ground-truth entries from a JSON array file or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_unanswerable(path: Optional[str]) -> List[Dict]:
    """Load unanswerable questions from a JSONL file (returns [] if path is None)."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        logger.warning(f"Unanswerable file not found: {path} — skipping")
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Gold SQL execution ──────────────────────────────────────────────────────────

def execute_gold(sql: str, db_name: str) -> Tuple[Any, Optional[str]]:
    """
    Execute the gold SQL directly via psycopg2.
    Returns (rows, error_message). rows is None on error.
    """
    try:
        from backend.retrieval.data_processing import connect_postgres
        import psycopg2.extras
        conn = connect_postgres(db_name=db_name)
        if conn is None:
            return None, "Could not connect to database"
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
            return rows, None
        finally:
            conn.close()
    except Exception as e:
        return None, str(e)


# ── Outcome classifier ──────────────────────────────────────────────────────────

def classify_outcome(
    agent_result: Dict,
    gold_rows: Any,
    gold_error: Optional[str],
    db_name: str,
) -> Tuple[str, Optional[str]]:
    """
    Returns (outcome, note):
        outcome  — "CORRECT" | "WRONG" | "ABSTAINED"
        note     — human-readable reason
    """
    intent = agent_result.get("intent", "")
    pred_sql = agent_result.get("sql")
    validation = agent_result.get("validation", {})

    # Agent chose not to run SQL
    if not pred_sql or intent in ("GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"):
        return "ABSTAINED", f"intent={intent}, no sql produced"

    if not validation.get("valid"):
        return "ABSTAINED", f"sql invalid: {validation.get('error', 'unknown')}"

    # If gold itself errored (bad fixture), skip comparison
    if gold_error:
        return "ABSTAINED", f"gold sql failed: {gold_error}"

    # Execute predicted SQL
    pred_rows, pred_error = execute_gold(pred_sql, db_name)
    if pred_error:
        return "WRONG", f"pred sql execution error: {pred_error}"

    if results_match(gold_rows, pred_rows):
        return "CORRECT", None
    return "WRONG", f"result mismatch: gold={len(_normalise(gold_rows))} rows, pred={len(_normalise(pred_rows))} rows"


# ── Core evaluation loop ────────────────────────────────────────────────────────

def run_eval(
    agent,
    ground_truth: List[Dict],
    unanswerable: List[Dict],
    db_name: str,
    session_prefix: str,
    model: Optional[str] = None,
) -> List[Dict]:
    """
    Run the full evaluation and return a list of per-question result dicts.

    ``model``, when given, overrides the agent's configured default LLM model
    for every turn — reuses OrchestrationAgent.run()'s existing per-request
    override (GH#52), no extra plumbing needed.
    """
    results: List[Dict] = []

    # ── Answerable questions (EX metric) ───────────────────────────────────────
    logger.info(f"Evaluating {len(ground_truth)} answerable questions …")
    for item in ground_truth:
        qid = item.get("question_id", str(uuid.uuid4()))
        question = item["question"]
        gold_sql = item["gold_sql"]
        difficulty = item.get("difficulty", "unknown")

        session_id = f"{session_prefix}-{qid}"
        t0 = time.perf_counter()

        try:
            agent_result = agent.run(session_id=session_id, user_query=question, model=model)
        except Exception as e:
            logger.error(f"[EVAL] agent.run failed for {qid}: {e}")
            agent_result = {"response": "", "sql": None, "intent": "", "validation": {}, "row_count": None, "csv_path": None}

        latency_ms = int((time.perf_counter() - t0) * 1000)

        gold_rows, gold_error = execute_gold(gold_sql, db_name)
        outcome, note = classify_outcome(agent_result, gold_rows, gold_error, db_name)

        row = {
            "question_id": qid,
            "type": "answerable",
            "difficulty": difficulty,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": agent_result.get("sql") or "",
            "intent": agent_result.get("intent", ""),
            "outcome": outcome,
            "note": note or "",
            "latency_ms": latency_ms,
        }
        results.append(row)
        logger.info(f"  [{outcome}] {qid}: {question[:60]}")

    # ── Unanswerable questions (Refusal Accuracy) ──────────────────────────────
    logger.info(f"Evaluating {len(unanswerable)} unanswerable questions …")
    for item in unanswerable:
        qid = item.get("question_id", str(uuid.uuid4()))
        question = item["question"]
        session_id = f"{session_prefix}-unans-{qid}"
        t0 = time.perf_counter()

        try:
            agent_result = agent.run(session_id=session_id, user_query=question, model=model)
        except Exception as e:
            logger.error(f"[EVAL] agent.run failed for {qid}: {e}")
            agent_result = {"response": "", "sql": None, "intent": "", "validation": {}, "row_count": None, "csv_path": None}

        latency_ms = int((time.perf_counter() - t0) * 1000)
        pred_sql = agent_result.get("sql")
        intent = agent_result.get("intent", "")
        correctly_refused = not pred_sql or intent in ("GENERAL", "WRITE_BLOCKED", "AMBIGUOUS")
        outcome = "ABSTAINED" if correctly_refused else "WRONG"

        row = {
            "question_id": qid,
            "type": "unanswerable",
            "difficulty": "n/a",
            "question": question,
            "gold_sql": "",
            "pred_sql": pred_sql or "",
            "intent": intent,
            "outcome": outcome,
            "note": "correctly refused" if correctly_refused else "hallucinated SQL for unanswerable question",
            "latency_ms": latency_ms,
        }
        results.append(row)
        logger.info(f"  [{outcome}] {qid}: {question[:60]}")

    return results


# ── Metrics summary ─────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    def rate(subset, outcome):
        return round(sum(1 for r in subset if r["outcome"] == outcome) / len(subset), 4) if subset else 0.0

    # Execution Accuracy: correct / total answerable
    ex = rate(answerable, "CORRECT")
    # Hallucination Rate: wrong / total answerable
    hr = rate(answerable, "WRONG")
    # Refusal Accuracy: abstained / total unanswerable
    ra = rate(unanswerable, "ABSTAINED")

    by_difficulty: Dict[str, Dict] = {}
    for diff in {"simple", "moderate", "challenging"}:
        subset = [r for r in answerable if r["difficulty"] == diff]
        by_difficulty[diff] = {
            "count": len(subset),
            "ex": rate(subset, "CORRECT"),
            "hr": rate(subset, "WRONG"),
        }

    return {
        "total": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "EX": ex,
        "HR": hr,
        "RA": ra,
        "by_difficulty": by_difficulty,
    }


# ── CSV writer ──────────────────────────────────────────────────────────────────

def write_csv(results: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "type", "difficulty", "question", "gold_sql",
                  "pred_sql", "intent", "outcome", "note", "latency_ms"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results written to {output_path}")


# ── CLI entry point ─────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Text-to-SQL evaluation harness")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to ground-truth JSON/JSONL file")
    parser.add_argument("--unanswerable", default=None,
                        help="Path to unanswerable questions JSONL file")
    parser.add_argument("--output", default="outputs/eval_results.csv",
                        help="Output CSV path (default: outputs/eval_results.csv)")
    parser.add_argument("--db", default="FYP",
                        help="PostgreSQL database name (default: FYP)")
    parser.add_argument("--session-prefix", default=None,
                        help="Prefix for synthetic session IDs (default: eval-<uuid>)")
    parser.add_argument("--model", default=None,
                        help="Override config.yaml's llm.model for this run (any OpenAI model id)")
    args = parser.parse_args(argv)

    session_prefix = args.session_prefix or f"eval-{uuid.uuid4().hex[:8]}"

    # Lazy import so tests can mock before the agent is imported
    from backend.config.config_loader import load_config
    from backend.agents.db_agent import OrchestrationAgent

    config = load_config()
    agent = OrchestrationAgent(config)

    ground_truth = load_ground_truth(args.ground_truth)
    unanswerable = load_unanswerable(args.unanswerable)

    logger.info(f"Ground truth: {len(ground_truth)} questions | Unanswerable: {len(unanswerable)}")

    results = run_eval(agent, ground_truth, unanswerable, args.db, session_prefix, model=args.model)
    metrics = compute_metrics(results)

    write_csv(results, args.output)

    # Print summary
    print("\n── Evaluation Summary ────────────────────────────")
    print(f"  Execution Accuracy (EX):   {metrics['EX']:.1%}  ({sum(1 for r in results if r['outcome']=='CORRECT' and r['type']=='answerable')}/{metrics['answerable']})")
    print(f"  Hallucination Rate (HR):   {metrics['HR']:.1%}  ({sum(1 for r in results if r['outcome']=='WRONG' and r['type']=='answerable')}/{metrics['answerable']})")
    if metrics["unanswerable"]:
        print(f"  Refusal Accuracy   (RA):   {metrics['RA']:.1%}  ({sum(1 for r in results if r['outcome']=='ABSTAINED' and r['type']=='unanswerable')}/{metrics['unanswerable']})")
    print()
    for diff, stats in metrics["by_difficulty"].items():
        if stats["count"]:
            print(f"  {diff:12s}  EX={stats['ex']:.1%}  HR={stats['hr']:.1%}  n={stats['count']}")
    print(f"\n  Full results → {args.output}")
    print("──────────────────────────────────────────────────\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
