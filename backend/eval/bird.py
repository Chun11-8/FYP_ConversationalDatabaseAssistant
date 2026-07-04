"""
backend/eval/bird.py — BIRD mini_dev benchmark evaluation harness.

Evaluates text-to-SQL generation against the BIRD mini-dev dataset using SQLite
as the execution backend (no PostgreSQL required).

Dataset: https://github.com/bird-bench/mini_dev

BIRD dev.json format
--------------------
[
  {
    "question_id": 0,
    "question":    "How many schools are there?",
    "evidence":    "schools refers to CDSCode",   # domain-knowledge hint
    "SQL":         "SELECT COUNT(*) FROM schools",
    "db_id":       "california_schools",
    "difficulty":  "simple"                       # simple | moderate | challenging
  },
  ...
]

Directory layout expected under --db-dir
-----------------------------------------
{db_dir}/
  california_schools/
    california_schools.db     ← SQLite file
  formula_1/
    formula_1.db
  ...

Metrics
-------
EX    Execution Accuracy    — predicted result set == gold (set-equal, order-insensitive)
F1    Soft F1-Score         — row-level precision × recall harmonic mean
VES   Valid Efficiency Score — Σ sqrt(reward) × 100 / N, reward based on exec-time ratio

Usage
-----
    python -m backend.eval.bird \\
        --dev-json  path/to/mini_dev_data/dev.json \\
        --db-dir    path/to/mini_dev_data/dev_databases \\
        --output    outputs/bird_results.json \\
        [--subset 50] \\
        [--difficulty simple|moderate|challenging] \\
        [--timeout 30]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Schema extraction ──────────────────────────────────────────────────────────

# Value-grounding limits (Issue #19): show the actual distinct values of a
# low-cardinality TEXT column so the model maps phrases like "medium" to the exact
# stored value 'Medium'. Columns with too many distinct values, or values that are
# too long (free text), are skipped to keep the schema context tight.
MAX_DISTINCT_VALUES = 20
MAX_VALUE_LEN = 40


def _is_text_type(col_type: str) -> bool:
    """SQLite TEXT affinity heuristic: TEXT/CHAR/CLOB (or untyped) is text-like."""
    t = (col_type or "").upper()
    if not t:
        return True
    return any(tok in t for tok in ("CHAR", "TEXT", "CLOB"))


def _distinct_values(cursor, table: str, col: str) -> Optional[List[str]]:
    """
    Return the distinct non-null values of a text column when it is low-cardinality
    and the values are short; otherwise None (high-cardinality / free-text / error).
    """
    try:
        cursor.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" '
            f'WHERE "{col}" IS NOT NULL LIMIT {MAX_DISTINCT_VALUES + 1}'
        )
        rows = [r[0] for r in cursor.fetchall()]
    except Exception:
        return None

    if not rows or len(rows) > MAX_DISTINCT_VALUES:
        return None

    values: List[str] = []
    for v in rows:
        s = str(v).strip()
        if s == "":
            continue
        if len(s) > MAX_VALUE_LEN:   # a long value → free text, not a category
            return None
        values.append(s)
    return values or None


def extract_sqlite_schema(db_path: str) -> str:
    """
    Introspect a SQLite database and return a plain-text schema context string
    suitable for passing to SQLCodeAgent and SQLValidatorAgent.

    For low-cardinality TEXT columns the actual distinct values are appended
    (Issue #19) so the model can map question phrases to exact stored values.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        sections: List[str] = []
        fk_lines: List[str] = []

        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = cursor.fetchall()
            # col tuple: (cid, name, type, notnull, dflt_value, pk)
            col_strs = []
            for col in cols:
                col_name = col[1]
                col_type = col[2] or "TEXT"
                markers = []
                if col[5]:
                    markers.append("PK")
                if col[3]:
                    markers.append("NOT NULL")
                suffix = f" [{', '.join(markers)}]" if markers else ""
                line = f"  - {col_name}: {col_type}{suffix}"
                # Value grounding: append distinct values for low-card text cols.
                if not col[5] and _is_text_type(col_type):  # skip PKs (ids)
                    vals = _distinct_values(cursor, table, col_name)
                    if vals:
                        line += " | values: {" + ", ".join(vals) + "}"
                col_strs.append(line)
            sections.append(f"Table: {table}\nColumns:\n" + "\n".join(col_strs))

            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            fks = cursor.fetchall()
            for fk in fks:
                # fk tuple: (id, seq, to_table, from_col, to_col, on_update, on_delete, match)
                fk_lines.append(f"  {table}.{fk[3]} -> {fk[2]}.{fk[4]}")

        schema = "\n\n".join(sections)
        if fk_lines:
            schema += "\n\nForeign Key Relationships:\n" + "\n".join(fk_lines)
        return schema
    finally:
        conn.close()


# ── SQLite execution with timeout ─────────────────────────────────────────────

def execute_sqlite(
    sql: str,
    db_path: str,
    timeout: float = 30.0,
) -> Tuple[List[tuple], Optional[str], float]:
    """
    Execute SQL against a SQLite database file.

    Returns
    -------
    (rows, error, elapsed_seconds)
        rows    — list of result tuples (empty on error or no results)
        error   — error message string, or None on success
        elapsed — wall-clock execution time in seconds
    """

    def _run() -> Tuple[List[tuple], float]:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            t0 = time.perf_counter()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return rows, time.perf_counter() - t0
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        try:
            rows, elapsed = fut.result(timeout=timeout)
            return rows, None, elapsed
        except concurrent.futures.TimeoutError:
            return [], f"Query timed out after {timeout}s", timeout
        except Exception as exc:
            return [], str(exc), 0.0


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rows_to_set(rows: List[tuple]) -> frozenset:
    """Normalise rows to a frozenset of string-tuples for order-insensitive comparison."""
    return frozenset(
        tuple("" if v is None else str(v) for v in row)
        for row in rows
    )


def compute_ex(pred_rows: List[tuple], gold_rows: List[tuple]) -> float:
    """Execution Accuracy: 1.0 when result sets are set-equal, else 0.0."""
    return 1.0 if _rows_to_set(pred_rows) == _rows_to_set(gold_rows) else 0.0


def compute_f1(pred_rows: List[tuple], gold_rows: List[tuple]) -> float:
    """
    Soft F1-Score: harmonic mean of row-level precision and recall.
    Returns 1.0 when both sets are empty.
    """
    if not pred_rows and not gold_rows:
        return 1.0
    if not pred_rows or not gold_rows:
        return 0.0
    pred_set = _rows_to_set(pred_rows)
    gold_set = _rows_to_set(gold_rows)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ves_reward(pred_elapsed: float, gold_elapsed: float) -> float:
    """
    VES reward for a single correct query.

    ratio = gold_elapsed / pred_elapsed  (how many times faster pred is than gold)

    Reward schedule:
      ratio ≥ 2.0  →  1.25  (pred is 2×+ faster than gold)
      ratio 1–2    →  1.00  (pred is faster or comparable)
      ratio 0.5–1  →  0.75  (pred is up to 2× slower)
      ratio 0.25–0.5 → 0.50 (pred is 2–4× slower)
      ratio < 0.25 →  0.25  (pred is 4×+ slower)
    """
    if gold_elapsed <= 0:
        return 1.0
    ratio = gold_elapsed / max(pred_elapsed, 1e-9)
    if ratio >= 2.0:
        return 1.25
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.75
    if ratio >= 0.25:
        return 0.5
    return 0.25


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dev_json(path: str) -> List[Dict]:
    """Load BIRD dev.json (a JSON array) from the given file path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BIRD dev.json not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_bird_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate per-question results into BIRD benchmark metrics.

    EX  = mean(ex_i) × 100
    F1  = mean(f1_i) × 100
    VES = mean(sqrt(ves_reward_i)) × 100
    """
    n = len(results)
    if not n:
        return {"total": 0, "EX": 0.0, "F1": 0.0, "VES": 0.0, "by_difficulty": {}}

    ex_score = sum(r["ex"] for r in results) / n * 100
    f1_score = sum(r["f1"] for r in results) / n * 100
    ves_score = sum(math.sqrt(r["ves_reward"]) for r in results) / n * 100

    by_difficulty: Dict[str, Dict] = {}
    for diff in ("simple", "moderate", "challenging"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            sn = len(subset)
            by_difficulty[diff] = {
                "count": sn,
                "EX":  round(sum(r["ex"] for r in subset) / sn * 100, 2),
                "F1":  round(sum(r["f1"] for r in subset) / sn * 100, 2),
                "VES": round(
                    sum(math.sqrt(r["ves_reward"]) for r in subset) / sn * 100, 2
                ),
            }

    return {
        "total": n,
        "EX":  round(ex_score, 2),
        "F1":  round(f1_score, 2),
        "VES": round(ves_score, 2),
        "by_difficulty": by_difficulty,
    }


# ── Execution-guided candidate selection (Tier 3, item 8) ─────────────────────

def select_by_execution(
    candidates: List[str],
    db_path: str,
    validator: Any,
    schema_context: str,
    timeout: float = 30.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Pick the best SQL candidate by EXECUTING them and voting on the results.

    Candidates that fail validation (safety/syntax — keeps execution SELECT-only)
    or error at execution get no vote. The remaining candidates are grouped by an
    order-insensitive fingerprint of their result set; the plurality group wins.
    Tie-breaks: a non-empty result beats an empty one (wrong queries dispropor-
    tionately return 0 rows); an unresolved tie returns None so the caller falls
    back to the text reranker.

    Returns (winning_sql | None, info) — info is a small diagnostics dict.
    """
    info: Dict[str, Any] = {
        "candidates": len(candidates), "executed": 0,
        "groups": 0, "winner_votes": 0, "fallback_reason": None,
    }
    if not candidates:
        info["fallback_reason"] = "no candidates"
        return None, info

    # fingerprint → list of SQLs that produced that exact result set
    groups: Dict[frozenset, List[str]] = {}
    for sql in candidates:
        try:
            verdict = validator.validate(sql, schema_context)
        except Exception:
            continue
        if not verdict.get("valid"):
            continue  # never execute non-SELECT / unparseable candidates
        rows, err, _ = execute_sqlite(sql, db_path, timeout)
        if err:
            continue
        info["executed"] += 1
        groups.setdefault(_rows_to_set(rows), []).append(sql)

    info["groups"] = len(groups)
    if not groups:
        info["fallback_reason"] = "no candidate executed successfully"
        return None, info

    best_size = max(len(sqls) for sqls in groups.values())
    tied = [(fp, sqls) for fp, sqls in groups.items() if len(sqls) == best_size]

    # Unanimity among survivors, or a strict plurality → clear winner.
    if len(tied) == 1 and (best_size >= 2 or len(groups) == 1):
        fp, sqls = tied[0]
        info["winner_votes"] = best_size
        return sqls[0], info

    # Tie-break: prefer the group with a non-empty result set.
    non_empty = [(fp, sqls) for fp, sqls in tied if len(fp) > 0]
    if len(non_empty) == 1:
        fp, sqls = non_empty[0]
        info["winner_votes"] = len(sqls)
        info["fallback_reason"] = None
        return sqls[0], info

    info["fallback_reason"] = "tie between distinct result sets"
    return None, info


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_bird_eval(
    dev_json_path: str,
    db_dir: str,
    sql_agent: Any,
    validator: Any,
    subset: Optional[int] = None,
    difficulty_filter: Optional[str] = None,
    timeout: float = 30.0,
    linker: Any = None,
    exec_voting: bool = True,
) -> List[Dict]:
    """
    Run BIRD evaluation and return a list of per-question result dicts.

    Parameters
    ----------
    dev_json_path    : path to BIRD dev.json
    db_dir           : root directory containing {db_id}/{db_id}.db files
    sql_agent        : SQLCodeAgent instance
    validator        : SQLValidatorAgent instance
    subset           : if set, evaluate only the first N questions
    difficulty_filter: if set, restrict to "simple", "moderate", or "challenging"
    linker           : optional SchemaLinkingAgent — when provided, a structured
                       query plan is produced and fed into generation (Issue #17)
    exec_voting      : when True, candidates are selected by executing them and
                       plurality-voting on result fingerprints; the text reranker
                       only breaks ties (Tier 3, item 8)
    timeout          : per-query SQLite execution timeout in seconds
    """
    questions = load_dev_json(dev_json_path)

    if difficulty_filter:
        questions = [q for q in questions if q.get("difficulty") == difficulty_filter]
    if subset:
        questions = questions[:subset]

    total = len(questions)
    logger.info(f"[BIRD] Starting evaluation — {total} question(s)")

    results: List[Dict] = []

    for idx, item in enumerate(questions, start=1):
        qid = item.get("question_id", idx - 1)
        question: str = item["question"]
        evidence: str = item.get("evidence", "")
        gold_sql: str = item["SQL"]
        db_id: str = item["db_id"]
        difficulty: str = item.get("difficulty", "unknown")

        db_path = str(Path(db_dir) / db_id / f"{db_id}.db")

        # ── Schema ───────────────────────────────────────────────────────────
        try:
            schema_context = extract_sqlite_schema(db_path)
        except Exception as exc:
            logger.error(f"[BIRD] Schema extraction failed for db={db_id}: {exc}")
            results.append(
                _error_record(qid, db_id, difficulty, question, evidence, gold_sql, str(exc))
            )
            continue

        # ── SQL generation ───────────────────────────────────────────────────
        # Promote the BIRD evidence hint into a structured block so the generator
        # gives it higher salience — it often names the exact column/table needed
        # for a correct JOIN.
        full_query = (
            f"<domain_context>{evidence.strip()}</domain_context>\n\n{question}"
            if evidence.strip()
            else question
        )
        # Schema-linking & question-understanding (Issue #17): produce a
        # structured plan first so generation follows resolved tables/joins.
        query_plan = None
        if linker is not None:
            try:
                query_plan = linker.analyze(full_query, schema_context)
            except Exception as exc:
                logger.warning(f"[BIRD] schema linking failed qid={qid}: {exc}")

        t_gen = time.perf_counter()
        pred_sql = ""
        vote_info: Optional[Dict[str, Any]] = None
        try:
            sqls = sql_agent.generate(
                query=full_query,
                schema_context=schema_context,
                few_shots=[],
                history=[],
                query_plan=query_plan,
            )
            if sqls:
                if exec_voting:
                    # Execution-guided selection; reranker only breaks ties.
                    winner, vote_info = select_by_execution(
                        sqls, db_path, validator, schema_context, timeout
                    )
                    if winner:
                        pred_sql = winner
                        logger.info(
                            f"[BIRD] qid={qid} exec-vote winner "
                            f"({vote_info['winner_votes']}/{vote_info['executed']} votes, "
                            f"{vote_info['groups']} distinct result sets)"
                        )
                    else:
                        pred_sql = sql_agent.rerank(sqls, full_query, schema_context)
                        logger.info(
                            f"[BIRD] qid={qid} exec-vote fallback → reranker "
                            f"({vote_info['fallback_reason']})"
                        )
                else:
                    pred_sql = sql_agent.rerank(sqls, full_query, schema_context)
        except Exception as exc:
            logger.warning(f"[BIRD] Generation failed qid={qid}: {exc}")
        gen_ms = int((time.perf_counter() - t_gen) * 1000)

        # ── Validation ───────────────────────────────────────────────────────
        validation: Dict[str, Any] = (
            validator.validate(pred_sql, schema_context)
            if pred_sql
            else {"valid": False, "error": "no SQL generated"}
        )

        # ── Execute gold ─────────────────────────────────────────────────────
        gold_rows, gold_error, gold_elapsed = execute_sqlite(gold_sql, db_path, timeout)

        # ── Execute predicted (only if valid and gold succeeded) ─────────────
        pred_rows: List[tuple] = []
        pred_error: Optional[str] = None
        pred_elapsed: float = 0.0
        if pred_sql and validation.get("valid") and not gold_error:
            pred_rows, pred_error, pred_elapsed = execute_sqlite(pred_sql, db_path, timeout)

        # ── Compute metrics ──────────────────────────────────────────────────
        can_score = (
            pred_sql
            and validation.get("valid")
            and not gold_error
            and not pred_error
        )
        if can_score:
            ex = compute_ex(pred_rows, gold_rows)
            f1 = compute_f1(pred_rows, gold_rows)
            ves_reward = compute_ves_reward(pred_elapsed, gold_elapsed) if ex == 1.0 else 0.0
        else:
            ex = f1 = ves_reward = 0.0

        record: Dict[str, Any] = {
            "question_id":    qid,
            "db_id":          db_id,
            "difficulty":     difficulty,
            "question":       question,
            "evidence":       evidence,
            "gold_sql":       gold_sql,
            "pred_sql":       pred_sql,
            "valid":          validation.get("valid", False),
            "gold_error":     gold_error,
            "pred_error":     pred_error,
            "ex":             ex,
            "f1":             f1,
            "ves_reward":     ves_reward,
            "gold_row_count": len(gold_rows),
            "pred_row_count": len(pred_rows),
            "gen_ms":         gen_ms,
            "gold_elapsed_ms": int(gold_elapsed * 1000),
            "pred_elapsed_ms": int(pred_elapsed * 1000),
            "exec_vote":      vote_info,
        }
        results.append(record)

        mark = "✓" if ex == 1.0 else "✗"
        logger.info(
            f"[{idx}/{total}] {mark} EX={ex} F1={f1:.2f} "
            f"db={db_id} diff={difficulty} | {question[:55]}"
        )

    return results


def _error_record(
    qid: Any,
    db_id: str,
    difficulty: str,
    question: str,
    evidence: str,
    gold_sql: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "question_id":    qid,
        "db_id":          db_id,
        "difficulty":     difficulty,
        "question":       question,
        "evidence":       evidence,
        "gold_sql":       gold_sql,
        "pred_sql":       "",
        "valid":          False,
        "gold_error":     error,
        "pred_error":     None,
        "ex":             0.0,
        "f1":             0.0,
        "ves_reward":     0.0,
        "gold_row_count": 0,
        "pred_row_count": 0,
        "gen_ms":         0,
        "gold_elapsed_ms": 0,
        "pred_elapsed_ms": 0,
    }


# ── JSON output ────────────────────────────────────────────────────────────────

def write_results(results: List[Dict], metrics: Dict, output_path: str) -> None:
    """Write per-question results and aggregate metrics to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "questions": results}
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"[BIRD] Results written to {output_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="BIRD mini-dev text-to-SQL evaluation harness"
    )
    parser.add_argument("--dev-json", required=True,
                        help="Path to BIRD dev.json")
    parser.add_argument("--db-dir", required=True,
                        help="Root directory containing {db_id}/{db_id}.db SQLite files")
    parser.add_argument("--output", default="outputs/bird_results.json",
                        help="Output JSON path (default: outputs/bird_results.json)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Evaluate only the first N questions (useful for smoke tests)")
    parser.add_argument(
        "--difficulty",
        choices=["simple", "moderate", "challenging"],
        default=None,
        help="Restrict evaluation to a single difficulty tier",
    )
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="SQLite execution timeout per query in seconds (default: 30)")
    parser.add_argument("--no-schema-linking", action="store_true",
                        help="Disable the schema-linking stage (#17) for A/B comparison")
    parser.add_argument("--no-exec-voting", action="store_true",
                        help="Disable execution-guided candidate voting for A/B comparison")
    args = parser.parse_args(argv)

    from backend.config.config_loader import load_config
    from backend.agents.sql_code_agent import SQLCodeAgent
    from backend.agents.sql_validator_agent import SQLValidatorAgent
    from backend.agents.schema_linking_agent import SchemaLinkingAgent
    from langchain_openai import ChatOpenAI

    config = load_config()
    # Override dialect to sqlite so the agent and validator target SQLite syntax
    bird_config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}

    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0,
        api_key=config["llm"]["api_key"],
    )
    sql_agent = SQLCodeAgent(bird_config, llm)
    validator = SQLValidatorAgent(bird_config)

    # Schema linking on by default; --no-schema-linking turns it off for A/B runs.
    use_linking = not args.no_schema_linking and bool(
        config.get("sql", {}).get("schema_linking", True)
    )
    linker = SchemaLinkingAgent(bird_config, llm) if use_linking else None
    logger.info("[BIRD] schema linking: %s", "ON" if linker else "OFF")

    use_voting = not args.no_exec_voting and bool(
        config.get("sql", {}).get("execution_voting", True)
    )
    logger.info("[BIRD] execution voting: %s (candidates per question: %s)",
                "ON" if use_voting else "OFF", sql_agent.top_k)

    results = run_bird_eval(
        dev_json_path=args.dev_json,
        db_dir=args.db_dir,
        sql_agent=sql_agent,
        validator=validator,
        subset=args.subset,
        difficulty_filter=args.difficulty,
        timeout=args.timeout,
        linker=linker,
        exec_voting=use_voting,
    )
    metrics = compute_bird_metrics(results)
    write_results(results, metrics, args.output)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── BIRD Evaluation Summary ───────────────────────────────────────────────")
    print(f"  Questions evaluated :   {metrics.get('total', 0)}")
    print(f"  Execution Accuracy  (EX)  :  {metrics.get('EX',  0):.2f}")
    print(f"  Soft F1-Score       (F1)  :  {metrics.get('F1',  0):.2f}")
    print(f"  Valid Efficiency    (VES) :  {metrics.get('VES', 0):.2f}")
    if metrics.get("by_difficulty"):
        print()
        for diff, stats in metrics["by_difficulty"].items():
            print(
                f"  {diff:12s}  EX={stats['EX']:.2f}  "
                f"F1={stats['F1']:.2f}  VES={stats['VES']:.2f}  n={stats['count']}"
            )
    print(f"\n  Full results → {args.output}")
    print("──────────────────────────────────────────────────────────────────────────\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
