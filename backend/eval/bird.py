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
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlglot

from backend.agents import usage as usage_tracking
from backend.eval.error_analysis import categorize, summarize

logger = logging.getLogger(__name__)


def resolve_model(cli_value: Optional[str], config: dict) -> str:
    """CLI --model wins when set (and non-blank); else config.yaml's llm.model.

    Any model id is accepted — this is a research CLI, not the public chat
    API, so we don't hard-block on config.yaml's llm.selectable_models (that
    allow-list exists to stop untrusted frontend input, not to constrain a
    user running their own eval). A model outside the list just gets a
    heads-up warning in case it's a typo.
    """
    model = (cli_value or "").strip() or config["llm"]["model"]
    selectable = config["llm"].get("selectable_models") or []
    if selectable and model not in selectable:
        logger.warning(
            "[EVAL] model %r is not in config.yaml's llm.selectable_models "
            "%s — proceeding anyway.", model, selectable)
    return model


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

    # Honesty metrics: a question whose GOLD errors on SQLite is unwinnable —
    # count them and report EX over the winnable subset too.
    gold_errors = sum(1 for r in results if r.get("gold_error"))
    winnable = [r for r in results if not r.get("gold_error")]
    ex_winnable = (
        round(sum(r["ex"] for r in winnable) / len(winnable) * 100, 2)
        if winnable else 0.0
    )

    usages = [r.get("usage") or {} for r in results]
    total_cost = sum(float(u.get("est_cost_usd") or 0.0) for u in usages)
    total_calls = sum(int(u.get("calls") or 0) for u in usages)

    return {
        "total": n,
        "EX":  round(ex_score, 2),
        "F1":  round(f1_score, 2),
        "VES": round(ves_score, 2),
        "gold_errors": gold_errors,
        "EX_winnable": ex_winnable,
        "error_categories": summarize(results),
        "llm_calls": total_calls,
        "llm_calls_per_question": round(total_calls / n, 2),
        "tokens_in": sum(int(u.get("input_tokens") or 0) for u in usages),
        "tokens_out": sum(int(u.get("output_tokens") or 0) for u in usages),
        "cost_usd": round(total_cost, 4),
        "cost_usd_per_question": round(total_cost / n, 6),
        "by_difficulty": by_difficulty,
    }


# ── Gold-dialect repair ────────────────────────────────────────────────────────
# Some mini-dev question files carry MySQL-flavoured gold SQL (DATE_FORMAT,
# YEAR(), CURRENT_TIMESTAMP()) that errors on SQLite — making those questions
# UNWINNABLE regardless of the system (observed: 8/50 on a moderate slice).
# When gold errors, we retry a mysql→sqlite transpile. One trap: SQLite's
# CAST(x AS DATETIME) coerces an ISO date string to a bare number, silently
# corrupting STRFTIME results — so those casts are stripped, not transpiled.

_CAST_DATETIME_RE = re.compile(r"CAST\(\s*([^()]+?)\s+AS\s+DATETIME\s*\)", re.IGNORECASE)


def gold_to_sqlite(gold_sql: str) -> Optional[str]:
    """Best-effort rewrite of a non-SQLite gold query. None when transpile fails."""
    try:
        out = sqlglot.transpile(gold_sql, read="mysql", write="sqlite")[0]
    except Exception:
        return None
    # STRFTIME('%Y', CAST(d AS DATETIME)) → STRFTIME('%Y', d): the cast breaks
    # date strings in SQLite (numeric coercion), it never helps.
    prev = None
    while prev != out:
        prev = out
        out = _CAST_DATETIME_RE.sub(r"\1", out)
    return out


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

    # Tie-break 1: prefer the group with a non-empty result set.
    non_empty = [(fp, sqls) for fp, sqls in tied if len(fp) > 0]
    if len(non_empty) == 1:
        fp, sqls = non_empty[0]
        info["winner_votes"] = len(sqls)
        info["fallback_reason"] = None
        return sqls[0], info

    # Tie-break 2 (Issue #27): soft-F1 consensus. A tied group whose rows overlap
    # the OTHER executed candidates most is likelier partially-correct than a
    # disjoint outlier. Vote-weighted; requires a STRICT max — fully disjoint
    # ties (all overlaps 0) still fall through to the text reranker.
    def _soft_f1(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        tp = len(a & b)
        p, r = tp / len(a), tp / len(b)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    pool = tied if len(tied) > 1 else list(groups.items())
    scores = {
        fp: sum(len(o_sqls) * _soft_f1(fp, o_fp)
                for o_fp, o_sqls in groups.items() if o_fp is not fp)
        for fp, _ in pool
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) >= 2 and ranked[0][1] > ranked[1][1] and ranked[0][1] > 0:
        winner_fp = ranked[0][0]
        sqls = dict(pool)[winner_fp]
        info["winner_votes"] = len(sqls)
        info["tie_break"] = "overlap"
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
    refine: bool = True,
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
    refine           : when True, one ReAct refine round runs on an observable
                       failure (invalid SQL, execution error, or 0 rows); the
                       refined SQL is kept only when it is strictly better
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

        # Official mini-dev ships {db_id}.sqlite; older/self-built sets use .db.
        db_path = str(Path(db_dir) / db_id / f"{db_id}.sqlite")
        if not Path(db_path).exists():
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
        # GH#57: account this question's LLM tokens/cost separately.
        q_usage = usage_tracking.start_request()
        if linker is not None:
            try:
                query_plan = linker.analyze(full_query, schema_context)
            except Exception as exc:
                logger.warning(f"[BIRD] schema linking failed qid={qid}: {exc}")

        def _select(candidates: List[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
            """Execution-vote (or plain rerank) over a candidate list."""
            if not candidates:
                return "", None
            if exec_voting:
                winner, v_info = select_by_execution(
                    candidates, db_path, validator, schema_context, timeout
                )
                if winner:
                    return winner, v_info
                return sql_agent.rerank(candidates, full_query, schema_context), v_info
            return sql_agent.rerank(candidates, full_query, schema_context), None

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
                tier=difficulty,   # BIRD provides the real tier; agent gates via config
            )
            pred_sql, vote_info = _select(sqls)
        except Exception as exc:
            logger.warning(f"[BIRD] Generation failed qid={qid}: {exc}")
        gen_ms = int((time.perf_counter() - t_gen) * 1000)

        # ── Execute gold (with dialect repair when the gold isn't SQLite) ────
        gold_rows, gold_error, gold_elapsed = execute_sqlite(gold_sql, db_path, timeout)
        gold_transpiled = False
        if gold_error:
            fixed = gold_to_sqlite(gold_sql)
            if fixed and fixed != gold_sql:
                rows_f, err_f, el_f = execute_sqlite(fixed, db_path, timeout)
                if not err_f:
                    logger.warning(
                        f"[BIRD] qid={qid} gold errored on SQLite "
                        f"({gold_error[:60]}) — using mysql→sqlite transpiled gold"
                    )
                    gold_sql, gold_rows, gold_error, gold_elapsed = fixed, rows_f, None, el_f
                    gold_transpiled = True

        def _attempt(sql: str) -> Tuple[Dict[str, Any], List[tuple], Optional[str], float]:
            """Validate a candidate and (if valid) execute it."""
            v: Dict[str, Any] = (
                validator.validate(sql, schema_context)
                if sql
                else {"valid": False, "error": "no SQL generated"}
            )
            rows: List[tuple] = []
            err: Optional[str] = None
            elapsed: float = 0.0
            if sql and v.get("valid") and not gold_error:
                rows, err, elapsed = execute_sqlite(sql, db_path, timeout)
            return v, rows, err, elapsed

        validation, pred_rows, pred_error, pred_elapsed = _attempt(pred_sql)

        # ── ReAct refine (one round): react to an observable failure ─────────
        # Triggers on outcomes visible WITHOUT looking at gold: no/invalid SQL,
        # an execution error, or a valid query that returned 0 rows.
        refined = False
        if refine and not gold_error:
            zero_rows = bool(
                pred_sql and validation.get("valid") and not pred_error and not pred_rows
            )
            if not pred_sql or not validation.get("valid") or pred_error or zero_rows:
                if pred_error:
                    observation = f"The query raised an execution error: {pred_error}"
                elif zero_rows:
                    observation = (
                        "The query is valid and executed successfully but returned 0 rows. "
                        "If a filter, JOIN, or value/format looks too strict or wrong, fix it; "
                        "if the query already looks correct, return it unchanged."
                    )
                else:
                    observation = (
                        f"The query failed validation: {validation.get('error', 'no SQL generated')}"
                    )
                try:
                    re_sqls = sql_agent.regenerate(
                        query=full_query,
                        schema_context=schema_context,
                        original_sql=pred_sql or "",
                        error=observation,
                        history=[],
                    )
                except Exception as exc:
                    logger.warning(f"[BIRD] refine failed qid={qid}: {exc}")
                    re_sqls = []
                new_sql, _ = _select(re_sqls)
                if new_sql:
                    v2, rows2, err2, t2 = _attempt(new_sql)
                    orig_ok = bool(pred_sql and validation.get("valid") and not pred_error)
                    accept = (
                        v2.get("valid") and not err2
                        and (not orig_ok or (zero_rows and rows2))
                    )
                    if accept:
                        reason = ("0 rows → " + str(len(rows2)) + " rows"
                                  if orig_ok else "was invalid/errored")
                        pred_sql, validation = new_sql, v2
                        pred_rows, pred_error, pred_elapsed = rows2, err2, t2
                        refined = True
                        logger.info(f"[BIRD] qid={qid} refine accepted ({reason})")

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
            "refined":        refined,
            "gold_transpiled": gold_transpiled,
            "query_plan":     query_plan,
            "usage":          q_usage.as_dict(),
            "error_category": categorize(
                ex, gold_error, pred_sql, bool(validation.get("valid")),
                pred_error, pred_rows, gold_rows,
            ),
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

def write_results(
    results: List[Dict],
    metrics: Dict,
    output_path: str,
    run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write per-question results + aggregate metrics to `output_path` (stable name,
    overwritten each run) AND to a versioned sibling
    `bird_<UTCstamp>_EX<score>.json` so A/B history is never lost (Issue #27).
    `run_config` embeds the knobs that produced the run for reproducibility.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_config": run_config or {}, "metrics": metrics, "questions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    out.write_text(text, encoding="utf-8")

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    versioned = out.parent / f"bird_{stamp}_EX{metrics.get('EX', 0):.1f}.json"
    versioned.write_text(text, encoding="utf-8")
    logger.info(f"[BIRD] Results written to {output_path} (versioned: {versioned.name})")


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
    parser.add_argument("--no-refine", action="store_true",
                        help="Disable the one-round ReAct refine on failure for A/B comparison")
    parser.add_argument("--model", default=None,
                        help="Override config.yaml's llm.model for this run (any model id)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-COMPATIBLE endpoint to benchmark against, e.g. "
                             "https://openrouter.ai/api/v1 or http://localhost:11434/v1. "
                             "Overrides $LLM_BASE_URL and config llm.base_url; "
                             "omit for OpenAI. Lets one BIRD run target Llama, "
                             "DeepSeek, Qwen, Mistral etc. (GH#65)")
    args = parser.parse_args(argv)

    from backend.config.config_loader import load_config
    from backend.agents.sql_code_agent import SQLCodeAgent
    from backend.agents.sql_validator_agent import SQLValidatorAgent
    from backend.agents.schema_linking_agent import SchemaLinkingAgent
    from backend.agents.llm_factory import (
        build_chat_llm, describe_target, resolve_base_url,
    )

    config = load_config()
    # Override dialect to sqlite so the agent and validator target SQLite syntax
    bird_config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}

    model_name = resolve_model(args.model, config)
    base_url = resolve_base_url(config, args.base_url)
    usage_tracking.set_pricing(config["llm"].get("pricing"))
    # Recorded in the run header: the same model id served by two providers can
    # differ in weights, quantisation and sampling defaults, so a benchmark
    # number without its endpoint is not reproducible.
    logger.info("[EVAL] target → %s", describe_target(model_name, base_url))
    llm = build_chat_llm(model_name, config["llm"]["api_key"], base_url)
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
    use_refine = not args.no_refine
    logger.info("[BIRD] execution voting: %s | refine: %s (candidates per question: %s)",
                "ON" if use_voting else "OFF", "ON" if use_refine else "OFF",
                sql_agent.top_k)

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
        refine=use_refine,
    )
    metrics = compute_bird_metrics(results)
    run_config = {
        "model": model_name,
        # GH#65: without the endpoint a cross-provider comparison isn't
        # attributable — the same id can be served with different weights,
        # quantisation or sampling defaults.
        "base_url": base_url or "openai-default",
        "dialect": "sqlite",
        "schema_linking": bool(linker),
        "exec_voting": use_voting,
        "refine": use_refine,
        "n_candidates": sql_agent.top_k,
        "adaptive_prompting": getattr(sql_agent, "_adaptive", None),
        "subset": args.subset,
        "difficulty": args.difficulty,
    }
    write_results(results, metrics, args.output, run_config=run_config)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── BIRD Evaluation Summary ───────────────────────────────────────────────")
    print(f"  Questions evaluated :   {metrics.get('total', 0)}")
    print(f"  Execution Accuracy  (EX)  :  {metrics.get('EX',  0):.2f}")
    print(f"  Soft F1-Score       (F1)  :  {metrics.get('F1',  0):.2f}")
    print(f"  Valid Efficiency    (VES) :  {metrics.get('VES', 0):.2f}")
    if metrics.get("gold_errors"):
        print(f"  ⚠️  UNWINNABLE (gold errored on SQLite): {metrics['gold_errors']} "
              f"question(s) — EX over winnable subset: {metrics.get('EX_winnable', 0):.2f}")
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
