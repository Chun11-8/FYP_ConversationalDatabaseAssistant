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
import os
import math
import re
import random
import sqlite3
import sys
import threading
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


# ── Question selection (GH#120) ───────────────────────────────────────────────

SAMPLE_STRATEGIES = ("head", "random", "stratified")


def select_questions(
    questions: List[Dict],
    difficulty_filter: Optional[str] = None,
    db_filter: Optional[str] = None,
    subset: Optional[int] = None,
    strategy: str = "head",
    seed: int = 42,
) -> List[Dict]:
    """
    Apply the filters, then take `subset` questions by `strategy` (GH#120).

    Filters run FIRST and `subset` LAST (GH#116): the other order would take
    the first N questions and then narrow them, so `--db-id x --subset 20`
    would evaluate however few of those 20 happened to belong to x — usually
    none, because dev.json is grouped by database.

    That grouping is also why `strategy` exists. mini_dev_sqlite.json stores
    each database as one contiguous block, so `head` — slicing the front of
    the file — returns questions from a SINGLE database. A 20-question run
    with no --db-id reads as an overall score and is really a
    debit_card_specializing score. The strategies:

      head        the file's own order. The pre-GH#120 behaviour, kept as the
                  DEFAULT so existing result files stay reproducible.
      random      a seeded shuffle before slicing — a genuine mix of domains,
                  though with 11 databases a small subset may miss some.
      stratified  proportional allocation across db_id (largest remainder,
                  at least one per database while the budget allows), shuffled
                  within each database and again at the end. Use this for a
                  cross-domain claim: it guarantees the coverage that `random`
                  only makes likely.

    `seed` makes every strategy reproducible; it is recorded in run_config,
    because a benchmark number that cannot be re-derived is not a measurement.
    """
    if strategy not in SAMPLE_STRATEGIES:
        raise ValueError(
            f"unknown sample strategy {strategy!r}; expected one of "
            f"{', '.join(SAMPLE_STRATEGIES)}")

    if difficulty_filter:
        questions = [q for q in questions if q.get("difficulty") == difficulty_filter]
    if db_filter:
        questions = [q for q in questions if q.get("db_id") == db_filter]

    if not subset or subset >= len(questions):
        return list(questions)

    rng = random.Random(seed)

    if strategy == "head":
        return questions[:subset]

    if strategy == "random":
        picked = list(questions)
        rng.shuffle(picked)
        return picked[:subset]

    # ── stratified ───────────────────────────────────────────────────────────
    by_db: Dict[str, List[Dict]] = {}
    for q in questions:
        by_db.setdefault(str(q.get("db_id")), []).append(q)
    for group in by_db.values():
        rng.shuffle(group)

    # Largest remainder: floor the proportional share, then hand the leftover
    # slots to the largest fractional parts. Ties break on the bigger database,
    # then on name, so the allocation does not depend on dict ordering.
    names = sorted(by_db)
    total = len(questions)
    exact = {db: subset * len(by_db[db]) / total for db in names}
    quota = {db: int(exact[db]) for db in names}

    # A database that rounds to zero still gets one slot while slots remain —
    # a "cross-domain" sample that silently omits a domain is the failure this
    # strategy exists to prevent.
    for db in sorted(names, key=lambda d: (-len(by_db[d]), d)):
        if sum(quota.values()) >= subset:
            break
        if quota[db] == 0:
            quota[db] = 1

    leftover = subset - sum(quota.values())
    if leftover > 0:
        order = sorted(names, key=lambda d: (-(exact[d] - int(exact[d])),
                                             -len(by_db[d]), d))
        for db in order:
            if leftover == 0:
                break
            if quota[db] < len(by_db[db]):
                quota[db] += 1
                leftover -= 1
    elif leftover < 0:
        # Over-allocated by the at-least-one pass; take back from the largest.
        order = sorted(names, key=lambda d: (-quota[d], d))
        for db in order:
            if leftover == 0:
                break
            if quota[db] > 1:
                quota[db] -= 1
                leftover += 1

    picked: List[Dict] = []
    for db in names:
        picked.extend(by_db[db][:quota[db]])
    rng.shuffle(picked)
    return picked[:subset]


# ── Per-question wall-clock budget (GH#119) ───────────────────────────────────

class _QuestionTimeout(Exception):
    """One question's LLM work ran past its wall-clock budget."""


class _Budget:
    """
    A wall-clock ceiling for one question, so a stalled call cannot hold the
    whole run (GH#119).

    A per-REQUEST timeout would not give the same guarantee: a moderate
    question makes roughly ten LLM calls, so a run can crawl for many minutes
    without any single request ever being slow enough to trip one. This bounds
    the question instead.

    The work runs on a DAEMON thread and is ABANDONED on expiry rather than
    cancelled — Python cannot interrupt a thread blocked in a socket read, and
    there is no honest way to pretend otherwise. Two consequences worth
    knowing: the abandoned call keeps running until the HTTP client gives up
    on its own (by default that is the OpenAI SDK's 600s per attempt, retries
    included), and it keeps billing until then. What this buys is that the
    LOOP moves on. Daemon, specifically, so an abandoned thread cannot hold
    the interpreter open at exit — which is exactly what a ThreadPoolExecutor
    would do, since its atexit hook joins every worker.
    """

    def __init__(self, seconds: Optional[float]) -> None:
        self._limit = float(seconds) if seconds and seconds > 0 else None
        self._t0 = time.perf_counter()

    @property
    def limit(self) -> Optional[float]:
        return self._limit

    def remaining(self) -> Optional[float]:
        if self._limit is None:
            return None
        return max(0.0, self._limit - (time.perf_counter() - self._t0))

    def expired(self) -> bool:
        left = self.remaining()
        return left is not None and left <= 0.0

    def run(self, fn: Any, label: str, tracker: Any = None) -> Any:
        """
        Call *fn* under the remaining budget, raising :class:`_QuestionTimeout`
        if it does not finish in time. An exception from *fn* propagates
        unchanged, so existing error handling around the call site still sees
        what it expects.

        *tracker* is re-bound inside the worker because token accounting is
        thread-local (``usage.bind`` exists for exactly this); without it every
        BIRD run would silently report zero tokens and zero cost.
        """
        if self._limit is None:
            return fn()

        box: Dict[str, Any] = {}

        def _target() -> None:
            if tracker is not None:
                usage_tracking.bind(tracker)
            try:
                box["value"] = fn()
            except BaseException as exc:            # noqa: BLE001 — re-raised below
                box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True,
                                  name=f"bird-{label}")
        thread.start()
        thread.join(self.remaining())
        if thread.is_alive():
            raise _QuestionTimeout(
                f"{label} exceeded the {self._limit:g}s question budget")
        if "error" in box:
            raise box["error"]
        return box.get("value")


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
        # GH#119: abandoned questions are scored 0 and stay in the denominator,
        # so a run with a non-zero count here UNDERSTATES the model. Surfaced
        # as its own number because the alternative — noticing it in the
        # error_categories dict — is how a slow run gets mistaken for a bad one.
        "timeouts": sum(1 for r in results if r.get("timed_out")),
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

def _verification_for(
    sql: str, rows: Any, question: str, evidence: str, schema_context: str,
    query_plan: Optional[dict], validation: Dict[str, Any],
    vote_info: Optional[Dict[str, Any]], refined: bool, judge_llm: Any,
) -> Dict[str, Any]:
    """
    GH#103: the live pipeline's verification block, assembled for one BIRD
    question so the detector can be scored against ground truth.

    Deterministic checks always run; they cost nothing. The back-translation
    round trip runs only when *judge_llm* is given, because it is two LLM calls
    per question and the literature's judge signal is the reason to pay them.

    Never raises: a failure here must cost the flags, not the benchmark run.
    The value-grounded checks stay silent — ValueIndex speaks to PostgreSQL and
    BIRD runs on SQLite — so those three codes remain unmeasured.
    """
    vote = vote_info or {}
    ev: Dict[str, Any] = {
        "candidates": int(vote.get("candidates") or 0),
        "executed": int(vote.get("executed") or 0),
        "agreement": int(vote.get("winner_votes") or 0),
        "distinct_result_sets": int(vote.get("groups") or 0),
        "decided_by": "plurality" if vote.get("executed") else "none_executed",
        "validated": bool(validation.get("valid")),
        "validation_checks": {k: bool(validation.get(k)) for k in
                              ("syntax_ok", "safety_ok", "semantic_ok")
                              if k in validation},
        "refine_rounds": 1 if refined else 0,
        "risks": [],
        "risk_score": 0,
        "back_translation": None,
    }
    if not sql:
        return ev

    try:
        from backend.agents import risk as risk_mod
        from backend.agents.sql_validator_agent import referenced_tables
        out = risk_mod.assess(
            sql=sql, rows=rows, question=question,
            schema_context=schema_context, query_plan=query_plan,
            tables=referenced_tables(sql, "sqlite"),
            value_index=None,          # PostgreSQL-only; BIRD runs on SQLite
        )
        ev["risks"] = out.get("risks") or []
        ev["risk_score"] = out.get("risk_score", 0)
    except Exception as exc:
        logger.warning(f"[BIRD] risk assessment failed: {exc}")

    if judge_llm is not None:
        try:
            from backend.agents.back_translation import round_trip
            # The question carries the BIRD evidence hint in the live prompt,
            # but the judge compares against what a USER asked — the hint is
            # not part of the question and would leak the intended reading.
            ev["back_translation"] = round_trip(
                describe_llm=judge_llm, compare_llm=judge_llm,
                sql=sql, question=question, schema_context=schema_context)
        except Exception as exc:
            logger.warning(f"[BIRD] back-translation failed: {exc}")
    return ev


def run_bird_eval(
    dev_json_path: str,
    db_dir: str,
    sql_agent: Any,
    validator: Any,
    subset: Optional[int] = None,
    difficulty_filter: Optional[str] = None,
    db_filter: Optional[str] = None,
    timeout: float = 30.0,
    linker: Any = None,
    exec_voting: bool = True,
    refine: bool = True,
    plan_diversity: bool = False,
    judge_llm: Any = None,
    question_timeout: Optional[float] = 60.0,
    sample: str = "head",
    seed: int = 42,
) -> List[Dict]:
    """
    Run BIRD evaluation and return a list of per-question result dicts.

    Parameters
    ----------
    dev_json_path    : path to BIRD dev.json
    db_dir           : root directory containing {db_id}/{db_id}.db files
    sql_agent        : SQLCodeAgent instance
    validator        : SQLValidatorAgent instance
    subset           : if set, evaluate only the first N questions — applied
                       LAST, so it slices what the filters already selected
    difficulty_filter: if set, restrict to "simple", "moderate", or "challenging"
    db_filter        : if set, restrict to one db_id (GH#116), e.g.
                       "debit_card_specializing"
    linker           : optional SchemaLinkingAgent — when provided, a structured
                       query plan is produced and fed into generation (Issue #17)
    exec_voting      : when True, candidates are selected by executing them and
                       plurality-voting on result fingerprints; the text reranker
                       only breaks ties (Tier 3, item 8)
    judge_llm        : GH#103 — when given, the winning SQL is read back into
                       plain English and compared with the question, and the
                       result is stored in the record's `verification` block.
                       None records the deterministic checks only.
    refine           : when True, one ReAct refine round runs on an observable
                       failure (invalid SQL, execution error, or 0 rows); the
                       refined SQL is kept only when it is strictly better
    timeout          : per-query SQLite execution timeout in seconds
    question_timeout : GH#119 — wall-clock ceiling in seconds for ONE question's
                       LLM work. On expiry the question is abandoned and scored
                       0, and the loop moves to the next one. 0 or None disables
                       it. This is NOT the same guard as `timeout`, which
                       bounds one SQLite execution: a question makes roughly
                       ten LLM calls, so a run can stall without any single
                       request being slow on its own.
    sample           : GH#120 — how `subset` picks its questions: "head" (the
                       file's order, the default and the old behaviour),
                       "random", or "stratified" (proportional across db_id).
                       dev.json is grouped by database, so "head" with no
                       db_filter returns ONE database's questions.
    seed             : RNG seed for "random" and "stratified", so a subset is
                       reproducible
    """
    all_questions = load_dev_json(dev_json_path)
    questions = select_questions(
        all_questions, difficulty_filter, db_filter, subset, sample, seed)

    total = len(questions)
    if total == 0:
        logger.warning(
            "[BIRD] No questions matched (difficulty=%s, db_id=%s). Check the "
            "spelling against the db_id values in dev.json.",
            difficulty_filter, db_filter)
    logger.info(f"[BIRD] Starting evaluation — {total} question(s)")

    # GH#120: say what was actually selected. A subset that quietly covers one
    # database reads in the results file exactly like an overall score.
    dbs = sorted({str(q.get("db_id")) for q in questions})
    if subset and total < subset:
        logger.warning(
            "[BIRD] --subset %d asked for more than the filters left (%d "
            "matched). Running all %d.", subset, total, total)
    if total:
        logger.info("[BIRD] Sampling: %s (seed=%d) — %d database(s): %s",
                    sample, seed, len(dbs), ", ".join(dbs))
    if sample == "head" and not db_filter and len(dbs) == 1 and subset:
        logger.warning(
            "[BIRD] All %d question(s) came from ONE database (%s). dev.json "
            "is grouped by database and --sample head slices the front of the "
            "file, so this is a single-domain run, not an overall score. Use "
            "--sample stratified for a cross-domain subset.", total, dbs[0])
    if question_timeout and question_timeout > 0:
        logger.info("[BIRD] Per-question budget: %gs (GH#119) — a question that "
                    "overruns is scored 0, not dropped", question_timeout)

    results: List[Dict] = []
    timed_out = 0

    for idx, item in enumerate(questions, start=1):
        qid = item.get("question_id", idx - 1)
        question: str = item["question"]
        evidence: str = item.get("evidence", "")
        gold_sql: str = item["SQL"]
        db_id: str = item["db_id"]
        difficulty: str = item.get("difficulty", "unknown")
        budget = _Budget(question_timeout)

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
                query_plan = budget.run(
                    lambda: linker.analyze(full_query, schema_context),
                    "schema-linking", q_usage)
            except _QuestionTimeout as exc:
                # Planning alone burned the budget. Generation without a plan
                # is the documented fallback (`query_plan=None` is a supported
                # input), but there is no time left to attempt it.
                logger.warning(f"[BIRD] qid={qid} SKIPPED — {exc}")
                results.append(_timeout_record(
                    qid, db_id, difficulty, question, evidence, gold_sql,
                    str(exc), q_usage))
                timed_out += 1
                continue
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
        timeout_note: Optional[str] = None
        try:
            def _gen(plan_for_batch):
                return sql_agent.generate(
                    query=full_query,
                    schema_context=schema_context,
                    few_shots=[],
                    history=[],
                    query_plan=plan_for_batch,
                    tier=difficulty,  # BIRD gives the real tier; agent gates via config
                )

            def _generate() -> Tuple[str, Optional[Dict[str, Any]]]:
                sqls = _gen(query_plan)
                # GH#76: candidates from a SECOND plan that disagrees on grain
                # or aggregation. Every candidate otherwise descends from one
                # plan, so voting confirms that plan rather than testing it.
                if plan_diversity and linker is not None and query_plan:
                    try:
                        alt = linker.analyze_alternative(
                            query=full_query, schema_context=schema_context,
                            first_plan=query_plan)
                        if alt:
                            sqls = sqls + [x for x in _gen(alt) if x not in sqls]
                    except Exception as exc:
                        logger.warning(f"[BIRD] alt-plan skipped qid={qid}: {exc}")
                return _select(sqls)

            pred_sql, vote_info = budget.run(_generate, "generation", q_usage)
        except _QuestionTimeout as exc:
            logger.warning(f"[BIRD] qid={qid} SKIPPED — {exc}")
            results.append(_timeout_record(
                qid, db_id, difficulty, question, evidence, gold_sql,
                str(exc), q_usage))
            timed_out += 1
            continue
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
        if refine and not gold_error and not budget.expired():
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
                # A refine that overruns keeps the ORIGINAL prediction rather
                # than discarding the question: generation already produced a
                # scorable answer, and throwing it away would turn a slow
                # optional improvement into a zero.
                try:
                    new_sql, _ = budget.run(
                        lambda: _select(sql_agent.regenerate(
                            query=full_query,
                            schema_context=schema_context,
                            original_sql=pred_sql or "",
                            error=observation,
                            history=[],
                        )),
                        "refine", q_usage)
                except _QuestionTimeout as exc:
                    logger.warning(
                        f"[BIRD] qid={qid} refine abandoned — {exc}; keeping "
                        f"the un-refined prediction")
                    timeout_note = str(exc)
                    new_sql = ""
                except Exception as exc:
                    logger.warning(f"[BIRD] refine failed qid={qid}: {exc}")
                    new_sql = ""
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

        # GH#103: the same evidence the live pipeline shows the user, stored
        # so trust_eval can score the detector against ground truth. Computed
        # here rather than reconstructed later because the executed ROWS and
        # the schema context exist only at this point — five of the ten checks
        # need them, and a harness replaying from disk cannot get them back.
        # The judge is the one remaining LLM call. Out of budget, it is
        # dropped rather than run past the ceiling — the ten deterministic
        # checks still run, and a dropped judge is recorded as NOT CHECKED
        # by _verification_for, never as nothing-wrong.
        judge_for_q = judge_llm
        if judge_llm is not None and budget.expired():
            logger.warning(f"[BIRD] qid={qid} judge skipped — question budget spent")
            timeout_note = timeout_note or "judge skipped: question budget spent"
            judge_for_q = None
        verification = _verification_for(
            sql=pred_sql, rows=pred_rows, question=question,
            evidence=evidence, schema_context=schema_context,
            query_plan=query_plan, validation=validation,
            vote_info=vote_info, refined=refined, judge_llm=judge_for_q,
        )

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
            "timeout_note":   timeout_note,
            "gold_transpiled": gold_transpiled,
            "query_plan":     query_plan,
            "verification":   verification,
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

    if timed_out:
        logger.warning(
            "[BIRD] %d/%d question(s) hit the %gs budget and were scored 0. "
            "They stay in the denominator — raise --question-timeout if the "
            "model is slow rather than stuck, or the score understates it.",
            timed_out, total, question_timeout)
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


def _timeout_record(
    qid: Any,
    db_id: str,
    difficulty: str,
    question: str,
    evidence: str,
    gold_sql: str,
    error: str,
    tracker: Any = None,
) -> Dict[str, Any]:
    """
    A question abandoned on its wall-clock budget (GH#119).

    Deliberately NOT ``_error_record``. That one reports through ``gold_error``,
    which means "this question is unwinnable — the BIRD gold SQL itself does
    not run on SQLite", and compute_bird_metrics drops those from the
    denominator entirely. Routing a timeout there would REMOVE the hardest,
    slowest questions from the score: a model that stalls on every challenging
    question would post a better EX than one that answers them slowly. The
    error belongs in ``pred_error`` — the model failed to produce an answer,
    which is a wrong answer, and it stays in the denominator scoring 0.
    """
    return {
        "question_id":    qid,
        "db_id":          db_id,
        "difficulty":     difficulty,
        "question":       question,
        "evidence":       evidence,
        "gold_sql":       gold_sql,
        "pred_sql":       "",
        "valid":          False,
        "gold_error":     None,
        "pred_error":     error,
        "timed_out":      True,
        "timeout_note":   error,
        "ex":             0.0,
        "f1":             0.0,
        "ves_reward":     0.0,
        "gold_row_count": 0,
        "pred_row_count": 0,
        "gen_ms":         0,
        "gold_elapsed_ms": 0,
        "pred_elapsed_ms": 0,
        "usage":          tracker.as_dict() if tracker is not None else None,
        "error_category": "timeout",
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
        "--db-id", default=None,
        help="Restrict evaluation to ONE database, e.g. debit_card_specializing.\n"
             "Applied before --subset, so --db-id X --subset 20 gives 20 of X's\n"
             "questions rather than however few of the first 20 belong to X.")
    parser.add_argument(
        "--difficulty",
        choices=["simple", "moderate", "challenging"],
        default=None,
        help="Restrict evaluation to a single difficulty tier",
    )
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="SQLite execution timeout per query in seconds (default: 30)")
    parser.add_argument("--sample", choices=SAMPLE_STRATEGIES, default="head",
                        help="GH#120: how --subset picks questions. dev.json is "
                             "GROUPED BY DATABASE, so 'head' (the default, and "
                             "the original behaviour) slices the front of the "
                             "file and returns ONE database's questions. Use "
                             "'stratified' for a cross-domain subset "
                             "(proportional across db_id, every database "
                             "represented), or 'random' for a seeded shuffle.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for --sample random/stratified, so a "
                             "subset is reproducible (default: 42)")
    parser.add_argument("--question-timeout", type=float, default=60.0,
                        help="GH#119: wall-clock budget in seconds for ONE "
                             "question's LLM work; on expiry the question is "
                             "abandoned, SCORED 0 (it stays in the denominator) "
                             "and the run moves on. 0 disables. Distinct from "
                             "--timeout, which bounds ONE SQLite execution: "
                             "this bounds a whole question's LLM work. "
                             "Default: 60")
    parser.add_argument("--no-schema-linking", action="store_true",
                        help="Disable the schema-linking stage (#17) for A/B comparison")
    parser.add_argument("--no-exec-voting", action="store_true",
                        help="Disable execution-guided candidate voting for A/B comparison")
    parser.add_argument("--no-refine", action="store_true",
                        help="Disable the one-round ReAct refine on failure for A/B comparison")
    parser.add_argument("--model", default=None,
                        help="Override config.yaml's llm.model for this run (any model id)")
    parser.add_argument("--plan-diversity", action="store_true",
                        help="GH#76: ask the linker for a SECOND plan that "
                             "disagrees on grain/aggregation and generate from "
                             "both. Default off — run WITH and WITHOUT to A/B.")
    parser.add_argument("--trust-eval", action="store_true",
                        help="GH#103: read the winning SQL back into plain "
                             "English and compare it with the question, storing "
                             "the result in each record's verification block. "
                             "Two extra LLM calls per question — it is the judge "
                             "signal trust_eval.py cannot measure without.")
    parser.add_argument("--judge-model", default=None,
                        help="Model for the --trust-eval round trip (default: "
                             "the run's own model). Point it at a DIFFERENT "
                             "provider to measure whether a cross-provider judge "
                             "sees errors the generator cannot.")
    parser.add_argument("--judge-base-url", default=None,
                        help="OpenAI-compatible endpoint for --judge-model.")
    parser.add_argument("--plan-rationale", action="store_true",
                        help="GH#76: make the linker name the answer's grain and "
                             "justify the aggregate against the evidence.")
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
        build_for_target, describe_target, resolve_api_key,
        resolve_judge_api_key, resolve_base_url, resolve_model_target,
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
    # GH#108: with no explicit --base-url, the model's own routing applies, so
    # `--model deepseek/deepseek-v4-flash` reaches OpenRouter without the
    # caller repeating the endpoint. An explicit --base-url still wins, which
    # is what benchmarking an id the config has never heard of needs.
    target = resolve_model_target(model_name, config)
    if base_url:
        target = target._replace(base_url=base_url,
                                 api_key=resolve_api_key(config, None))
    logger.info("[EVAL] target → %s", describe_target(target.model, target.base_url))
    llm = build_for_target(target)
    sql_agent = SQLCodeAgent(bird_config, llm)
    validator = SQLValidatorAgent(bird_config)

    # Schema linking on by default; --no-schema-linking turns it off for A/B runs.
    use_linking = not args.no_schema_linking and bool(
        config.get("sql", {}).get("schema_linking", True)
    )
    linker = (SchemaLinkingAgent(bird_config, llm,
                                 rationale=bool(args.plan_rationale))
              if use_linking else None)
    logger.info("[BIRD] schema linking: %s", "ON" if linker else "OFF")

    use_voting = not args.no_exec_voting and bool(
        config.get("sql", {}).get("execution_voting", True)
    )
    use_refine = not args.no_refine
    logger.info("[BIRD] execution voting: %s | refine: %s (candidates per question: %s)",
                "ON" if use_voting else "OFF", "ON" if use_refine else "OFF",
                sql_agent.top_k)

    # GH#103: judging with the generator's own model measures self-agreement;
    # a second provider is what the literature reports reaching 0.82 AUROC.
    judge_llm = None
    if args.trust_eval:
        judge_model = args.judge_model or model_name
        # GH#106/#108. Three cases, in order:
        #   --judge-base-url given  → it wins outright;
        #   --judge-model given     → that model's own routing (GH#108), which
        #                             is how a cross-provider judge is reached
        #                             without repeating the endpoint;
        #   neither                 → the generator's own target, i.e. a
        #                             self-judge measuring self-agreement.
        if args.judge_base_url:
            judge_target = resolve_model_target(judge_model, config)._replace(
                base_url=args.judge_base_url)
        elif args.judge_model:
            judge_target = resolve_model_target(judge_model, config)
        else:
            judge_target = target._replace(model=judge_model)
        # $JUDGE_API_KEY overrides; otherwise the target's own key is right —
        # it already came from that provider's env var.
        judge_target = judge_target._replace(
            api_key=(os.getenv("JUDGE_API_KEY") or "").strip()
                    or judge_target.api_key or resolve_judge_api_key(config))
        judge_llm = build_for_target(judge_target)
        logger.info("[BIRD] trust eval: ON | judge → %s",
                    describe_target(judge_model, judge_target.base_url))

    results = run_bird_eval(
        dev_json_path=args.dev_json,
        db_dir=args.db_dir,
        sql_agent=sql_agent,
        validator=validator,
        subset=args.subset,
        difficulty_filter=args.difficulty,
        db_filter=args.db_id,
        timeout=args.timeout,
        linker=linker,
        exec_voting=use_voting,
        refine=use_refine,
        plan_diversity=bool(args.plan_diversity),
        judge_llm=judge_llm,
        question_timeout=args.question_timeout,
        sample=args.sample,
        seed=args.seed,
    )
    metrics = compute_bird_metrics(results)
    run_config = {
        "model": model_name,
        # GH#65: without the endpoint a cross-provider comparison isn't
        # attributable — the same id can be served with different weights,
        # quantisation or sampling defaults.
        "base_url": base_url or "openai-default",
        "plan_diversity": bool(args.plan_diversity),
        "plan_rationale": bool(args.plan_rationale),
        "trust_eval": bool(args.trust_eval),
        "judge_model": (args.judge_model or model_name) if args.trust_eval else None,
        "judge_base_url": ((args.judge_base_url or base_url or "openai-default")
                           if args.trust_eval else None),
        "dialect": "sqlite",
        "schema_linking": bool(linker),
        "exec_voting": use_voting,
        "refine": use_refine,
        "n_candidates": sql_agent.top_k,
        "adaptive_prompting": getattr(sql_agent, "_adaptive", None),
        "subset": args.subset,
        "difficulty": args.difficulty,
        "db_id": args.db_id,
        # GH#119: recorded because it can CHANGE the score — two runs at
        # different budgets are not comparable if either one abandoned work.
        "question_timeout": args.question_timeout,
        # GH#120: which questions a subset actually contained. Without these
        # two, a results file cannot be re-derived or even interpreted — the
        # same --subset 20 means a different 20 under a different strategy.
        "sample": args.sample,
        "seed": args.seed,
        "databases_evaluated": sorted({str(r.get("db_id")) for r in results}),
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
    if metrics.get("timeouts"):
        print(f"  ⏱  ABANDONED (hit the per-question budget): {metrics['timeouts']} "
              f"question(s) — scored 0 and still counted; raise "
              f"--question-timeout if the model is slow rather than stuck")
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
