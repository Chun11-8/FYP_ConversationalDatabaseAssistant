"""
error_analysis.py — automatic failure categorization for BIRD runs.

Every evaluated question gets an `error_category` tag computed WHILE the result
rows are still in hand (the JSON only persists counts, so categories like
arity-mismatch cannot be reconstructed afterwards). This turns each run into a
self-clustering failure report instead of manual JSON archaeology.

Categories (first match wins):
    correct          — EX = 1.0
    gold_error       — gold SQL failed on SQLite even after dialect repair (unwinnable)
    no_pred          — no SQL was generated
    pred_invalid     — generated SQL failed validation
    pred_exec_error  — valid SQL errored at execution
    zero_rows        — valid SQL executed, returned 0 rows while gold has rows
    arity_mismatch   — rows returned but with a different column COUNT than gold
                       (fingerprint of wrong output shape, e.g. concatenated names)
    superset         — every gold row present plus extras (missing a filter)
    subset           — only gold rows returned but some missing (over-filtering)
    partial_overlap  — some rows shared (right neighbourhood, wrong logic)
    disjoint         — non-empty result sharing nothing with gold (wrong query)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _row_set(rows: List[tuple]) -> frozenset:
    return frozenset(
        tuple("" if v is None else str(v) for v in row) for row in rows
    )


def categorize(
    ex: float,
    gold_error: Optional[str],
    pred_sql: str,
    valid: bool,
    pred_error: Optional[str],
    pred_rows: List[tuple],
    gold_rows: List[tuple],
) -> str:
    """Assign one error category to an evaluated question (see module docstring)."""
    if ex == 1.0:
        return "correct"
    if gold_error:
        return "gold_error"
    if not pred_sql:
        return "no_pred"
    if not valid:
        return "pred_invalid"
    if pred_error:
        return "pred_exec_error"
    if not pred_rows:
        return "zero_rows" if gold_rows else "disjoint"

    # Both sides returned rows — compare shape then content.
    if gold_rows and len(pred_rows[0]) != len(gold_rows[0]):
        return "arity_mismatch"

    pred_set, gold_set = _row_set(pred_rows), _row_set(gold_rows)
    shared = pred_set & gold_set
    if not shared:
        return "disjoint"
    if gold_set <= pred_set:
        return "superset"
    if pred_set <= gold_set:
        return "subset"
    return "partial_overlap"


def summarize(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count error categories across a run's records (skips untagged records)."""
    return dict(Counter(
        r["error_category"] for r in results if r.get("error_category")
    ))
