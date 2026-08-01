"""
backend/memory/skeleton.py — DAIL-SQL-style structural-shape signatures for
blending into few-shot selection (GH#48-50 companion; see semantic_cache.py's
retrieve_few_shots_blended).

Pure embedding-distance few-shot retrieval can surface a topically-similar
but structurally-mismatched example (a plain COUNT question for a grouped-
superlative one, or vice versa) while missing a structurally-matching
example that's phrased differently. This module gives both sides of the
comparison a small, comparable "shape" — a frozenset of tags — cheap to
compute and never raising on bad input.

Two different sources feed the two sides, by necessity: candidate few-shot
pairs have real, stored SQL (skeletonized precisely via sqlglot's AST), but
the CURRENT question has no SQL yet at retrieval time (retrieve_schema runs
before link_schema/generate_sql in the pipeline — see db_agent.py) — only
its raw NL text, so its shape is a best-effort keyword heuristic. Both sides
share the same tag vocabulary so they're comparable via skeleton_overlap().
"""

from __future__ import annotations

import re
from typing import FrozenSet

import sqlglot

# Shared tag vocabulary between the two sides.
_AGG_TAGS = {"Count": "AGG:COUNT", "Sum": "AGG:SUM", "Avg": "AGG:AVG",
             "Max": "AGG:MAX", "Min": "AGG:MIN"}


def sql_skeleton(sql: str, dialect: str = "sqlite") -> FrozenSet[str]:
    """
    Structural-shape tags for a real SQL query, via sqlglot AST parsing.
    Empty frozenset on any parse failure — never raises, so a malformed
    stored pair just contributes no structural signal (embedding distance
    alone decides for it).
    """
    if not sql or not sql.strip():
        return frozenset()
    try:
        ast = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return frozenset()

    tags = set()
    for agg in ast.find_all(sqlglot.exp.AggFunc):
        tag = _AGG_TAGS.get(type(agg).__name__)
        if tag:
            tags.add(tag)

    if ast.find(sqlglot.exp.Group) is not None:
        tags.add("GROUP_BY")
    if ast.find(sqlglot.exp.Having) is not None:
        tags.add("HAVING")
    # ORDER BY + LIMIT together is the grouped/plain-superlative shape.
    if ast.find(sqlglot.exp.Order) is not None and ast.find(sqlglot.exp.Limit) is not None:
        tags.add("SUPERLATIVE")
    if ast.find(sqlglot.exp.Except) is not None or ast.find(sqlglot.exp.Not) is not None:
        tags.add("EXCLUSION")

    n_joins = sum(1 for _ in ast.find_all(sqlglot.exp.Join))
    tags.add(f"JOINS:{'3+' if n_joins >= 3 else n_joins}")

    select = ast.find(sqlglot.exp.Select)
    n_cols = len(select.expressions) if select else 0
    tags.add(f"COLS:{'3+' if n_cols >= 3 else n_cols}")

    return frozenset(tags)


# Ordered so a more specific phrase (e.g. "how many") isn't shadowed by a
# generic one; each regex is checked independently, not mutually exclusive.
_QUESTION_CUES = (
    (re.compile(r"\bhow many\b|\bcount of\b|\bnumber of\b", re.IGNORECASE), "AGG:COUNT"),
    (re.compile(r"\btotal\b|\bsum of\b", re.IGNORECASE), "AGG:SUM"),
    (re.compile(r"\baverage\b|\bmean\b", re.IGNORECASE), "AGG:AVG"),
    (re.compile(r"\bhighest\b|\bmost\b|\btop\b|\bmaximum\b|\bpeak\b|\bmax\b",
                re.IGNORECASE), "SUPERLATIVE"),
    (re.compile(r"\blowest\b|\bleast\b|\bminimum\b|\bmin\b", re.IGNORECASE), "SUPERLATIVE"),
    (re.compile(r"\beach\b|\bper\b|\bby\s+\w+\s*$|\bgrouped by\b", re.IGNORECASE), "GROUP_BY"),
    (re.compile(r"\bnot\b|\bwithout\b|\bexcept\b|\bexcluding\b|\bother than\b",
                re.IGNORECASE), "EXCLUSION"),
    (re.compile(r"\band also\b|\bas well as\b|\?.*\?", re.IGNORECASE), "MULTI_COL"),
)


def question_shape_signature(question: str) -> FrozenSet[str]:
    """
    Best-effort structural-shape tags derived from a NL question's surface
    phrasing — no SQL/query-plan available yet at retrieval time. Coarse by
    nature; a question matching no cue returns an empty set (falls back to
    pure embedding-distance ranking for that comparison).
    """
    if not question:
        return frozenset()
    return frozenset(tag for pattern, tag in _QUESTION_CUES if pattern.search(question))


def skeleton_overlap(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard similarity; 0.0 when either side is empty (no false 'perfect match')."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
