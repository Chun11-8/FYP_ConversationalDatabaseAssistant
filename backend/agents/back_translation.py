"""
back_translation.py — does the query describe what was actually asked? (GH#91)

Every other check reads the SQL against the schema, the plan or itself. All of
them pass on the failure that matters most: a query that is safe, fully
schema-grounded, plan-conformant, and unanimous across candidates — and
answers a DIFFERENT question. Two observed examples:

  * "who had the least consumption in LAM?" → MIN() over single monthly rows
    instead of SUM() per customer. Valid, grounded, unanimous, wrong question.
  * "customers from the country Wakanda" → `segment = 'LAM' AND currency =
    'EUR'`. Real columns, real values, internally consistent, wrong question.

The round trip closes that gap. The SQL is described in plain English WITHOUT
showing the model the original question — otherwise it paraphrases the
question rather than reading the query — and the description is then compared
with what the user asked.

Two calls, the comparison on the cheap intent model. Never raises: any failure
returns "unknown", which the grader treats as "not checked", never as a fault.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.usage import invoke_tracked
from backend.agents.utils import strip_reasoning

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 400

_DESCRIBE_SYSTEM = (
    "You explain SQL to someone who cannot read it. You are given a query and "
    "the schema it runs against — and deliberately NOT the question it was "
    "written for.\n\n"
    "In ONE sentence, state what the query returns: the unit of each row, the "
    "quantity computed, and every filter applied. Name real column values "
    "where the query filters on them.\n\n"
    "Be literal. Describe what the SQL DOES, not what it was probably meant to "
    "do — if it takes the smallest single row, say so; do not say 'the total'. "
    "No preamble, no SQL, one sentence."
)

_COMPARE_SYSTEM = (
    "You are checking whether a database query answers the question it was "
    "written for.\n\n"
    "You are given the user's QUESTION and a plain-English description of what "
    "a query actually returns. Decide whether the description answers that "
    "question.\n\n"
    "Differences that MATTER: a different quantity (a total vs a single "
    "smallest value), a different unit per row (per customer vs per month), a "
    "missing filter the question asked for, an extra filter it did not, "
    "counting rows vs counting distinct things.\n\n"
    "Differences that DO NOT matter: wording, column names, row ordering, "
    "extra descriptive columns, how a date range is expressed.\n\n"
    "Reply on two lines, exactly:\n"
    "VERDICT: MATCH or MISMATCH\n"
    "REASON: one short sentence a non-technical user would understand. For "
    "MISMATCH, say what the query does instead of what was asked."
)


def describe_sql(llm: Any, sql: str, schema_context: str = "") -> Optional[str]:
    """One-sentence plain-English reading of *sql*. None on any failure."""
    if not sql or not sql.strip():
        return None
    schema = (schema_context or "")[:3000]
    prompt = (f"<schema>\n{schema}\n</schema>\n\n" if schema else "") + \
             f"<sql>\n{sql}\n</sql>\n\nWhat does this query return?"
    try:
        raw = invoke_tracked(
            llm, [SystemMessage(content=_DESCRIBE_SYSTEM),
                  HumanMessage(content=prompt)], "back_translate").content
    except Exception as exc:
        logger.warning(f"[BACKTRANS] describe failed: {exc}")
        return None
    text = " ".join(strip_reasoning(raw or "").split())
    if not text:
        return None
    return text[:MAX_DESCRIPTION_CHARS]


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(MATCH|MISMATCH)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def compare(llm: Any, question: str, description: str) -> Dict[str, Any]:
    """
    {"verdict": "match"|"mismatch"|"unknown", "reason": str|None}.

    "unknown" on an LLM failure or an unparseable reply — an unavailable check
    must read as "not checked", never as a fault found.
    """
    if not question or not description:
        return {"verdict": "unknown", "reason": None}
    prompt = (f"QUESTION: {question}\n\n"
              f"THE QUERY RETURNS: {description}\n\nVERDICT:")
    try:
        raw = invoke_tracked(
            llm, [SystemMessage(content=_COMPARE_SYSTEM),
                  HumanMessage(content=prompt)], "back_translate_compare").content
    except Exception as exc:
        logger.warning(f"[BACKTRANS] compare failed: {exc}")
        return {"verdict": "unknown", "reason": None}

    text = strip_reasoning(raw or "")
    m = _VERDICT_RE.search(text)
    if not m:
        logger.info(f"[BACKTRANS] unparseable verdict: {text[:120]!r}")
        return {"verdict": "unknown", "reason": None}
    reason_m = _REASON_RE.search(text)
    reason = " ".join(reason_m.group(1).split())[:300] if reason_m else None
    return {"verdict": m.group(1).lower(), "reason": reason}


def round_trip(describe_llm: Any, compare_llm: Any, sql: str, question: str,
               schema_context: str = "") -> Dict[str, Any]:
    """
    {"description": str|None, "verdict": ..., "reason": ...}

    The description is returned even when the comparison is inconclusive: it
    is the single most useful thing on the evidence panel, because a user who
    reads "the lowest single monthly reading" against a question that meant
    "the lowest yearly total" needs no verdict at all.
    """
    description = describe_sql(describe_llm, sql, schema_context)
    if not description:
        return {"description": None, "verdict": "unknown", "reason": None}
    result = compare(compare_llm, question, description)
    return {"description": description, **result}
