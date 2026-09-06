"""
risk.py — gold-free risk checks for a query that ran but may be wrong (GH#69).

A text-to-SQL system gets no natural signal telling it whether its answer is
right. Syntax errors are self-reporting and safety violations are matchable,
but a query that joins the wrong table parses cleanly, executes, and returns
plausible rows. On the 150-question BIRD runs in ``outputs/`` this is not the
long tail, it IS the failure mode: gpt-5-mini had 0 validation failures and 0
execution failures, and 57 semantically wrong answers.

Phase 1 (GH#67) surfaced candidate agreement, which catches ~28 % of those at
76 % precision. Its ceiling is that 41 of the 57 errors occurred on turns where
every candidate agreed — the candidates are wrong *together*. This module
supplies signals that do not depend on agreement, and (critically) do not
depend on a gold answer, since at serving time there isn't one.

Design rules
------------
* **Annotate, never block.** Nothing here suppresses, rewrites or retries a
  query. A false alarm that hides a correct answer is worse than the disease.
* **Every check is a pure function** over already-computed inputs — no LLM
  calls, no extra database round-trips — so the whole pass is free and each
  check is unit-testable without a database.
* **Specific messages.** "'Brazil' doesn't appear in country" is actionable;
  "low confidence" is not.
* **Silence is the default.** A check that cannot decide returns nothing
  rather than guessing, because precision matters far more than recall here:
  users stop reading warnings that cry wolf.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Severity ranks. Kept coarse on purpose — a finer scale would imply a
# calibration this has not yet earned.
HIGH, MEDIUM, LOW = "high", "medium", "low"
_WEIGHT = {HIGH: 3, MEDIUM: 2, LOW: 1}


def _risk(code: str, severity: str, message: str, **detail: Any) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message,
            "detail": detail or None}


# ── 1. zero rows ──────────────────────────────────────────────────────────────

def check_zero_rows(rows: Optional[Sequence]) -> List[Dict[str, Any]]:
    """
    An empty result is the one failure the user can already see, but not
    interpret: "no matching data" and "my filter is wrong" look identical.
    7 of the 41 unanimous-but-wrong gpt-5-mini errors were this.
    """
    if isinstance(rows, list) and len(rows) == 0:
        return [_risk("zero_rows", MEDIUM,
                      "The query ran but matched no rows — this may mean no "
                      "such data exists, or that a filter is too narrow or "
                      "misspelled.")]
    return []


# ── 2. ungrounded WHERE literal ───────────────────────────────────────────────

_STRING_LITERAL_RE = re.compile(r"'([^']{2,64})'")
# Dates and numeric-ish strings are not the kind of literal the value index can
# adjudicate. Neither is a LIKE pattern: '%abc%' is a substring match, so it is
# SUPPOSED not to equal any stored value — looking it up guarantees a false
# alarm, hence the separate `%` test rather than a whole-string character class.
_SKIP_LITERAL_RE = re.compile(r"^[\d\s\-/:.%_]+$")


def _unadjudicable(lit: str) -> bool:
    return bool(_SKIP_LITERAL_RE.match(lit)) or "%" in lit


def check_ungrounded_literals(
    sql: str, tables: Sequence[str], value_index: Any,
) -> List[Dict[str, Any]]:
    """
    A string literal in the SQL that does not appear in any indexed column of
    the tables being read. This is the single highest-value check available:
    ``disjoint`` — the wrong-rows category — was 25 of those 41 errors, and a
    literal that matches nothing is a direct, checkable cause.

    Uses the existing GH#49 value index, so it costs nothing extra. Stays
    silent whenever the index cannot answer (column not indexed, non-text
    column, index unavailable) rather than guessing.
    """
    if not sql or value_index is None or not tables:
        return []
    out: List[Dict[str, Any]] = []
    try:
        seen: set = set()
        for m in _STRING_LITERAL_RE.finditer(sql):
            lit = m.group(1).strip()
            if not lit or lit in seen or _unadjudicable(lit):
                continue
            seen.add(lit)
            # ground() reports spans whose CLOSEST real value differs from how
            # they were written — exactly "this literal is not in the data".
            hints = value_index.ground(f"'{lit}'", list(tables))
            if hints:
                out.append(_risk(
                    "ungrounded_literal", HIGH,
                    f"The value {lit!r} does not appear in the data — "
                    f"{hints[0]}",
                    literal=lit, suggestion=hints[0]))
    except Exception as exc:                    # never break a turn
        logger.warning("[RISK] literal grounding failed: %s", exc)
        return []
    return out[:3]


# ── 2b. question ↔ SQL coverage (GH#73) ───────────────────────────────────────

# Spans that routinely appear in the SQL in a DIFFERENT form, so their textual
# absence proves nothing. Dates are the big one: "between August and November
# 2013" is correctly answered with a '201308'-style literal, and on the
# labelled data every one of these produced a false alarm.
# \b on the bare-year alternative is load-bearing: unanchored `\d{4}` matches
# "3850" inside the customer id 38508, so an entity id was silently skipped as
# "date-shaped" and a dropped-constraint went unflagged (GH#76).
_DATE_SHAPED_RE = re.compile(r"\d{1,4}\s*[/-]\s*\d{1,2}|\b\d{4}\b|\b\d{1,2}:\d{2}")
_SENTENCE_STARTERS = {
    "is", "if", "in", "on", "at", "what", "which", "who", "how", "the", "a",
    "an", "and", "or", "of", "for", "list", "show", "give", "state", "name",
    "please", "find", "calculate", "among", "between", "from", "to",
}
_MONTH_WORDS = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
}


def _span_is_untestable(span: str) -> bool:
    """True when a span's absence from the SQL carries no information."""
    low = span.casefold().strip()
    if low in _SENTENCE_STARTERS:            # capitalised sentence opener, not a value
        return True
    if low in _MONTH_WORDS:                  # rendered as a date literal
        return True
    if _DATE_SHAPED_RE.search(span):         # reformatted (2012/8/24 → 2012-08-24)
        return True
    # A multi-word span whose first word is a sentence opener ("Is it true…")
    first = low.split()[0] if low.split() else ""
    return first in _SENTENCE_STARTERS



def check_question_coverage(
    sql: str, question: str, tables: Sequence[str], value_index: Any,
) -> List[Dict[str, Any]]:
    """
    Every other check reads the SQL and asks whether it is well-formed against
    the schema. This one reads the QUESTION and asks whether the SQL actually
    used it.

    The failure it exists for (GH#73): asked for "customers from the country
    Wakanda" against a schema with no country column, the model dropped the
    constraint entirely and substituted `segment = 'LAM' AND currency = 'EUR'`
    — real columns, real values, internally consistent, and an answer to a
    different question. Nothing downstream could see it, because every literal
    in the SQL was legitimate: the constraint was not corrupted, it was
    deleted.

    Two distinct verdicts:
      * ``unknown_value``    — the span matches nothing in the database at all.
      * ``dropped_constraint`` — the span exists (or could not be checked) but
        the SQL neither contains it nor the value it grounds to.

    The grounded value must be checked, not just the literal span: value
    grounding rewrites 'Brazil' to 'BRA' on purpose, and a naive
    span-not-in-SQL test would fire on a correctly grounded query.
    """
    if not sql or not question:
        return []
    try:
        from backend.retrieval.value_index import extract_value_spans
        spans = extract_value_spans(question)
        if not spans:
            return []

        # {} means the index could not answer — treated as "unknown", never as
        # "nothing matched", so an index outage cannot manufacture alarms.
        resolved: Dict[str, Any] = {}
        if value_index is not None and tables:
            try:
                resolved = value_index.resolve_spans(question, list(tables)) or {}
            except Exception as exc:
                logger.warning("[RISK] span resolution failed: %s", exc)
                resolved = {}

        low_sql = sql.casefold()
        out: List[Dict[str, Any]] = []
        for span in spans:
            if _span_is_untestable(span):
                continue
            if span.casefold() in low_sql:
                continue                                  # used as written
            actual = resolved.get(span)
            if actual and actual.casefold() in low_sql:
                continue                                  # used, correctly grounded
            if span not in resolved:
                # The index could not be consulted, so "absent from the SQL"
                # cannot be distinguished from "expressed another way".
                # Measured on the 150 labelled questions, guessing here drove
                # false alarms to 54% and precision to 42% — silence is the
                # only defensible answer without the data to check against.
                continue
            if actual is None:
                out.append(_risk(
                    "unknown_value", HIGH,
                    f"{span!r} does not appear anywhere in this database, so "
                    f"no query can filter on it.", span=span))
            else:
                out.append(_risk(
                    "dropped_constraint", HIGH,
                    f"Your question mentions {span!r} (stored as {actual!r}), "
                    f"but the query does not filter on it — the answer may be "
                    f"for a different question.", span=span, stored_as=actual))
        return out[:3]
    except Exception as exc:
        logger.warning("[RISK] question-coverage check failed: %s", exc)
        return []


# ── 3. join fan-out ───────────────────────────────────────────────────────────

def check_join_fanout(
    sql: str, rows: Optional[Sequence], row_limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    A join with no ON condition, or an ON that doesn't constrain, multiplies
    rows. The classic symptom of a wrong join path — and a wrong join path is
    the usual cause of a ``disjoint`` result.

    Detects only the unambiguous case: a multi-table FROM with no JOIN ... ON
    and no WHERE linking them. A row-count heuristic is deliberately avoided —
    without the base table sizes it cannot distinguish a fan-out from a
    legitimately large table.
    """
    if not sql:
        return []
    s = " ".join(sql.split())
    # Comma-joined FROM (implicit cross join) with no WHERE at all.
    m = re.search(r"\bFROM\b(.+?)(\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
                  s, re.IGNORECASE)
    if not m:
        return []
    from_clause, terminator = m.group(1), m.group(2).upper()
    comma_joined = from_clause.count(",") >= 1
    has_join_on = re.search(r"\bJOIN\b.+?\bON\b", s, re.IGNORECASE) is not None
    if comma_joined and not has_join_on and terminator != "WHERE":
        return [_risk("cross_join", HIGH,
                      "Several tables are combined without a join condition, "
                      "which multiplies rows together — totals and counts "
                      "from this query are likely inflated.")]
    if re.search(r"\bCROSS\s+JOIN\b", s, re.IGNORECASE):
        return [_risk("cross_join", MEDIUM,
                      "This query uses an explicit CROSS JOIN — every row is "
                      "paired with every other row.")]
    return []


# ── 4. aggregate without GROUP BY on a per-group question ─────────────────────

# A bare "by <word>" was tried here and removed: on the labelled data it fired
# on the ordinary preposition ("spent by customer", "done by Ahmed") and
# produced 3 false alarms against 1 real catch. Only genuine per-group
# phrasing is matched now.
_PER_GROUP_RE = re.compile(
    r"\b(each|per\s+\w+|for\s+every|respectively|breakdown|"
    r"grouped\s+by|for\s+each)\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


def check_aggregate_grain(sql: str, question: str) -> List[Dict[str, Any]]:
    """
    The question asks for a value *per* something, but the SQL aggregates the
    whole table into one row. A grain error: the number is real, correctly
    computed, and answers a different question.
    """
    if not sql or not question:
        return []
    if not _PER_GROUP_RE.search(question):
        return []
    if not _AGG_RE.search(sql):
        return []
    if re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
        return []
    return [_risk("missing_group_by", MEDIUM,
                  "The question asks for a value per group, but the query "
                  "aggregates everything into a single total.")]


# ── 4b. aggregate function vs the question's verb (GH#76) ─────────────────────
#
# The largest single failure mode: of the 33 gpt-5-mini failures that had the
# RIGHT tables and joins, 17 (52 %) differed from gold only in which aggregate
# was applied. The structure is correct and the number is wrong, which is
# exactly the class no other check can see.

_SUM_WORDS = re.compile(
    r"\b(total|sum|altogether|combined|in\s+total|how\s+much\s+.*\bspent\b|"
    r"amount\s+spent)\b", re.IGNORECASE)
_EXTREMUM_WORDS = re.compile(
    r"\b(highest|lowest|maximum|minimum|peak|most|least|largest|smallest|top)\b",
    re.IGNORECASE)
_PER_UNIT_RE = re.compile(
    r"\b(monthly|daily|yearly|annual|weekly|per\s+\w+|each\s+\w+)\b",
    re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(percentage|percent|proportion|ratio|%)\b",
                         re.IGNORECASE)


def _aggs_in(sql: str) -> set:
    return {m.group(1).upper() for m in _AGG_RE.finditer(sql or "")}


def check_aggregate_intent(sql: str, question: str) -> List[Dict[str, Any]]:
    """
    The aggregate the SQL applies against the one the question asks for.

    Three signatures taken straight from the labelled failures:

    * "the amount **spent**" answered with MAX/MIN and no SUM — the query
      returns one transaction rather than the total.
    * "the **highest monthly** consumption" answered with a bare MAX and no
      GROUP BY. Gold sums per month and takes the largest month; MAX over raw
      rows returns the largest single row, a different and smaller number that
      looks entirely plausible.
    * "what **percentage**" answered without any division.

    Each requires an explicit contradiction between question and SQL, never an
    absence, so a question the patterns do not recognise stays silent.
    """
    if not sql or not question:
        return []
    out: List[Dict[str, Any]] = []
    aggs = _aggs_in(sql)
    has_group = bool(re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE))

    if (_SUM_WORDS.search(question) and not _EXTREMUM_WORDS.search(question)
            and aggs and "SUM" not in aggs and aggs & {"MAX", "MIN"}):
        out.append(_risk(
            "aggregate_mismatch", HIGH,
            "The question asks for a total, but the query takes a "
            f"{sorted(aggs & {'MAX', 'MIN'})[0]} — that returns a single "
            "extreme value, not the sum."))

    if (_EXTREMUM_WORDS.search(question) and _PER_UNIT_RE.search(question)
            and aggs & {"MAX", "MIN"} and not has_group):
        out.append(_risk(
            "extremum_without_grouping", HIGH,
            "The question asks for the highest/lowest value per group, but the "
            "query takes an extreme over individual rows without grouping — "
            "these give different answers."))

    if (_PERCENT_RE.search(question) and "/" not in sql
            and not re.search(r"\bCAST\b|\bNULLIF\b", sql, re.IGNORECASE)):
        out.append(_risk(
            "percentage_without_division", MEDIUM,
            "The question asks for a percentage or proportion, but the query "
            "computes no ratio."))
    return out[:3]


# ── 5. plural question answered with LIMIT 1 ──────────────────────────────────

# "what/which <plural noun>" needs the auxiliary-verb exclusion: without it
# `\w+s` matches the copula in "What is the name of player 3?" and "What was
# the total", turning every singular question into a false alarm.
_PLURAL_RE = re.compile(
    r"\b(list|all|every|each|names)\b"
    r"|\b(?:which|what)\s+(?!is\b|was\b|has\b|does\b|its\b)\w{3,}s\b",
    re.IGNORECASE)
_SUPERLATIVE_RE = re.compile(
    r"\b(most|least|highest|lowest|top|best|worst|maximum|minimum|oldest|newest)\b",
    re.IGNORECASE)


def check_limit_one_on_plural(sql: str, question: str) -> List[Dict[str, Any]]:
    """
    "List the players who…" answered with LIMIT 1 returns one row of a set.
    Superlatives ("the highest-paid player") legitimately want one row, so
    they suppress the check — that exclusion is what keeps this from firing
    constantly.
    """
    if not sql or not question:
        return []
    if not re.search(r"\bLIMIT\s+1\b", sql, re.IGNORECASE):
        return []
    if _SUPERLATIVE_RE.search(question):
        return []
    if not _PLURAL_RE.search(question):
        return []
    return [_risk("limit_one_on_plural", MEDIUM,
                  "The question asks for a list, but the query returns only "
                  "one row.")]


# ── 6. column count vs the attributes asked for ───────────────────────────────

def check_arity(rows: Optional[Sequence], query_plan: Optional[dict]
                ) -> List[Dict[str, Any]]:
    """
    The linker's plan names the target attributes; the result has a different
    column count. ``arity_mismatch`` was 5 of the 41 errors — usually an
    identifier returned without the value asked for, or vice versa.

    Only fires when the plan states a comma-separated target, so a vague plan
    never produces a warning.
    """
    if not rows or not isinstance(rows, list) or not isinstance(query_plan, dict):
        return []
    target = query_plan.get("target")
    if not isinstance(target, str) or "," not in target:
        return []
    try:
        actual = len(rows[0].keys())
    except Exception:
        return []
    wanted = len([t for t in target.split(",") if t.strip()])
    if wanted and actual and actual != wanted:
        return [_risk("arity_mismatch", LOW,
                      f"The plan asked for {wanted} attributes but the result "
                      f"has {actual} column{'s' if actual != 1 else ''}.",
                      wanted=wanted, actual=actual)]
    return []


# ── 7. identifier outside the retrieved schema ────────────────────────────────

def check_out_of_schema_tables(
    tables: Sequence[str], schema_context: str,
) -> List[Dict[str, Any]]:
    """
    The query reads a table that never appeared in the schema context the
    model was given — it was invented, or retrieval missed it. Either way the
    answer rests on something unverified.

    Matching is case-insensitive and word-bounded against the context text
    rather than a parsed schema, because the context format varies (verbose vs
    M-Schema). An empty context means "cannot tell" → silence.
    """
    if not tables or not schema_context:
        return []
    missing = [t for t in tables
               if not re.search(rf"\b{re.escape(t)}\b", schema_context, re.IGNORECASE)]
    if not missing:
        return []
    return [_risk("out_of_schema", HIGH,
                  "The query reads "
                  + ", ".join(f"`{t}`" for t in missing[:3])
                  + ", which was not in the schema shown to the model.",
                  tables=missing[:3])]


# ── 10. plan conformance ──────────────────────────────────────────────────────

_AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


def _plan_aggregates(plan: Dict[str, Any]) -> set:
    """Aggregate functions the plan asked for, from its own wording."""
    text = " ".join(str(plan.get(k) or "") for k in ("aggregation", "target"))
    return {m.group(1).lower() for m in _AGG_RE.finditer(text)} | {
        w for w in ("count", "sum", "avg", "average", "min", "minimum",
                    "max", "maximum")
        if re.search(rf"\b{w}\b", text, re.IGNORECASE)
    }


_AGG_SYNONYM = {"average": "avg", "minimum": "min", "maximum": "max"}


def check_plan_conformance(
    sql: str, query_plan: Optional[dict],
) -> List[Dict[str, Any]]:
    """
    GH#91: did the SQL actually implement the plan the linker produced?

    SQLCodeAgent._plan_penalty already diffs tables and join paths, but only
    to RANK candidates — the winner's own discrepancies are then discarded.
    Reported here instead, and extended to the two dimensions that caused most
    of the observed failures: which aggregate was applied, and whether the
    GROUP BY the plan named survived.

    This is not a correctness proof. It says the query and the reasoning
    behind it disagree, which is a fact the user can check in one glance —
    "your plan said one row per customer, the SQL returns one row per month".
    """
    if not sql or not isinstance(query_plan, dict) or not query_plan:
        return []

    out: List[Dict[str, Any]] = []
    low_sql = sql.casefold()

    # ── aggregate ──
    plan_aggs = {_AGG_SYNONYM.get(a, a) for a in _plan_aggregates(query_plan)}
    sql_aggs = {m.group(1).lower() for m in _AGG_RE.finditer(sql)}
    if plan_aggs and sql_aggs and not (plan_aggs & sql_aggs):
        out.append(_risk(
            "plan_aggregate_mismatch", HIGH,
            f"The plan called for {'/'.join(sorted(plan_aggs)).upper()} but the "
            f"query uses {'/'.join(sorted(sql_aggs)).upper()} — these answer "
            f"different questions.",
            plan=sorted(plan_aggs), sql=sorted(sql_aggs)))

    # ── grouping ──
    plan_group = str(query_plan.get("group_by") or "").strip()
    if plan_group and plan_group.lower() not in ("none", "null", "n/a", "(none)"):
        if "group by" not in low_sql:
            out.append(_risk(
                "plan_group_by_dropped", MEDIUM,
                f"The plan groups by {plan_group!r}, but the query has no "
                f"GROUP BY — it returns a single overall figure instead of "
                f"one row per group.",
                plan_group_by=plan_group))

    # ── grain (GH#76 emits it when plan_rationale is on) ──
    grain = str(query_plan.get("grain") or "").strip()
    if grain and "group by" not in low_sql and re.search(
            r"\bper\b|\beach\b|one row per", grain, re.IGNORECASE):
        out.append(_risk(
            "plan_grain_mismatch", MEDIUM,
            f"The plan describes the answer as {grain!r}, which needs one row "
            f"per group, but the query returns an ungrouped result.",
            grain=grain))
    return out[:3]


# ── combiner ──────────────────────────────────────────────────────────────────

def assess(
    sql: str,
    rows: Optional[Sequence],
    question: str,
    schema_context: str = "",
    query_plan: Optional[dict] = None,
    tables: Optional[Sequence[str]] = None,
    value_index: Any = None,
) -> Dict[str, Any]:
    """
    Run every check and return ``{"risks": [...], "risk_score": int}``.

    ``risk_score`` is a severity-weighted count, NOT a probability — nothing
    yet establishes what a given score implies about correctness, and dressing
    it up as a percentage would imply a calibration this has not earned. It
    exists to rank turns, and to be measured against the labelled data.

    Never raises: a failing check is dropped, because a broken heuristic must
    not cost the user their answer.
    """
    risks: List[Dict[str, Any]] = []
    tables = list(tables or [])
    checks = (
        lambda: check_zero_rows(rows),
        lambda: check_ungrounded_literals(sql, tables, value_index),
        lambda: check_question_coverage(sql, question, tables, value_index),
        lambda: check_join_fanout(sql, rows),
        lambda: check_aggregate_grain(sql, question),
        lambda: check_aggregate_intent(sql, question),
        lambda: check_limit_one_on_plural(sql, question),
        lambda: check_arity(rows, query_plan),
        lambda: check_out_of_schema_tables(tables, schema_context),
        lambda: check_plan_conformance(sql, query_plan),
    )
    for run in checks:
        try:
            risks.extend(run() or [])
        except Exception as exc:
            logger.warning("[RISK] check failed: %s", exc)
    return {
        "risks": risks,
        "risk_score": sum(_WEIGHT.get(r.get("severity"), 1) for r in risks),
    }
