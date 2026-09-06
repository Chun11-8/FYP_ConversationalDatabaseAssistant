"""
trust.py — turn the evidence for one answer into a verdict a user can act on.

GH#67 and GH#69 put real evidence on screen; GH#91 makes it mean something.
"3 of 5 candidates agreed" does not tell a person whether to use the number in
front of them.

Two rules govern everything here.

**Only this query.** Every signal is computed from this SQL, this schema and
this result set. Nothing is looked up, remembered, or read from a benchmark.
Base rates measured on the BIRD databases say nothing about a question over
the user's own data, and printing one next to an answer implies a guarantee it
cannot carry. Benchmark numbers evaluate the DETECTOR; they are not a property
of an individual answer.

**Claim only what was checked.** The top verdict is "every check passed", not
"this is correct" — the checks are necessary conditions, not sufficient ones,
and the wording says so on screen.

Pure: no LLM call, no database, no I/O. Given the verification block the graph
already assembles, it returns the same verdict every time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Verdicts, worst last — the level is the worst gate's outcome.
PASSED = "passed"
REVIEW = "review"
UNTRUSTED = "untrusted"
NOT_GRADED = "not_graded"
VERDICTS = (PASSED, REVIEW, UNTRUSTED, NOT_GRADED)

_RANK = {PASSED: 0, REVIEW: 1, UNTRUSTED: 2}

# Gate outcomes.
OK, WARN, FAIL, SKIPPED = "ok", "warn", "fail", "skipped"
_STATUS_LEVEL = {OK: PASSED, WARN: REVIEW, FAIL: UNTRUSTED, SKIPPED: PASSED}

HEADLINE = {
    PASSED: "Checks passed",
    REVIEW: "Needs your review",
    UNTRUSTED: "Do not rely on this",
    NOT_GRADED: "No query to check",
}

STATEMENT = {
    PASSED: ("Every check we can run without knowing the right answer passed. "
             "That makes this answer trustworthy enough to act on, but it is "
             "not a guarantee — no check can prove a query matches what you "
             "meant."),
    REVIEW: ("Something did not line up. The answer may well be right, but one "
             "of our checks could not confirm it — read the points below "
             "before you use this."),
    UNTRUSTED: ("A check failed in a way that usually means the query answers a "
                "different question from the one you asked. Do not use this "
                "answer without rewriting the question or checking the SQL "
                "yourself."),
    NOT_GRADED: ("This turn did not run a query, so there is nothing to check."),
}

# Risk severities that make an answer untrustworthy rather than merely
# questionable. High-severity flags name something concrete and checkable —
# a value that does not exist, a constraint that was dropped, a table that was
# never in the schema — so they are reported as failures, not doubts.
_FAIL_SEVERITIES = {"high"}


def _gate(gid: str, label: str, status: str, detail: str,
          remedy: Optional[str] = None) -> Dict[str, Any]:
    return {"id": gid, "label": label, "status": status,
            "detail": detail, "remedy": remedy}


def _gate_safety(v: Dict[str, Any]) -> Dict[str, Any]:
    checks = v.get("validation_checks") or {}
    if not v.get("validated") and not checks:
        return _gate("safety", "Safe and well-formed", SKIPPED,
                     "No query was validated on this turn.")
    failed = [k.replace("_ok", "") for k, ok in checks.items() if not ok]
    if failed or not v.get("validated"):
        names = ", ".join(failed) or "validation"
        return _gate("safety", "Safe and well-formed", FAIL,
                     f"The query failed {names} validation.",
                     "The assistant will not run a query that fails this — "
                     "try rephrasing the question.")
    return _gate("safety", "Safe and well-formed", OK,
                 "Read-only, parses cleanly, and every table and column it "
                 "uses exists in the schema.")


def _gate_question_match(v: Dict[str, Any]) -> Dict[str, Any]:
    risks = [r for r in (v.get("risks") or []) if isinstance(r, dict)]
    # Plan conformance is reported as its own gate, so it is excluded here.
    risks = [r for r in risks if not str(r.get("code", "")).startswith("plan_")]
    if not risks:
        return _gate("question_match", "Matches your question", OK,
                     "No dropped filters, unknown values, accidental cross "
                     "joins or suspicious row counts.")
    worst = FAIL if any(r.get("severity") in _FAIL_SEVERITIES
                        for r in risks) else WARN
    messages = [str(r.get("message", "")).strip() for r in risks[:3]]
    return _gate("question_match", "Matches your question", worst,
                 " ".join(m for m in messages if m),
                 "Name the exact value or column you meant, and ask again.")


def _gate_plan(v: Dict[str, Any]) -> Dict[str, Any]:
    plan_risks = [r for r in (v.get("risks") or [])
                  if isinstance(r, dict) and str(r.get("code", "")).startswith("plan_")]
    if not plan_risks:
        return _gate("plan", "Follows its own plan", OK,
                     "The query implements the tables, joins and aggregation "
                     "the assistant planned before writing it.")
    worst = FAIL if any(r.get("severity") in _FAIL_SEVERITIES
                        for r in plan_risks) else WARN
    return _gate("plan", "Follows its own plan", worst,
                 " ".join(str(r.get("message", "")).strip() for r in plan_risks[:2]),
                 "Say explicitly which figure you want — a total, an average, "
                 "or one row per group.")


def _gate_agreement(v: Dict[str, Any]) -> Dict[str, Any]:
    executed = int(v.get("executed") or 0)
    agreement = int(v.get("agreement") or 0)
    if v.get("decided_by") == "cache_hit":
        return _gate("agreement", "Agrees with itself", SKIPPED,
                     "Reused a query answered earlier in this database; it was "
                     "re-validated and re-run against current data.")
    if executed == 0:
        return _gate("agreement", "Agrees with itself", FAIL,
                     "No candidate query ran successfully.",
                     "Rephrase the question, or name the tables you mean.")
    if executed == 1:
        return _gate("agreement", "Agrees with itself", WARN,
                     "Only one query was produced, so there was nothing to "
                     "cross-check it against.")
    if agreement == executed:
        return _gate("agreement", "Agrees with itself", OK,
                     f"All {executed} independently written queries returned "
                     f"the same result.")
    return _gate("agreement", "Agrees with itself", WARN,
                 f"{agreement} of {executed} independently written queries "
                 f"returned this result; the others disagreed.",
                 "Where the wording allows two readings, spell out which you "
                 "mean.")


def _gate_back_translation(v: Dict[str, Any]) -> Dict[str, Any]:
    bt = v.get("back_translation") or {}
    verdict = str(bt.get("verdict") or "unknown").lower()
    description = bt.get("description")
    if verdict == "mismatch":
        return _gate("back_translation", "Describes what you asked", FAIL,
                     bt.get("reason") or
                     "Read back in plain English, the query answers a "
                     "different question.",
                     "Compare the description above with your question and "
                     "rephrase the part that differs.")
    if verdict == "match":
        return _gate("back_translation", "Describes what you asked", OK,
                     "Read back in plain English, the query answers the "
                     "question you asked.")
    return _gate("back_translation", "Describes what you asked", SKIPPED,
                 "Not checked on this turn."
                 if not description else
                 "The read-back could not be compared with your question.")


def grade(verification: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    {"level", "headline", "statement", "gates": [...], "failed": [...]}

    The level is the worst gate outcome. A gate that could not run is SKIPPED
    and never drags the verdict down: "not checked" is not "found wrong".
    """
    v = verification if isinstance(verification, dict) else {}

    # A turn that never generated SQL — GENERAL, a blocked write, a clarifying
    # question — is not a failing answer; there is simply nothing to grade.
    if v.get("decided_by") in ("not_generated", "generation_failed"):
        return {"level": NOT_GRADED, "headline": HEADLINE[NOT_GRADED],
                "statement": STATEMENT[NOT_GRADED], "gates": [], "failed": []}

    gates: List[Dict[str, Any]] = [
        _gate_safety(v),
        _gate_question_match(v),
        _gate_plan(v),
        _gate_agreement(v),
        _gate_back_translation(v),
    ]

    level = PASSED
    for g in gates:
        level = max(level, _STATUS_LEVEL[g["status"]], key=lambda x: _RANK[x])

    return {
        "level": level,
        "headline": HEADLINE[level],
        "statement": STATEMENT[level],
        "gates": gates,
        "failed": [g["label"] for g in gates if g["status"] in (WARN, FAIL)],
    }
