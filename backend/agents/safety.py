"""
safety.py — Centralized SQL safety regexes and enforcement helpers.

Shared by SQLValidatorAgent (pre-execution) and ResponseOptimizer (post-execution
output guardrail). Both layers import from here so changes stay in one place.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Blocked DML / DDL keywords ────────────────────────────────────────────────
BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|CALL|EXEC)\b",
    re.IGNORECASE,
)

# ── SQL injection heuristics ──────────────────────────────────────────────────
SQL_INJECTION = re.compile(
    r"(--|;.+|\/\*|\*\/|xp_|UNION\s+SELECT|1\s*=\s*1|'OR'|\"OR\")",
    re.IGNORECASE,
)

# Default LIMIT applied when none is found in the query
DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def is_write_blocked(text: str) -> tuple[bool, str | None]:
    """Return (True, keyword) if a DML/DDL keyword is detected, else (False, None)."""
    m = BLOCKED.search(text)
    if m:
        return True, m.group()
    return False, None


def is_injection(text: str) -> bool:
    """Return True if an injection-like pattern is found in *text*."""
    return bool(SQL_INJECTION.search(text))


_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def enforce_limit(sql: str, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> str:
    """
    Ensure a SELECT query has a LIMIT clause capped at *maximum*.

    If the SQL already contains LIMIT n:
      - n > maximum → rewrite to maximum.
      - n ≤ maximum → leave as-is.
    If no LIMIT → append LIMIT *default*.
    Non-SELECT statements pass through unchanged (they will be blocked upstream).

    We inspect the LAST LIMIT in the query text because the outermost SELECT's
    LIMIT appears last when subqueries are involved (e.g. subquery LIMIT 5
    appears before outer LIMIT 600).
    """
    sql_strip = sql.strip().rstrip(";")

    # Only touch SELECT statements
    if not re.match(r"^\s*SELECT\b", sql_strip, re.IGNORECASE):
        return sql

    matches = list(_LIMIT_RE.finditer(sql_strip))

    if matches:
        # Use the LAST match — the outermost LIMIT in the query
        last = matches[-1]
        current = int(last.group(1))
        if current > maximum:
            logger.info(f"[SAFETY] LIMIT {current} exceeds max {maximum}; capping.")
            # Replace only this specific occurrence by position
            sql_strip = (
                sql_strip[: last.start()]
                + f"LIMIT {maximum}"
                + sql_strip[last.end() :]
            )
        return sql_strip
    else:
        logger.info(f"[SAFETY] No LIMIT found; appending LIMIT {default}.")
        return f"{sql_strip} LIMIT {default}"


# Output-only injection patterns: stricter subset to avoid markdown false-positives.
# Note: bare '--' is excluded here because '---' is a valid markdown table separator.
_OUTPUT_INJECTION = re.compile(
    r"(\/\*|\*\/|xp_|UNION\s+SELECT|1\s*=\s*1|'OR'|\"OR\"|;\s*DROP\b|;\s*DELETE\b|;\s*INSERT\b|;\s*UPDATE\b)",
    re.IGNORECASE,
)


def scan_output(text: str) -> bool:
    """
    Output guardrail: return True if the response text contains patterns that
    look like injected SQL or DML keywords smuggled into a natural-language answer.

    Uses a stricter subset vs is_injection() to avoid false-positives in
    markdown (e.g. '---' table separators match '--' but are not injections).
    """
    return bool(_OUTPUT_INJECTION.search(text))
