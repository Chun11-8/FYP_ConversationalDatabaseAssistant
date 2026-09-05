from __future__ import annotations

import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.trace import trace
from backend.agents.usage import invoke_tracked
from backend.agents import usage as usage_tracking
from backend.agents.utils import PromptHarness, strip_reasoning
from backend.agents.schema_linking_agent import render_query_plan

logger = logging.getLogger(__name__)

# Reranker schema context cap — large enough to include FK-expanded schemas
_RERANKER_SCHEMA_CAP = 3500


# ─────────────────────────────────────────────
# SQLite dialect normalization
# ─────────────────────────────────────────────
# The agent generates PostgreSQL-flavoured SQL. When the target backend is
# SQLite (e.g. the BIRD benchmark harness), several PostgreSQL-only constructs
# are syntactically valid but fail at execution — producing EX/F1/VES = 0 even
# when the query logic is correct. These transforms rewrite the common ones.
# All are idempotent so they are safe to apply after a sqlglot transpile pass.

def _to_sqlite(sql: str) -> str:
    """Deterministic regex rewrites of PostgreSQL-only constructs to SQLite."""
    out = sql

    # ILIKE → LIKE (SQLite LIKE is already case-insensitive for ASCII)
    out = re.sub(r"\bILIKE\b", "LIKE", out, flags=re.IGNORECASE)

    # STRING_AGG(expr, sep) → GROUP_CONCAT(expr, sep)
    out = re.sub(r"\bSTRING_AGG\s*\(", "GROUP_CONCAT(", out, flags=re.IGNORECASE)

    # x::NUMERIC / x::numeric(10,2) → CAST(x AS REAL); other ::TYPE → CAST(x AS TYPE)
    def _cast_repl(m: "re.Match") -> str:
        expr, typ = m.group(1), m.group(2).upper()
        if typ in ("NUMERIC", "DECIMAL"):
            typ = "REAL"
        return f"CAST({expr} AS {typ})"

    out = re.sub(
        r"(\w+)\s*::\s*([A-Za-z]+)(?:\s*\([0-9,\s]*\))?",
        _cast_repl,
        out,
    )

    # CAST(x AS NUMERIC[(p,s)]) / DECIMAL → CAST(x AS REAL)
    out = re.sub(
        r"CAST\((.*?)\s+AS\s+(?:NUMERIC|DECIMAL)(?:\s*\([0-9,\s]*\))?\)",
        r"CAST(\1 AS REAL)",
        out,
        flags=re.IGNORECASE,
    )

    # Boolean literals → integer literals
    out = re.sub(r"\bTRUE\b", "1", out, flags=re.IGNORECASE)
    out = re.sub(r"\bFALSE\b", "0", out, flags=re.IGNORECASE)

    # YEAR(x)/MONTH(x)/DAY(x) — evidence pseudo-code and MySQL-isms that SQLite
    # lacks (observed: `year(Description) = '1997'` emitted literally). Rewrite
    # to STRFTIME with balanced-paren scanning (handles YEAR(CURRENT_TIMESTAMP)).
    out = _rewrite_date_funcs(out)

    return out


_DATE_FUNCS = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}


def _rewrite_date_funcs(sql: str) -> str:
    out = sql
    for _ in range(10):
        changed = False
        for name, fmt in _DATE_FUNCS.items():
            m = re.search(rf"\b{name}\s*\(", out, re.IGNORECASE)
            if not m:
                continue
            i, depth = m.end(), 1
            while i < len(out) and depth:
                if out[i] == "(":
                    depth += 1
                elif out[i] == ")":
                    depth -= 1
                i += 1
            if depth:
                continue
            arg = out[m.end(): i - 1].strip()
            out = out[: m.start()] + f"STRFTIME('{fmt}', {arg})" + out[i:]
            changed = True
        if not changed:
            break
    return out


# ─────────────────────────────────────────────
# Evidence pseudo-function translation
# ─────────────────────────────────────────────
# BIRD evidence writes formulas in pseudo-code — SUBTRACT(a,b), DIVIDE(a,b),
# ADD(a,b), MULTIPLY(a,b), Total(x) — and the evidence-verbatim rule makes the
# model copy them literally into SQL, where they are not functions (observed:
# `SELECT SUBTRACT(YEAR(...), YEAR(...))` → "no such function"). These rewrites
# translate them deterministically to real operators. Balanced-paren scanning
# handles nesting (DIVIDE(SUM(SUBTRACT(...)), COUNT(...))).

_PSEUDO_OPS = {"SUBTRACT": "-", "DIVIDE": "/", "ADD": "+", "MULTIPLY": "*"}


def _split_two_args(body: str) -> Optional[Tuple[str, str]]:
    """Split 'a, b' at the top-level comma; None when there isn't exactly one."""
    depth = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return body[:i], body[i + 1:]
    return None


def _mask_string_literals(sql: str) -> Tuple[str, List[str]]:
    """Replace each 'string literal' with a paren-free sentinel; return (masked, literals)."""
    literals: List[str] = []

    def _repl(m: "re.Match") -> str:
        literals.append(m.group(0))
        return f"\x00L{len(literals) - 1}\x00"

    return _STRING_LITERAL_SPLIT.sub(_repl, sql), literals


def _unmask_string_literals(masked: str, literals: List[str]) -> str:
    for idx, lit in enumerate(literals):
        masked = masked.replace(f"\x00L{idx}\x00", lit)
    return masked


def translate_pseudo_functions(sql: str) -> str:
    """Rewrite evidence pseudo-functions into real SQL operators (idempotent).

    String literals are masked out first so a value like ``'Grand Total(2020)'``
    or ``'DIVIDE(x)'`` is never mistaken for a pseudo-function call.
    """
    out, _literals = _mask_string_literals(sql)
    for _ in range(20):  # bounded — each pass rewrites one call
        changed = False
        for name, op in _PSEUDO_OPS.items():
            m = re.search(rf"\b{name}\s*\(", out, re.IGNORECASE)
            if not m:
                continue
            i, depth = m.end(), 1
            while i < len(out) and depth:
                if out[i] == "(":
                    depth += 1
                elif out[i] == ")":
                    depth -= 1
                i += 1
            if depth:            # unbalanced parens — leave untouched
                continue
            args = _split_two_args(out[m.end(): i - 1])
            if not args:
                continue
            a, b = args[0].strip(), args[1].strip()
            if name == "DIVIDE":
                # Cast the numerator to REAL so integer/integer does not truncate
                # (BIRD ratio evidence like DIVIDE(male, total) means a real
                # quotient, e.g. 0.42 — not 0 from integer division).
                replacement = f"(CAST({a} AS REAL) / {b})"
            else:
                replacement = f"({a} {op} {b})"
            out = out[: m.start()] + replacement + out[i:]
            changed = True
        if not changed:
            break
    # Total(x) → SUM(x): evidence shorthand, not a portable SQL function.
    out = re.sub(r"\bTotal\s*\(", "SUM(", out, flags=re.IGNORECASE)
    return _unmask_string_literals(out, _literals)


# ─────────────────────────────────────────────
# Identifier reconciliation (space/hyphen column names)
# ─────────────────────────────────────────────
# Real BIRD columns include spaces and hyphens (`Examination Date`, `aCL IgM`,
# `T-CHO`). The model mangles them — `Examination_Date`, `aCL_IgM`, `T - CHO`
# (which even parses as subtraction!) — and the query dies. The schema context
# knows the real names, so mangled references are deterministically rewritten
# back to the exact, double-quoted identifier. String literals are never touched.

_BIRD_COL_RE = re.compile(r"^\s*-\s*([^:\n]+?):", re.MULTILINE)       # "  - aCL IgM: REAL"
_CHUNK_COL_RE = re.compile(r"^\s{2}([A-Za-z][^(\n]*?)\s*\(", re.MULTILINE)  # "  aCL IgM (REAL"
_MSCHEMA_COL_RE = re.compile(r"^\(([^:,)\n]+?):", re.MULTILINE)       # "(aCL IgM:REAL, …)"

_STRING_LITERAL_SPLIT = re.compile(r"('(?:[^']|'')*')")


def _special_schema_columns(schema_context: str) -> List[str]:
    """Schema column names containing non-word chars (spaces, hyphens, dots)."""
    names = set()
    for pat in (_BIRD_COL_RE, _CHUNK_COL_RE, _MSCHEMA_COL_RE):
        for m in pat.findall(schema_context or ""):
            name = m.strip()
            if name and re.search(r"[^\w]", name):
                names.add(name)
    return sorted(names, key=len, reverse=True)   # longest first — no partial shadowing


def reconcile_identifiers(sql: str, schema_context: str) -> str:
    """
    Rewrite mangled references to schema columns whose real names contain
    spaces/hyphens into the exact double-quoted identifier:
        Examination_Date  →  "Examination Date"
        T - CHO / T_CHO   →  "T-CHO"
    Only applied OUTSIDE string literals; already-quoted identifiers are left alone.
    """
    if not sql or not schema_context:
        return sql
    specials = _special_schema_columns(schema_context)
    if not specials:
        return sql

    patterns = []
    for real in specials:
        parts = [p for p in re.split(r"[^\w]+", real) if p]
        if len(parts) < 2:
            continue
        # Match the parts joined by any single run of separator chars — covers
        # the real name itself, the underscore variant, and spaced hyphens.
        # Guards: not preceded/followed by word chars or an opening quote.
        body = r"[ \t]*[-_. ][ \t]*".join(re.escape(p) for p in parts)
        patterns.append((re.compile(rf"(?<![\w\"`\[]){body}(?![\w])"), f'"{real}"'))

    # Never rewrite inside single-quoted string literals.
    segments = _STRING_LITERAL_SPLIT.split(sql)
    for idx in range(0, len(segments), 2):
        seg = segments[idx]
        for pat, repl in patterns:
            seg = pat.sub(repl, seg)
        segments[idx] = seg
    return "".join(segments)


# ─────────────────────────────────────────────
# String-literal apostrophe escaping (Issue #21)
# ─────────────────────────────────────────────
# Grounded values with apostrophes (e.g. Women's Soccer, O'Brien) are frequently
# emitted as raw literals — `'Women's Soccer'` — which close the string early and
# produce invalid SQL (sqlglot TokenError). In SQL an interior apostrophe must be
# doubled: `'Women''s Soccer'`.

def escape_sql_literal(value: str) -> str:
    """Escape a raw string value for a single-quoted SQL literal (double the ')."""
    return value.replace("'", "''")


def _double_interior_apostrophes(sql: str) -> str:
    """
    Repair unescaped apostrophes inside single-quoted string literals.

    Heuristic: while inside a literal, a `'` is the CLOSING quote unless the
    character immediately after it is alphanumeric (glued to a word) — in which
    case it is an interior apostrophe and gets doubled. Already-doubled `''` pairs
    are preserved. This is intentionally only used as a parse-gated fallback
    (see :meth:`SQLCodeAgent._repair_literals`) so it never touches valid SQL.
    """
    out: List[str] = []
    i, n = 0, len(sql)
    in_str = False
    while i < n:
        ch = sql[i]
        if not in_str:
            out.append(ch)
            if ch == "'":
                in_str = True
            i += 1
            continue
        # inside a string literal
        if ch == "'":
            if i + 1 < n and sql[i + 1] == "'":   # already-escaped pair
                out.append("''")
                i += 2
                continue
            nxt = sql[i + 1] if i + 1 < n else ""
            if nxt.isalnum():                     # interior apostrophe → double
                out.append("''")
                i += 1
            else:                                 # closing quote
                out.append("'")
                in_str = False
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ─────────────────────────────────────────────
# Default few-shot examples (decomposition + correct aggregation grain)
# ─────────────────────────────────────────────
# The semantic cache supplies dynamic few-shots, but it is empty until enough
# successful queries accumulate. Until then `few_shots` is []. These static
# examples teach the failure modes we keep hitting on JOIN questions:
#   1. filter-in-one-table / metric-in-another → must JOIN before filtering;
#   2. aggregate GRAIN → one AVG over joined rows, NOT AVG of per-group AVGs,
#      and honor an evidence formula (e.g. /12) verbatim;
#   3. evidence applied VERBATIM as filters → an EXACT date value (Date = '201208',
#      not a year-only SUBSTR) and a FORMULA used in WHERE (Price / Amount > 29).
# Kept in sync with prompt.md [SECTION: FEW_SHOT_TEMPLATE].

DEFAULT_FEWSHOTS: List[str] = [
    (
        "USER_QUESTION: In 2012, which customer in the LAM segment had the least consumption?\n"
        "REASONING: segment is in customers, consumption/date in yearmonth → JOIN on FK; "
        "sum per customer over joined+filtered rows; 'least' → ORDER BY SUM ASC LIMIT 1.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT t1.customerid\n"
        "FROM customers AS t1\n"
        "INNER JOIN yearmonth AS t2 ON t1.customerid = t2.customerid\n"
        "WHERE t1.segment = 'LAM' AND SUBSTR(t2.date, 1, 4) = '2012'\n"
        "GROUP BY t1.customerid\n"
        "ORDER BY SUM(t2.consumption) ASC\n"
        "LIMIT 1;\n"
        "```"
    ),
    (
        "USER_QUESTION: What is the average monthly consumption of SME customers in 2013? "
        "(evidence: average monthly consumption = AVG(consumption) / 12)\n"
        "REASONING: ONE scalar number. Copy the evidence formula EXACTLY. JOIN customers↔"
        "yearmonth, filter, then a SINGLE AVG over joined rows divided by 12 — do NOT "
        "average per customer first (AVG(x)/12 != AVG(per-customer AVG(x))). Scalar → no "
        "GROUP BY, no LIMIT.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT AVG(t2.consumption) / 12\n"
        "FROM customers AS t1\n"
        "INNER JOIN yearmonth AS t2 ON t1.customerid = t2.customerid\n"
        "WHERE SUBSTR(t2.date, 1, 4) = '2013' AND t1.segment = 'SME';\n"
        "```"
    ),
    (
        "USER_QUESTION: Give the consumption in August 2012 for people who paid more "
        "than 29.00 per unit of product id 5. "
        "(evidence: August of 2012 refers to Date = '201208'; price per unit = Price / Amount)\n"
        "REASONING: apply the evidence VERBATIM — the date is an EXACT value "
        "(date = '201208', NOT a year-only SUBSTR), and 'per unit' is the FORMULA "
        "price / amount used as a WHERE filter (guard division with NULLIF). JOIN "
        "transactions_1k↔yearmonth on customerid; return the consumption.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT t2.consumption\n"
        "FROM transactions_1k AS t1\n"
        "INNER JOIN yearmonth AS t2 ON t1.customerid = t2.customerid\n"
        "WHERE t1.price / NULLIF(t1.amount, 0) > 29.00 AND t1.productid = 5 "
        "AND t2.date = '201208';\n"
        "```"
    ),
    (
        "USER_QUESTION: List the names of events that had an attendance of over 20 "
        "students but were not fundraisers.\n"
        "REASONING: 'over 20 attendance' is a per-group COUNT threshold → GROUP BY "
        "the PK event_id (NOT event_name — names recur across weekly events, grouping "
        "by name merges distinct events) + HAVING COUNT(...) > 20 (NOT WHERE). "
        "'were not fundraisers' is an exclusion of a set of rows → EXCEPT the events "
        "whose type = 'Fundraiser'.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT t1.event_name\n"
        "FROM event AS t1\n"
        "INNER JOIN attendance AS t2 ON t1.event_id = t2.link_to_event\n"
        "GROUP BY t1.event_id\n"
        "HAVING COUNT(t2.link_to_event) > 20\n"
        "EXCEPT\n"
        "SELECT event_name FROM event WHERE type = 'Fundraiser';\n"
        "```"
    ),
    (
        "USER_QUESTION: Among students who attended the event 'Women's Soccer', how "
        "many want a medium-size t-shirt? "
        "(evidence: t-shirt in medium size refers to t_shirt_size = 'Medium')\n"
        "REASONING: event and member connect only through the junction table "
        "attendance → join THROUGH it on its FK columns (link_to_event/link_to_member, "
        "named differently from event_id/member_id). Copy exact values: event_name = "
        "'Women''s Soccer' (double the apostrophe), t_shirt_size = 'Medium'. 'how many' "
        "→ COUNT.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT COUNT(t3.member_id)\n"
        "FROM event AS t1\n"
        "INNER JOIN attendance AS t2 ON t1.event_id = t2.link_to_event\n"
        "INNER JOIN member AS t3 ON t2.link_to_member = t3.member_id\n"
        "WHERE t1.event_name = 'Women''s Soccer' AND t3.t_shirt_size = 'Medium';\n"
        "```"
    ),
    (
        "USER_QUESTION: List the names of students who attended more than 7 events. "
        "(evidence: full name refers to first_name, last_name; attended more than 7 "
        "events refers to COUNT(link_to_event) > 7)\n"
        "REASONING: 'full name refers to first_name, last_name' → select the TWO "
        "columns SEPARATELY (never first_name || ' ' || last_name — concatenation "
        "changes the result shape and fails). Per-member count → GROUP BY the PK "
        "member_id, threshold in HAVING.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT t2.first_name, t2.last_name\n"
        "FROM attendance AS t1\n"
        "INNER JOIN member AS t2 ON t1.link_to_member = t2.member_id\n"
        "GROUP BY t2.member_id\n"
        "HAVING COUNT(t1.link_to_event) > 7;\n"
        "```"
    ),
    (
        "USER_QUESTION: Based on the total cost of all events, what is the percentage "
        "of cost for the Yearly Kickoff event?\n"
        "REASONING: percentage OF A TOTAL → the condition goes INSIDE the aggregate "
        "(CASE WHEN), NEVER in WHERE — filtering rows to the event makes the ratio "
        "100%. Denominator = the SAME unfiltered rows. One scalar → no GROUP BY, no "
        "LIMIT.\n"
        "SQL:\n"
        "```sql\n"
        "SELECT CAST(SUM(CASE WHEN t3.event_name = 'Yearly Kickoff' THEN t1.cost "
        "ELSE 0 END) AS REAL) * 100 / SUM(t1.cost)\n"
        "FROM expense AS t1\n"
        "INNER JOIN budget AS t2 ON t1.link_to_budget = t2.budget_id\n"
        "INNER JOIN event AS t3 ON t2.link_to_event = t3.event_id;\n"
        "```"
    ),
]


# ─────────────────────────────────────────────
# Difficulty-adaptive strategy blocks (Issue #27/GH#44)
# ─────────────────────────────────────────────
# One prompt for all difficulties wastes effort on simple questions and gives
# challenging ones no decomposition scaffold. The BIRD path passes the real
# difficulty label; the live pipeline infers a tier from the linker plan.
# Moderate (and tier=None) keeps today's default prompt exactly.

TIER_STRATEGIES: Dict[str, str] = {
    "simple": (
        "STRATEGY (SIMPLE tier): this question needs at most one join and no "
        "composed aggregates. Write the query DIRECTLY from the schema — minimal "
        "clauses, no CTEs, no elaborate planning. Correctness checks still apply."
    ),
    "challenging": (
        "STRATEGY (CHALLENGING tier): decompose BEFORE writing the final query:\n"
        "  1. Write each required sub-result as its own named SELECT.\n"
        "  2. State how the sub-results combine (join / EXCEPT / CASE-sum / "
        "scalar comparison).\n"
        "  3. Compose the final query — CTEs ARE permitted here (this is the "
        "genuine multi-step exception to the simplest-shape rule).\n"
        "  4. Re-trace the GRAIN of every sub-query before composing: do NOT "
        "pre-aggregate and aggregate again unless the question explicitly asks "
        "for a per-group-then-overall value."
    ),
}

# ─────────────────────────────────────────────
# Reasoning-path diversity (GH#48, CHASE-SQL)
# ─────────────────────────────────────────────
# Temperature resamples of ONE prompt produce correlated errors — execution
# voting then confidently agrees on the same wrong answer. Distinct reasoning
# paths disagree in useful ways: the direct prompt, a divide-and-conquer
# decomposition, and a query-execution-plan chain-of-thought each fail
# differently, so the vote/selector has real signal to work with.

PATH_STRATEGIES: Dict[str, str] = {
    "divide_conquer": (
        "STRATEGY (DIVIDE & CONQUER): solve by decomposition, in the answer text\n"
        "BEFORE the final SQL block:\n"
        "  1. Split the question into the smallest sub-questions.\n"
        "  2. For each sub-question write a minimal sub-SELECT and state its\n"
        "     GRAIN (one row per what?).\n"
        "  3. Merge the sub-SELECTs into ONE final query (JOIN / IN / CASE-sum /\n"
        "     CTE), re-checking the grain at every merge.\n"
        "Then output the single merged query as the ONLY ```sql block."
    ),
    "query_plan": (
        "STRATEGY (QUERY EXECUTION PLAN): reason like a database engine, in the\n"
        "answer text BEFORE the final SQL block:\n"
        "  1. SCAN: which tables, which columns.\n"
        "  2. JOIN: the exact FK pairs connecting them.\n"
        "  3. FILTER: every WHERE predicate, with literal values copied exactly.\n"
        "  4. AGGREGATE: function + GROUP BY key (a PK, never a name column).\n"
        "  5. SORT/LIMIT: only if the question asks for it.\n"
        "Then output the query implementing exactly that plan as the ONLY\n"
        "```sql block."
    ),
}


# ─────────────────────────────────────────────
# SQL Code Agent
# ─────────────────────────────────────────────

class SQLCodeAgent:
    """
    Generates top-K SQL candidates from schema context + few-shot examples,
    then reranks using an LLM judge to select the best query.
    """

    TOP_K = 3  # default number of SQL candidates (config `sql.n_candidates` overrides)

    def __init__(self, config: dict, llm: ChatOpenAI):
        self.llm = llm
        self.dialect = config.get("db", {}).get("dialect", "postgres")
        # More candidates → better execution-guided voting (Tier 3, item 8).
        try:
            self.top_k = max(1, int(config.get("sql", {}).get("n_candidates", self.TOP_K)))
        except (TypeError, ValueError):
            self.top_k = self.TOP_K
        self._distinct_postpass = bool(
            config.get("sql", {}).get("distinct_postpass", True))
        # Difficulty-adaptive prompting (Issue #27): per-tier candidate counts.
        self._adaptive = bool(config.get("sql", {}).get("adaptive_prompting", True))
        tiers = config.get("sql", {}).get("candidates_by_tier") or {}
        self._tier_candidates = {
            "simple": int(tiers.get("simple", 2)),
            "moderate": int(tiers.get("moderate", self.top_k)),
            "challenging": int(tiers.get("challenging", 7)),
        }
        # GH#48: reasoning-path-diverse candidates for moderate/challenging.
        self._path_diversity = bool(
            config.get("sql", {}).get("path_diversity", True))

    # ── Generation ──────────────────────────────

    def generate(
        self,
        query: str,
        schema_context: str,
        few_shots: List[str],
        history: List[Dict[str, str]],
        query_plan: Optional[dict] = None,
        tier: Optional[str] = None,
        previous_sql: Optional[str] = None,
    ) -> List[str]:
        """Generate top-k SQL candidates.

        ``query_plan`` is the optional structured plan from SchemaLinkingAgent
        (Issue #17). ``tier`` ("simple"/"moderate"/"challenging", Issue #27)
        selects a strategy block and per-tier candidate count; None or adaptive
        prompting disabled → today's default behavior exactly.
        """
        tier = (tier or "").strip().lower() or None
        if not self._adaptive or tier not in self._tier_candidates:
            tier = None
        n = self._tier_candidates[tier] if tier else self.top_k

        # Fall back to the static decomposition examples when the semantic cache
        # has no dynamic few-shots yet — they teach the JOIN/grain patterns.
        if not few_shots:
            few_shots = DEFAULT_FEWSHOTS

        # GH#48: for moderate/challenging, replace two direct candidates with one
        # divide-and-conquer and one query-plan-CoT candidate — decorrelated
        # errors give execution voting real signal. Simple tier keeps the single
        # cheap direct call.
        diverse = self._path_diversity and tier in ("moderate", "challenging")
        n_direct = max(1, n - len(PATH_STRATEGIES)) if diverse else n

        prompt = self._build_prompt(query, schema_context, few_shots, history,
                                    query_plan, tier=tier, n_candidates=n_direct,
                                    previous_sql=previous_sql)

        system_p = self._system_prompt()

        trace("\n  ┌─ LLM SYSTEM PROMPT (sql_code_agent) ─────────────────────")
        for line in system_p.splitlines():
            trace(f"  │ {line}")
        trace("  ├─ GENERATION PROMPT ──────────────────────────────────────")
        for line in prompt.splitlines():
            trace(f"  │ {line}")
        trace("  └───────────────────────────────────────────────────────────")

        # Captured so the pool workers below attribute their tokens to the
        # request that spawned them (thread-locals don't cross into workers).
        tracker = usage_tracking.current()

        def _invoke(p: str) -> str:
            return invoke_tracked(self.llm, [
                SystemMessage(content=system_p), HumanMessage(content=p),
            ], "generate_sql", tracker=tracker).content

        if diverse:
            path_prompts = {
                name: self._build_prompt(
                    query, schema_context, few_shots, history, query_plan,
                    tier=tier, n_candidates=1, previous_sql=previous_sql,
                    strategy_override=block)
                for name, block in PATH_STRATEGIES.items()
            }
            # The direct + path calls are independent I/O — run them concurrently.
            with ThreadPoolExecutor(max_workers=1 + len(path_prompts)) as pool:
                direct_f = pool.submit(_invoke, prompt)
                path_fs = {name: pool.submit(_invoke, p)
                           for name, p in path_prompts.items()}
                raw = direct_f.result()
                sqls = self._extract_sqls(raw)
                tagged = [("direct", s) for s in sqls]
                for name, f in path_fs.items():
                    try:
                        extra = self._extract_sqls(f.result())
                    except Exception as exc:
                        logger.warning(f"[SQL_GEN] {name} path failed: {exc}")
                        continue
                    # One candidate per extra path — first block wins.
                    tagged += [(name, s) for s in extra[:1]]
            sqls = [s for _, s in tagged]
            for strat, s in tagged:
                trace(f"  🧭 [{strat}] {' '.join(s.split())[:100]}")
            logger.info("[SQL_GEN] path mix: "
                        + ", ".join(f"{strat}×1" if strat != "direct" else
                                    f"direct×{sum(1 for t, _ in tagged if t == 'direct')}"
                                    for strat in dict.fromkeys(t for t, _ in tagged)))
        else:
            raw = _invoke(prompt)
            trace("\n  ┌─ RAW LLM OUTPUT ──────────────────────────────────────────")
            for line in raw.splitlines():
                trace(f"  │ {line}")
            trace("  └───────────────────────────────────────────────────────────")
            sqls = self._extract_sqls(raw)
        # Deterministic repairs BEFORE dialect normalization: fix unescaped
        # apostrophes (Issue #21) and mangled space/hyphen column names, so the
        # sqlite transpile step receives parseable, schema-true SQL.
        sqls = [
            self.normalize_dialect(
                reconcile_identifiers(self._repair_literals(s), schema_context)
            )
            for s in sqls
        ]
        if self._distinct_postpass:
            sqls = [self._inject_distinct(s, query) for s in sqls]
        # Demote wrong-grain (GROUP BY name-only) and plan-deviating candidates.
        sqls = self._order_candidates(sqls, schema_context, query_plan)

        if not sqls:
            logger.warning("[SQL_GEN] No SQL extracted from LLM output")
            trace("  ⚠️  No SQL code blocks found in LLM output!")
        return sqls[: n]

    def normalize_dialect(self, sql: str) -> str:
        """
        Rewrite generated SQL to the configured target dialect.

        No-op unless ``self.dialect == "sqlite"``. For SQLite, runs a best-effort
        sqlglot transpile (postgres → sqlite) followed by a deterministic regex
        pass (:func:`_to_sqlite`) that guarantees the named PostgreSQL-only
        constructs are rewritten even when sqlglot leaves them untouched or fails.
        """
        if not sql:
            return sql

        # Evidence pseudo-functions (SUBTRACT/DIVIDE/Total…) are not SQL in ANY
        # dialect — translate them first, for the live pipeline too.
        out = translate_pseudo_functions(sql)
        if self.dialect != "sqlite":
            return out

        try:
            transpiled = sqlglot.transpile(out, read="postgres", write="sqlite")
            if transpiled and transpiled[0].strip():
                out = transpiled[0]
        except Exception as exc:  # malformed candidate — fall back to regex only
            logger.debug(f"[DIALECT] sqlglot transpile failed: {exc}")

        return _to_sqlite(out)

    # ── GROUP BY grain demotion ─────────────────
    # Entity names recur (weekly events, duplicate person names): GROUP BY a
    # name column merges DISTINCT entities and corrupts per-entity counts
    # (BIRD q1323: GROUP BY event_name summed attendance across recurring
    # events). When the schema shows a PK, candidates that group ONLY by
    # name-like columns are demoted to the back of the candidate list so the
    # vote-winner / reranker default favours PK-grained candidates.

    _PK_PATTERNS = (
        re.compile(r"-\s*(\w+):[^\n]*\[PK", re.IGNORECASE),      # bird.py schema format
        re.compile(r"(\w+)\s*\([^)\n]*\[PK", re.IGNORECASE),     # embedding chunk format
        re.compile(r"\(\s*(\w+):[^,\n)]*,\s*Primary Key", re.IGNORECASE),  # M-Schema
    )

    def _grain_penalty(self, sql: str, schema_context: str) -> int:
        """1 when the query groups ONLY by name-like columns although the schema
        exposes primary keys (wrong grain risk); 0 otherwise / on parse failure."""
        if not sql or not schema_context:
            return 0
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            group = ast.args.get("group")
            if group is None:
                return 0
            group_cols = {c.name.lower() for c in group.find_all(sqlglot.exp.Column) if c.name}
            if not group_cols:
                return 0
            pks = {
                m.lower()
                for pat in self._PK_PATTERNS
                for m in pat.findall(schema_context)
            }
            if not pks or group_cols & pks:
                return 0  # no PK info, or already grouping by a PK — fine
            if all("name" in c for c in group_cols):
                return 1
        except Exception:
            return 0
        return 0

    def _order_by_grain(self, sqls: List[str], schema_context: str) -> List[str]:
        """Stable-sort candidates so wrong-grain (GROUP BY name-only) ones go last."""
        if len(sqls) < 2:
            return sqls
        return sorted(sqls, key=lambda s: self._grain_penalty(s, schema_context))

    # ── Plan-conformance demotion (closes the #17 loop) ──────────────────────
    # The SchemaLinkingAgent resolves tables + the FK join path, but nothing
    # verified the generated SQL actually followed it. Candidates that use a
    # different table set, or miss a join condition the plan resolved, are
    # demoted so plan-conforming candidates win vote-ties and reranker defaults.

    _JOIN_COND_RE = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)")

    def _plan_penalty(self, sql: str, query_plan: Optional[dict]) -> int:
        """0 = conforms; +1 for a table-set mismatch; +1 for a missing plan join."""
        if not sql or not query_plan or not isinstance(query_plan, dict):
            return 0
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception:
            return 0
        penalty = 0

        # Alias → real table map so t1.customerid resolves to its table.
        alias_map: Dict[str, str] = {}
        sql_tables: set = set()
        for t in ast.find_all(sqlglot.exp.Table):
            if not t.name:
                continue
            real = t.name.lower()
            sql_tables.add(real)
            alias_map[(t.alias or t.name).lower()] = real
            alias_map[real] = real

        plan_tables = {
            str(t).strip().lower() for t in (query_plan.get("tables") or []) if t
        }
        if plan_tables and sql_tables and sql_tables != plan_tables:
            penalty += 1

        # Every join condition the plan resolved should appear in the SQL
        # (compared as unordered {(table, column), (table, column)} pairs,
        # with SQL-side aliases resolved to real table names).
        plan_joins = []
        for cond in (query_plan.get("join_path") or []):
            m = self._JOIN_COND_RE.search(str(cond))
            if m:
                a_t, a_c, b_t, b_c = (s.lower() for s in m.groups())
                plan_joins.append(frozenset({(a_t, a_c), (b_t, b_c)}))
        if plan_joins:
            sql_joins = set()
            for eq in ast.find_all(sqlglot.exp.EQ):
                left, right = eq.left, eq.right
                if isinstance(left, sqlglot.exp.Column) and isinstance(right, sqlglot.exp.Column):
                    lt = alias_map.get((left.table or "").lower(), (left.table or "").lower())
                    rt = alias_map.get((right.table or "").lower(), (right.table or "").lower())
                    sql_joins.add(frozenset({(lt, left.name.lower()), (rt, right.name.lower())}))
            if any(pj not in sql_joins for pj in plan_joins):
                penalty += 1
        return penalty

    # ── Join-minimality demotion (Issue #50) ──────────────────────────────────
    # A join that contributes no column anywhere OUTSIDE its own ON clause is
    # pure risk: if the join key isn't fully referentially covered, an INNER
    # JOIN silently drops otherwise-valid rows for no logical reason (BIRD
    # q1189: joining `Patient` when every needed column already lives on
    # `Examination` alone corrupted a COUNT via referential gaps). This is the
    # inverse of _plan_penalty's missing-join check — here a table is present
    # that the query never actually needed.

    def _join_penalty(self, sql: str) -> int:
        """+1 for each joined table whose columns are never referenced outside
        its own ON clause (spurious join risk); 0 on parse failure / no joins."""
        if not sql:
            return 0
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            joins = list(ast.find_all(sqlglot.exp.Join))
            if not joins:
                return 0

            alias_map: Dict[str, str] = {}
            for t in ast.find_all(sqlglot.exp.Table):
                if not t.name:
                    continue
                real = t.name.lower()
                alias_map[(t.alias or t.name).lower()] = real
                alias_map[real] = real

            def _resolve(col: sqlglot.exp.Column) -> Optional[str]:
                key = (col.table or "").lower()
                return alias_map.get(key, key) if key else None

            penalty = 0
            for j in joins:
                table_node = j.this if isinstance(j.this, sqlglot.exp.Table) else None
                if table_node is None or not table_node.name:
                    continue
                joined_table = table_node.name.lower()
                on_ids = {id(c) for c in j.args.get("on").find_all(sqlglot.exp.Column)} \
                    if j.args.get("on") is not None else set()
                used_elsewhere = any(
                    _resolve(c) == joined_table
                    for c in ast.find_all(sqlglot.exp.Column)
                    if id(c) not in on_ids
                )
                if not used_elsewhere:
                    penalty += 1
            return penalty
        except Exception:
            return 0

    def _order_candidates(
        self,
        sqls: List[str],
        schema_context: str,
        query_plan: Optional[dict] = None,
    ) -> List[str]:
        """Stable-sort: plan-conforming, correctly-grained, minimally-joined
        candidates first."""
        if len(sqls) < 2:
            return sqls
        return sorted(
            sqls,
            key=lambda s: (
                self._grain_penalty(s, schema_context)
                + self._plan_penalty(s, query_plan)
                + self._join_penalty(s)
            ),
        )

    # ── DISTINCT post-pass (Issue #28/GH#45 step 4) ──────────────────────────
    # "List …" questions over joins keep leaking duplicate rows (q1239 failed
    # at F1 0.80 missing ONLY a DISTINCT). Deterministic, conservative injection:
    # only for "list" questions, only when the query joins, has no GROUP BY, no
    # aggregates, and no DISTINCT already.

    def _inject_distinct(self, sql: str, question: str) -> str:
        if not sql or not question or not re.search(r"\blist\b", question, re.IGNORECASE):
            return sql
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            if not isinstance(ast, sqlglot.exp.Select):
                return sql
            if ast.args.get("distinct") or ast.args.get("group"):
                return sql
            if not any(True for _ in ast.find_all(sqlglot.exp.Join)):
                return sql
            if any(e.find(sqlglot.exp.AggFunc) for e in ast.expressions):
                return sql
            ast.set("distinct", sqlglot.exp.Distinct())
            out = ast.sql(dialect=self.dialect)
            logger.info("[SQL_GEN] DISTINCT injected for a 'list' question over joins")
            return out
        except Exception:
            return sql

    # ── Apostrophe repair (Issue #21) ───────────

    def _parses(self, sql: str) -> bool:
        try:
            sqlglot.parse_one(sql, dialect=self.dialect)
            return True
        except Exception:
            return False

    def _repair_literals(self, sql: str) -> str:
        """
        Parse-gated repair of unescaped apostrophes in string literals. No-op when
        the SQL already parses, so valid SQL is never altered. Only keeps the
        repaired form if it actually parses.
        """
        if not sql or self._parses(sql):
            return sql
        repaired = _double_interior_apostrophes(sql)
        if repaired != sql and self._parses(repaired):
            logger.info("[SQL_GEN] Repaired unescaped apostrophe(s) in a string literal")
            trace("  🔧 Repaired an unescaped apostrophe in a string literal")
            return repaired
        return sql

    # NOTE (Issue #50): the "CAST the numerator AS REAL" rule below is a known,
    # accepted source of a small number of BIRD gold mismatches — some gold SQL
    # divides two integer aggregates with NO cast at all, which under SQLite
    # truncates to integer division; our REAL cast produces the mathematically
    # correct (but gold-mismatching) ratio in that minority of cases. This is
    # judged net-positive: the same rule fixes the much larger, well-documented
    # integer-division-truncation failure cluster elsewhere. Do not remove it to
    # chase those few mismatches.
    def _build_prompt(
        self,
        query: str,
        schema_context: str,
        few_shots: List[str],
        history: List[Dict],
        query_plan: Optional[dict] = None,
        tier: Optional[str] = None,
        n_candidates: Optional[int] = None,
        previous_sql: Optional[str] = None,
        strategy_override: Optional[str] = None,
    ) -> str:
        few_shot_block = "\n\n".join(few_shots) if few_shots else "No examples available."
        history_block = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) or "No prior context."
        n = n_candidates or self.top_k

        # Optional structured plan from SchemaLinkingAgent (Issue #17).
        plan_block = render_query_plan(query_plan)
        plan_section = f"\n{plan_block}\n" if plan_block else ""
        # Difficulty-adaptive strategy block (Issue #27); moderate/None → none.
        # A reasoning-path override (GH#48) replaces the tier block entirely.
        strategy = strategy_override or TIER_STRATEGIES.get(tier or "")
        strategy_section = f"\n{strategy}\n" if strategy else ""
        # GH#40: the session's last validated SQL — follow-ups usually REFINE it.
        prev_section = (
            f"\n<previous_sql>\n{previous_sql.strip()}\n</previous_sql>\n"
            "If the question is a follow-up that refines the previous request "
            "(adds a filter, changes a limit, narrows the scope), prefer MODIFYING "
            "<previous_sql> over re-deriving from scratch. Ignore it when the "
            "question is unrelated.\n"
            if previous_sql else ""
        )

        dialect = self.dialect.upper()
        return f"""
<schema_context>
{schema_context}
</schema_context>

<few_shot_examples>
{few_shot_block}
</few_shot_examples>

<conversation_history>
{history_block}
</conversation_history>
{prev_section}{plan_section}
<user_question>
{query}
</user_question>

Target SQL dialect: {dialect}
{strategy_section}
Decompose the question silently before writing SQL:
1. FINAL VALUE: exactly what to return. If a <domain_context>/evidence FORMULA is
   given (e.g. "= AVG(consumption)/12"), copy it VERBATIM — do not reinterpret it.
   (An evidence formula may instead define a FILTER — see Step 4.) When the
   formula computes the EXACT quantity the question asks for — "how many times
   more/less", "ratio of", "percentage of", "difference in" — the formula IS the
   SELECT expression itself; it is NOT a HAVING/WHERE threshold to satisfy and
   NOT something to count occurrences of. Concrete trap: "how many times was X
   more than Y" with evidence `DIVIDE(SUM(...X...), SUM(...Y...))` means
   `SELECT SUM(...X...) / SUM(...Y...)` (one ratio number) — NOT
   `SELECT COUNT(*) ... GROUP BY ... HAVING SUM(X) > SUM(Y)` (a count of groups
   where a comparison holds); "how many times" here is idiomatic for a
   multiplier, not a literal COUNT.
   Output shape: "full name refers to first_name, last_name" → TWO separate
   columns, never concatenated. Yes/no question → CASE WHEN ... THEN 1 ELSE 0
   (never 'Yes'/'No' strings). "List X and Y" → exactly those columns, DISTINCT
   when entities repeat. "each X" → per-row values, not a count.
2. TABLES & JOINS: locate each needed column's table. If a filter column lives in
   one table and the metric in another, JOIN them on the FK columns shown in the
   schema. You cannot filter on a column that is not in your FROM/JOIN. If two
   tables connect only through a junction/bridge table, join THROUGH it on its FK
   columns (often named differently, e.g. attendance.link_to_event = event.event_id).
3. GRAIN (most important): JOIN first → FILTER in WHERE → aggregate ONCE over the
   resulting rows. Do NOT pre-aggregate in a CTE/subquery and aggregate again —
   AVG(x) over joined rows is NOT the same number as AVG(per-group AVG(x)). Only
   nest aggregates when the question explicitly asks "average per X then overall".
4. FILTERS: every term → a predicate; map a literal value (e.g. 'SME') to its
   categorical column (exact casing — copy from the column's `values: {{...}}` list
   when shown; escape apostrophes by doubling). Apply every <domain_context>/
   evidence hint VERBATIM — exact values (date = '201208'), exact formats, and
   formulas used as filters (price / NULLIF(amount,0) > 29) go in WHERE/HAVING, not
   only SELECT. Dates: when an EXACT value is given, match it exactly; use
   SUBSTR(date,1,4)=... or date LIKE '2013%' ONLY for a year-only / partial match.
   Evidence formulas are PSEUDO-CODE: SUBTRACT(a,b)=a-b, DIVIDE(a,b)=a/b,
   Total(x)=SUM(x) — never emit them as function calls. Columns with spaces or
   hyphens must be double-quoted exactly as the schema shows ("Examination Date",
   "T-CHO") — never underscored, never left unquoted.
5. COMPOSE: a threshold on a per-group COUNT/SUM/AVG is HAVING after GROUP BY (not
   WHERE). "not / without / except X" is an exclusion — EXCEPT <subquery> for a set
   of rows, or NOT IN (...) / <> for a value. When aggregating per entity, GROUP BY
   the entity's PRIMARY KEY and project the name (GROUP BY t.event_id ... SELECT
   t.event_name) — never GROUP BY a name column when a PK exists: names recur and
   grouping by name merges distinct entities. Percentage OF A TOTAL → condition
   INSIDE the aggregate: CAST(SUM(CASE WHEN cond THEN 1 ELSE 0 END) AS REAL)*100 /
   COUNT(*) — never WHERE-filter the numerator condition (ratio becomes 100%).
   CAST the numerator AS REAL in EVERY division of aggregates (integer division
   truncates). Difference between two periods → two CASE sums in ONE query.
   Multi-part question → one row, multiple columns (extra parts via CASE WHEN).
6. SHAPE: a single scalar aggregate has NO GROUP BY and NO LIMIT. Superlatives →
   ORDER BY <agg> + LIMIT 1 (or N). Grouped superlative ("peak month") → SELECT
   the GROUP KEY grouped+ordered by the aggregate, NOT the aggregate value.
   "Who is older, X or Y?" → the entity name via ORDER BY, never a sentence.

Prefer the simplest correct shape:
  SELECT <agg>(...) FROM a JOIN b ON a.k=b.k WHERE <filters>;
Use a CTE only for genuine multi-step logic (e.g. rank-then-filter).

Generate {n} candidate queries. Each must:
- Use only tables/columns in <schema_context>; JOIN on the shown FK columns.
- Be read-only (SELECT only) with explicit columns — never SELECT *.
- Return ONLY the FINAL VALUE; honor any evidence formula exactly.
- Aggregate at the correct grain (Step 3); no spurious GROUP BY/LIMIT on scalars.

Return ONLY SQL blocks, each wrapped in ```sql ... ``` fences.
""".strip()

    # ── ReAct refine: regenerate from an execution observation ──────────────

    def regenerate(
        self,
        query: str,
        schema_context: str,
        original_sql: str,
        error: str,
        history: List[Dict[str, str]],
    ) -> List[str]:
        """
        Produce corrected SQL candidates given a failed query and the observation
        (an execution error or a 0-row result). Uses the standalone
        ``sql_regeneration.md`` prompt. Never raises a missing-template error to
        the caller — falls back to a built-in template if the file is absent.
        """
        try:
            template = PromptHarness.get_file("sql_regeneration.md")
        except Exception:
            template = (
                "Fix the failed SQL so it answers the question. "
                "Question: {user_question}\nObservation: {validation_error}\n"
                "Original SQL:\n{original_sql}\nSchema:\n{schema_context}\n"
                "Return {top_k} corrected ```sql``` blocks."
            )

        # Use replace (not str.format) so literal braces in the template/SQL
        # examples can never raise KeyError/IndexError.
        prompt = (
            template
            .replace("{validation_error}", error)
            .replace("{original_sql}", original_sql or "")
            .replace("{top_k}", str(self.top_k))
            .replace("{schema_context}", schema_context or "")
            .replace("{user_question}", query)
        )

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=prompt),
        ]
        raw = invoke_tracked(self.llm, messages, "refine_sql").content
        sqls = [
            self.normalize_dialect(
                reconcile_identifiers(self._repair_literals(s), schema_context)
            )
            for s in self._extract_sqls(raw)
        ]
        sqls = self._order_by_grain(sqls, schema_context)
        if not sqls:
            logger.warning("[REFINE] No SQL extracted from regeneration output")
        return sqls[: self.top_k]

    def _system_prompt(self) -> str:
        try:
            base = PromptHarness.get("SQL_CODE_AGENT")
        except Exception:
            base = (
                f"You are an expert {self.dialect.upper()} SQL engineer. "
                "Write correct, safe, read-only SELECT queries only. "
                "Always LIMIT ≤500 rows. Explicit columns only. No explanations."
            )
        return base

    # ── Reranking ───────────────────────────────

    def rerank(self, sqls: List[str], query: str, schema_context: str) -> str:
        """LLM judge picks best SQL from candidates. Falls back to first if rerank fails."""
        if not sqls:
            return ""
        if len(sqls) == 1:
            return sqls[0]

        candidates_block = "\n\n".join(
            f"Candidate {i+1}:\n```sql\n{s}\n```" for i, s in enumerate(sqls)
        )

        try:
            reranker_prompt = PromptHarness.get_file("sql_reranker.md")
        except Exception:
            reranker_prompt = (
                "You are a SQL quality judge. Select the BEST candidate that is most "
                "accurate, efficient, and correct. Reply with ONLY the number: 1, 2, or 3."
            )

        messages = [
            SystemMessage(content=reranker_prompt),
            HumanMessage(content=(
                f"User query: {query}\n\n"
                f"Schema:\n{schema_context[:_RERANKER_SCHEMA_CAP]}\n\n"
                f"{candidates_block}\n\n"
                "Which candidate number is best?"
            )),
        ]

        try:
            raw_choice = invoke_tracked(self.llm, messages, "rerank").content.strip()
            trace(f"\n  ┌─ RERANKER OUTPUT ──────────────────────────────────────────")
            trace(f"  │ {raw_choice}")
            trace(f"  └────────────────────────────────────────────────────────────")
            idx = int(re.search(r"\d", raw_choice).group()) - 1
            chosen = sqls[max(0, min(idx, len(sqls) - 1))]
            trace(f"  🏆 Reranker chose candidate {idx + 1}")
            return chosen
        except Exception:
            trace("  ⚠️  Reranker failed to parse choice — defaulting to candidate 1")
            return sqls[0]

    # ── Helpers ─────────────────────────────────

    def _extract_sqls(self, raw: str) -> List[str]:
        """Extract SQL blocks from an LLM reply.

        Tries ```sql ... ``` (preferred), then bare ``` ... ``` containing a
        SELECT, then an unfenced SELECT (GH#65).
        """
        # GH#65: strip a reasoning model's think block first, or we'd collect
        # candidate SQL the model was only considering, not proposing.
        raw = strip_reasoning(raw or "")

        # Primary: explicit ```sql tag
        blocks = re.findall(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # Fallback: bare ``` fences that contain a SELECT statement
            blocks = re.findall(r"```\s*(SELECT[\s\S]*?)```", raw, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # GH#65 last resort: unfenced SQL. Open-weight models honour the
            # "wrap in ```sql fences" instruction far less reliably than
            # gpt-4o-mini, and returning [] here sends a perfectly good query
            # to fallback_handler for two wasted rounds. Anchored at a line
            # start and terminated by a blank line or `;` so surrounding prose
            # ("Here is the query:") isn't swallowed.
            blocks = re.findall(r"(?im)^[ \t]*(SELECT\b[\s\S]*?)(?:;|\n[ \t]*\n|\Z)", raw)
        return [b.strip() for b in blocks if b.strip()]