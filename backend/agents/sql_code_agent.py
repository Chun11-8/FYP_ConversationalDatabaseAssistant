from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.utils import PromptHarness
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

    return out


# ─────────────────────────────────────────────
# Default few-shot examples (decomposition + correct aggregation grain)
# ─────────────────────────────────────────────
# The semantic cache supplies dynamic few-shots, but it is empty until enough
# successful queries accumulate. Until then `few_shots` is []. These static
# examples teach the two failure modes we keep hitting on JOIN questions:
#   1. filter-in-one-table / metric-in-another → must JOIN before filtering;
#   2. aggregate GRAIN → one AVG over joined rows, NOT AVG of per-group AVGs,
#      and honor an evidence formula (e.g. /12) verbatim.
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
]


# ─────────────────────────────────────────────
# SQL Code Agent
# ─────────────────────────────────────────────

class SQLCodeAgent:
    """
    Generates top-K SQL candidates from schema context + few-shot examples,
    then reranks using an LLM judge to select the best query.
    """

    TOP_K = 3  # number of SQL candidates to generate

    def __init__(self, config: dict, llm: ChatOpenAI):
        self.llm = llm
        self.dialect = config.get("db", {}).get("dialect", "postgres")

    # ── Generation ──────────────────────────────

    def generate(
        self,
        query: str,
        schema_context: str,
        few_shots: List[str],
        history: List[Dict[str, str]],
        query_plan: Optional[dict] = None,
    ) -> List[str]:
        """Generate TOP_K SQL candidates.

        ``query_plan`` is the optional structured plan from SchemaLinkingAgent
        (Issue #17). When present it is injected as a <query_plan> block so the
        generator follows resolved tables/joins/filters instead of re-deriving
        them. None → behaves exactly as before.
        """
        # Fall back to the static decomposition examples when the semantic cache
        # has no dynamic few-shots yet — they teach the JOIN/grain patterns.
        if not few_shots:
            few_shots = DEFAULT_FEWSHOTS
        prompt = self._build_prompt(query, schema_context, few_shots, history, query_plan)

        system_p = self._system_prompt()
        messages = [
            SystemMessage(content=system_p),
            HumanMessage(content=prompt),
        ]

        print("\n  ┌─ LLM SYSTEM PROMPT (sql_code_agent) ─────────────────────")
        for line in system_p.splitlines():
            print(f"  │ {line}")
        print("  ├─ GENERATION PROMPT ──────────────────────────────────────")
        for line in prompt.splitlines():
            print(f"  │ {line}")
        print("  └───────────────────────────────────────────────────────────")

        raw = self.llm.invoke(messages).content

        print("\n  ┌─ RAW LLM OUTPUT ──────────────────────────────────────────")
        for line in raw.splitlines():
            print(f"  │ {line}")
        print("  └───────────────────────────────────────────────────────────")

        sqls = self._extract_sqls(raw)
        sqls = [self.normalize_dialect(s) for s in sqls]

        if not sqls:
            logger.warning("[SQL_GEN] No SQL extracted from LLM output")
            print("  ⚠️  No SQL code blocks found in LLM output!")
        return sqls[: self.TOP_K]

    def normalize_dialect(self, sql: str) -> str:
        """
        Rewrite generated SQL to the configured target dialect.

        No-op unless ``self.dialect == "sqlite"``. For SQLite, runs a best-effort
        sqlglot transpile (postgres → sqlite) followed by a deterministic regex
        pass (:func:`_to_sqlite`) that guarantees the named PostgreSQL-only
        constructs are rewritten even when sqlglot leaves them untouched or fails.
        """
        if not sql or self.dialect != "sqlite":
            return sql

        out = sql
        try:
            transpiled = sqlglot.transpile(out, read="postgres", write="sqlite")
            if transpiled and transpiled[0].strip():
                out = transpiled[0]
        except Exception as exc:  # malformed candidate — fall back to regex only
            logger.debug(f"[DIALECT] sqlglot transpile failed: {exc}")

        return _to_sqlite(out)

    def _build_prompt(
        self,
        query: str,
        schema_context: str,
        few_shots: List[str],
        history: List[Dict],
        query_plan: Optional[dict] = None,
    ) -> str:
        few_shot_block = "\n\n".join(few_shots) if few_shots else "No examples available."
        history_block = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) or "No prior context."

        # Optional structured plan from SchemaLinkingAgent (Issue #17).
        plan_block = render_query_plan(query_plan)
        plan_section = f"\n{plan_block}\n" if plan_block else ""

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
{plan_section}
<user_question>
{query}
</user_question>

Target SQL dialect: {dialect}

Decompose the question silently before writing SQL:
1. FINAL VALUE: exactly what to return. If a <domain_context>/evidence FORMULA is
   given (e.g. "= AVG(consumption)/12"), copy it VERBATIM — do not reinterpret it.
2. TABLES & JOINS: locate each needed column's table. If a filter column lives in
   one table and the metric in another, JOIN them on the FK columns shown in the
   schema. You cannot filter on a column that is not in your FROM/JOIN.
3. GRAIN (most important): JOIN first → FILTER in WHERE → aggregate ONCE over the
   resulting rows. Do NOT pre-aggregate in a CTE/subquery and aggregate again —
   AVG(x) over joined rows is NOT the same number as AVG(per-group AVG(x)). Only
   nest aggregates when the question explicitly asks "average per X then overall".
4. FILTERS: every term → a predicate. Map a literal value (e.g. 'SME') to its
   categorical column. Dates: SUBSTR(date,1,4)='2013' or date LIKE '2013%'.
5. SHAPE: a single scalar aggregate has NO GROUP BY and NO LIMIT. Superlatives →
   ORDER BY <agg> + LIMIT 1 (or N).

Prefer the simplest correct shape:
  SELECT <agg>(...) FROM a JOIN b ON a.k=b.k WHERE <filters>;
Use a CTE only for genuine multi-step logic (e.g. rank-then-filter).

Generate {self.TOP_K} candidate queries. Each must:
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
            .replace("{top_k}", str(self.TOP_K))
            .replace("{schema_context}", schema_context or "")
            .replace("{user_question}", query)
        )

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=prompt),
        ]
        raw = self.llm.invoke(messages).content
        sqls = [self.normalize_dialect(s) for s in self._extract_sqls(raw)]
        if not sqls:
            logger.warning("[REFINE] No SQL extracted from regeneration output")
        return sqls[: self.TOP_K]

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
            raw_choice = self.llm.invoke(messages).content.strip()
            print(f"\n  ┌─ RERANKER OUTPUT ──────────────────────────────────────────")
            print(f"  │ {raw_choice}")
            print(f"  └────────────────────────────────────────────────────────────")
            idx = int(re.search(r"\d", raw_choice).group()) - 1
            chosen = sqls[max(0, min(idx, len(sqls) - 1))]
            print(f"  🏆 Reranker chose candidate {idx + 1}")
            return chosen
        except Exception:
            print("  ⚠️  Reranker failed to parse choice — defaulting to candidate 1")
            return sqls[0]

    # ── Helpers ─────────────────────────────────

    def _extract_sqls(self, raw: str) -> List[str]:
        """Extract SQL blocks from fenced code blocks.

        Handles both ```sql ... ``` (preferred) and bare ``` ... ``` fallback.
        """
        # Primary: explicit ```sql tag
        blocks = re.findall(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # Fallback: bare ``` fences that contain a SELECT statement
            blocks = re.findall(r"```\s*(SELECT[\s\S]*?)```", raw, re.DOTALL | re.IGNORECASE)
        return [b.strip() for b in blocks if b.strip()]