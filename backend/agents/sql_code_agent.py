from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.utils import PromptHarness

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
    ) -> List[str]:
        """Generate TOP_K SQL candidates."""
        prompt = self._build_prompt(query, schema_context, few_shots, history)

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
    ) -> str:
        few_shot_block = "\n\n".join(few_shots) if few_shots else "No examples available."
        history_block = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) or "No prior context."

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

        <user_query>
        {query}
        </user_query>

        Generate {self.TOP_K} distinct SQL query candidates. Each must:
        - Use only tables/columns present in <schema_context>
        - Be read-only (SELECT only, no DML)
        - Include LIMIT clause (max 500 rows)
        - Handle NULL values with COALESCE where appropriate
        - Use explicit column names, never SELECT *

        Return ONLY SQL blocks, each wrapped in ```sql ... ``` fences.
        """.strip()

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