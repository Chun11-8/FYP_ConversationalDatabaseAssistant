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

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=prompt),
        ]

        raw = self.llm.invoke(messages).content
        
        #print(raw)
        sqls = self._extract_sqls(raw)

        if not sqls:
            logger.warning("[SQL_GEN] No SQL extracted from LLM output")
        return sqls[: self.TOP_K]

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
            choice = self.llm.invoke(messages).content.strip()
            idx = int(re.search(r"\d", choice).group()) - 1
            return sqls[max(0, min(idx, len(sqls) - 1))]
        except Exception:
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