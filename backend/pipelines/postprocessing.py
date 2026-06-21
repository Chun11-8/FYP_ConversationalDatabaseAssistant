"""
postprocessing.py — Response Optimizer
Location: backend/pipelines/postprocessing.py

Decides how to return results based on size:
  ≤ 10 rows    → natural language summary (inline)
  11–500 rows  → markdown table + summary
  > 500 rows   → save as CSV file, return download path
  Error        → plain English error message

Also provides:
  - output guardrail: re-scan final response for injection patterns
  - truncate_general: summarize over-long GENERAL/web answers
"""

import logging
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.safety import scan_output

logger = logging.getLogger(__name__)

# Thresholds
INLINE_LIMIT  = 10     # rows → natural language
TABLE_LIMIT   = 500    # rows → markdown table; above this → file

# Max chars for a GENERAL / web-search synthesized answer before summarization
GENERAL_MAX_CHARS = 3000


class ResponseOptimizer:

    RESPONSE_PROMPT = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_response.md")
    ).read()
    
    # Absolute default so CSVs land in the same place regardless of CWD
    _DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"

    def __init__(self, llm: ChatOpenAI, output_dir: Path | str | None = None):
        self.llm = llm
        self.output_dir = str(output_dir or self._DEFAULT_OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def format(
        self, query: str, sql: str, results: Any
    ) -> tuple[str, str | None]:
        """
        Main entry. Routes to correct formatter based on result size.
        results: list of dicts (from psycopg2 or sqlalchemy)

        Returns:
            (response_text, csv_path) — csv_path is set only when results
            exceed TABLE_LIMIT and are written to a file; None otherwise.
        """
        # ── Error ──────────────────────────────────────────────────────
        if results is None:
            return (
                "No results returned. The query executed successfully but found no matching records.",
                None,
            )

        if isinstance(results, str):
            return f"Error: {results}", None

        rows = results if isinstance(results, list) else list(results)
        count = len(rows)

        # ── Empty ───────────────────────────────────────────────────────
        if count == 0:
            return self._format_empty(query), None

        # ── Small → LLM natural language summary ────────────────────────
        if count <= INLINE_LIMIT:
            response, csv_path = self._format_inline(query, sql, rows), None

        # ── Medium → markdown table ─────────────────────────────────────
        elif count <= TABLE_LIMIT:
            response, csv_path = self._format_table(query, sql, rows, count), None

        # ── Large → save CSV, return path ───────────────────────────────
        else:
            response, csv_path = self._format_file(query, sql, rows, count)

        # ── Output guardrail: re-check for injection patterns ───────────
        response = self._guardrail(response)
        return response, csv_path

    def _guardrail(self, text: str) -> str:
        """
        Output-layer safety check. If injection patterns are found in the
        generated response text, replace it with a safe message and log a warning.
        This is a second-pass defense in case the LLM echoed unsafe content.
        """
        if scan_output(text):
            logger.warning("[GUARDRAIL] Injection pattern detected in output; suppressing.")
            return (
                "⚠️ The generated response was suppressed by the output safety filter. "
                "Please rephrase your question."
            )
        return text

    def truncate_general(self, text: str, max_chars: int = GENERAL_MAX_CHARS) -> str:
        """
        If a GENERAL / web-search answer exceeds *max_chars*, ask the LLM to
        summarize it to a tighter length. Falls back to hard truncation if the
        LLM call fails.
        """
        if len(text) <= max_chars:
            return text

        logger.info(f"[OPTIMIZER] GENERAL answer {len(text)} chars > {max_chars}; summarizing.")
        try:
            summary = self.llm.invoke([
                SystemMessage(content=(
                    "You are a concise technical writer. "
                    "Summarize the following answer to at most 400 words "
                    "while keeping all key facts, sources, and code examples."
                )),
                HumanMessage(content=text),
            ]).content
            return summary
        except Exception as exc:
            logger.warning(f"[OPTIMIZER] summarization failed: {exc}; hard-truncating.")
            return text[:max_chars] + "\n\n_(Answer truncated for length.)_"

    # ── Formatters ───────────────────────────────────────────────────────

    def _format_empty(self, query: str) -> str:
        return (
            "No matching records found for your query.\n"
            "This could mean the data doesn't exist or the filters are too specific. "
            "Try broadening your search criteria."
        )

    def _format_inline(self, query: str, sql: str, rows: list) -> str:
        """≤10 rows: ask LLM to summarise in natural language."""
        rows_text = "\n".join(str(r) for r in rows)

        raw = self.llm.invoke([
            SystemMessage(content = self.RESPONSE_PROMPT),
            HumanMessage(content=(
                f"User asked: {query}\n\n"
                f"Results ({len(rows)} rows):\n{rows_text}\n\n"
                f"Write the Answer section ONLY. "
                f"Do NOT include SQL Used or Record Count — those will be added separately."
            )),
        ]).content

        return raw.strip()

    def _format_table(self, query: str, sql: str, rows: list, count: int) -> str:
        """11–500 rows: markdown table with a short LLM summary header."""
        headers = list(rows[0].keys()) if isinstance(rows[0], dict) else [f"col_{i}" for i in range(len(rows[0]))]

        # Build markdown table
        header_row = "| " + " | ".join(headers) + " |"
        divider    = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows  = []
        for row in rows:
            values = row.values() if isinstance(row, dict) else row
            data_rows.append("| " + " | ".join(str(v) for v in values) + " |")

        table = "\n".join([header_row, divider] + data_rows)

        # Short summary from LLM
        summary = self.llm.invoke([
            SystemMessage(content=self.RESPONSE_PROMPT),
            HumanMessage(content=(
                f"User asked: {query}\n"
                f"Row count: {count}\n"
                f"First 3 rows: {rows[:3]}\n\n"
                f"Write the Answer section ONLY. "
                f"Do NOT include SQL Used or Record Count — those will be added separately."
            )),
        ]).content

        return f"{summary.strip()}\n\n{table}"
        
    def _format_file(
        self, query: str, sql: str, rows: list, count: int
    ) -> tuple[str, str]:
        """
        >500 rows: write to CSV file, return (response_text, csv_filepath).
        The caller (FastAPI route) should serve the file as a download.
        """
        # Belt-and-suspenders: recreate the output dir in case it was removed
        # after construction, so a large result never crashes the response path.
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"query_results_{timestamp}.csv"
        filepath  = os.path.join(self.output_dir, filename)

        headers = (
            list(rows[0].keys())
            if isinstance(rows[0], dict)
            else [f"col_{i}" for i in range(len(rows[0]))]
        )

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(
                rows if isinstance(rows[0], dict)
                else [dict(zip(headers, r)) for r in rows]
            )

        text = (
            f"Your query returned **{count} records** — too large to display inline. "
            f"A CSV file has been prepared for download."
        )
        return text, filename
