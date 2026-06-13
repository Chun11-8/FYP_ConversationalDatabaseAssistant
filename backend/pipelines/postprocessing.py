"""
postprocessing.py — Response Optimizer
Location: backend/pipelines/postprocessing.py

Decides how to return results based on size:
  ≤ 10 rows    → natural language summary (inline)
  11–500 rows  → markdown table + summary
  > 500 rows   → save as CSV file, return download path
  Error        → plain English error message
"""

import os
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Thresholds
INLINE_LIMIT  = 10     # rows → natural language
TABLE_LIMIT   = 500    # rows → markdown table; above this → file


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
            return self._format_inline(query, sql, rows), None

        # ── Medium → markdown table ─────────────────────────────────────
        if count <= TABLE_LIMIT:
            return self._format_table(query, sql, rows, count), None

        # ── Large → save CSV, return path ───────────────────────────────
        return self._format_file(query, sql, rows, count)

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

        return (
            f"{raw}\n\n"
            f"**SQL Used:**\n```sql\n{sql}\n```\n\n"
            f"_{len(rows)} record(s) returned_"
        )
 
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

        return (
            f"{summary}\n\n"
            f"{table}\n\n"
            f"**SQL Used:**\n```sql\n{sql}\n```\n\n"
            f"_{count} record(s) returned_"
        )
        
    def _format_file(
        self, query: str, sql: str, rows: list, count: int
    ) -> tuple[str, str]:
        """
        >500 rows: write to CSV file, return (response_text, csv_filepath).
        The caller (FastAPI route) should serve the file as a download.
        """
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
            f"Your query returned **{count} records** — too large to display inline.\n\n"
            f"A CSV file has been prepared for download.\n\n"
            f"**SQL Used:**\n```sql\n{sql}\n```\n\n"
            f"_{count} record(s) returned_"
        )
        return text, filename
