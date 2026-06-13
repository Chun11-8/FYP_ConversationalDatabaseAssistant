"""
db_agent.py — SQL Code Agent + SQL Validator Agent
SQLCodeAgent   : prompt construction → top-K generation → reranking
SQLValidatorAgent: syntax → safety → semantic checks
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from backend.agents.sql_code_agent import SQLCodeAgent
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Blocked DML patterns (safety enforcement)
# ─────────────────────────────────────────────
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|CALL|EXEC)\b",
    re.IGNORECASE,
)
_SQL_INJECTION = re.compile(
    r"(--|;.+|\/\*|\*\/|xp_|UNION\s+SELECT|1\s*=\s*1|'OR'|\"OR\")",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# SQL Validator Agent
# ─────────────────────────────────────────────

class SQLValidatorAgent:
    """
    Three-stage validation pipeline:
      1. Syntax check    — sqlglot parse
      2. Safety check    — DML block + injection detection
      3. Semantic check  — tables/columns exist in schema
    """

    def __init__(self, config: dict):
        self.dialect = config.get("db", {}).get("dialect", "postgres")

    def validate(self, sql: str, schema_context: str) -> Dict[str, Any]:
        result = {
            "valid": False,
            "syntax_ok": False,
            "safety_ok": False,
            "semantic_ok": False,
            "error": None,
        }

        if not sql:
            result["error"] = "Empty SQL"
            return result

        # ── Stage 1: Syntax ──────────────────────
        syntax_ok, syntax_err = self._check_syntax(sql)
        result["syntax_ok"] = syntax_ok
        if not syntax_ok:
            result["error"] = f"Syntax: {syntax_err}"
            return result

        # ── Stage 2: Safety ──────────────────────
        safety_ok, safety_err = self._check_safety(sql)
        result["safety_ok"] = safety_ok
        if not safety_ok:
            result["error"] = f"Safety: {safety_err}"
            return result

        # ── Stage 3: Semantic ────────────────────
        sem_ok, sem_err = self._check_semantic(sql, schema_context)
        result["semantic_ok"] = sem_ok
        if not sem_ok:
            result["error"] = f"Semantic: {sem_err}"
            # Don't hard-block on semantic — let it through with warning
            logger.warning(f"[SEMANTIC_CHECK] {sem_err}")

        result["valid"] = syntax_ok and safety_ok  # semantic is advisory
        return result

    # ── Stage Implementations ────────────────────

    def _check_syntax(self, sql: str):
        try:
            sqlglot.parse_one(sql, dialect=self.dialect)
            return True, None
        except sqlglot.errors.ParseError as e:
            return False, str(e)

    def _check_safety(self, sql: str):
        if _BLOCKED.search(sql):
            match = _BLOCKED.search(sql).group()
            return False, f"Blocked keyword detected: {match}"
        if _SQL_INJECTION.search(sql):
            return False, "Potential SQL injection pattern detected"
        # Ensure only SELECT at top level
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            if ast.__class__.__name__ != "Select":
                return False, f"Non-SELECT statement detected: {ast.__class__.__name__}"
        except Exception:
            pass  # already caught in syntax check
        return True, None

    def _check_semantic(self, sql: str, schema_context: str):
        """
        Lightweight semantic check: extract table names from SQL and verify
        they appear in schema_context. 
        Note: this is best-effort — SchemaRetriever is the authoritative source.
        """
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            tables_in_sql = {
                t.name.lower()
                for t in ast.find_all(sqlglot.exp.Table)
                if t.name
            }
            schema_lower = schema_context.lower()
            missing = [t for t in tables_in_sql if t not in schema_lower]
            if missing:
                return False, f"Tables not found in schema: {missing}"
        except Exception as e:
            return False, f"Could not parse for semantic check: {e}"

        return True, None


