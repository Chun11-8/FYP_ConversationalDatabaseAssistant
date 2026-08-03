"""
db_agent.py — SQL Code Agent + SQL Validator Agent
SQLCodeAgent   : prompt construction → top-K generation → reranking
SQLValidatorAgent: syntax → safety → semantic checks
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.safety import BLOCKED as _BLOCKED, SQL_INJECTION as _SQL_INJECTION
logger = logging.getLogger(__name__)


def referenced_tables(sql: str, dialect: str = "postgres") -> List[str]:
    """
    Real table names a query reads from, excluding CTE names (which are defined
    by the query itself, not by the schema). Empty list on a parse failure —
    callers treat "no names" as "can't narrow", never as "no tables".
    """
    try:
        ast = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return []
    cte_names = {c.alias_or_name.lower()
                 for c in ast.find_all(sqlglot.exp.CTE) if c.alias_or_name}
    return sorted({t.name for t in ast.find_all(sqlglot.exp.Table)
                   if t.name and t.name.lower() not in cte_names})


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
        # Strict mode (GH #19): a query referencing tables/columns absent from
        # the schema context is BLOCKED (triggers regeneration) instead of
        # executing and silently returning wrong rows. CTE names and aliases
        # defined inside the query itself are always tolerated.
        self.strict_semantic = bool(config.get("sql", {}).get("strict_semantic", True))

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
            logger.warning(f"[SEMANTIC_CHECK] {sem_err}")

        # Strict mode blocks on unknown tables/columns (so the regeneration
        # fallback fires with the missing name); advisory mode only warns.
        result["valid"] = syntax_ok and safety_ok and (sem_ok or not self.strict_semantic)
        return result

    # ── Stage Implementations ────────────────────

    def _check_syntax(self, sql: str):
        try:
            sqlglot.parse_one(sql, dialect=self.dialect)
            return True, None
        except sqlglot.errors.SqlglotError as e:
            # Base class covers ParseError AND TokenError (e.g. an unescaped
            # apostrophe in a string literal). TokenError is NOT a ParseError
            # subclass, so catching only ParseError would let it 500 upstream.
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
        Lightweight semantic check: verify table and column names from the SQL
        appear in schema_context using word-boundary matching.
        Note: advisory only — SchemaRetriever is the authoritative source.
        """
        import re as _re
        try:
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            schema_lower = schema_context.lower()

            # ── Names DEFINED inside the query are always legitimate ─────────
            # CTE names act as tables; SELECT/CTE aliases act as columns.
            cte_names = {
                c.alias_or_name.lower()
                for c in ast.find_all(sqlglot.exp.CTE)
                if c.alias_or_name
            }
            defined_aliases = {
                a.alias.lower()
                for a in ast.find_all(sqlglot.exp.Alias)
                if isinstance(a.alias, str) and a.alias
            }
            # Table aliases (e.g. `players p`) may be referenced as qualifiers,
            # and subquery aliases behave like tables too.
            table_aliases = {
                t.alias.lower()
                for t in ast.find_all(sqlglot.exp.Table)
                if t.alias
            } | {
                s.alias.lower()
                for s in ast.find_all(sqlglot.exp.Subquery)
                if s.alias
            }

            # ── Table check (word-boundary to avoid substring false-positives) ──
            tables_in_sql = {
                t.name.lower()
                for t in ast.find_all(sqlglot.exp.Table)
                if t.name
            } - cte_names
            missing_tables = [
                t for t in tables_in_sql
                if not _re.search(rf"\b{_re.escape(t)}\b", schema_lower)
            ]
            if missing_tables:
                return False, f"Tables not found in schema: {missing_tables}"

            # ── Column check (word-boundary; alias/CTE-derived names skipped) ──
            columns_in_sql = {
                c.name.lower()
                for c in ast.find_all(sqlglot.exp.Column)
                if c.name
            } - defined_aliases - table_aliases - cte_names
            missing_cols = [
                c for c in columns_in_sql
                if not _re.search(rf"\b{_re.escape(c)}\b", schema_lower)
            ]
            if missing_cols:
                return False, f"Columns not found in schema: {missing_cols}"

        except Exception as e:
            return False, f"Could not parse for semantic check: {e}"

        return True, None


