"""
db_agent.py — SQL Code Agent + SQL Validator Agent
SQLCodeAgent   : prompt construction → top-K generation → reranking
SQLValidatorAgent: syntax → safety → semantic checks
"""

from __future__ import annotations

import logging
import re as _re
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.safety import BLOCKED as _BLOCKED, SQL_INJECTION as _SQL_INJECTION


_TABLE_DECL = _re.compile(
    r"(?:^|\n)\s*(?:#\s*)?Table:\s*([A-Za-z_][\w]*)|CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[\"`\[]?([A-Za-z_][\w]*)", _re.IGNORECASE)


def schema_table_names(schema_context: str, limit: int = 40) -> List[str]:
    """
    The table names a schema context declares, for error messages.

    Covers both renderings in use: the BIRD harness emits ``Table: name``
    (`eval/bird.py::extract_sqlite_schema`), the live M-Schema path emits
    ``# Table: name`` (`retrieval/embedding.py`), and a raw DDL context emits
    ``CREATE TABLE name``. Best-effort by design — an empty result just means
    the hint is omitted, never that validation changes.
    """
    names: List[str] = []
    for m in _TABLE_DECL.finditer(schema_context or ""):
        name = m.group(1) or m.group(2)
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _boundary_pattern(name: str) -> str:
    r"""
    A regex that finds `name` in a schema string without matching it as part of
    a longer identifier.

    ``\b`` cannot do this job here. It asserts a word/non-word transition, so
    it only means anything next to a word character — and a quoted SQL
    identifier often ends in one that is not. ``"Enrollment (K-12)"`` ends in
    ``)``, and the character after it in a CREATE TABLE is ``"``; both are
    non-word, so ``\b`` finds no transition and the pattern NEVER matches. The
    column is then reported as missing from a schema that plainly defines it.

    That is not a corner case. Every prediction against BIRD's
    california_schools database was rejected this way and scored 0 while the
    SQL was correct, because that is the one database whose columns are
    parenthesised. A leading space is harmless (``"Low Grade"`` ends in ``e``),
    which is why the bug looked like it was about spaces and was not.

    So the guard is applied only on the side where it can mean something: a
    lookaround when the adjacent character of `name` is a word character, and
    nothing when it is already punctuation and cannot run into a neighbour.
    """
    pattern = _re.escape(name)
    if name[:1].isalnum() or name[:1] == "_":
        pattern = r"(?<!\w)" + pattern
    if name[-1:].isalnum() or name[-1:] == "_":
        pattern = pattern + r"(?!\w)"
    return pattern

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
        # Strict mode: a query referencing tables/columns absent from
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
        appear in schema_context, guarded against partial-identifier
        matches by _boundary_pattern.
        Note: advisory only — SchemaRetriever is the authoritative source.
        """
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
                if not _re.search(_boundary_pattern(t), schema_lower)
            ]
            if missing_tables:
                # Name what IS available. Without it the message is unactionable
                # and actively misleading: the table usually exists somewhere,
                # just not in THIS question's schema — a few-shot example or
                # another database is the usual source. The refine round feeds
                # this string straight back to the model as its observation
                # (eval/bird.py), so the list is what lets it self-correct.
                available = schema_table_names(schema_context)
                hint = f". Schema has: {available}" if available else ""
                return False, f"Tables not found in schema: {missing_tables}{hint}"

            # ── Column check (word-boundary; alias/CTE-derived names skipped) ──
            columns_in_sql = {
                c.name.lower()
                for c in ast.find_all(sqlglot.exp.Column)
                if c.name
            } - defined_aliases - table_aliases - cte_names
            missing_cols = [
                c for c in columns_in_sql
                if not _re.search(_boundary_pattern(c), schema_lower)
            ]
            if missing_cols:
                return False, f"Columns not found in schema: {missing_cols}"

        except Exception as e:
            return False, f"Could not parse for semantic check: {e}"

        return True, None


