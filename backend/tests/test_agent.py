"""
Tests for backend/agents/ routing and safety (Issues #9, #10, #11).

All tests mock LLM, schema retrieval, and SQL execution so they run offline.

Run from repo root:
    pytest backend/tests/test_agent.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from backend.agents.sql_validator_agent import SQLValidatorAgent


# ── SQLValidatorAgent — DML / injection blocking ──────────────────────────────

class TestSQLValidatorSafety:

    @pytest.fixture()
    def validator(self):
        return SQLValidatorAgent({"db": {"dialect": "postgres"}})

    @pytest.mark.parametrize("sql", [
        "INSERT INTO players VALUES (1, 'test')",
        "UPDATE players SET name='x' WHERE id=1",
        "DELETE FROM players WHERE id=1",
        "DROP TABLE players",
        "ALTER TABLE players ADD COLUMN foo TEXT",
        "TRUNCATE TABLE players",
        "CREATE TABLE evil (id INT)",
    ])
    def test_dml_ddl_blocked(self, validator, sql):
        result = validator.validate(sql, schema_context="players(id, name)")
        assert result["valid"] is False
        assert result["safety_ok"] is False

    @pytest.mark.parametrize("sql", [
        "SELECT 1 -- injected",
        "SELECT * FROM t UNION SELECT * FROM users",
        "SELECT * WHERE 1=1",
    ])
    def test_injection_patterns_blocked(self, validator, sql):
        result = validator.validate(sql, schema_context="t(id)")
        # Injection OR syntax failure → not valid
        assert result["valid"] is False

    def test_valid_select_passes_safety(self, validator):
        sql = "SELECT id, name FROM players LIMIT 10"
        result = validator.validate(sql, schema_context="table players columns id name")
        assert result["syntax_ok"] is True
        assert result["safety_ok"] is True

    def test_empty_sql_fails(self, validator):
        result = validator.validate("", schema_context="")
        assert result["valid"] is False
        assert "Empty" in result["error"]

    def test_bad_syntax_fails(self, validator):
        result = validator.validate("SELECT FROM", schema_context="t(id)")
        assert result["syntax_ok"] is False

    def test_semantic_advisory_only(self, validator):
        # Unknown table name → semantic warning but still valid (advisory)
        sql = "SELECT id FROM nonexistent_table LIMIT 5"
        result = validator.validate(sql, schema_context="players(id, name)")
        assert result["syntax_ok"] is True
        assert result["safety_ok"] is True
        # semantic_ok is False but valid should still be True (advisory)
        assert result["valid"] is True


# ── enforce_limit integration ─────────────────────────────────────────────────

class TestEnforceLimitIntegration:
    """Verify the centralized enforce_limit is correctly imported by both modules."""

    def test_safety_module_importable(self):
        from backend.agents.safety import enforce_limit, BLOCKED, SQL_INJECTION
        assert callable(enforce_limit)

    def test_validator_uses_centralized_blocked(self):
        from backend.agents import sql_validator_agent as mod
        from backend.agents.safety import BLOCKED
        # The validator's _BLOCKED must be the same object from safety.py
        assert getattr(mod, "_BLOCKED", None) is BLOCKED

    def test_enforce_limit_adds_default(self):
        from backend.agents.safety import enforce_limit, DEFAULT_LIMIT
        result = enforce_limit("SELECT id FROM players")
        assert f"LIMIT {DEFAULT_LIMIT}" in result

    def test_enforce_limit_caps_over_max(self):
        from backend.agents.safety import enforce_limit, MAX_LIMIT
        result = enforce_limit(f"SELECT id FROM players LIMIT {MAX_LIMIT * 2}")
        assert f"LIMIT {MAX_LIMIT}" in result


# ── Tool registry integration ─────────────────────────────────────────────────

class TestToolRegistryIntegration:

    def test_registry_starts_with_web_search(self):
        """OrchestrationAgent registers WebSearchTool by default."""
        # We import just the registry + tool without spinning up the full agent
        from backend.tools.registry import ToolRegistry
        from backend.tools.web_search import WebSearchTool
        reg = ToolRegistry()
        reg.register(WebSearchTool())
        assert reg.get("web_search") is not None

    def test_web_search_returns_string(self, monkeypatch):
        import sys, types
        # Stub out duckduckgo_search
        fake_ddgs_mod = types.ModuleType("duckduckgo_search")

        class FakeDDGS:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def text(self, q, max_results=5):
                return [{"title": "PG docs", "body": "SELECT syntax.", "href": "https://pg.org"}]

        fake_ddgs_mod.DDGS = FakeDDGS
        monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_ddgs_mod)

        from backend.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        result = tool.run({"query": "postgres select syntax"})
        assert isinstance(result, str)
        assert len(result) > 0
