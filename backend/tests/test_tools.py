"""
Tests for backend/tools/ (Issue #9) and backend/agents/safety.py (Issue #10).

All tests run offline (no network calls, no DB, no LLM). Web search is tested
with a monkey-patched DDGS so the network is never touched.

Run from repo root:
    pytest backend/tests/test_tools.py -v
"""

from __future__ import annotations

import pytest

from backend.tools.base import Tool
from backend.tools.registry import ToolRegistry
from backend.tools.web_search import WebSearchTool
from backend.agents.safety import (
    BLOCKED,
    SQL_INJECTION,
    enforce_limit,
    is_write_blocked,
    is_injection,
    scan_output,
    DEFAULT_LIMIT,
    MAX_LIMIT,
)


# ── Safety helpers ────────────────────────────────────────────────────────────

class TestSafetyHelpers:

    @pytest.mark.parametrize("kw", [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "TRUNCATE", "CREATE", "REPLACE", "MERGE", "CALL", "EXEC",
    ])
    def test_blocked_detects_dml_ddl(self, kw):
        blocked, match = is_write_blocked(f"SELECT 1; {kw} INTO foo")
        assert blocked
        assert match is not None

    def test_blocked_ignores_select(self):
        blocked, _ = is_write_blocked("SELECT id FROM players LIMIT 10")
        assert not blocked

    def test_injection_detects_comment(self):
        assert is_injection("SELECT 1 -- injected comment")

    def test_injection_detects_union(self):
        assert is_injection("SELECT a FROM t UNION SELECT b FROM secret")

    def test_injection_detects_or_tautology(self):
        # 'OR' (no spaces) is the pattern; spaced variants use 1=1 or UNION checks
        assert is_injection("SELECT * FROM users WHERE id='1'OR'1'='1")

    def test_injection_clean_query(self):
        assert not is_injection("SELECT name FROM players WHERE country = 'Brazil'")

    def test_scan_output_with_dml(self):
        assert scan_output("Here is the answer: ; DROP TABLE players;")

    def test_scan_output_clean(self):
        assert not scan_output("There are 42 players from Brazil.")


# ── enforce_limit ────────────────────────────────────────────────────────────

class TestEnforceLimit:

    def test_adds_limit_when_missing(self):
        sql = "SELECT name FROM players"
        result = enforce_limit(sql)
        assert f"LIMIT {DEFAULT_LIMIT}" in result

    def test_preserves_existing_limit_within_max(self):
        sql = "SELECT name FROM players LIMIT 50"
        assert "LIMIT 50" in enforce_limit(sql)

    def test_caps_limit_over_max(self):
        sql = f"SELECT name FROM players LIMIT {MAX_LIMIT + 100}"
        result = enforce_limit(sql)
        assert f"LIMIT {MAX_LIMIT}" in result
        assert str(MAX_LIMIT + 100) not in result

    def test_non_select_passthrough(self):
        sql = "INSERT INTO t VALUES (1)"
        assert enforce_limit(sql) == sql

    def test_case_insensitive_limit(self):
        sql = "select name from players limit 10"
        result = enforce_limit(sql)
        assert "limit 10" in result.lower()

    def test_custom_default(self):
        sql = "SELECT 1"
        result = enforce_limit(sql, default=25, maximum=200)
        assert "LIMIT 25" in result

    def test_custom_max(self):
        sql = "SELECT 1 LIMIT 1000"
        result = enforce_limit(sql, default=100, maximum=200)
        assert "LIMIT 200" in result

    def test_subquery_inner_limit_preserved_outer_capped(self):
        # Inner LIMIT 5 must stay; outer LIMIT 600 must be capped to 500.
        sql = "SELECT * FROM (SELECT id FROM t LIMIT 5) sub LIMIT 600"
        result = enforce_limit(sql)
        assert "LIMIT 5" in result   # inner unchanged
        assert f"LIMIT {MAX_LIMIT}" in result
        assert "LIMIT 600" not in result

    def test_subquery_inner_limit_not_replaced_when_outer_ok(self):
        # Both inner and outer are within max; neither should change.
        sql = "SELECT * FROM (SELECT id FROM t LIMIT 5) sub LIMIT 10"
        result = enforce_limit(sql)
        assert "LIMIT 5" in result
        assert "LIMIT 10" in result


# ── Tool protocol ─────────────────────────────────────────────────────────────

class TestToolProtocol:

    def test_web_search_tool_is_tool(self):
        t = WebSearchTool()
        assert isinstance(t, Tool)

    def test_web_search_has_name_and_description(self):
        t = WebSearchTool()
        assert t.name == "web_search"
        assert len(t.description) > 10


# ── ToolRegistry ──────────────────────────────────────────────────────────────

class TestToolRegistry:

    def _make_stub(self, name="stub"):
        class _Stub:
            def __init__(self, n):
                self.name = n
                self.description = "stub"
            def run(self, args):
                return f"result:{args.get('x', '')}"
        return _Stub(name)

    def test_register_and_get(self):
        reg = ToolRegistry()
        stub = self._make_stub("echo")
        reg.register(stub)
        assert reg.get("echo") is stub

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_run_dispatches(self):
        reg = ToolRegistry()
        reg.register(self._make_stub("echo"))
        assert reg.run("echo", {"x": "hello"}) == "result:hello"

    def test_run_missing_tool_returns_error_string(self):
        reg = ToolRegistry()
        result = reg.run("no_such_tool", {})
        assert "not found" in result

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(self._make_stub("echo"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(self._make_stub("echo"))

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(self._make_stub("a"))
        reg.register(self._make_stub("b"))
        names = [t["name"] for t in reg.list_tools()]
        assert "a" in names and "b" in names

    def test_run_catches_tool_exception(self):
        class Boom:
            name = "boom"
            description = "crashes"
            def run(self, args):
                raise RuntimeError("intentional")
        reg = ToolRegistry()
        reg.register(Boom())
        result = reg.run("boom", {})
        assert "error" in result.lower()


# ── WebSearchTool (monkey-patched) ───────────────────────────────────────────

class FakeDDGS:
    """Offline stand-in for duckduckgo_search.DDGS."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def text(self, query, max_results=5):
        return [
            {
                "title": f"Result for {query}",
                "body": "PostgreSQL SELECT syntax is straightforward.",
                "href": "https://www.postgresql.org/docs/",
            }
        ]


class TestWebSearchTool:

    @pytest.fixture(autouse=True)
    def patch_ddgs(self, monkeypatch):
        import backend.tools.web_search as ws_mod
        monkeypatch.setattr(ws_mod, "DDGS", FakeDDGS, raising=False)
        # Also patch the import path used inside run()
        import sys
        import types
        fake_mod = types.ModuleType("duckduckgo_search")
        fake_mod.DDGS = FakeDDGS
        sys.modules["duckduckgo_search"] = fake_mod

    def test_returns_result_string(self):
        tool = WebSearchTool()
        result = tool.run({"query": "how to write SELECT in postgres"})
        assert "PostgreSQL" in result or "Result for" in result

    def test_empty_query(self):
        tool = WebSearchTool()
        result = tool.run({"query": ""})
        assert "No query" in result

    def test_source_url_included(self):
        tool = WebSearchTool()
        result = tool.run({"query": "postgres docs"})
        assert "postgresql.org" in result

    def test_max_results_respected(self):
        tool = WebSearchTool()
        # Should not raise even when max_results > internal cap
        result = tool.run({"query": "test", "max_results": 100})
        assert isinstance(result, str)
