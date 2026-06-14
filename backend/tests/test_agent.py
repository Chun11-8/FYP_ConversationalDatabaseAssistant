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


# ── FK-aware retrieval expansion ──────────────────────────────────────────────

class TestExpandFkTables:
    """
    Unit tests for expand_fk_tables(). All DB calls are mocked so these run
    offline without a live PostgreSQL connection.
    """

    def _make_retrieve_result(self, table_names: list[str]) -> dict:
        """Minimal retrieve() output for the given table names."""
        tables = [
            {"table_name": n, "text": f"Table: {n}\nColumns: id (int) [PK]"}
            for n in table_names
        ]
        context = "\n\n".join(t["text"] for t in tables)
        return {"tables": tables, "knowledge_base": [], "prompt_context": context}

    def _patch_pgstore(self, monkeypatch, fk_edges: list[dict], chunks: list[dict]):
        """Monkey-patch pgvector_store functions used by expand_fk_tables."""
        import backend.retrieval.pgvector_store as pgstore

        def fake_get_store(*a, **kw):
            return object()  # sentinel — never actually used

        def fake_get_fk_graph(conn):
            graph: dict[str, set] = {}
            for e in fk_edges:
                graph.setdefault(e["from_table"], set()).add(e["to_table"])
                graph.setdefault(e["to_table"], set()).add(e["from_table"])
            return graph, fk_edges

        def fake_get_chunks_by_tables(conn, table_names, db_name):
            return [c for c in chunks if c["table_name"] in table_names]

        monkeypatch.setattr(pgstore, "get_store", fake_get_store)
        monkeypatch.setattr(pgstore, "get_fk_graph", fake_get_fk_graph)
        monkeypatch.setattr(pgstore, "get_chunks_by_tables", fake_get_chunks_by_tables)

    def test_no_fk_edges_returns_original_context(self, monkeypatch):
        self._patch_pgstore(monkeypatch, fk_edges=[], chunks=[])
        from backend.retrieval.indexing import expand_fk_tables
        result_in = self._make_retrieve_result(["players"])
        result = expand_fk_tables(result_in, "FYP")
        assert result["fk_tables"] == []
        assert result["fk_edges"] == []
        assert "players" in result["prompt_context"]

    def test_single_hop_fk_table_added(self, monkeypatch):
        edges = [{"from_table": "players", "from_col": "team_id",
                  "to_table": "teams", "to_col": "team_id"}]
        chunks = [{"table_name": "teams", "content": "Table: teams\nColumns: team_id (int) [PK], team_name (text)"}]
        self._patch_pgstore(monkeypatch, fk_edges=edges, chunks=chunks)

        from backend.retrieval.indexing import expand_fk_tables
        result_in = self._make_retrieve_result(["players"])
        result = expand_fk_tables(result_in, "FYP")

        assert len(result["fk_tables"]) == 1
        assert result["fk_tables"][0]["table_name"] == "teams"
        assert "FK-Related Tables" in result["prompt_context"]
        assert "Foreign Key Relationships" in result["prompt_context"]
        assert "players.team_id → teams.team_id" in result["prompt_context"]

    def test_already_retrieved_table_not_duplicated(self, monkeypatch):
        edges = [{"from_table": "players", "from_col": "team_id",
                  "to_table": "teams", "to_col": "team_id"}]
        chunks = [{"table_name": "teams", "content": "Table: teams\n..."}]
        self._patch_pgstore(monkeypatch, fk_edges=edges, chunks=chunks)

        from backend.retrieval.indexing import expand_fk_tables
        # Both players AND teams already retrieved — teams should not appear twice
        result_in = self._make_retrieve_result(["players", "teams"])
        result = expand_fk_tables(result_in, "FYP")

        assert result["fk_tables"] == []   # nothing new to add

    def test_fk_graph_failure_returns_safe_fallback(self, monkeypatch):
        import backend.retrieval.pgvector_store as pgstore
        monkeypatch.setattr(pgstore, "get_store", lambda *a, **kw: object())
        monkeypatch.setattr(pgstore, "get_fk_graph", lambda conn: (_ for _ in ()).throw(RuntimeError("DB down")))

        from backend.retrieval.indexing import expand_fk_tables
        result_in = self._make_retrieve_result(["players"])
        result = expand_fk_tables(result_in, "FYP")
        # Must not raise; returns original result with empty expansion
        assert result["fk_tables"] == []
        assert "players" in result["prompt_context"]

    def test_primary_section_header_present(self, monkeypatch):
        edges = [{"from_table": "players", "from_col": "league_id",
                  "to_table": "leagues", "to_col": "league_id"}]
        chunks = [{"table_name": "leagues", "content": "Table: leagues\nColumns: league_id (int) [PK]"}]
        self._patch_pgstore(monkeypatch, fk_edges=edges, chunks=chunks)

        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._make_retrieve_result(["players"]), "FYP")

        assert "### Relevant Table Schemas (Retrieved by Query)" in result["prompt_context"]
        assert "### FK-Related Tables" in result["prompt_context"]

    def test_empty_primary_tables_returns_early(self, monkeypatch):
        self._patch_pgstore(monkeypatch, fk_edges=[], chunks=[])
        from backend.retrieval.indexing import expand_fk_tables
        result_in = {"tables": [], "knowledge_base": [], "prompt_context": ""}
        result = expand_fk_tables(result_in, "FYP")
        assert result["fk_tables"] == []

    def test_bidirectional_expansion(self, monkeypatch):
        # FK goes player_stats → players (backward direction)
        # Retrieving 'players' should still pull in 'player_stats'
        edges = [{"from_table": "player_stats", "from_col": "player_id",
                  "to_table": "players", "to_col": "player_id"}]
        chunks = [{"table_name": "player_stats", "content": "Table: player_stats\n..."}]
        self._patch_pgstore(monkeypatch, fk_edges=edges, chunks=chunks)

        from backend.retrieval.indexing import expand_fk_tables
        result = expand_fk_tables(self._make_retrieve_result(["players"]), "FYP")
        assert any(c["table_name"] == "player_stats" for c in result["fk_tables"])
