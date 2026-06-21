"""
Tests for graceful degradation (Issue #13).

All tests run offline. They confirm the system stays usable — no 500s, no
exceptions — when optional components (DB, vector index, LLM, cache) are absent.

Run from repo root:
    pytest backend/tests/test_startup_degraded.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import backend.api.main as api_main


# ── /health survives a DB outage ──────────────────────────────────────────────

class TestHealthDegraded:

    @pytest.fixture()
    def client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("PG_PASSWORD", "test-pw")

        mock_agent = MagicMock()
        mock_agent.semantic_cache.is_enabled = False
        mock_agent.memory.is_persistent = False

        mock_config = {
            "llm": {"model": "gpt-4o-mini", "api_key": "sk-test"},
            "db": {"host": "127.0.0.1", "port": 5432, "name": "FYP",
                   "user": "postgres", "password": "test-pw", "dialect": "postgres"},
        }

        with patch.object(api_main, "load_config", return_value=mock_config), \
             patch.object(api_main, "OrchestrationAgent", return_value=mock_agent), \
             patch.object(api_main, "_check_db", return_value=False), \
             patch.object(api_main, "_check_vector_index", return_value=False):
            with TestClient(api_main.app, raise_server_exceptions=True) as c:
                yield c

    def test_health_returns_200_when_db_down(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_degraded_components(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["db_connected"] is False
        assert body["vector_index"] is False
        assert body["cache_enabled"] is False
        assert body["memory_persistent"] is False


# ── generate_sql degrades on LLM failure instead of raising ───────────────────

class TestGenerateSqlDegraded:

    def test_llm_failure_yields_empty_sql_not_exception(self):
        from backend.agents.db_agent import OrchestrationAgent

        # Bypass __init__ (which needs config/LLM) and inject a failing sql_agent.
        agent = OrchestrationAgent.__new__(OrchestrationAgent)
        failing = MagicMock()
        failing.generate.side_effect = RuntimeError("OpenAI down")
        agent.sql_agent = failing

        state = {
            "user_query": "show all players",
            "schema_context": "players(id, name)",
            "few_shot_examples": [],
            "generated_sqls": [],
            "conversation_history": [],
            "fallback_level": 0,
        }

        # Must not raise; best_sql empty so validate_sql → fallback path.
        result = agent.generate_sql(state)
        assert result["best_sql"] == ""
        assert result["generated_sqls"] == []


# ── SemanticCache is a true no-op without a DB ────────────────────────────────

class TestSemanticCacheNoop:

    def test_disabled_without_db_config(self):
        from backend.memory.semantic_cache import SemanticCache
        cache = SemanticCache(db_config=None)
        assert cache.is_enabled is False

    def test_all_ops_are_noops(self):
        from backend.memory.semantic_cache import SemanticCache
        cache = SemanticCache(db_config=None)
        # None of these should raise or touch a DB.
        assert cache.retrieve_few_shots("show players") == []
        assert cache.embed("anything") is None
        cache.save_pair(user_query="q", sql="SELECT 1", embedding=None)
        cache.log_interaction(
            session_id="s", user_query="q", sql=None, response="r",
            intent="GENERAL", valid=None, row_count=None, latency_ms=1,
        )


# ── SessionMemory falls back to RAM without a DB ──────────────────────────────

class TestSessionMemoryRamFallback:

    def test_ram_backend_when_no_db(self):
        from backend.memory.session_memory import SessionMemory
        mem = SessionMemory(db_config=None)
        assert mem.is_persistent is False

    def test_add_and_read_roundtrip_in_ram(self):
        from backend.memory.session_memory import SessionMemory
        mem = SessionMemory(db_config=None)
        mem.add("sess-1", "hello", "hi there", sql=None, valid=None)
        history = mem.get_history("sess-1")
        assert any(t["content"] == "hello" for t in history)
        assert any(t["content"] == "hi there" for t in history)


# ── ResponseOptimizer recreates outputs/ if it was removed ────────────────────

class TestOutputsDirRecreated:

    def test_format_file_recreates_missing_dir(self, tmp_path):
        from backend.pipelines.postprocessing import ResponseOptimizer

        out_dir = tmp_path / "outputs"
        optimizer = ResponseOptimizer(MagicMock(), output_dir=out_dir)

        # Simulate the directory being deleted after construction.
        shutil.rmtree(out_dir)
        assert not out_dir.exists()

        rows = [{"id": i, "name": f"p{i}"} for i in range(3)]
        text, filename = optimizer._format_file("q", "SELECT 1", rows, len(rows))

        assert filename is not None
        assert (out_dir / filename).exists()
        assert "record(s) returned" in text
