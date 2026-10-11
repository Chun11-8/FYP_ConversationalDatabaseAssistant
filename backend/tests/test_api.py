"""
Tests for backend/api/main.py.

Run from repo root:
    pytest backend/tests/test_api.py -v

All LLM and DB calls are mocked — no live services required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import the module once so patch.object can reference it
import backend.api.main as api_main


# ── Fixtures ───────────────────────────────────────────────────────────────────

MOCK_RUN_RESULT = {
    "response": "There are 42 players with overall rating ≥ 67.",
    "sql": "SELECT player_name FROM players WHERE overall >= 67 LIMIT 5;",
    "intent": "READ",
    "validation": {"valid": True, "syntax_ok": True, "safety_ok": True},
    "row_count": 5,
    "csv_path": None,
    "turn_id": 7,
}


@pytest.fixture()
def client(monkeypatch):
    """
    Create a TestClient with the agent fully mocked.
    Uses patch.object so the already-imported module is patched correctly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PG_PASSWORD", "test-pw")

    mock_agent = MagicMock()
    mock_agent.run.return_value = MOCK_RUN_RESULT
    mock_agent.memory.get_history.return_value = []

    mock_config = {
        "llm": {"model": "gpt-4o-mini", "api_key": "sk-test"},
        "db": {"host": "127.0.0.1", "port": 5432, "name": "FYP",
               "user": "postgres", "password": "test-pw", "dialect": "postgres"},
    }

    with patch.object(api_main, "load_config", return_value=mock_config), \
         patch.object(api_main, "OrchestrationAgent", return_value=mock_agent), \
         patch.object(api_main, "_check_db", return_value=True):

        with TestClient(api_main.app, raise_server_exceptions=True) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_db_connected_field_present(self, client):
        resp = client.get("/health")
        assert "db_connected" in resp.json()


# ── /session ───────────────────────────────────────────────────────────────────

class TestCreateSession:

    def test_returns_session_id(self, client):
        resp = client.post("/session")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_session_id_is_uuid(self, client):
        import uuid
        resp = client.post("/session")
        session_id = resp.json()["session_id"]
        # Should not raise
        uuid.UUID(session_id)

    def test_each_call_returns_unique_id(self, client):
        ids = {client.post("/session").json()["session_id"] for _ in range(5)}
        assert len(ids) == 5


# ── /chat ──────────────────────────────────────────────────────────────────────

class TestChat:

    def test_happy_path_returns_structured_response(self, client):
        resp = client.post("/chat", json={
            "session_id": "sess-abc",
            "message": "Show top 5 players by rating",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == MOCK_RUN_RESULT["response"]
        assert data["sql"] == MOCK_RUN_RESULT["sql"]
        assert data["intent"] == "READ"
        assert data["row_count"] == 5
        assert data["csv_path"] is None
        assert data["turn_id"] == 7

    def test_missing_message_returns_422(self, client):
        resp = client.post("/chat", json={"session_id": "sess-abc"})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client):
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 422

    def test_empty_message_returns_422(self, client):
        resp = client.post("/chat", json={"session_id": "sess-abc", "message": ""})
        assert resp.status_code == 422

    def test_agent_exception_returns_500(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.run.side_effect = RuntimeError("LLM timeout")
            resp = client.post("/chat", json={
                "session_id": "sess-abc",
                "message": "break things",
            })
        assert resp.status_code == 500
        assert "Agent error" in resp.json()["detail"]


# ── /history ───────────────────────────────────────────────────────────────────

class TestSessions:

    def test_lists_sessions(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.memory.list_sessions.return_value = [
                {"session_id": "s1", "title": "First chat",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "updated_at": "2026-01-02T00:00:00+00:00"},
            ]
            resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "s1"
        assert data[0]["title"] == "First chat"

    def test_lists_empty(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.memory.list_sessions.return_value = []
            resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []


class TestHistory:

    def test_returns_list_for_known_session(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.memory.get_history.return_value = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            resp = client.get("/history/sess-abc")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 2

    def test_returns_empty_list_for_unknown_session(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.memory.get_history.return_value = []
            resp = client.get("/history/unknown-session")
        assert resp.status_code == 200
        assert resp.json() == []


# ── /download ──────────────────────────────────────────────────────────────────

class TestDownload:

    def test_serves_existing_csv(self, client, tmp_path):
        # Write a real CSV into the outputs dir
        from backend.api.main import _OUTPUTS_DIR
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_file = _OUTPUTS_DIR / "query_results_test.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")

        resp = client.get("/download/query_results_test.csv")
        assert resp.status_code == 200
        assert "col1" in resp.text

        csv_file.unlink(missing_ok=True)

    def test_missing_file_returns_404(self, client):
        resp = client.get("/download/does_not_exist.csv")
        assert resp.status_code == 404

    def test_path_traversal_double_dot_rejected(self, client):
        resp = client.get("/download/../secrets.env")
        # FastAPI URL routing will 404 on path with .., but our guard is belt-and-suspenders
        assert resp.status_code in (400, 404)

    def test_path_traversal_encoded_rejected(self, client):
        resp = client.get("/download/..%2Fsecrets.env")
        assert resp.status_code in (400, 404)


# ── /chat with api_key ─────────────────────────────────────────────────────────

class TestChatApiKey:

    def test_custom_api_key_forwarded_to_agent(self, client):
        """The api_key from the request body must be passed to agent.run()."""
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.run.return_value = MOCK_RUN_RESULT
            client.post("/chat", json={
                "session_id": "sess-key",
                "message": "hello",
                "api_key": "sk-custom-key-123",
            })
            call_kwargs = mock_agent.run.call_args.kwargs
            assert call_kwargs.get("api_key") == "sk-custom-key-123"

    def test_omitted_api_key_passes_none_to_agent(self, client):
        """When api_key is omitted, agent.run() receives None."""
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.run.return_value = MOCK_RUN_RESULT
            client.post("/chat", json={"session_id": "sess-key", "message": "hello"})
            call_kwargs = mock_agent.run.call_args.kwargs
            assert call_kwargs.get("api_key") is None

    def test_empty_api_key_treated_as_none(self, client):
        """Empty string api_key should be coerced to None by the endpoint."""
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.run.return_value = MOCK_RUN_RESULT
            client.post("/chat", json={
                "session_id": "sess-key",
                "message": "hello",
                "api_key": "",
            })
            call_kwargs = mock_agent.run.call_args.kwargs
            assert call_kwargs.get("api_key") is None


# ── /schema ────────────────────────────────────────────────────────────────────

MOCK_SCHEMA = {
    "db_name": "FYP",
    "table_count": 2,
    "tables": [
        {
            "table_name": "players",
            "column_count": 3,
            "columns": [
                {"name": "id", "type": "integer", "not_null": True, "primary_key": True, "foreign_key": None},
                {"name": "name", "type": "text", "not_null": True, "primary_key": False, "foreign_key": None},
                {"name": "overall", "type": "integer", "not_null": False, "primary_key": False, "foreign_key": None},
            ],
        },
        {
            "table_name": "teams",
            "column_count": 2,
            "columns": [
                {"name": "id", "type": "integer", "not_null": True, "primary_key": True, "foreign_key": None},
                {"name": "name", "type": "text", "not_null": True, "primary_key": False, "foreign_key": None},
            ],
        },
    ],
}


class TestSchema:

    def test_schema_returns_db_info(self, client):
        with patch.object(api_main, "get_schema_info", return_value=MOCK_SCHEMA):
            resp = client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["db_name"] == "FYP"
        assert data["table_count"] == 2
        assert len(data["tables"]) == 2

    def test_schema_tables_have_columns(self, client):
        with patch.object(api_main, "get_schema_info", return_value=MOCK_SCHEMA):
            resp = client.get("/schema")
        players = next(t for t in resp.json()["tables"] if t["table_name"] == "players")
        assert players["column_count"] == 3
        assert any(c["primary_key"] for c in players["columns"])

    def test_schema_db_error_returns_500(self, client):
        with patch.object(api_main, "get_schema_info", side_effect=RuntimeError("DB down")):
            resp = client.get("/schema")
        assert resp.status_code == 500
        assert "Schema fetch failed" in resp.json()["detail"]

    def test_schema_in_openapi(self, client):
        resp = client.get("/openapi.json")
        assert "/schema" in resp.json()["paths"]


# ── /feedback ──────────────────────────────────────────────────────────────────

class TestFeedback:

    def test_thumbs_up_returns_ok(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.semantic_cache.add_feedback.return_value = None
            resp = client.post("/feedback", json={
                "turn_id": 7,
                "session_id": "sess-abc",
                "rating": "up",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_thumbs_down_returns_ok(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.semantic_cache.add_feedback.return_value = None
            resp = client.post("/feedback", json={
                "turn_id": 7,
                "session_id": "sess-abc",
                "rating": "down",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_feedback_with_comment(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.semantic_cache.add_feedback.return_value = None
            resp = client.post("/feedback", json={
                "turn_id": 7,
                "session_id": "sess-abc",
                "rating": "down",
                "comment": "SQL was wrong",
            })
        assert resp.status_code == 200
        mock_agent.semantic_cache.add_feedback.assert_called_once_with(
            turn_id=7,
            session_id="sess-abc",
            rating="down",
            comment="SQL was wrong",
        )

    def test_invalid_rating_returns_422(self, client):
        resp = client.post("/feedback", json={
            "turn_id": 7,
            "session_id": "sess-abc",
            "rating": "maybe",
        })
        assert resp.status_code == 422

    def test_missing_turn_id_returns_422(self, client):
        resp = client.post("/feedback", json={
            "session_id": "sess-abc",
            "rating": "up",
        })
        assert resp.status_code == 422

    def test_cache_exception_returns_500(self, client):
        with patch("backend.api.main._agent") as mock_agent:
            mock_agent.semantic_cache.add_feedback.side_effect = RuntimeError("DB down")
            resp = client.post("/feedback", json={
                "turn_id": 7,
                "session_id": "sess-abc",
                "rating": "up",
            })
        assert resp.status_code == 500
        assert "Feedback error" in resp.json()["detail"]

    def test_feedback_in_openapi(self, client):
        resp = client.get("/openapi.json")
        assert "/feedback" in resp.json()["paths"]


# ── docs ───────────────────────────────────────────────────────────────────────

class TestDocs:

    def test_openapi_docs_render(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_has_all_endpoints(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/health" in paths
        assert "/session" in paths
        assert "/chat" in paths
        assert "/history/{session_id}" in paths
        assert "/download/{filename}" in paths
        assert "/schema" in paths
        assert "/sessions" in paths
        assert "/feedback" in paths
