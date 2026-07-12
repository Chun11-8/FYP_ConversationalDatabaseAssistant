"""
Tests for GH#42: per-stage timings, request-ID correlation, banner gating,
and /metrics-lite. All offline.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import backend.api.main as api_main
from backend.agents.db_agent import _staged
from backend.agents.trace import trace, set_pipeline_trace, pipeline_trace_enabled


class TestStageTimings:

    def test_staged_records_ms_in_state(self):
        node = _staged("generate_sql", lambda s: {"best_sql": "SELECT 1"})
        out = node({"timings": {"classify_intent": 5}})
        assert "generate_sql" in out["timings"]
        assert out["timings"]["classify_intent"] == 5        # prior stages kept
        assert isinstance(out["timings"]["generate_sql"], int)

    def test_reentry_accumulates(self):
        node = _staged("generate_sql", lambda s: {"best_sql": "SELECT 1"})
        first = node({"timings": {}})
        second = node({"timings": first["timings"]})
        assert second["timings"]["generate_sql"] >= first["timings"]["generate_sql"]

    def test_chat_response_accepts_new_fields(self):
        from backend.api.models import ChatResponse
        r = ChatResponse(response="x", intent="READ", latency_ms=123,
                         timings={"generate_sql": 88})
        assert r.latency_ms == 123 and r.timings["generate_sql"] == 88


class TestTraceGate:

    def test_disabled_suppresses_output(self, capsys):
        set_pipeline_trace(False)
        try:
            trace("BANNER LINE")
            assert "BANNER" not in capsys.readouterr().out
        finally:
            set_pipeline_trace(True)
        trace("BANNER BACK")
        assert "BANNER BACK" in capsys.readouterr().out
        assert pipeline_trace_enabled() is True


class TestRequestId:

    def test_header_assigned_and_echoed(self):
        client = TestClient(api_main.app)
        resp = client.post("/session")
        assert resp.headers.get("X-Request-ID")

    def test_provided_header_is_kept(self):
        client = TestClient(api_main.app)
        resp = client.post("/session", headers={"X-Request-ID": "abc123"})
        assert resp.headers["X-Request-ID"] == "abc123"

    def test_filter_stamps_records(self):
        rec = logging.LogRecord("n", logging.INFO, "p", 1, "m", (), None)
        assert api_main.RequestIdFilter().filter(rec) is True
        assert rec.request_id == "-"        # default outside a request


class TestMetricsLite:

    def _reset(self):
        with api_main._METRICS_LOCK:
            api_main._METRICS["turns"] = 0
            api_main._METRICS["errors"] = 0
            api_main._METRICS["cache_hits"] = 0
            api_main._METRICS["latencies_ms"].clear()
            api_main._METRICS["stage_ms"].clear()
            api_main._METRICS["stage_n"].clear()

    def test_turns_latency_and_cache_rate(self):
        self._reset()
        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            {"response": "a", "intent": "READ", "latency_ms": 100,
             "timings": {"generate_sql": 60}, "cache_hit": False},
            {"response": "b", "intent": "READ", "latency_ms": 300,
             "timings": {"generate_sql": 20}, "cache_hit": True},
        ]
        with patch.object(api_main, "_agent", mock_agent):
            client = TestClient(api_main.app)
            for _ in range(2):
                assert client.post("/chat", json={"session_id": "s", "message": "q"}).status_code == 200
            m = client.get("/metrics-lite").json()
        assert m["turns"] == 2
        assert m["cache_hits"] == 1 and m["cache_hit_rate"] == 0.5
        assert m["latency_avg_ms"] == 200
        assert m["avg_stage_ms"]["generate_sql"] == 40

    def test_error_counted(self):
        self._reset()
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("boom")
        with patch.object(api_main, "_agent", mock_agent):
            client = TestClient(api_main.app)
            assert client.post("/chat", json={"session_id": "s", "message": "q"}).status_code == 500
            m = client.get("/metrics-lite").json()
        assert m["errors"] == 1 and m["turns"] == 0
