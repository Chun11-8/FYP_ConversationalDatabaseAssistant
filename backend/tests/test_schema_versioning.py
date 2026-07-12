"""
Tests for the Schema Versioning Layer (Issue #7 / GH#8).

Covers the pure diff/hash functions, the check()/sync() orchestration with all
DB/embedding boundaries mocked, and the /schema/diff + /schema/sync endpoints.
All offline.

Run from repo root:
    pytest backend/tests/test_schema_versioning.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import backend.api.main as api_main
from backend.agents.schema_versioning_agent import (
    SchemaVersioningAgent, hash_schema, diff_schemas, snapshot_from_tables,
)


# ── pure functions ────────────────────────────────────────────────────────────

class TestDiff:

    OLD = {"players": {"id": "integer", "name": "text"},
           "teams": {"id": "integer"}}

    def test_add_table(self):
        new = {**self.OLD, "leagues": {"id": "integer"}}
        assert diff_schemas(self.OLD, new) == [{"action": "add_table", "table": "leagues"}]

    def test_drop_table(self):
        new = {"players": self.OLD["players"]}
        assert {"action": "drop_table", "table": "teams"} in diff_schemas(self.OLD, new)

    def test_add_and_drop_column(self):
        new = {**self.OLD, "players": {"id": "integer", "age": "integer"}}
        d = diff_schemas(self.OLD, new)
        assert {"action": "add_column", "table": "players", "column": "age"} in d
        assert {"action": "drop_column", "table": "players", "column": "name"} in d

    def test_modify_column(self):
        new = {**self.OLD, "players": {"id": "bigint", "name": "text"}}
        d = diff_schemas(self.OLD, new)
        assert any(c["action"] == "modify_column" and c["column"] == "id"
                   and "→" in c["detail"] for c in d)

    def test_no_change(self):
        assert diff_schemas(self.OLD, dict(self.OLD)) == []

    def test_hash_stable_and_order_independent(self):
        a = {"t": {"a": "int", "b": "text"}}
        b = {"t": {"b": "text", "a": "int"}}
        assert hash_schema(a) == hash_schema(b)
        assert hash_schema(a) != hash_schema({"t": {"a": "bigint"}})

    def test_snapshot_from_tables(self):
        tables = [{"table_name": "p",
                   "columns": [{"name": "id", "type": "integer"},
                               {"name": "name", "type": "text"}]}]
        assert snapshot_from_tables(tables) == {"p": {"id": "integer", "name": "text"}}


# ── check() / sync() orchestration ─────────────────────────────────────────────

class TestAgentOrchestration:

    def _agent(self, current, last_snap, indexed):
        a = SchemaVersioningAgent(db_name="FYP")
        a.current_snapshot = MagicMock(return_value=current)
        a.last_snapshot = MagicMock(
            return_value=({"hash": hash_schema(last_snap), "snapshot": last_snap,
                           "created_at": "2026-01-01"} if last_snap is not None else None))
        a._indexed_tables = MagicMock(return_value=indexed)
        a._reembed_tables = MagicMock(side_effect=lambda ts: list(ts))
        a._remove_tables = MagicMock(side_effect=lambda ts: list(ts))
        a._persist = MagicMock(return_value="newhash")
        return a

    def test_check_reports_changes_and_unindexed(self):
        a = self._agent(
            current={"players": {"id": "int"}, "teams": {"id": "int"}},
            last_snap={"players": {"id": "int"}},          # teams is new
            indexed=["players"],                            # teams unindexed
        )
        out = a.check()
        assert out["in_sync"] is False
        assert {"action": "add_table", "table": "teams"} in out["changes"]
        assert "teams" in out["unindexed_tables"]

    def test_check_in_sync(self):
        a = self._agent(current={"players": {"id": "int"}},
                        last_snap={"players": {"id": "int"}},
                        indexed=["players"])
        assert a.check()["in_sync"] is True

    def test_check_survives_vector_store_down(self):
        a = self._agent(current={"players": {"id": "int"}},
                        last_snap={"players": {"id": "int"}}, indexed=[])
        a._indexed_tables = MagicMock(side_effect=RuntimeError("store down"))
        out = a.check()          # must not raise
        assert out["unindexed_tables"] == ["players"]   # can't confirm indexed → treated unindexed

    def test_sync_surgical_only_touches_changed_and_unindexed(self):
        a = self._agent(
            current={"players": {"id": "int", "age": "int"}, "teams": {"id": "int"}},
            last_snap={"players": {"id": "int"}, "old": {"x": "int"}},  # +age, +teams, -old
            indexed=["players", "old"],
        )
        out = a.sync()
        # re-embed the modified 'players' and the new 'teams'; NOT untouched ones
        assert set(out["reembedded_tables"]) == {"players", "teams"}
        assert out["removed_tables"] == ["old"]         # dropped table's chunks removed
        a._persist.assert_called_once()

    def test_sync_full_reembeds_everything(self):
        a = self._agent(current={"players": {"id": "int"}, "teams": {"id": "int"}},
                        last_snap={"players": {"id": "int"}, "teams": {"id": "int"}},
                        indexed=["players", "teams"])
        out = a.sync(full=True)
        assert set(out["reembedded_tables"]) == {"players", "teams"}
        assert out["full"] is True

    def test_sync_tables_filter(self):
        a = self._agent(current={"players": {"id": "int"}, "teams": {"id": "int"}},
                        last_snap=None, indexed=[])
        out = a.sync(tables=["teams"])
        assert out["reembedded_tables"] == ["teams"]

    def test_first_sync_no_snapshot_treats_all_as_new(self):
        a = self._agent(current={"players": {"id": "int"}, "teams": {"id": "int"}},
                        last_snap=None, indexed=[])
        out = a.check()
        assert {c["table"] for c in out["changes"]} == {"players", "teams"}


# ── API endpoints ──────────────────────────────────────────────────────────────

class TestEndpoints:

    def _client(self):
        return TestClient(api_main.app)

    def test_diff_endpoint(self):
        fake = MagicMock()
        fake.check.return_value = {
            "db_name": "FYP", "in_sync": False, "last_synced_at": None,
            "changes": [{"action": "add_table", "table": "teams"}],
            "unindexed_tables": ["teams"],
        }
        with patch.object(api_main, "_versioner", return_value=fake):
            r = self._client().get("/schema/diff")
        assert r.status_code == 200
        body = r.json()
        assert body["in_sync"] is False
        assert body["changes"][0]["table"] == "teams"

    def test_sync_endpoint_passes_full_flag(self):
        fake = MagicMock()
        fake.sync.return_value = {
            "db_name": "FYP", "schema_hash": "h", "changes_applied": [],
            "reembedded_tables": ["players"], "removed_tables": [], "full": True,
        }
        with patch.object(api_main, "_versioner", return_value=fake):
            r = self._client().post("/schema/sync", json={"full": True})
        assert r.status_code == 200
        assert r.json()["reembedded_tables"] == ["players"]
        fake.sync.assert_called_once_with(full=True, tables=None)

    def test_diff_endpoint_500_on_error(self):
        fake = MagicMock()
        fake.check.side_effect = RuntimeError("db down")
        with patch.object(api_main, "_versioner", return_value=fake):
            r = self._client().get("/schema/diff")
        assert r.status_code == 500
