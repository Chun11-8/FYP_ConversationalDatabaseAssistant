"""
Tests for GH#81 — tag each schema chunk with the logical database (BIRD db_id)
that owns its table, and scope retrieval to one of them. All offline: the DB
boundary is faked and the catalog is built from a temp JSON file.

Run from repo root:
    pytest backend/tests/test_db_catalog_scoping.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import backend.retrieval.db_catalog as catmod
import backend.retrieval.pgvector_store as pgstore
from backend.retrieval.data_processing import extract_schema_pg
from backend.retrieval.db_catalog import DBCatalog, group_by_db, load_catalog

_TABLES_JSON = [
    {"db_id": "debit_card_specializing",
     "table_names_original": ["customers", "gasstations", "yearmonth"]},
    {"db_id": "european_football_2",
     "table_names_original": ["Player_Attributes", "Team"]},
]


@pytest.fixture
def catalog_file(tmp_path):
    p = tmp_path / "dev_tables.json"
    p.write_text(json.dumps(_TABLES_JSON), encoding="utf-8")
    catmod.reset_cache()
    yield p
    catmod.reset_cache()


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parses_bird_tables_json(catalog_file):
    cat = DBCatalog.from_tables_json(catalog_file)
    assert len(cat) == 5
    assert cat.databases() == ["debit_card_specializing", "european_football_2"]
    assert cat.tables_for_db("debit_card_specializing") == [
        "customers", "gasstations", "yearmonth"]


def test_lookup_is_case_insensitive_because_postgres_folds_identifiers(catalog_file):
    cat = DBCatalog.from_tables_json(catalog_file)
    # BIRD says Player_Attributes; PostgreSQL stores player_attributes.
    assert cat.db_for_table("player_attributes", "FYP") == "european_football_2"
    assert cat.db_for_table("PLAYER_ATTRIBUTES", "FYP") == "european_football_2"
    assert cat.db_for_table('"Team"', "FYP") == "european_football_2"


def test_unknown_table_keeps_the_fallback(catalog_file):
    cat = DBCatalog.from_tables_json(catalog_file)
    assert cat.db_for_table("some_local_table", "FYP") == "FYP"
    assert cat.db_for_table("", "FYP") == "FYP"


def test_a_table_claimed_twice_keeps_the_first_claim(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([
        {"db_id": "a", "table_names_original": ["shared"]},
        {"db_id": "b", "table_names_original": ["shared"]},
    ]), encoding="utf-8")
    assert DBCatalog.from_tables_json(p).db_for_table("shared", "FYP") == "a"


def test_missing_file_degrades_to_an_empty_catalog(tmp_path, monkeypatch):
    catmod.reset_cache()
    monkeypatch.delenv("BIRD_TABLES_JSON", raising=False)
    monkeypatch.setattr(catmod, "DEFAULT_CATALOG_PATHS", ())
    cat = load_catalog(str(tmp_path / "nope.json"))
    assert not cat
    # An empty catalog must reproduce the pre-GH#81 labelling exactly.
    assert cat.db_for_table("customers", "FYP") == "FYP"
    catmod.reset_cache()


def test_malformed_file_degrades_instead_of_raising(tmp_path):
    catmod.reset_cache()
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert not load_catalog(str(p))
    catmod.reset_cache()


def test_group_by_db_reports_the_scopes(catalog_file):
    cat = DBCatalog.from_tables_json(catalog_file)
    assert group_by_db(["yearmonth", "Team", "sessions_local"], "FYP", cat) == {
        "FYP": ["sessions_local"],
        "debit_card_specializing": ["yearmonth"],
        "european_football_2": ["Team"],
    }


# ── the real catalog shipped in outputs/ ──────────────────────────────────────

def test_the_repos_own_dev_tables_json_covers_the_bird_minidev_set():
    catmod.reset_cache()
    cat = load_catalog()
    catmod.reset_cache()
    if not cat:                      # not checked out in this environment
        pytest.skip("outputs/dev_tables.json not present")
    assert len(cat.databases()) == 11
    assert cat.db_for_table("customers", "FYP") == "debit_card_specializing"
    assert cat.db_for_table("yearmonth", "FYP") == "debit_card_specializing"
    # The 4 tables the failing UI question needed must share one scope.
    assert cat.db_for_table("gasstations", "FYP") == "debit_card_specializing"
    assert cat.db_for_table("transactions_1k", "FYP") == "debit_card_specializing"


# ── tagging at build time ─────────────────────────────────────────────────────

class _SchemaCursor:
    def __init__(self, table_names):
        self._table_names = table_names
        self._rows = []

    def execute(self, sql, params=None):
        if "information_schema.tables" in sql:
            self._rows = [(t,) for t in self._table_names]
        elif "information_schema.columns" in sql:
            self._rows = [("id", "integer", "NO", None)]
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


def test_extract_schema_pg_tags_each_table_with_its_owning_db(catalog_file):
    cat = DBCatalog.from_tables_json(catalog_file)
    conn = MagicMock()
    conn.cursor.return_value = _SchemaCursor(
        ["customers", "player_attributes", "local_notes"])

    tables = extract_schema_pg(conn, "FYP", catalog=cat)

    assert {t["table_name"]: t["db_name"] for t in tables} == {
        "customers": "debit_card_specializing",
        "player_attributes": "european_football_2",
        # unknown to the catalog → keeps the build's --db name
        "local_notes": "FYP",
    }


# ── pruning the old labels ────────────────────────────────────────────────────

def test_prune_relabelled_chunks_deletes_only_the_stale_label():
    cur = MagicMock()
    cur.rowcount = 1
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    n = pgstore.prune_relabelled_chunks(conn, [
        {"table_name": "customers", "db_name": "debit_card_specializing"},
        {"table_name": "yearmonth", "db_name": "debit_card_specializing"},
    ])

    assert n == 2
    for call in cur.execute.call_args_list:
        sql, params = call[0]
        assert "db_name <> %s" in sql
        assert params[1] == "debit_card_specializing"
    conn.commit.assert_called_once()


def test_prune_relabelled_chunks_is_a_noop_without_chunks():
    conn = MagicMock()
    assert pgstore.prune_relabelled_chunks(conn, []) == 0
    conn.cursor.assert_not_called()


# ── retrieval scope ───────────────────────────────────────────────────────────

def _agent_with(retrieval_cfg):
    from backend.agents.db_agent import OrchestrationAgent
    cfg = {"llm": {"model": "gpt-4o-mini", "api_key": "x"},
           "db": {"name": "FYP", "dialect": "postgres"},
           "retrieval": retrieval_cfg}
    return OrchestrationAgent(cfg)


def test_active_db_overrides_the_postgres_database_name():
    agent = _agent_with({"active_db": "debit_card_specializing"})
    assert agent._db_name == "debit_card_specializing"
    assert agent.active_db == "debit_card_specializing"


def test_without_active_db_every_scope_is_searched():
    """None reproduces the pre-GH#81 behaviour: the PostgreSQL database name
    was never a real filter, because every chunk carried it."""
    agent = _agent_with({})
    assert agent._db_name is None


def test_unscoped_chunk_lookup_spans_every_database():
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    pgstore.get_chunks_by_tables(conn, ["customers"], None)
    sql, params = cur.execute.call_args[0]
    assert "db_name" not in sql.split("WHERE", 1)[1]
    assert params == (["customers"],)

    cur.reset_mock()
    pgstore.get_chunks_by_tables(conn, ["customers"], "debit_card_specializing")
    sql, params = cur.execute.call_args[0]
    assert "db_name = %s" in sql
    assert params[0] == "debit_card_specializing"


def test_unscoped_list_tables_spans_every_database():
    cur = MagicMock()
    cur.fetchall.return_value = [("customers",)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    assert pgstore.list_tables(conn, None) == ["customers"]
    assert "DISTINCT" in cur.execute.call_args[0][0]


# ── GET /databases ────────────────────────────────────────────────────────────

class TestDatabasesEndpoint:

    def _client(self):
        from fastapi.testclient import TestClient
        import backend.api.main as api_main
        return TestClient(api_main.app), api_main

    def test_lists_the_scopes_and_marks_the_active_one(self):
        from unittest.mock import patch
        client, api_main = self._client()
        agent = MagicMock()
        agent.active_db = "debit_card_specializing"
        store = MagicMock()
        with patch.object(api_main, "_get_agent", return_value=agent), \
             patch.object(pgstore, "get_store", return_value=store), \
             patch.object(pgstore, "list_databases_with_counts",
                          return_value=[("debit_card_specializing", 5),
                                        ("superhero", 10)]):
            r = client.get("/databases")

        assert r.status_code == 200
        body = r.json()
        assert body["active"] == "debit_card_specializing"
        assert body["databases"] == [
            {"name": "debit_card_specializing", "tables": 5, "active": True},
            {"name": "superhero", "tables": 10, "active": False},
        ]
        store.close.assert_called_once()

    def test_index_unavailable_is_a_503_not_a_500(self):
        from unittest.mock import patch
        client, api_main = self._client()
        agent = MagicMock()
        agent.active_db = "FYP"
        with patch.object(api_main, "_get_agent", return_value=agent), \
             patch.object(pgstore, "get_store", side_effect=OSError("no pg")):
            r = client.get("/databases")
        assert r.status_code == 503
