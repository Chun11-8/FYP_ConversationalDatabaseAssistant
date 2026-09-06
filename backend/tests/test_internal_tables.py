"""
Tests for GH#79 — the assistant's own tables must never be indexed or scanned
as user data. All offline: the DB boundary is faked.

Run from repo root:
    pytest backend/tests/test_internal_tables.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import backend.retrieval.pgvector_store as pgstore
from backend.retrieval.data_processing import extract_schema_pg
from backend.retrieval.internal_tables import (
    INTERNAL_TABLES, filter_internal, is_internal_table,
)
from backend.retrieval.value_index import ValueIndex


# ── the list itself ───────────────────────────────────────────────────────────

def test_every_table_the_app_creates_is_listed():
    """The list is the single source of truth; a new app table must join it."""
    for name in ("sessions", "turns", "query_sql_pairs", "query_interactions",
                 "feedback", "schema_embeddings", "kb_embeddings",
                 "schema_version_store"):
        assert is_internal_table(name), name


def test_domain_and_bird_tables_are_not_internal():
    # The BIRD benchmark tables must stay indexed — they are the user's data.
    for name in ("customers", "yearmonth", "transactions_1k", "gasstations",
                 "players", "teams", "sessions_log", "user_feedback_form"):
        assert not is_internal_table(name), name


def test_matching_is_case_and_quote_insensitive():
    assert is_internal_table("Sessions")
    assert is_internal_table('"TURNS"')
    assert is_internal_table("  feedback  ")
    assert not is_internal_table(None)
    assert not is_internal_table("")


def test_filter_internal_preserves_order_of_the_rest():
    assert filter_internal(
        ["customers", "sessions", "yearmonth", "feedback"]
    ) == ["customers", "yearmonth"]
    assert filter_internal(None) == []


# ── indexing source ───────────────────────────────────────────────────────────

class _SchemaCursor:
    """Answers the four information_schema queries extract_schema_pg makes."""

    def __init__(self, table_names):
        self._table_names = table_names
        self._rows = []
        self.queried_tables = []

    def execute(self, sql, params=None):
        if "information_schema.tables" in sql:
            self._rows = [(t,) for t in self._table_names]
        elif "information_schema.columns" in sql:
            self.queried_tables.append(params[0])
            self._rows = [("id", "integer", "NO", None)]
        else:                                    # PK / FK constraint lookups
            self._rows = []

    def fetchall(self):
        return self._rows


def test_extract_schema_pg_skips_the_assistants_own_tables():
    cur = _SchemaCursor(["customers", "sessions", "schema_embeddings",
                         "yearmonth", "turns", "feedback"])
    conn = MagicMock()
    conn.cursor.return_value = cur

    tables = extract_schema_pg(conn, "FYP")

    assert [t["table_name"] for t in tables] == ["customers", "yearmonth"]
    # and no wasted column/PK/FK round-trips for the excluded ones
    assert "sessions" not in cur.queried_tables


# ── value index ───────────────────────────────────────────────────────────────

def test_value_index_refuses_internal_tables_without_touching_the_db():
    factory = MagicMock(side_effect=AssertionError("must not connect"))
    idx = ValueIndex(factory)

    assert idx._load_table("query_sql_pairs") == {}
    factory.assert_not_called()


def test_value_index_still_loads_domain_tables():
    cur = MagicMock()
    cur.execute.return_value = None
    cur.fetchall.side_effect = [[("segment",)], [("LAM",), ("SME",)]]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    idx = ValueIndex(lambda: conn)
    assert idx._load_table("customers") == {"segment": ["LAM", "SME"]}


# ── FK graph ──────────────────────────────────────────────────────────────────

def test_fk_graph_drops_edges_touching_internal_tables():
    cur = MagicMock()
    cur.fetchall.return_value = [
        ("yearmonth", "customerid", "customers", "customerid"),
        ("turns", "session_id", "sessions", "session_id"),
        ("query_interactions", "session_id", "sessions", "session_id"),
    ]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    graph, edges = pgstore.get_fk_graph(conn)

    assert set(graph) == {"yearmonth", "customers"}
    assert [e["from_table"] for e in edges] == ["yearmonth"]


# ── purge of a pre-existing index ─────────────────────────────────────────────

def test_purge_internal_chunks_deletes_only_internal_rows():
    cur = MagicMock()
    cur.rowcount = 7
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    assert pgstore.purge_internal_chunks(conn, "FYP") == 7

    sql, params = cur.execute.call_args[0]
    assert "DELETE FROM schema_embeddings" in sql
    assert params[0] == "FYP"
    assert set(params[1]) == set(INTERNAL_TABLES)
    conn.commit.assert_called_once()
