"""
Tests for GH#85 — make the live pipeline retrieve schema the way the BIRD
harness does, and stop guessing an effort tier. All offline: the vector store
and the LLM boundaries are mocked.

Run from repo root:
    pytest backend/tests/test_full_schema_alignment.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.agents.db_agent as dbm
import backend.retrieval.indexing as idx
from backend.agents.db_agent import OrchestrationAgent


def _agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_db_name="debit_card_specializing", _full_schema=True,
                    _top_k_tables=5, _fk_max_hops=2, _fk_max_expansion=5,
                    _max_schema_tokens=4000, _live_tier="moderate")
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


_FULL = {
    "tables": [{"chunk_id": "c1", "db_name": "debit_card_specializing",
                "table_name": t, "text": f"Table: {t}", "distance": 0.0}
               for t in ("customers", "gasstations", "products",
                         "transactions_1k", "yearmonth")],
    "knowledge_base": [],
    "prompt_context": "### Relevant Table Schemas\nTable: customers\nTable: yearmonth",
    "full_schema": True,
}


# ── fetch_full_schema ─────────────────────────────────────────────────────────

class TestFetchFullSchema:

    def _store(self, names, chunks):
        backend = MagicMock()
        backend.list_tables.return_value = names
        backend.get_chunks_by_tables.return_value = chunks
        return backend

    def test_returns_every_table_of_the_scope(self):
        chunks = [{"chunk_id": f"c{i}", "db_name": "d", "table_name": n,
                   "content": f"Table: {n}"} for i, n in enumerate(("a", "b"))]
        backend = self._store(["a", "b"], chunks)
        with patch.object(idx, "_get_backend", return_value=backend):
            out = idx.fetch_full_schema("d")
        assert [t["table_name"] for t in out["tables"]] == ["a", "b"]
        assert out["full_schema"] is True
        assert "Table: a" in out["prompt_context"]
        backend.list_tables.assert_called_once_with(backend.get_store(), "d")

    def test_distance_is_zero_because_nothing_was_ranked(self):
        chunks = [{"chunk_id": "c", "db_name": "d", "table_name": "a",
                   "content": "Table: a"}]
        with patch.object(idx, "_get_backend",
                          return_value=self._store(["a"], chunks)):
            out = idx.fetch_full_schema("d")
        assert all(t["distance"] == 0.0 for t in out["tables"])

    def test_no_scope_returns_none(self):
        assert idx.fetch_full_schema("") is None
        assert idx.fetch_full_schema(None) is None

    def test_nothing_indexed_for_the_scope_returns_none(self):
        with patch.object(idx, "_get_backend",
                          return_value=self._store([], [])):
            assert idx.fetch_full_schema("d") is None

    def test_a_store_failure_degrades_to_none_not_an_exception(self):
        backend = MagicMock()
        backend.get_store.side_effect = OSError("no pg")
        with patch.object(idx, "_get_backend", return_value=backend):
            assert idx.fetch_full_schema("d") is None

    def test_a_store_without_name_lookup_returns_none(self):
        """Pinecone has no list_tables — cosine is the only option there."""
        backend = MagicMock(spec=["get_store", "retrieve"])
        with patch.object(idx, "_get_backend", return_value=backend):
            assert idx.fetch_full_schema("d") is None


# ── the agent's preference order ──────────────────────────────────────────────

class TestSchemaPreference:

    def test_a_scoped_question_gets_the_complete_schema_not_top_k(self):
        a = _agent()
        with patch.object(dbm, "fetch_full_schema", return_value=_FULL) as full, \
             patch.object(dbm, "retrieve") as ret, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("who consumed least in LAM?", [0.1], top_k=5)
        full.assert_called_once()
        ret.assert_not_called()          # no cosine search at all
        assert len(out["tables"]) == 5

    def test_an_unscoped_question_falls_back_to_cosine(self):
        a = _agent(_db_name=None)
        with patch.object(dbm, "fetch_full_schema") as full, \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "prompt_context": "ctx"}) as ret, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            a._fetch_schema("q", [0.1], top_k=5)
        full.assert_not_called()
        assert ret.call_args[1]["top_k_tables"] == 5

    def test_an_empty_index_for_the_scope_falls_back_to_cosine(self):
        a = _agent()
        with patch.object(dbm, "fetch_full_schema", return_value=None), \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "prompt_context": "ctx"}) as ret, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        ret.assert_called_once()
        assert out["prompt_context"] == "ctx"

    def test_a_scope_over_the_token_budget_falls_back_to_cosine(self):
        """Primary tables are never dropped by expand_fk_tables, so an
        oversized scope must be caught here or the generator receives a
        context larger than max_schema_tokens."""
        huge = {**_FULL, "prompt_context": "word " * 20000}
        a = _agent(_max_schema_tokens=100)
        with patch.object(dbm, "fetch_full_schema", return_value=huge), \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "prompt_context": "ctx"}) as ret, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        ret.assert_called_once()
        assert out["prompt_context"] == "ctx"

    def test_the_preference_can_be_switched_off_for_an_ab(self):
        a = _agent(_full_schema=False)
        with patch.object(dbm, "fetch_full_schema") as full, \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "prompt_context": "ctx"}), \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            a._fetch_schema("q", [0.1], top_k=5)
        full.assert_not_called()

    def test_fk_relationships_are_still_assembled_for_a_full_schema(self):
        """The BIRD schema dump includes foreign keys; dropping the expansion
        step would lose the '### Foreign Key Relationships' block that tells
        the generator how to join."""
        a = _agent()
        with patch.object(dbm, "fetch_full_schema", return_value=_FULL), \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *_, **__: c) as exp:
            a._fetch_schema("q", [0.1], top_k=5)
        exp.assert_called_once()
        assert exp.call_args[0][1] == "debit_card_specializing"


# ── the fallback path uses the same strategy ──────────────────────────────────

def test_the_broaden_fallback_uses_the_same_schema_strategy():
    """Two retrieval sites drifting apart is how the UI/eval gap started."""
    import inspect
    src = inspect.getsource(OrchestrationAgent.fallback_handler)
    assert "_fetch_schema" in src
    assert "retrieve(" not in src
