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
                    _max_schema_tokens=12000, _live_tier="moderate",
                    _auto_scope=True)
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


# ── automatic domain routing (GH#95) ──────────────────────────────────────────

def _ranking(*db_names):
    return [{"chunk_id": f"c{i}", "db_name": d, "table_name": f"t{i}",
             "text": "T", "distance": 0.1 * i}
            for i, d in enumerate(db_names)]


class TestInferScope:
    """The ranking already names its own domain — every chunk carries the
    db_id that owns its table, so routing costs nothing."""

    def test_the_reported_question_routes_to_student_club(self):
        # The retrieval ranking from the reported failure, verbatim.
        assert OrchestrationAgent._infer_scope(_ranking(
            "student_club", "student_club", "california_schools",
            "debit_card_specializing", "european_football_2")
        ) == "student_club"

    def test_two_agreeing_tables_beat_a_rank_one_distractor(self):
        assert OrchestrationAgent._infer_scope(_ranking(
            "card_games", "student_club", "student_club", "x", "y")
        ) == "student_club"

    def test_a_lone_chunk_never_commits_a_whole_domain(self):
        """Five domains, one chunk each: the ranking has no view, and guessing
        would exclude the right table from the context entirely."""
        assert OrchestrationAgent._infer_scope(
            _ranking("a", "b", "c", "d", "e")) is None

    def test_a_single_domain_install_always_routes(self):
        assert OrchestrationAgent._infer_scope(_ranking("a", "a", "a")) == "a"

    def test_empty_or_untagged_rankings_are_safe(self):
        assert OrchestrationAgent._infer_scope([]) is None
        assert OrchestrationAgent._infer_scope([{"table_name": "t"}]) is None


class TestRoutingInFetchSchema:

    def test_an_unscoped_question_is_routed_then_gets_the_full_schema(self):
        a = _agent(_db_name=None, _auto_scope=True)
        cosine = {"tables": _ranking("student_club", "student_club", "frpm_db"),
                  "prompt_context": "top-k"}
        with patch.object(dbm, "retrieve", return_value=cosine) as ret, \
             patch.object(dbm, "fetch_full_schema",
                          return_value={**_FULL, "prompt_context": "full"}) as full, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        ret.assert_called_once()                       # one cosine query
        full.assert_called_once_with("student_club", store="pgvector")
        assert out["prompt_context"] == "full"

    def test_no_clear_winner_keeps_the_cosine_result(self):
        a = _agent(_db_name=None, _auto_scope=True)
        cosine = {"tables": _ranking("a", "b", "c"), "prompt_context": "top-k"}
        with patch.object(dbm, "retrieve", return_value=cosine), \
             patch.object(dbm, "fetch_full_schema") as full, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        full.assert_not_called()
        assert out["prompt_context"] == "top-k"

    def test_a_configured_scope_is_not_overridden_by_the_ranking(self):
        """Explicit intent — config or the UI picker — always wins."""
        a = _agent(_db_name="superhero", _auto_scope=True)
        with patch.object(dbm, "fetch_full_schema", return_value=_FULL) as full, \
             patch.object(dbm, "retrieve") as ret, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            a._fetch_schema("q", [0.1], top_k=5)
        full.assert_called_once_with("superhero", store="pgvector")
        ret.assert_not_called()

    def test_routing_can_be_switched_off(self):
        a = _agent(_db_name=None, _auto_scope=False)
        cosine = {"tables": _ranking("a", "a"), "prompt_context": "top-k"}
        with patch.object(dbm, "retrieve", return_value=cosine), \
             patch.object(dbm, "fetch_full_schema") as full, \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        full.assert_not_called()
        assert out["prompt_context"] == "top-k"

    def test_an_over_budget_routed_domain_falls_back_to_its_cosine_result(self):
        a = _agent(_db_name=None, _auto_scope=True, _max_schema_tokens=50)
        cosine = {"tables": _ranking("a", "a"), "prompt_context": "top-k"}
        with patch.object(dbm, "retrieve", return_value=cosine), \
             patch.object(dbm, "fetch_full_schema",
                          return_value={**_FULL, "prompt_context": "x" * 5000}), \
             patch.object(dbm, "expand_fk_tables", side_effect=lambda c, *_, **__: c):
            out = a._fetch_schema("q", [0.1], top_k=5)
        assert out["prompt_context"] == "top-k"
