"""
Tests for the GH#57 cost-routing changes:
  * cache lookup happens BEFORE the RAG stage, with a by-name schema fetch
  * run() resolves a repeat question pre-graph, so the turn enters at
    validate_sql and skips intent classification + condensation
  * a stored answer is reused only when the re-executed rows are identical

All offline: no Postgres, no OpenAI. The agent is built with __new__ and given
just the attributes the code under test reads, matching the existing
test_hardening_latency.py / test_skeleton_fewshots.py fixtures.

Run from repo root:
    pytest backend/tests/test_fast_path.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent, rows_digest
from backend.agents.sql_validator_agent import referenced_tables


def _agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_cache_shortcut=True, _cache_hit_threshold=0.03,
                    _fast_path=True, _answer_reuse=True,
                    _dialect="postgres", _db_name="FYP",
                    _fk_max_hops=2, _fk_max_expansion=5, _max_schema_tokens=4000,
                    # These tests are about the cache path, not GH#85's
                    # schema strategy — keep them on cosine retrieval so the
                    # patched dbm.retrieve is what runs.
                    _full_schema=False, _top_k_tables=5,
                    _fewshot_skeleton_blend=False)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    a.semantic_cache = MagicMock()
    a.semantic_cache.retrieve_few_shots.return_value = []
    a.semantic_cache.best_match.return_value = None
    return a


HIT = {"user_query": "how many players?", "sql": "SELECT COUNT(*) FROM players",
       "confidence": 1.0, "distance": 0.01,
       "rows_fingerprint": "abc", "response": "There are 12 players."}


@pytest.fixture(autouse=True)
def _clean_tls():
    yield
    for attr in ("embed_future", "stage_cb", "llm", "intent_llm",
                 "sql_agent", "linker", "optimizer"):
        if hasattr(dbm._tls, attr):
            delattr(dbm._tls, attr)


# ── referenced_tables ────────────────────────────────────────────────────────

class TestReferencedTables:

    def test_extracts_plain_table(self):
        assert referenced_tables("SELECT a FROM players") == ["players"]

    def test_extracts_joined_tables(self):
        sql = "SELECT p.a FROM players p JOIN teams t ON p.tid = t.id"
        assert referenced_tables(sql) == ["players", "teams"]

    def test_cte_names_are_not_real_tables(self):
        sql = ("WITH recent AS (SELECT * FROM matches) "
               "SELECT * FROM recent JOIN teams ON recent.tid = teams.id")
        assert referenced_tables(sql) == ["matches", "teams"]

    def test_unparseable_returns_empty(self):
        assert referenced_tables("SELECT FROM ((( bad") == []


# ── _cache_lookup ────────────────────────────────────────────────────────────

class TestCacheLookup:

    def test_hit_within_threshold(self):
        a = _agent()
        a.semantic_cache.best_match.return_value = HIT
        assert a._cache_lookup([0.1]) == HIT

    def test_miss_beyond_threshold(self):
        a = _agent()
        a.semantic_cache.best_match.return_value = {**HIT, "distance": 0.9}
        assert a._cache_lookup([0.1]) is None

    def test_disabled_shortcut_never_looks_up(self):
        a = _agent(_cache_shortcut=False)
        assert a._cache_lookup([0.1]) is None
        a.semantic_cache.best_match.assert_not_called()

    def test_no_embedding_returns_none(self):
        a = _agent()
        assert a._cache_lookup(None) is None

    def test_cache_failure_degrades_to_none(self):
        a = _agent()
        a.semantic_cache.best_match.side_effect = RuntimeError("db down")
        assert a._cache_lookup([0.1]) is None

    def test_non_numeric_distance_is_rejected(self):
        a = _agent()
        a.semantic_cache.best_match.return_value = {**HIT, "distance": "close"}
        assert a._cache_lookup([0.1]) is None


# ── _schema_for_sql ──────────────────────────────────────────────────────────

class TestSchemaForSql:

    def test_fetches_only_the_tables_the_sql_reads(self):
        a = _agent()
        with patch.object(dbm, "fetch_table_chunks",
                          return_value=[{"table_name": "players",
                                         "content": "players(id, name)"}]) as fetch:
            ctx = a._schema_for_sql("SELECT id FROM players")
        assert ctx == "players(id, name)"
        assert fetch.call_args[0][0] == ["players"]

    def test_partial_fetch_is_rejected(self):
        """A half-fetched schema would fail strict-semantic validation on a
        perfectly good cached query — better to fall back to full retrieval."""
        a = _agent()
        with patch.object(dbm, "fetch_table_chunks",
                          return_value=[{"table_name": "players", "content": "x"}]):
            assert a._schema_for_sql(
                "SELECT p.a FROM players p JOIN teams t ON p.tid = t.id") == ""

    def test_unparseable_sql_returns_empty(self):
        a = _agent()
        assert a._schema_for_sql("not sql (((") == ""


# ── retrieve_schema: cache checked before the RAG stage ──────────────────────

class TestCacheBeforeRetrieval:

    def _state(self):
        return {"user_query": "how many players?", "conversation_history": [],
                "query_embedding": [0.1] * 3, "fallback_level": 0}

    def test_hit_skips_retrieval_entirely(self):
        a = _agent()
        a.semantic_cache.best_match.return_value = HIT
        with patch.object(dbm, "fetch_table_chunks",
                          return_value=[{"table_name": "players", "content": "p"}]), \
             patch.object(dbm, "retrieve") as ret:
            out = a.retrieve_schema(self._state())
        ret.assert_not_called()                      # no embedding search
        a.semantic_cache.retrieve_few_shots.assert_not_called()
        assert out["cache_hit"] is True
        assert out["best_sql"] == HIT["sql"]
        assert out["cached_response"] == HIT["response"]

    def test_hit_still_applies_when_by_name_fetch_comes_back_empty(self):
        """A hit must never be lost just because the cheap path didn't work."""
        a = _agent()
        a.semantic_cache.best_match.return_value = HIT
        with patch.object(dbm, "fetch_table_chunks", return_value=[]), \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "fk_tables": [],
                                        "prompt_context": "full schema"}), \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *a_, **k: c):
            out = a.retrieve_schema(self._state())
        assert out["cache_hit"] is True
        assert out["schema_context"] == "full schema"   # fell back to full RAG

    def test_miss_takes_the_normal_path(self):
        a = _agent()
        with patch.object(dbm, "fetch_table_chunks") as fetch, \
             patch.object(dbm, "retrieve",
                          return_value={"tables": [], "fk_tables": [],
                                        "prompt_context": "full schema"}), \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *a_, **k: c):
            out = a.retrieve_schema(self._state())
        fetch.assert_not_called()
        assert not out.get("cache_hit")
        assert out["schema_context"] == "full schema"

    def test_state_embedding_is_reused_for_retrieval(self):
        a = _agent()
        with patch.object(dbm, "retrieve",
                          return_value={"tables": [], "fk_tables": [],
                                        "prompt_context": ""}) as ret, \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *a_, **k: c):
            a.retrieve_schema(self._state())
        assert ret.call_args.kwargs["embedding"] == [0.1] * 3
        a.semantic_cache.embed.assert_not_called()

    def test_condensed_followup_embeds_its_own_text(self):
        a = _agent()
        a.semantic_cache.embed.return_value = [0.9] * 3
        state = {**self._state(), "standalone_query": "how many players in 2013?"}
        with patch.object(dbm, "retrieve",
                          return_value={"tables": [], "fk_tables": [],
                                        "prompt_context": ""}), \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *a_, **k: c):
            out = a.retrieve_schema(state)
        a.semantic_cache.embed.assert_called_once_with("how many players in 2013?")
        assert out["query_embedding"] == [0.9] * 3


# ── _probe_fast_path + route_entry ───────────────────────────────────────────

class TestFastPathState:

    def test_hit_returns_prefilled_state(self):
        a = _agent()
        with patch.object(dbm, "fetch_table_chunks",
                          return_value=[{"table_name": "players", "content": "p"}]):
            fast = a._fast_path_state(HIT)
        assert fast["best_sql"] == HIT["sql"]
        assert fast["cached_fingerprint"] == "abc"

    def test_by_name_fetch_failure_falls_back_to_the_graph(self):
        a = _agent()
        with patch.object(dbm, "fetch_table_chunks", return_value=[]):
            assert a._fast_path_state(HIT) is None

    def test_anaphoric_followup_is_never_probed(self):
        """'what about 2013?' means nothing until condensed against history."""
        a = _agent()
        history = [{"role": "user", "content": "how many players?"}]
        assert a._looks_anaphoric("what about 2013?", history) is True

    def test_first_turn_is_never_treated_as_anaphoric(self):
        assert _agent()._looks_anaphoric("top 5?", []) is False


class TestProbedOrLookup:

    def test_reuses_the_pre_graph_probe_instead_of_searching_again(self):
        a = _agent()
        state = {"user_query": "q", "cache_probed": True, "cache_pair": HIT}
        assert a._probed_or_lookup(state, "q", [0.1]) == HIT
        a.semantic_cache.best_match.assert_not_called()

    def test_reuses_a_probed_miss(self):
        a = _agent()
        state = {"user_query": "q", "cache_probed": True, "cache_pair": None}
        assert a._probed_or_lookup(state, "q", [0.1]) is None
        a.semantic_cache.best_match.assert_not_called()

    def test_condensed_query_gets_its_own_lookup(self):
        # The condensed text differs from user_query, so the pre-graph probe
        # (done against the RAW follow-up) must not be reused — a fresh
        # best_match runs instead. Text kept literal-compatible with HIT so
        # this stays a test of the re-lookup, not of the GH#63 literal guard
        # (which has its own tests in TestLiteralGuard).
        a = _agent()
        a.semantic_cache.best_match.return_value = HIT
        state = {"user_query": "q", "cache_probed": True, "cache_pair": None}
        assert a._probed_or_lookup(state, "how many players?", [0.9]) == HIT
        a.semantic_cache.best_match.assert_called_once()

    def test_unprobed_state_looks_up(self):
        a = _agent()
        a.semantic_cache.best_match.return_value = HIT
        assert a._probed_or_lookup({"user_query": "q"}, "q", [0.1]) == HIT

    def test_rejected_cache_is_never_handed_back(self):
        a = _agent()
        state = {"user_query": "q", "cache_probed": True, "cache_pair": HIT,
                 "cache_rejected": True}
        assert a._probed_or_lookup(state, "q", [0.1]) is None


class TestLiteralGuard:
    """
    GH#63 — the reported multi-turn bug. Turn 1 asks about 2013; the follow-up
    'what about 2014?' condenses to the same sentence with one digit changed,
    which sits inside the 0.03 hit threshold. Without the guard the cached 2013
    SQL is returned and the assistant replays the previous answer forever.
    """

    Q2013 = "How much did customer 6 consume in total between August and November 2013?"
    Q2014 = "How much did customer 6 consume in total between August and November 2014?"

    def _agent_hitting(self, cached_question: str, distance: float = 0.01):
        a = _agent()
        a.semantic_cache.best_match.return_value = {
            "user_query": cached_question, "sql": "SELECT 1", "confidence": 1.0,
            "distance": distance, "rows_fingerprint": "f", "response": "old answer",
        }
        return a

    def test_the_reported_bug_year_change_is_not_a_cache_hit(self):
        a = self._agent_hitting(self.Q2013)
        assert a._cache_lookup([0.1], self.Q2014) is None

    def test_the_same_question_still_hits(self):
        """The GH#57 cost saving must survive the fix."""
        a = self._agent_hitting(self.Q2013)
        assert a._cache_lookup([0.1], self.Q2013) is not None

    def test_rewording_without_changing_values_still_hits(self):
        a = self._agent_hitting("How many players are there?")
        assert a._cache_lookup([0.1], "How many players do we have?") is not None

    @pytest.mark.parametrize("asked", [
        "How much did customer 7 consume in total between August and November 2013?",
        "How much did customer 6 consume in total between August and September 2013?",
    ])
    def test_other_changed_literals_are_also_rejected(self, asked):
        """Not year-specific: entity ids and month names matter just as much."""
        a = self._agent_hitting(self.Q2013)
        assert a._cache_lookup([0.1], asked) is None

    def test_quoted_string_change_is_rejected(self):
        a = self._agent_hitting("players from 'Brazil'")
        assert a._cache_lookup([0.1], "players from 'Spain'") is None

    def test_month_abbreviation_is_not_a_difference(self):
        a = self._agent_hitting("revenue in Aug 2013")
        assert a._cache_lookup([0.1], "revenue in August 2013") is not None

    def test_distance_still_gates_before_the_literal_check(self):
        a = self._agent_hitting(self.Q2013, distance=0.9)
        assert a._cache_lookup([0.1], self.Q2013) is None

    def test_omitting_query_text_keeps_the_old_distance_only_behaviour(self):
        a = self._agent_hitting(self.Q2013)
        assert a._cache_lookup([0.1]) is not None

    def test_probed_or_lookup_applies_the_guard_to_a_condensed_followup(self):
        """End-to-end through the node's own helper, not just _cache_lookup."""
        a = self._agent_hitting(self.Q2013)
        state = {"user_query": "what about 2014?", "cache_probed": True,
                 "cache_pair": None}
        assert a._probed_or_lookup(state, self.Q2014, [0.9]) is None


class TestStaleCacheUnwind:

    def test_validate_marks_a_stale_cached_sql(self):
        a = _agent()
        a.validator = MagicMock()
        a.validator.validate.return_value = {"valid": False, "error": "no such table"}
        out = a.validate_sql({"best_sql": "SELECT 1 FROM gone",
                              "schema_context": "x", "cache_hit": True})
        assert out["cache_rejected"] is True

    def test_validate_does_not_mark_a_normal_failure(self):
        a = _agent()
        a.validator = MagicMock()
        a.validator.validate.return_value = {"valid": False}
        out = a.validate_sql({"best_sql": "SELECT 1", "schema_context": "x",
                              "cache_hit": False})
        assert "cache_rejected" not in out

    def test_stale_hit_routes_back_through_retrieval_not_linking(self):
        """Regenerating needs the FULL schema the cheap by-name fetch skipped."""
        a = _agent()
        assert a.route_after_validation(
            {"validation": {"valid": False}, "cache_hit": True}) == "retrieve_schema"

    def test_reentry_drops_the_hit_so_routing_cannot_loop(self):
        a = _agent()
        with patch.object(dbm, "retrieve",
                          return_value={"tables": [], "fk_tables": [],
                                        "prompt_context": "full schema"}), \
             patch.object(dbm, "expand_fk_tables",
                          side_effect=lambda c, *a_, **k: c):
            out = a.retrieve_schema(
                {"user_query": "q", "conversation_history": [],
                 "query_embedding": [0.1], "fallback_level": 0,
                 "cache_hit": True, "best_sql": "SELECT 1 FROM gone",
                 "cache_rejected": True})
        assert out["cache_hit"] is False
        assert out["schema_context"] == "full schema"


class TestRouteEntry:

    def test_prefilled_cache_hit_enters_at_validation(self):
        a = _agent()
        assert a.route_entry({"cache_hit": True, "best_sql": "SELECT 1"}) == "validate_sql"

    def test_normal_turn_enters_at_classification(self):
        a = _agent()
        assert a.route_entry({"cache_hit": False, "best_sql": ""}) == "classify_intent"

    def test_cache_hit_without_sql_still_classifies(self):
        a = _agent()
        assert a.route_entry({"cache_hit": True, "best_sql": ""}) == "classify_intent"


# ── row fingerprinting + answer reuse ────────────────────────────────────────

class TestRowsDigest:

    def test_same_rows_same_digest(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert rows_digest(rows) == rows_digest([dict(r) for r in rows])

    def test_row_order_does_not_matter(self):
        a = [{"a": 1}, {"a": 2}]
        assert rows_digest(a) == rows_digest(list(reversed(a)))

    def test_changed_value_changes_digest(self):
        assert rows_digest([{"a": 1}]) != rows_digest([{"a": 2}])

    def test_none_and_empty_string_are_equivalent_as_in_execution_voting(self):
        assert rows_digest([{"a": None}]) == rows_digest([{"a": ""}])

    def test_non_list_returns_none(self):
        assert rows_digest(None) is None
        assert rows_digest("rows") is None

    def test_empty_result_set_has_a_digest(self):
        assert isinstance(rows_digest([]), str)


class TestAnswerReuse:

    def _state(self, **over):
        s = {"cached_response": "There are 12 players.",
             "cached_fingerprint": "abc"}
        s.update(over)
        return s

    def test_reuses_when_fingerprint_matches(self):
        assert _agent()._can_reuse_answer(self._state(), "abc") is True

    def test_reformats_when_rows_changed(self):
        assert _agent()._can_reuse_answer(self._state(), "def") is False

    def test_no_reuse_without_a_stored_answer(self):
        a = _agent()
        assert a._can_reuse_answer(self._state(cached_response=None), "abc") is False

    def test_no_reuse_without_a_stored_fingerprint(self):
        a = _agent()
        assert a._can_reuse_answer(self._state(cached_fingerprint=None), "abc") is False

    def test_disabled_flag_always_reformats(self):
        a = _agent(_answer_reuse=False)
        assert a._can_reuse_answer(self._state(), "abc") is False

    def test_format_response_skips_the_llm_on_reuse(self):
        a = _agent()
        a.memory = MagicMock()
        a.memory.add.return_value = 7
        optimizer = MagicMock()
        a._get_optimizer = MagicMock(return_value=optimizer)
        rows = [{"n": 12}]
        state = {"user_query": "how many players?", "best_sql": "SELECT COUNT(*) FROM p",
                 "execution_result": rows, "validation": {"valid": True},
                 "session_id": "s", "cached_response": "There are 12 players.",
                 "cached_fingerprint": rows_digest(rows)}
        out = a.format_response(state)
        optimizer.format.assert_not_called()
        assert out["response"] == "There are 12 players."

    def test_format_response_calls_the_llm_when_rows_moved(self):
        a = _agent()
        a.memory = MagicMock()
        a.memory.add.return_value = 7
        optimizer = MagicMock()
        optimizer.format.return_value = ("There are 13 players.", None)
        a._get_optimizer = MagicMock(return_value=optimizer)
        state = {"user_query": "how many players?", "best_sql": "SELECT COUNT(*) FROM p",
                 "execution_result": [{"n": 13}], "validation": {"valid": True},
                 "session_id": "s", "cached_response": "There are 12 players.",
                 "cached_fingerprint": "stale-hash"}
        out = a.format_response(state)
        optimizer.format.assert_called_once()
        assert out["response"] == "There are 13 players."
        assert out["rows_fingerprint"] == rows_digest([{"n": 13}])


class TestCacheWriteBack:

    def _final(self, **over):
        f = {"validation": {"valid": True}, "best_sql": "SELECT 1",
             "execution_result": [{"x": 1}], "intent": "READ", "response": "ok",
             "cache_hit": False, "rows_fingerprint": "new", "cached_fingerprint": None}
        f.update(over)
        return f

    def test_stores_fingerprint_and_response(self):
        a = _agent()
        a._update_cache("s", "q", self._final(), 1, 10)
        kwargs = a.semantic_cache.save_pair.call_args.kwargs
        assert kwargs["rows_fingerprint"] == "new"
        assert kwargs["response"] == "ok"

    def test_stores_the_condensed_question_not_the_raw_followup(self):
        """GH#63: the embedding saved alongside is the CONDENSED text's, so
        storing 'what about 2014?' here left text and vector describing
        different questions — and made the pair useless as a few-shot."""
        a = _agent()
        a._update_cache("s", "what about 2014?",
                        self._final(standalone_query="revenue in 2014"), 1, 10)
        assert a.semantic_cache.save_pair.call_args.kwargs["user_query"] \
            == "revenue in 2014"

    def test_the_audit_log_still_records_what_the_user_actually_typed(self):
        a = _agent()
        a._update_cache("s", "what about 2014?",
                        self._final(standalone_query="revenue in 2014"), 1, 10)
        assert a.semantic_cache.log_interaction.call_args.kwargs["user_query"] \
            == "what about 2014?"

    def test_falls_back_to_the_raw_query_when_nothing_was_condensed(self):
        a = _agent()
        a._update_cache("s", "revenue in 2014", self._final(), 1, 10)
        assert a.semantic_cache.save_pair.call_args.kwargs["user_query"] \
            == "revenue in 2014"

    def test_refreshes_a_reused_pair_when_the_rows_moved(self):
        a = _agent()
        a._update_cache("s", "q", self._final(cache_hit=True,
                                             cached_fingerprint="old"), 1, 10)
        a.semantic_cache.save_pair.assert_called_once()

    def test_does_not_rewrite_an_unchanged_reused_pair(self):
        a = _agent()
        a._update_cache("s", "q", self._final(cache_hit=True,
                                              cached_fingerprint="new"), 1, 10)
        a.semantic_cache.save_pair.assert_not_called()

    def test_logs_usage_columns(self):
        a = _agent(_model="gpt-4o-mini")
        a._update_cache("s", "q", self._final(), 1, 10,
                        usage={"input_tokens": 100, "output_tokens": 20,
                               "est_cost_usd": 0.001})
        kwargs = a.semantic_cache.log_interaction.call_args.kwargs
        assert kwargs["tokens_in"] == 100
        assert kwargs["cost_usd"] == 0.001
        assert kwargs["model"] == "gpt-4o-mini"

    def test_logs_the_per_request_override_model_not_the_static_default(self):
        # Self-review finding: query_interactions.model must match whatever
        # model the tokens/cost in that same row were actually billed
        # against, not the agent's static config default.
        a = _agent(_model="gpt-4o-mini")
        a._update_cache("s", "q", self._final(), 1, 10,
                        usage={"input_tokens": 50, "output_tokens": 5,
                               "est_cost_usd": 0.002},
                        model="gpt-4o")
        kwargs = a.semantic_cache.log_interaction.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"

    def test_model_falls_back_to_the_default_when_not_passed(self):
        a = _agent(_model="gpt-4o-mini")
        a._update_cache("s", "q", self._final(), 1, 10)
        kwargs = a.semantic_cache.log_interaction.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
