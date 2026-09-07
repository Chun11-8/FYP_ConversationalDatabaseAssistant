"""
Tests for the unanswerable-question path and the two defects it exposed
(GH#75).

The reported failure: "Show me customers from the country Wakanda" against a
schema with no country column returned 100 rows of `segment='LAM' AND
currency='EUR'` customers, headed "Here are the customers from Wakanda:",
above a warning saying Wakanda does not exist — and the chips read `Cached`
and `0 LLM calls`, meaning the wrong answer had already been stored and was
being replayed.

Three separate defects:
  1. `_update_cache` had no risk gate, so a flagged answer was memorised.
  2. `format_response` never saw the risks, so the prose contradicted them.
  3. Once a question was classified READ there was no exit producing no SQL,
     so an unsatisfiable question forced the model to invent a filter.

Run from repo root:
    pytest backend/tests/test_unanswerable.py -v
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock


import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent, _prepend_caution


def _agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    for k, v in dict(value_index=None, _dialect="postgres").items():
        setattr(a, k, v)
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


class _Index:
    """resolve_spans stand-in: {span: matched value | None}. {} = couldn't check."""

    def __init__(self, mapping, tables=None):
        self.mapping = mapping
        self._tables = tables or {}

    def resolve_spans(self, text, tables):
        return self.mapping

    def _load_table(self, table):
        return self._tables.get(table, {})

    def nearest_values(self, span, tables, limit=6):
        """Mirrors ValueIndex.nearest_values (GH#97): stored values ranked by
        similarity to the span the user actually typed."""
        import difflib
        scored = []
        for t in tables:
            for values in self._tables.get(t, {}).values():
                for v in values:
                    scored.append((difflib.SequenceMatcher(
                        None, str(span).casefold(), str(v).casefold()).ratio(), v))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _s, v in scored:
            if v not in out:
                out.append(v)
            if len(out) >= limit:
                break
        return out


# ── Fix 1: never cache an answer the system flagged ───────────────────────────

class TestFlaggedAnswersAreNotCached:

    def _final(self, **over):
        f = {"validation": {"valid": True}, "best_sql": "SELECT 1",
             "execution_result": [{"x": 1}], "intent": "READ", "response": "ok",
             "cache_hit": False, "rows_fingerprint": "new",
             "cached_fingerprint": None, "verification": {"risk_score": 0}}
        f.update(over)
        return f

    def _agent(self):
        a = _agent(_model="gpt-4o-mini")
        a.semantic_cache = MagicMock()
        return a

    def test_a_clean_answer_is_still_cached(self):
        a = self._agent()
        a._update_cache("s", "q", self._final(), 1, 10)
        a.semantic_cache.save_pair.assert_called_once()

    def test_a_flagged_answer_is_not_cached(self):
        """The reported bug: the system detected its own error and then
        memorised it, so every later ask replayed it with zero LLM calls."""
        a = self._agent()
        a._update_cache("s", "q", self._final(
            verification={"risk_score": 3,
                          "risks": [{"code": "unknown_value", "severity": "high"}]}),
            1, 10)
        a.semantic_cache.save_pair.assert_not_called()

    def test_the_interaction_is_still_logged(self):
        """Not caching must not also lose the audit row — cost accounting and
        error analysis both read query_interactions."""
        a = self._agent()
        a._update_cache("s", "q", self._final(verification={"risk_score": 3}), 1, 10)
        a.semantic_cache.log_interaction.assert_called_once()

    def test_missing_verification_does_not_block_caching(self):
        """A turn from before risk checks, or with them disabled, must still
        cache — absence of evidence is not a flag."""
        a = self._agent()
        a._update_cache("s", "q", self._final(verification=None), 1, 10)
        a.semantic_cache.save_pair.assert_called_once()


# ── Fix 2: the prose must not contradict the warning ──────────────────────────

class TestPrependCaution:

    HIGH = {"risks": [{"code": "unknown_value", "severity": "high",
                       "message": "'Wakanda' does not appear anywhere in this database."}]}

    def test_a_high_severity_flag_leads_the_answer(self):
        out = _prepend_caution("Here are the customers from Wakanda:", self.HIGH)
        assert out.startswith("⚠️")
        assert "Wakanda' does not appear" in out
        assert "Here are the customers" in out

    def test_medium_and_low_flags_do_not_add_a_caveat(self):
        """A caveat on every zero-row turn would appear on roughly a fifth of
        all answers and be tuned out — the exact failure the risk feature
        exists to avoid. Those still render as structured flags in the UI."""
        for sev in ("medium", "low"):
            v = {"risks": [{"code": "zero_rows", "severity": sev, "message": "m"}]}
            assert _prepend_caution("answer", v) == "answer"

    def test_no_risks_leaves_the_answer_untouched(self):
        assert _prepend_caution("answer", {"risks": []}) == "answer"
        assert _prepend_caution("answer", None) == "answer"
        assert _prepend_caution("answer", {}) == "answer"

    def test_an_empty_response_is_left_alone(self):
        assert _prepend_caution("", self.HIGH) == ""

    def test_a_flag_without_a_message_still_cautions(self):
        v = {"risks": [{"code": "x", "severity": "high"}]}
        assert _prepend_caution("answer", v).startswith("⚠️")


# ── Fix 3: an unsatisfiable question exits without SQL ────────────────────────

class TestUnsatisfiableDetection:

    def test_a_value_absent_from_the_data_is_reported(self):
        a = _agent(value_index=_Index({"Wakanda": None}))
        assert a._unsatisfiable_spans("customers from Wakanda", ["customers"]) == ["Wakanda"]

    def test_a_value_present_in_the_data_is_not(self):
        a = _agent(value_index=_Index({"Brazil": "BRA"}))
        assert a._unsatisfiable_spans("customers from Brazil", ["customers"]) == []

    def test_untestable_spans_never_trigger_a_refusal(self):
        """Dates and sentence openers are absent from the data as VALUES but
        say nothing about answerability. Refusing on these would be far worse
        than answering."""
        a = _agent(value_index=_Index({"August": None, "Is": None, "2012/8/24": None}))
        assert a._unsatisfiable_spans("Is it August 2012/8/24?", ["t"]) == []

    def test_no_index_means_no_refusal(self):
        assert _agent(value_index=None)._unsatisfiable_spans("from Wakanda", ["t"]) == []

    def test_an_index_that_could_not_check_means_no_refusal(self):
        """resolve_spans returns {} on an outage. A database being down must
        never be reported to the user as "that value does not exist"."""
        a = _agent(value_index=_Index({}))
        assert a._unsatisfiable_spans("customers from Wakanda", ["customers"]) == []

    def test_an_index_failure_is_swallowed(self):
        class Boom:
            def resolve_spans(self, *a, **k):
                raise RuntimeError("down")
        assert _agent(value_index=Boom())._unsatisfiable_spans("q", ["t"]) == []

    def test_no_tables_means_no_refusal(self):
        a = _agent(value_index=_Index({"Wakanda": None}))
        assert a._unsatisfiable_spans("customers from Wakanda", []) == []


class TestRoutingToUnanswerable:

    def _agent(self):
        return _agent(_clarify=True)

    def test_an_unsatisfiable_question_skips_generation(self):
        assert self._agent().route_after_retrieval(
            {"schema_context": "T", "unsatisfiable": ["Wakanda"]}) == "handle_unanswerable"

    def test_an_ordinary_question_still_generates(self):
        assert self._agent().route_after_retrieval(
            {"schema_context": "T", "unsatisfiable": []}) == "generate_sql"

    def test_a_cache_hit_still_takes_precedence(self):
        """A stored, previously validated answer is evidence the question IS
        answerable — it must not be second-guessed by the span check."""
        assert self._agent().route_after_retrieval(
            {"cache_hit": True, "best_sql": "SELECT 1",
             "unsatisfiable": ["Wakanda"]}) == "validate_sql"

    def test_missing_schema_still_falls_back_first(self):
        assert self._agent().route_after_retrieval(
            {"schema_context": "", "fallback_level": 0,
             "unsatisfiable": ["Wakanda"]}) == "fallback_handler"


class TestHandleUnanswerableNode:

    def _agent(self):
        a = _agent(value_index=_Index({}, {"customers": {"country": ["BRA", "ESP"]}}))
        a.memory = MagicMock()
        a.memory.add.return_value = 42
        return a

    def _state(self, **over):
        s = {"session_id": "s", "user_query": "customers from Wakanda",
             "unsatisfiable": ["Wakanda"], "retrieved_tables": ["customers"]}
        s.update(over)
        return s

    def test_produces_no_sql(self):
        out = self._agent().handle_unanswerable(self._state())
        assert out["best_sql"] == ""

    def test_names_the_missing_value(self):
        out = self._agent().handle_unanswerable(self._state())
        assert "Wakanda" in out["response"]

    def test_suggests_real_values_so_the_user_can_correct_it(self):
        out = self._agent().handle_unanswerable(self._state())
        assert "BRA" in out["response"]

    def test_records_the_turn_as_invalid(self):
        a = self._agent()
        a.handle_unanswerable(self._state())
        assert a.memory.add.call_args.kwargs["valid"] is False

    def test_survives_a_value_index_that_cannot_suggest(self):
        a = _agent(value_index=None)
        a.memory = MagicMock()
        a.memory.add.return_value = 1
        out = a.handle_unanswerable(self._state())
        assert "Wakanda" in out["response"]      # still useful without suggestions


class TestRetrievedTablesArePopulated:
    """`_nearest_values` reads state["retrieved_tables"]. Nothing set that key
    when the helper was written, so the suggestion silently never appeared —
    the unit test above masked it by supplying the key by hand. This pins the
    producer side."""

    def test_retrieve_schema_declares_and_populates_the_key(self):
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent.retrieve_schema)
        assert '"retrieved_tables": primary + fk_extra' in src
        assert "retrieved_tables" in dbm.AgentState.__annotations__

    def test_the_initial_state_seeds_it(self):
        src = inspect.getsource(dbm.OrchestrationAgent.run)
        assert '"retrieved_tables": []' in src


class TestGraphWiring:

    def test_the_node_is_registered_and_terminal(self):
        """Guards the wiring itself: a node that exists but is unreachable, or
        one that falls through to generation, would silently restore the bug."""
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent._build_graph)
        assert 'g.add_node("handle_unanswerable"' in src
        assert '"handle_unanswerable": "handle_unanswerable"' in src
        assert 'g.add_edge("handle_unanswerable", END)' in src
