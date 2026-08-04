"""
Tests for intent normalization (_normalize_intent) and the routing it protects.

The classifier is asked for a bare label, but a chatty reply used to become the
intent verbatim. Because every router compares by equality, that silently fell
through to the READ path — and the UI rendered the whole sentence as a badge.

Run from repo root:
    pytest backend/tests/test_intent_normalization.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent, _normalize_intent


class TestNormalizeIntent:

    @pytest.mark.parametrize("raw", ["READ", "read", "  Read  ", "read\n"])
    def test_bare_label_any_case(self, raw):
        assert _normalize_intent(raw) == "READ"

    @pytest.mark.parametrize("label", ["GENERAL", "AMBIGUOUS", "WRITE_BLOCKED"])
    def test_each_canonical_label_round_trips(self, label):
        assert _normalize_intent(label) == label

    def test_the_verbose_reply_from_the_bug_report(self):
        raw = ("Classification: READ. The user is requesting a query to retrieve "
               "member full names and contact numbers based on a condition "
               "involving average spending on expenses, which is a data "
               "retrieval task.")
        assert _normalize_intent(raw) == "READ"

    @pytest.mark.parametrize("raw", ["WRITE BLOCKED", "write-blocked", "Write_Blocked"])
    def test_spacing_variants_of_write_blocked(self, raw):
        assert _normalize_intent(raw) == "WRITE_BLOCKED"

    def test_most_restrictive_label_wins(self):
        """A reply naming several labels must resolve to the safest reading."""
        raw = "The user wants to remove rows, so WRITE_BLOCKED, not READ."
        assert _normalize_intent(raw) == "WRITE_BLOCKED"

    def test_ambiguous_beats_general_and_read(self):
        assert _normalize_intent("This is AMBIGUOUS or possibly READ") == "AMBIGUOUS"

    @pytest.mark.parametrize("raw", ["", "   ", "hello there", "42", None])
    def test_unrecognized_falls_back_to_ambiguous(self, raw):
        # Safety-first: better to ask than to guess a route.
        assert _normalize_intent(raw) == "AMBIGUOUS"

    def test_label_inside_a_longer_word_does_not_match(self):
        # "ALREADY" contains "READ" but must not classify as READ.
        assert _normalize_intent("The answer is ALREADY computed") == "AMBIGUOUS"

    def test_always_returns_a_routable_token(self):
        for raw in ["READ", "nonsense", "", "WRITE BLOCKED", "general chat"]:
            assert _normalize_intent(raw) in {
                "READ", "GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"}


class TestClassifyIntentNode:

    def _agent(self, reply: str) -> OrchestrationAgent:
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a.INTENT_PROMPT = "classify"
        a._clarify = True
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=reply)
        a._get_intent_llm = MagicMock(return_value=llm)
        a._format_history = MagicMock(return_value="")
        return a

    def _state(self, query="show me the players"):
        return {"user_query": query, "conversation_history": [], "session_id": "s"}

    def test_verbose_reply_is_stored_normalized(self):
        a = self._agent("Classification: READ — the user wants data.")
        assert a.classify_intent(self._state())["intent"] == "READ"

    def test_dml_keyword_override_still_wins(self):
        """The hard safety net must survive normalization."""
        a = self._agent("READ")
        out = a.classify_intent(self._state("DELETE FROM players"))
        assert out["intent"] == "WRITE_BLOCKED"

    def test_llm_failure_defaults_to_ambiguous(self):
        a = self._agent("READ")
        a._get_intent_llm.return_value.invoke.side_effect = RuntimeError("boom")
        assert a.classify_intent(self._state())["intent"] == "AMBIGUOUS"

    def test_stored_intent_is_always_routable(self):
        a = self._agent("some unparseable nonsense")
        intent = a.classify_intent(self._state())["intent"]
        assert intent in {"READ", "GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"}


class TestRoutingOnNormalizedIntent:
    """Every canonical token must reach its intended node — the regression that
    an unnormalized string caused was a silent fallthrough to the READ path."""

    def _agent(self, clarify=True):
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a._clarify = clarify
        return a

    @pytest.mark.parametrize("intent,expected", [
        ("READ", "retrieve_schema"),
        ("GENERAL", "handle_general"),
        ("WRITE_BLOCKED", "handle_write_blocked"),
        ("AMBIGUOUS", "ask_clarification"),
    ])
    def test_each_token_routes_correctly(self, intent, expected):
        assert self._agent().route_by_intent({"intent": intent}) == expected

    def test_ambiguous_falls_through_to_read_when_clarify_disabled(self):
        assert self._agent(clarify=False).route_by_intent(
            {"intent": "AMBIGUOUS"}) == "retrieve_schema"

    def test_verbose_write_blocked_would_have_leaked_to_read(self):
        """Documents the bug: the raw string does NOT route, the normalized one does."""
        raw = "Classification: WRITE_BLOCKED - the user wants to remove members."
        a = self._agent()
        assert a.route_by_intent({"intent": raw}) == "retrieve_schema"      # the bug
        assert a.route_by_intent(
            {"intent": _normalize_intent(raw)}) == "handle_write_blocked"   # the fix
