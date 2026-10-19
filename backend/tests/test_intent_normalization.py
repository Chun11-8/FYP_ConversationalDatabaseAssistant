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

    def test_reasoning_style_reply_takes_the_final_line_as_the_verdict(self):
        raw = ("The user is asking to retrieve data about vice president funds.\n"
               "This requires checking the schema for a matching table.\n\n"
               "READ")
        assert _normalize_intent(raw) == "READ"

    def test_final_line_verdict_ignores_trailing_punctuation(self):
        raw = "Some reasoning about the request.\n\nREAD."
        assert _normalize_intent(raw) == "READ"

    def test_exact_match_short_circuits_before_any_scan(self):
        # A compliant reply never needs the fallback scan at all.
        assert _normalize_intent("GENERAL") == "GENERAL"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_none_and_empty_do_not_crash_the_fast_paths(self, raw):
        assert _normalize_intent(raw) == "AMBIGUOUS"

    def test_most_restrictive_label_wins(self):
        """A reply naming several labels must resolve to the safest reading."""
        raw = "The user wants to remove rows, so WRITE_BLOCKED, not READ."
        assert _normalize_intent(raw) == "WRITE_BLOCKED"

    def test_actionable_label_beats_an_incidental_ambiguous_mention(self):
        # A verbose/reasoning-style reply that hedges before concluding must
        # not be derailed by the word "ambiguous" appearing along the way —
        # this was the root cause of a real bug: a non-gpt-4o-mini model kept
        # getting classified AMBIGUOUS on every turn because its raw reply
        # discussed the possibility of ambiguity before landing on READ.
        raw = ("This could be somewhat ambiguous without more context, but "
               "since it's asking to retrieve a value, I'll classify this as READ.")
        assert _normalize_intent(raw) == "READ"

    def test_write_blocked_still_wins_over_an_incidental_read_mention(self):
        # Safety-critical: WRITE_BLOCKED must still be checked first, even
        # though READ now outranks AMBIGUOUS.
        raw = "The user wants to remove rows (not just read them), so WRITE_BLOCKED."
        assert _normalize_intent(raw) == "WRITE_BLOCKED"

    def test_bare_ambiguous_with_no_actionable_label_stays_ambiguous(self):
        assert _normalize_intent("This is AMBIGUOUS, no clear target table.") == "AMBIGUOUS"

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

    def test_llm_failure_sets_a_real_error_not_just_ambiguous(self):
        """A call failure (bad model, auth, network) must be distinguishable
        from a genuinely unclear question — not silently masked as one."""
        a = self._agent("READ")
        a._get_intent_llm.return_value.invoke.side_effect = RuntimeError(
            "model_not_found: gpt-5o-mini")
        out = a.classify_intent(self._state())
        assert out["intent"] == "AMBIGUOUS"
        assert out["error"] is not None
        assert "gpt-5o-mini" in out["error"]

    def test_successful_unmatched_reply_sets_no_error(self):
        """A reply that came back but didn't parse to a label is a genuine
        AMBIGUOUS result, not a failure — must not carry an error."""
        a = self._agent("I'm not sure what you mean.")
        out = a.classify_intent(self._state())
        assert out["intent"] == "AMBIGUOUS"
        assert out["error"] is None

    def test_dml_keyword_override_clears_a_pending_error(self):
        a = self._agent("READ")
        a._get_intent_llm.return_value.invoke.side_effect = RuntimeError("boom")
        out = a.classify_intent(self._state("DELETE FROM players"))
        assert out["intent"] == "WRITE_BLOCKED"
        assert out["error"] is None

    def test_short_reply_to_own_clarification_short_circuits_to_read(self):
        """The reported bug: a one-word answer to our own clarifying question
        must not go back through the LLM (and risk AMBIGUOUS again) — it
        should be forced to READ so condense_query can merge it with history."""
        a = self._agent("this should never be called")
        history = [
            {"role": "user", "content": "What is the amount of the funds "
                                         "that the Vice President received?"},
            {"role": "assistant", "content": "I want to make sure I query the "
                                              "right data — your question is a "
                                              "little ambiguous. Could you name "
                                              "the table, entity, or metric you "
                                              "mean? Are you asking about data "
                                              "in schools, satscores, frpm?"},
        ]
        state = {"user_query": "frpm", "conversation_history": history, "session_id": "s"}
        out = a.classify_intent(state)
        assert out["intent"] == "READ"
        assert out["error"] is None
        a._get_intent_llm.return_value.invoke.assert_not_called()

    def test_long_reply_to_own_clarification_still_goes_through_the_llm(self):
        """Only SHORT replies are treated as continuing the clarification —
        a full new question after one should be classified normally."""
        a = self._agent("READ")
        history = [
            {"role": "user", "content": "vague question"},
            {"role": "assistant", "content": "I want to make sure I query the "
                                              "right data — your question is a "
                                              "little ambiguous. Could you name "
                                              "the table, entity, or metric you mean?"},
        ]
        state = {"user_query": "Actually, show me all players from Brazil please",
                 "conversation_history": history, "session_id": "s"}
        out = a.classify_intent(state)
        assert out["intent"] == "READ"
        a._get_intent_llm.return_value.invoke.assert_called_once()

    def test_short_reply_when_last_turn_was_not_a_clarification_goes_through_llm(self):
        """The short-circuit is specific to OUR clarification template — a
        short reply to a normal assistant answer must classify normally."""
        a = self._agent("AMBIGUOUS")
        history = [
            {"role": "user", "content": "how many players are there?"},
            {"role": "assistant", "content": "There are 500 players."},
        ]
        state = {"user_query": "frpm", "conversation_history": history, "session_id": "s"}
        out = a.classify_intent(state)
        assert out["intent"] == "AMBIGUOUS"
        a._get_intent_llm.return_value.invoke.assert_called_once()

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
