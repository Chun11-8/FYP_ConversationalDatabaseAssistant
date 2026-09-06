"""
Tests for GH#89 — a topic switch drops conversation history everywhere, not
just in the condenser.

GH#87 stopped the condenser MERGING an unrelated question into the previous
turn's constraints. History still reached classify_intent, the planner and
generation, where it degrades a turn silently — no refusal, just a worse plan.

All offline: no LLM, no database.

Run from repo root:
    pytest backend/tests/test_topic_switch_history.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent as A

HISTORY = [{"role": "user", "content": "consumption in SME for 2013"},
           {"role": "assistant", "content": "The average was 1234."},
           {"role": "user", "content": "and for 2014?"},
           {"role": "assistant", "content": "It was 2345."},
           {"role": "user", "content": "which segment spends most?"},
           {"role": "assistant", "content": "LAM."}]


def _agent(**attrs) -> A:
    a = A.__new__(A)
    defaults = dict(_dialect="postgres", _topic_switch_detection=True)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


# ── detection ─────────────────────────────────────────────────────────────────

class TestDetectTopicSwitch:

    PREV = "SELECT AVG(consumption) FROM yearmonth JOIN customers ON 1=1"

    def test_disjoint_tables_are_a_switch(self):
        assert _agent()._detect_topic_switch(
            {"previous_sql": self.PREV}, ["member", "event"]) is True

    def test_overlapping_tables_are_not(self):
        assert _agent()._detect_topic_switch(
            {"previous_sql": self.PREV}, ["yearmonth", "gasstations"]) is False

    def test_matching_is_case_insensitive(self):
        assert _agent()._detect_topic_switch(
            {"previous_sql": "SELECT x FROM Member"}, ["member"]) is False

    def test_a_first_turn_is_never_a_switch(self):
        assert _agent()._detect_topic_switch(
            {"previous_sql": None}, ["member"]) is False

    def test_no_retrieved_tables_is_never_a_switch(self):
        assert _agent()._detect_topic_switch(
            {"previous_sql": self.PREV}, []) is False

    def test_unparseable_previous_sql_degrades_to_no_switch(self):
        """Dropping history on a parse failure would silently break real
        follow-ups — the safe default is to keep it."""
        assert _agent()._detect_topic_switch(
            {"previous_sql": "not sql at all !!"}, ["member"]) is False

    def test_the_flag_can_be_turned_off(self):
        a = _agent(_topic_switch_detection=False)
        # detection itself is unconditional; the node consults the flag
        import inspect
        assert "_topic_switch_detection" in inspect.getsource(a.retrieve_schema)


# ── the history each stage sees ───────────────────────────────────────────────

class TestHistoryFor:

    def test_a_switch_hides_the_conversation(self):
        assert A._history_for({"conversation_history": HISTORY,
                               "topic_switch": True}) == []

    def test_a_normal_turn_keeps_the_last_four(self):
        assert A._history_for({"conversation_history": HISTORY,
                               "topic_switch": False}) == HISTORY[-4:]

    def test_absent_flag_behaves_as_no_switch(self):
        assert A._history_for({"conversation_history": HISTORY}) == HISTORY[-4:]

    def test_empty_history_is_safe(self):
        assert A._history_for({"topic_switch": False}) == []

    def test_every_generation_site_goes_through_it(self):
        """Four call sites passed conversation_history[-4:] directly; each one
        left un-migrated is a leak this issue exists to close."""
        import inspect
        src = inspect.getsource(dbm.OrchestrationAgent)
        assert 'state["conversation_history"][-4:]' not in src
        assert src.count("history=self._history_for(state)") >= 4


# ── the intent classifier runs before retrieval ───────────────────────────────

class TestIntentHistoryGate:

    def _classify(self, agent, query, history):
        reply = MagicMock()
        reply.content = "READ"
        with patch.object(dbm, "invoke_tracked", return_value=reply) as llm:
            agent.classify_intent({"user_query": query,
                                   "conversation_history": history,
                                   "session_id": "s"})
        return llm.call_args[0][1][1].content     # the HumanMessage

    def _agent(self):
        a = _agent()
        a.INTENT_PROMPT = "x"
        a.intent_llm = MagicMock()
        a.llm = MagicMock()
        return a

    def test_a_self_contained_question_is_classified_without_history(self):
        sent = self._classify(
            self._agent(),
            "List the first names of members that wear a Medium t-shirt.",
            HISTORY)
        assert "SME" not in sent and "consumption" not in sent

    def test_a_follow_up_still_sees_the_conversation(self):
        sent = self._classify(self._agent(), "what about 2014?", HISTORY)
        assert "consumption" in sent

    def test_a_first_turn_is_unaffected(self):
        sent = self._classify(self._agent(), "what about 2014?", [])
        assert "what about 2014?" in sent
