"""
Tests for GH#138 — "how do I …?" questions about write operations.

The project claims it answers two kinds of question: those answered FROM the
data, and those answered ABOUT the database, explaining how an operation would
be performed rather than performing it.

The second half did not hold. handle_write_blocked was a fixed refusal ending
"Please contact your database administrator", and handle_general's prompt said
"Do NOT generate SQL unless explicitly asked" — which made the commonest
question of this kind unanswerable, because the useful answer IS the statement.

Showing the statement is safe: handle_general never touches a database, and the
validator rejects non-SELECT on the paths that do. Read-only is unchanged.

Offline — the LLM is stubbed. Run from repo root:
    pytest backend/tests/test_general_howto.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.db_agent import OrchestrationAgent


@pytest.fixture()
def agent():
    """An orchestrator with no config load, no DB, no LLM."""
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    a.memory = MagicMock()
    a.memory.add.return_value = "turn-1"
    a.tool_registry = MagicMock()
    a.tool_registry.run.return_value = ""      # web search returns nothing
    a._SEARCH_RESULT_PREFIXES = ("1.",)
    optimizer = MagicMock()
    optimizer.truncate_general.side_effect = lambda t: t
    optimizer._guardrail.side_effect = lambda t: t
    a._get_optimizer = lambda: optimizer
    return a


def _run_general(agent, question, llm_reply="ALTER TABLE customer ADD COLUMN x TEXT;"):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=llm_reply)
    agent._get_llm = lambda: llm
    with patch("backend.agents.db_agent.invoke_tracked",
               side_effect=lambda m, msgs, stage: m.invoke(msgs)):
        state = agent.handle_general(
            {"session_id": "s", "user_query": question})
    return state, llm.invoke.call_args[0][0][0].content


class TestIntentRoutingIsByPhrasing:
    """
    "How do I add a column?" is a request for an explanation about a write.
    Routing it by the SQL verb sends it to WRITE_BLOCKED and withholds an
    answer that is safe to give; routing it by phrasing answers it.
    """

    def test_the_prompt_states_the_rule_explicitly(self, agent):
        p = agent._intent_system_prompt()
        assert "PHRASING" in p
        assert "not by the SQL verb" in p

    def test_both_sides_are_shown_as_examples(self, agent):
        """A rule without a worked pair either way is one a model guesses at."""
        p = agent._intent_system_prompt()
        assert "How do I add a column to the customer table?" in p
        assert "Add a column to the customer table." in p

    def test_general_is_described_as_covering_writes(self, agent):
        assert "how a write or schema change would be done" in agent._intent_system_prompt()

    def test_write_blocked_is_described_as_an_instruction(self, agent):
        """'attempts … data modification' is what made a how-to look like one."""
        p = agent._intent_system_prompt()
        assert "INSTRUCTS" in p

    def test_the_four_labels_are_unchanged(self, agent):
        """Routing elsewhere in the graph keys off these exact strings."""
        p = agent._intent_system_prompt()
        for label in ("READ", "GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"):
            assert label in p


class TestGeneralExplainsTheOperation:

    def test_the_prompt_asks_for_the_statement(self, agent):
        _, system = _run_general(agent, "How do I add a column to customer?")
        assert "```sql" in system
        assert "WOULD be done" in system

    def test_the_prompt_requires_saying_it_was_not_run(self, agent):
        """The safety property the claim rests on: shown, not performed."""
        _, system = _run_general(agent, "How do I add a column to customer?")
        assert "NOT run it" in system
        assert "only reads" in system

    def test_the_prompt_asks_what_to_check_first(self, agent):
        _, system = _run_general(agent, "How do I add a column to customer?")
        for consideration in ("locking", "defaults", "backfill"):
            assert consideration in system

    def test_the_old_blanket_sql_ban_is_gone(self, agent):
        """It is what made this class of question unanswerable."""
        _, system = _run_general(agent, "How do I add a column to customer?")
        assert "Do NOT generate SQL" not in system

    def test_it_must_never_claim_to_have_changed_anything(self, agent):
        _, system = _run_general(agent, "How do I drop an index?")
        assert "Never claim to have changed anything" in system

    def test_the_answer_is_returned_and_recorded(self, agent):
        state, _ = _run_general(agent, "How do I add a column to customer?")
        assert "ALTER TABLE" in state["response"]
        assert state["turn_id"] == "turn-1"
        agent.memory.add.assert_called_once()

    def test_no_sql_is_executed_on_this_path(self, agent):
        """handle_general must never reach a database — that is what makes
        showing a DDL statement safe."""
        with patch("backend.agents.db_agent.execute_query") as ex:
            _run_general(agent, "How do I delete duplicate rows?")
        ex.assert_not_called()

    def test_the_sql_recorded_for_the_turn_is_none(self, agent):
        """A shown-not-run statement must not be stored as this turn's SQL, or
        the UI and the cache would treat it as an executed query."""
        _run_general(agent, "How do I add a column to customer?")
        kwargs = agent.memory.add.call_args.kwargs
        assert kwargs.get("sql") is None


class TestWriteInstructionsStayBlocked:

    def test_an_instruction_is_still_refused(self, agent):
        state = agent.handle_write_blocked(
            {"session_id": "s", "user_query": "Delete all players from France"})
        assert "read-only" in state["response"]
        assert "cannot insert, update, delete or alter" in state["response"]

    def test_the_refusal_points_at_the_answerable_form(self, agent):
        """'Contact your database administrator' was a dead end when the same
        question, asked as a how-to, is one this assistant answers in full."""
        state = agent.handle_write_blocked(
            {"session_id": "s", "user_query": "Add a column to customer"})
        assert "how do i add a column" in state["response"].lower()
        assert "administrator" not in state["response"].lower()

    def test_the_refusal_records_no_sql(self, agent):
        agent.handle_write_blocked({"session_id": "s", "user_query": "drop table x"})
        assert agent.memory.add.call_args.kwargs.get("sql") is None
