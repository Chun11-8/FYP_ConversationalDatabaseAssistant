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

    These assert against OrchestrationAgent.INTENT_PROMPT — the string actually
    sent to the model (db_agent.py:792) — and NOT against a helper that returns
    the same text. GH#138 edited a `_intent_system_prompt()` method that nothing
    called, so the tests passed while the running classifier was unchanged and
    kept blocking how-to questions. The prompt lives in prompt.md under
    [SECTION: ORCHESTRATION_AGENT]; that dead method has been removed.
    """

    @property
    def prompt(self) -> str:
        return OrchestrationAgent.INTENT_PROMPT

    def test_the_prompt_is_loaded_from_prompt_md(self):
        """If PromptHarness silently returned "", every rule below would be
        vacuously true."""
        assert len(self.prompt) > 200
        assert "ORCHESTRATION" not in self.prompt.split("\n")[0]

    def test_the_rule_is_stated_explicitly(self):
        assert "PHRASING" in self.prompt
        assert "not by the sql verb" in self.prompt.lower()

    def test_both_sides_are_shown_as_worked_examples(self):
        """A rule without a pair either way is one a model guesses at."""
        assert "How do I add a column to the customer table?" in self.prompt
        assert "Add a column to the customer table." in self.prompt

    def test_general_covers_how_an_operation_is_done(self):
        assert "how an operation is done" in self.prompt

    def test_write_blocked_is_described_as_an_instruction(self):
        """'Any INSERT, UPDATE, DELETE, DROP, ALTER' is what made a how-to look
        like one: the question names the verb."""
        assert "INSTRUCTS" in self.prompt

    def test_naming_a_write_verb_is_explicitly_not_enough(self):
        assert "does NOT by itself mean WRITE_BLOCKED" in self.prompt

    def test_the_safety_first_default_no_longer_points_at_write_blocked(self):
        """'when in doubt, classify as WRITE_BLOCKED' overrode every example
        below it. Genuine uncertainty belongs in AMBIGUOUS, which asks the user
        rather than refusing them."""
        assert "when in doubt, classify as WRITE_BLOCKED" not in self.prompt
        assert "prefer AMBIGUOUS" in self.prompt

    def test_the_prompt_says_blocking_a_how_to_protects_nothing(self):
        """The reason the rule is safe, stated where the model can use it."""
        flat = " ".join(self.prompt.split())
        assert "read-only" in flat
        assert "withholds an explanation that is safe to give" in flat

    def test_the_four_labels_are_unchanged(self):
        """Routing elsewhere in the graph keys off these exact strings."""
        for label in ("READ", "GENERAL", "WRITE_BLOCKED", "AMBIGUOUS"):
            assert label in self.prompt


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


class TestGeneralSeesTheSchema:
    """
    GH#139: "How do I add a column to the customer table?" is only answerable
    ABOUT THIS DATABASE if the answer can see `customer`. Without it the model
    returns correct generic DDL that may contradict the real table — proposing
    a column that already exists, or a NOT NULL default on a populated one.
    """

    CHUNKS = [
        {"table_name": "customer", "content": "# Table: customer\n[(id:INT, Primary Key), (name:TEXT)]"},
        {"table_name": "orders", "content": "# Table: orders\n[(order_id:INT, Primary Key)]"},
    ]

    @staticmethod
    def _with_lookup(agent, chunks):
        agent._get_scope = lambda: "FYP"
        return patch("backend.agents.db_agent.fetch_table_chunks", return_value=chunks)

    def test_a_named_table_is_attached(self, agent):
        with self._with_lookup(agent, self.CHUNKS[:1]):
            out = agent._schema_for_question("How do I add a column to the customer table?")
        assert "# Table: customer" in out

    def test_the_lookup_is_by_name_not_by_embedding(self, agent):
        """The question already contains the table name; an embedding call to
        recover it would be paid for nothing."""
        agent._get_scope = lambda: "FYP"
        with patch("backend.agents.db_agent.fetch_table_chunks",
                   return_value=self.CHUNKS[:1]) as fetch, \
             patch("backend.agents.db_agent.retrieve") as retrieve:
            agent._schema_for_question("How do I index the customer table?")
        fetch.assert_called_once()
        retrieve.assert_not_called()

    def test_a_plural_question_finds_a_singular_table(self, agent):
        """Schemas name tables both ways and users follow the schema's lead
        only half the time."""
        agent._get_scope = lambda: "FYP"
        with patch("backend.agents.db_agent.fetch_table_chunks",
                   return_value=self.CHUNKS[:1]) as fetch:
            agent._schema_for_question("How do I add a column to customers?")
        assert "customer" in fetch.call_args[0][0]

    def test_generic_words_are_not_treated_as_table_names(self, agent):
        """'add a column to the table' must not pull a table called `table`."""
        agent._get_scope = lambda: "FYP"
        with patch("backend.agents.db_agent.fetch_table_chunks",
                   return_value=[]) as fetch:
            agent._schema_for_question("How do I add a column to a table?")
        asked = {w.casefold() for w in fetch.call_args[0][0]}
        assert "table" not in asked and "column" not in asked

    def test_the_attached_schema_is_capped(self, agent):
        """A question full of common words must not paste the whole database
        into the prompt."""
        many = [{"table_name": f"t{i}", "content": f"# Table: t{i}"} for i in range(12)]
        with self._with_lookup(agent, many):
            out = agent._schema_for_question("How do I join alpha beta gamma delta epsilon?")
        assert out.count("# Table:") == agent._GENERAL_MAX_TABLES

    def test_an_unindexed_name_costs_nothing(self, agent):
        with self._with_lookup(agent, []):
            assert agent._schema_for_question("How do I alter the widgets table?") == ""

    def test_a_lookup_failure_falls_back_silently(self, agent):
        """A broken index must degrade to the generic answer, not lose the turn."""
        agent._get_scope = lambda: "FYP"
        with patch("backend.agents.db_agent.fetch_table_chunks",
                   side_effect=RuntimeError("index down")):
            assert agent._schema_for_question("How do I alter customer?") == ""

    def test_the_schema_reaches_the_prompt(self, agent):
        with self._with_lookup(agent, self.CHUNKS[:1]):
            _, system = _run_general(agent, "How do I add a column to customer?")
        assert "actual schema is given below" in system

    def test_the_schema_is_in_the_user_message_too(self, agent):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="ALTER TABLE customer ...")
        agent._get_llm = lambda: llm
        with self._with_lookup(agent, self.CHUNKS[:1]), \
             patch("backend.agents.db_agent.invoke_tracked",
                   side_effect=lambda m, msgs, stage: m.invoke(msgs)):
            agent.handle_general({"session_id": "s", "user_query":
                                  "How do I add a column to customer?"})
        user_msg = llm.invoke.call_args[0][0][1].content
        assert "<schema_context>" in user_msg and "# Table: customer" in user_msg

    def test_no_schema_means_the_prompt_says_nothing_about_one(self, agent):
        """The schema-aware instruction must not appear when there is no schema
        — it would tell the model to use tables it was never given."""
        with self._with_lookup(agent, []):
            _, system = _run_general(agent, "What is a foreign key?")
        assert "actual schema is given below" not in system
