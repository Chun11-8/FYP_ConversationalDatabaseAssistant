"""
Tests for the history-injection fix in db_agent.py.

Verifies the bug where the literal string "history_str" was sent to the LLM is
fixed, and that formatted prior context (including rolling summaries) is injected
into the intent classifier.

Run from repo root:
    pytest backend/tests/test_db_agent_history.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.agents.db_agent as dbmod


def _make_agent():
    """Construct an OrchestrationAgent with no network and an in-RAM memory."""
    with patch.object(dbmod, "ChatOpenAI", return_value=MagicMock()):
        agent = dbmod.OrchestrationAgent(
            {"llm": {"model": "gpt-4o-mini", "api_key": "k"}, "db": {}}
        )
    # Wipe any thread-local LLM override leaked from other tests.
    dbmod._tls.__dict__.clear()
    return agent


class TestFormatHistory:

    def test_empty_history(self):
        agent = _make_agent()
        assert agent._format_history([]) == "No prior context."

    def test_user_and_assistant_rendered(self):
        agent = _make_agent()
        out = agent._format_history([
            {"role": "user", "content": "show 5 players"},
            {"role": "assistant", "content": "Here they are"},
        ])
        assert "User: show 5 players" in out
        assert "Assistant: Here they are" in out

    def test_summary_role_rendered(self):
        agent = _make_agent()
        out = agent._format_history([
            {"role": "summary", "content": "User asked about players earlier"},
            {"role": "user", "content": "now the top 3"},
        ])
        assert "[Earlier conversation summary]: User asked about players earlier" in out
        assert "User: now the top 3" in out


class TestClassifyIntentHistoryInjection:

    def test_history_is_injected_not_literal(self):
        agent = _make_agent()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="READ")
        agent.llm = mock_llm
        agent.intent_llm = mock_llm  # classify_intent routes via _get_intent_llm
        dbmod._tls.__dict__.clear()  # force _get_llm() → agent.llm

        state = {
            "user_query": "now only the top 3",
            "conversation_history": [
                {"role": "user", "content": "show 5 players"},
                {"role": "assistant", "content": "Here are 5 players",
                 "sql": "SELECT name FROM players LIMIT 5"},
            ],
        }
        agent.classify_intent(state)

        # Inspect the HumanMessage actually sent to the LLM
        sent_messages = mock_llm.invoke.call_args.args[0]
        human_content = sent_messages[1].content

        assert "show 5 players" in human_content       # real history injected
        assert "history_str" not in human_content        # bug is gone
        assert "now only the top 3" in human_content     # new query present

    def test_no_history_uses_placeholder(self):
        agent = _make_agent()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="READ")
        agent.llm = mock_llm
        agent.intent_llm = mock_llm
        dbmod._tls.__dict__.clear()

        agent.classify_intent({"user_query": "list teams", "conversation_history": []})
        human_content = mock_llm.invoke.call_args.args[0][1].content
        assert "No prior context." in human_content
        assert "history_str" not in human_content

    def test_dml_keyword_forces_write_blocked_regardless_of_history(self):
        agent = _make_agent()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="READ")
        agent.llm = mock_llm
        agent.intent_llm = mock_llm
        dbmod._tls.__dict__.clear()

        result = agent.classify_intent(
            {"user_query": "DELETE FROM players", "conversation_history": []}
        )
        assert result["intent"] == "WRITE_BLOCKED"
