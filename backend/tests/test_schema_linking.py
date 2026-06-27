"""
Tests for the Schema-Linking & Question-Understanding stage (Issue #17).

Covers:
  - SchemaLinkingAgent.analyze() JSON parsing + graceful degradation
  - render_query_plan() block rendering
  - plan injection into SQLCodeAgent generation prompt
  - OrchestrationAgent.link_schema node + graph wiring

All LLM calls are mocked, so these run offline.

Run from repo root:
    pytest backend/tests/test_schema_linking.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.agents.schema_linking_agent import (
    SchemaLinkingAgent,
    render_query_plan,
    _extract_json,
)
from backend.agents.sql_code_agent import SQLCodeAgent


def _linker(dialect: str = "sqlite") -> SchemaLinkingAgent:
    return SchemaLinkingAgent({"db": {"dialect": dialect}}, MagicMock())


_PLAN_JSON = (
    '{"target": "COUNT(member_id)",'
    ' "tables": ["event", "attendance", "member"],'
    ' "join_path": ["event.event_id = attendance.link_to_event",'
    '               "attendance.link_to_member = member.member_id"],'
    ' "filters": ["member.t_shirt_size = \'Medium\'",'
    '             "event.event_name = \'Women\'\'s Soccer\'"],'
    ' "aggregation": "COUNT", "group_by": null, "having": null, "set_op": null}'
)


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:

    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_amid_prose(self):
        assert _extract_json('here: {"tables": ["t"]} done') == {"tables": ["t"]}

    def test_no_json_returns_none(self):
        assert _extract_json("no json at all") is None

    def test_non_dict_json_returns_none(self):
        assert _extract_json("[1, 2, 3]") is None


# ── analyze() ─────────────────────────────────────────────────────────────────

class TestAnalyze:

    def test_parses_plan_with_join_path(self):
        agent = _linker()
        agent.llm.invoke.return_value = MagicMock(content=_PLAN_JSON)
        plan = agent.analyze("how many medium shirts at Women's Soccer", "schema")
        assert plan["tables"] == ["event", "attendance", "member"]
        assert any("link_to_member" in j for j in plan["join_path"])

    def test_maps_value_phrase_to_column(self):
        agent = _linker()
        agent.llm.invoke.return_value = MagicMock(content=_PLAN_JSON)
        plan = agent.analyze("q", "schema")
        assert any("t_shirt_size = 'Medium'" in f for f in plan["filters"])

    def test_passes_schema_and_question_to_llm(self):
        agent = _linker()
        agent.llm.invoke.return_value = MagicMock(content=_PLAN_JSON)
        agent.analyze("MY_QUESTION", "MY_SCHEMA")
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "MY_QUESTION" in sent and "MY_SCHEMA" in sent

    def test_empty_query_returns_none(self):
        agent = _linker()
        assert agent.analyze("", "schema") is None

    def test_unparseable_output_returns_none(self):
        agent = _linker()
        agent.llm.invoke.return_value = MagicMock(content="I cannot help")
        assert agent.analyze("q", "schema") is None

    def test_llm_error_returns_none(self):
        agent = _linker()
        agent.llm.invoke.side_effect = RuntimeError("LLM down")
        assert agent.analyze("q", "schema") is None

    def test_plan_with_no_useful_fields_returns_none(self):
        agent = _linker()
        agent.llm.invoke.return_value = MagicMock(content='{"foo": "bar"}')
        assert agent.analyze("q", "schema") is None


# ── render_query_plan ─────────────────────────────────────────────────────────

class TestRenderQueryPlan:

    def test_renders_tables_and_joins(self):
        plan = {
            "tables": ["event", "member"],
            "join_path": ["event.event_id = attendance.link_to_event"],
        }
        block = render_query_plan(plan)
        assert "<query_plan>" in block and "</query_plan>" in block
        assert "event.event_id = attendance.link_to_event" in block

    def test_omits_empty_and_null_fields(self):
        block = render_query_plan({"target": "x", "having": None, "group_by": "null", "tables": []})
        assert "target: x" in block
        # null/empty fields produce no field line (check the labelled lines only)
        assert "having:" not in block and "group_by:" not in block
        assert "tables:" not in block  # empty list omitted

    def test_none_and_empty_return_empty_string(self):
        assert render_query_plan(None) == ""
        assert render_query_plan({}) == ""
        assert render_query_plan({"having": None, "tables": []}) == ""


# ── generation prompt injection ───────────────────────────────────────────────

class TestPlanInjection:

    def _sql_agent(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        return agent

    def test_plan_injected_when_provided(self):
        agent = self._sql_agent()
        plan = {"tables": ["a", "b"], "join_path": ["a.id = b.a_id"]}
        agent.generate(query="q", schema_context="s", few_shots=[], history=[], query_plan=plan)
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "<query_plan>" in sent
        assert "a.id = b.a_id" in sent

    def test_no_plan_omits_block(self):
        agent = self._sql_agent()
        agent.generate(query="q", schema_context="s", few_shots=[], history=[])
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "<query_plan>" not in sent


# ── OrchestrationAgent.link_schema node + graph wiring ────────────────────────

class TestLinkSchemaNode:

    def _agent(self, schema_linking: bool = True):
        from backend.agents.db_agent import OrchestrationAgent
        agent = OrchestrationAgent.__new__(OrchestrationAgent)
        agent._schema_linking = schema_linking
        return agent

    def test_node_sets_query_plan_from_linker(self):
        agent = self._agent()
        linker = MagicMock()
        linker.analyze.return_value = {"tables": ["t"]}
        agent.linker = linker
        state = {"user_query": "q", "schema_context": "s", "conversation_history": []}
        out = agent.link_schema(state)
        assert out["query_plan"] == {"tables": ["t"]}

    def test_node_disabled_sets_none(self):
        agent = self._agent(schema_linking=False)
        agent.linker = MagicMock()
        state = {"user_query": "q", "schema_context": "s", "conversation_history": []}
        out = agent.link_schema(state)
        assert out["query_plan"] is None
        agent.linker.analyze.assert_not_called()

    def test_node_degrades_to_none_on_empty_plan(self):
        agent = self._agent()
        linker = MagicMock()
        linker.analyze.return_value = None
        agent.linker = linker
        state = {"user_query": "q", "schema_context": "s", "conversation_history": []}
        out = agent.link_schema(state)
        assert out["query_plan"] is None

    def test_graph_contains_link_schema_node(self):
        import backend.agents.db_agent as m
        with patch.object(m, "ChatOpenAI", MagicMock()), \
             patch.object(m, "SQLCodeAgent", MagicMock()), \
             patch.object(m, "SQLValidatorAgent", MagicMock()), \
             patch.object(m, "SchemaLinkingAgent", MagicMock()), \
             patch.object(m, "SessionMemory", MagicMock()), \
             patch.object(m, "SemanticCache", MagicMock()), \
             patch.object(m, "ResponseOptimizer", MagicMock()), \
             patch.object(m, "ToolRegistry", MagicMock()):
            agent = m.OrchestrationAgent({"llm": {"model": "x", "api_key": "k"}, "db": {}})
            assert "link_schema" in agent.graph.get_graph().nodes
            assert agent._schema_linking is True  # default on
