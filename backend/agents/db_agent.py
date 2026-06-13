"""
main_agent.py — LangGraph Orchestration Agent
Coordinates: Intent Classification → RAG Schema Retrieval → SQL Generation → Validation → Execution → Response
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent

from backend.memory.session_memory import SessionMemory
from backend.retrieval.indexing import retrieve , execute_query
from backend.pipelines.postprocessing import ResponseOptimizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────
"""
class AgentState(TypedDict):
    session_id: str
    user_query: str
    intent: str                      # READ | GENERAL | WRITE_BLOCKED | AMBIGUOUS
    schema_context: str
    few_shot_examples: List[str]
    generated_sqls: List[str]
    best_sql: str
    validation: Dict[str, Any]       # {valid, syntax_ok, safety_ok, semantic_ok, error}
    execution_result: Any
    response: str
    fallback_level: int              # 0=normal, 1=broader_search, 2=clarification
    error: Optional[str]
"""
class AgentState(TypedDict):
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, str]]
    intent: str                      # READ | GENERAL | WRITE_BLOCKED | AMBIGUOUS
    schema_context: str
    few_shot_examples: List[str]
    generated_sqls: List[str]
    best_sql: str
    validation: Dict[str, Any]       # {valid, syntax_ok, safety_ok, semantic_ok, error}
    execution_result: Any
    response: str
    fallback_level: int              # 0=normal, 1=broader_search, 2=clarification
    error: Optional[str]


# ─────────────────────────────────────────────
# Node Implementations
# ─────────────────────────────────────────────

class OrchestrationAgent:
    """
    Master agent — routes queries, enforces safety, coordinates sub-agents.
    Uses LangGraph stateful graph for deterministic flow with conditional edges.
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    INTENT_PROMPT = open(os.path.join(current_file_dir, "prompt.md")).read()  # loaded once

    def __init__(self, config: dict):
        self.llm = ChatOpenAI(
            model=config["llm"]["model"],
            temperature=0,
            api_key=config["llm"]["api_key"],
        )
        self.sql_agent = SQLCodeAgent(config, self.llm)
        self.validator = SQLValidatorAgent(config)
        self.memory = SessionMemory()
        self.optimizer = ResponseOptimizer(self.llm)
        self.graph = self._build_graph()

    # ── Nodes ──────────────────────────────────

    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify query as READ / GENERAL / WRITE_BLOCKED / AMBIGUOUS."""
        #history_str = self._format_history(state["conversation_history"][-6:])

        messages = [
            SystemMessage(content=self.INTENT_PROMPT),
            HumanMessage(content=(
                f"Conversation so far:\nhistory_str\n\n"
                f"New query: {state['user_query']}"
            )),
        ]
        result = self.llm.invoke(messages).content.strip().upper()

        # Hard safety: detect DML keywords regardless of LLM output
        dml_keywords = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"}
        if any(kw in state["user_query"].upper() for kw in dml_keywords):
            result = "WRITE_BLOCKED"

        logger.info(f"[INTENT] {result} | query={state['user_query'][:60]}")
        return {**state, "intent": result}

    
    def retrieve_schema(self, state: AgentState) -> AgentState:
        """Agentic RAG: retrieve relevant schema chunks + few-shot SQL pairs."""
        try:
            schema_chunks = retrieve(state["user_query"],"FYP",top_k_tables=5 ,store="pgvector")
            schema_context = schema_chunks["prompt_context"]
            
            #few_shots = self.retriever.retrieve_few_shots(state["user_query"], top_k=3)
            logger.info(f"[RAG] Retrieved {len(schema_chunks['tables'])} tables: "
                    f"{[t['table_name'] for t in schema_chunks['tables']]}")
             
            return {**state, "schema_context": schema_context, "few_shot_examples": []}
        except Exception as e:
            logger.warning(f"[RAG] retrieval failed: {e}")
            return {**state, "schema_context": "", "few_shot_examples": [], "fallback_level": 1}

    def generate_sql(self, state: AgentState) -> AgentState:
        """SQL Code Agent: generate top-K candidates then rerank."""
        sqls = self.sql_agent.generate(
            query=state["user_query"],
            schema_context=state["schema_context"],
            few_shots=state["few_shot_examples"],
            history=state["conversation_history"][-4:],
        )
        print(f"Generated SQLs: {sqls}")
        best = self.sql_agent.rerank(sqls, state["user_query"], state["schema_context"])
        logger.info(f"[SQL_GEN] best={best[:80]}")
        return {**state, "generated_sqls": sqls, "best_sql": best}

    def validate_sql(self, state: AgentState) -> AgentState:
        """Validation: syntax → safety → semantic checks."""
        result = self.validator.validate(state["best_sql"], state["schema_context"])
        logger.info(f"[VALIDATE] {result}")
        return {**state, "validation": result}

    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute validated SELECT-only query against PostgreSQL."""
        try:
            rows = execute_query(state["best_sql"])
            print(rows)
            return {**state, "execution_result": rows, "error": None}
        except Exception as e:
            logger.error(f"[EXECUTE] {e}")
            return {**state, "execution_result": None, "error": str(e)}

    def handle_general(self, state: AgentState) -> AgentState:
        """Handle non-SQL questions (DB how-to, general help) via LLM directly."""
        messages = [
            SystemMessage(content=(
                "You are a helpful database assistant. "
                "Answer the user's general question clearly and concisely. "
                "Do NOT generate SQL unless explicitly asked."
            )),
            HumanMessage(content=state["user_query"]),
        ]
        answer = self.llm.invoke(messages).content
        return {**state, "response": answer}

    def handle_write_blocked(self, state: AgentState) -> AgentState:
        return {**state, "response": (
            "⚠️ This system is read-only. Data modification queries (INSERT/UPDATE/DELETE/DROP) "
            "are not permitted. Please contact your database administrator."
        )}

    def format_response(self, state: AgentState) -> AgentState:
        """Format execution results into natural language response."""
        if state.get("error"):
            response = f"I encountered an error executing the query: {state['error']}"
        else:
            response = self.optimizer.format(
                query=state["user_query"],
                sql=state["best_sql"],
                results=state["execution_result"],
            )

        # Persist to memory
        self.memory.add(
            session_id=state["session_id"],
            user_query=state["user_query"],
            sql=state["best_sql"],
            response=response,
            valid=state["validation"].get("valid", False),
        )
        return {**state, "response": response}

    def fallback_handler(self, state: AgentState) -> AgentState:
        """Tiered fallback: broader schema search → ask for clarification."""
        level = state.get("fallback_level", 0) + 1

        if level == 1:
            # Broaden schema scan
            schema_chunks = retrieve(state["user_query"],"FYP",top_k_tables=10 ,store="pgvector")
            schema_context = schema_chunks["prompt_context"]
            return {**state, "schema_context": schema_context, "fallback_level": 1}

        if level == 2:
            return {**state, "response": (
                "I'm having trouble finding the relevant tables for your query. "
                "Could you clarify which tables or columns you're referring to? "
                "For example: 'Show me students from the **enrollments** table...'"
            ), "fallback_level": 2}

        return {**state, "response": "Unable to process the query. Please rephrase.", "fallback_level": 3}

    # ── Conditional Edges ───────────────────────

    def route_by_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "READ")
        if intent == "WRITE_BLOCKED":
            return "handle_write_blocked"
        if intent == "GENERAL":
            return "handle_general"
        return "retrieve_schema"  # READ or AMBIGUOUS → attempt SQL

    def route_after_validation(self, state: AgentState) -> str:
        v = state.get("validation", {})
        if v.get("valid"):
            return "execute_sql"
        if state.get("fallback_level", 0) < 2:
            return "fallback_handler"
        return "format_response"  # give up gracefully

    def route_after_retrieval(self, state: AgentState) -> str:
        if not state.get("schema_context") and state.get("fallback_level", 0) == 0:
            return "fallback_handler"
        return "generate_sql"

    # ── Graph Builder ───────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("classify_intent", self.classify_intent)
        g.add_node("retrieve_schema", self.retrieve_schema)
        g.add_node("generate_sql", self.generate_sql)
        g.add_node("validate_sql", self.validate_sql)
        g.add_node("execute_sql", self.execute_sql)
        g.add_node("format_response", self.format_response)
        g.add_node("handle_general", self.handle_general)
        g.add_node("handle_write_blocked", self.handle_write_blocked)
        g.add_node("fallback_handler", self.fallback_handler)

        g.set_entry_point("classify_intent")

        g.add_conditional_edges("classify_intent", self.route_by_intent, {
            "retrieve_schema": "retrieve_schema",
            "handle_general": "handle_general",
            "handle_write_blocked": "handle_write_blocked",
        })

        g.add_conditional_edges("retrieve_schema", self.route_after_retrieval, {
            "generate_sql": "generate_sql",
            "fallback_handler": "fallback_handler",
        })

        g.add_edge("generate_sql", "validate_sql")

        g.add_conditional_edges("validate_sql", self.route_after_validation, {
            "execute_sql": "execute_sql",
            "fallback_handler": "fallback_handler",
            "format_response": "format_response",
        })

        g.add_edge("execute_sql", "format_response")
        g.add_edge("handle_general", END)
        g.add_edge("handle_write_blocked", END)
        g.add_edge("format_response", END)

        # Fallback loops back to generate_sql with broadened context (max 2 rounds)
        g.add_conditional_edges("fallback_handler", self._route_fallback, {
            "generate_sql": "generate_sql",
            "end": END,
        })

        return g.compile()

    def _route_fallback(self, state: AgentState) -> str:
        if state.get("fallback_level", 0) >= 2 or state.get("response"):
            return "end"
        return "generate_sql"

    """
    # ── Public API ──────────────────────────────
    def run(self, session_id: str, user_query: str) -> str:
        #history = self.memory.get_history(session_id)
        initial_state: AgentState = {
            "session_id": session_id,
            "user_query": user_query,
            "intent": "",
            "schema_context": "",
            "few_shot_examples": [],
            "generated_sqls": [],
            "best_sql": "",
            "validation": {},
            "execution_result": None,
            "response": "",
            "fallback_level": 0,
            "error": None,
        }
        final = self.graph.invoke(initial_state)
        return final["response"]
    """
    def run(self, session_id: str, user_query: str) -> str:
        history = self.memory.get_history(session_id)
        initial_state: AgentState = {
            "session_id": session_id,
            "user_query": user_query,
            "conversation_history": history,
            "intent": "",
            "schema_context": "",
            "few_shot_examples": [],
            "generated_sqls": [],
            "best_sql": "",
            "validation": {},
            "execution_result": None,
            "response": "",
            "fallback_level": 0,
            "error": None,
        }
        final = self.graph.invoke(initial_state)
        return final["response"]

    # ── Helpers ─────────────────────────────────

    def _format_history(self, history: List[Dict]) -> str:
        return "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ) or "No prior context."

    def _intent_system_prompt(self) -> str:
        return (
            "You are an intent classifier for a read-only database assistant.\n"
            "Classify the user query into exactly one of:\n"
            "  READ      — wants to query/retrieve data from the database\n"
            "  GENERAL   — asks for help, explanations, or SQL syntax questions\n"
            "  WRITE_BLOCKED — attempts INSERT/UPDATE/DELETE/DROP or data modification\n"
            "  AMBIGUOUS — unclear, needs clarification\n\n"
            "Reply with ONLY the label. No explanation."
        )
        
        
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv("backend/config/secrets.env")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    
    config = {
        "llm": {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY
        },
    }
    
    agent = OrchestrationAgent(config)
    response = agent.run(session_id="user-123", user_query="How many distinct players have overall rating of 67 and above and show me the top 5 names of those players?")
    print(response)
