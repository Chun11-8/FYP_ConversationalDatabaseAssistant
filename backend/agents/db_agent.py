"""
main_agent.py — LangGraph Orchestration Agent
Coordinates: Intent Classification → RAG Schema Retrieval → SQL Generation → Validation → Execution → Response
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent
from backend.agents.safety import enforce_limit
from backend.agents.utils import PromptHarness

from backend.memory.session_memory import SessionMemory
from backend.memory.semantic_cache import SemanticCache
from backend.retrieval.indexing import retrieve, execute_query, expand_fk_tables
from backend.pipelines.postprocessing import ResponseOptimizer
from backend.tools.registry import ToolRegistry
from backend.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

# Thread-local storage for per-request LLM/agent overrides (api_key support)
_tls: threading.local = threading.local()

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
    query_embedding: Optional[List[float]]   # cached once per turn, reused for cache R/W
    generated_sqls: List[str]
    best_sql: str
    validation: Dict[str, Any]       # {valid, syntax_ok, safety_ok, semantic_ok, error}
    execution_result: Any
    response: str
    turn_id: Optional[int]           # DB id of the assistant turn (for feedback)
    csv_path: Optional[str]          # set when results exceed TABLE_LIMIT
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
    # Load only the intent-classification section — not the full 250-line file
    INTENT_PROMPT: str = PromptHarness.get("ORCHESTRATION_AGENT")

    def __init__(self, config: dict):
        self._config = config
        self._model = config["llm"]["model"]
        self._default_api_key = config["llm"]["api_key"]
        self.llm = ChatOpenAI(
            model=self._model,
            temperature=0,
            api_key=self._default_api_key,
        )
        self.sql_agent = SQLCodeAgent(config, self.llm)
        self.validator = SQLValidatorAgent(config)
        self.memory = SessionMemory(
            db_config=config.get("db"),
            summarizer=self._summarize_history,
        )
        self.semantic_cache = SemanticCache(db_config=config.get("db"))
        self.optimizer = ResponseOptimizer(self.llm)
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(WebSearchTool())
        self.graph = self._build_graph()

    # ── Per-request LLM helpers (thread-safe via threading.local) ──────────

    def _get_llm(self) -> ChatOpenAI:
        """Return the per-request LLM if an api_key override is active, else the default."""
        return getattr(_tls, "llm", self.llm)

    def _get_sql_agent(self) -> SQLCodeAgent:
        return getattr(_tls, "sql_agent", self.sql_agent)

    def _get_optimizer(self) -> ResponseOptimizer:
        return getattr(_tls, "optimizer", self.optimizer)

    # ── Nodes ──────────────────────────────────

    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify query as READ / GENERAL / WRITE_BLOCKED / AMBIGUOUS."""
        history_str = self._format_history(state.get("conversation_history", [])[-6:])

        print("\n" + "━" * 70)
        print("🔍  PIPELINE START — classify_intent")
        print("━" * 70)
        print(f"  Query   : {state['user_query']}")
        print(f"  Session : {state.get('session_id', 'n/a')}")
        print(f"  History : {history_str[:200]}{'…' if len(history_str) > 200 else ''}")

        messages = [
            SystemMessage(content=self.INTENT_PROMPT),
            HumanMessage(content=(
                f"Conversation so far:\n{history_str}\n\n"
                f"New query: {state['user_query']}"
            )),
        ]
        try:
            result = self._get_llm().invoke(messages).content.strip().upper()
        except Exception as e:
            logger.warning(f"[INTENT] LLM call failed, defaulting to AMBIGUOUS: {e}")
            result = "AMBIGUOUS"

        # Hard safety: detect DML keywords regardless of LLM output
        dml_keywords = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"}
        if any(kw in state["user_query"].upper() for kw in dml_keywords):
            result = "WRITE_BLOCKED"

        print(f"  ✅ Intent: {result}")
        logger.info(f"[INTENT] {result} | query={state['user_query'][:60]}")
        return {**state, "intent": result}

    
    def retrieve_schema(self, state: AgentState) -> AgentState:
        """Agentic RAG: retrieve relevant schema chunks + few-shot SQL pairs."""
        print("\n" + "─" * 70)
        print("📚  NODE: retrieve_schema")
        print("─" * 70)

        # Embed the query once and reuse it for both cache read (now) and cache
        # write (after execution) — avoids a redundant embedding call per turn.
        query_embedding = self.semantic_cache.embed(state["user_query"])
        few_shots = self._retrieve_few_shots(state["user_query"], query_embedding)

        if few_shots:
            print(f"  💡 Few-shot examples injected ({len(few_shots)}):")
            for i, ex in enumerate(few_shots, 1):
                print(f"     [{i}] {ex[:120]}{'…' if len(ex) > 120 else ''}")
        else:
            print("  💡 Few-shot examples: none")

        try:
            schema_chunks = retrieve(state["user_query"], "FYP", top_k_tables=5, store="pgvector")
            # FK-aware expansion: pull in tables linked by FK to the retrieved set
            schema_chunks = expand_fk_tables(schema_chunks, "FYP")
            schema_context = schema_chunks["prompt_context"]

            primary = [t["table_name"] for t in schema_chunks["tables"]]
            fk_extra = [c["table_name"] for c in schema_chunks.get("fk_tables", [])]
            print(f"  📋 Primary tables  : {primary}")
            if fk_extra:
                print(f"  🔗 FK-expanded     : {fk_extra}")

            # Print similarity distances for debugging retrieval quality
            print("  📊 Retrieval distances (lower = more relevant):")
            for t in schema_chunks["tables"]:
                print(f"       {t['distance']:.4f}  {t['table_name']}")

            print("\n  📄 Schema context sent to LLM:")
            print("  " + "·" * 60)
            for line in schema_context.splitlines():
                print(f"  {line}")
            print("  " + "·" * 60)

            logger.info(
                f"[RAG] Retrieved {len(primary)} table(s): {primary}"
                + (f" | FK-expanded: {fk_extra}" if fk_extra else "")
            )

            return {**state, "schema_context": schema_context,
                    "few_shot_examples": few_shots, "query_embedding": query_embedding}
        except Exception as e:
            print(f"  ❌ RAG retrieval failed: {e}")
            logger.warning(f"[RAG] retrieval failed: {e}")
            return {**state, "schema_context": "", "few_shot_examples": few_shots,
                    "query_embedding": query_embedding, "fallback_level": 1}

    def _retrieve_few_shots(
        self, user_query: str, embedding: Optional[List[float]] = None
    ) -> List[str]:
        """Pull similar cached pairs as few-shot examples (never raises)."""
        try:
            few_shots = self.semantic_cache.retrieve_few_shots(
                user_query, top_k=3, embedding=embedding
            )
            if few_shots:
                logger.info(f"[CACHE] Injected {len(few_shots)} few-shot example(s).")
            return few_shots
        except Exception as e:
            logger.warning(f"[CACHE] few-shot retrieval failed: {e}")
            return []

    def generate_sql(self, state: AgentState) -> AgentState:
        """SQL Code Agent: generate top-K candidates then rerank."""
        print("\n" + "─" * 70)
        print(f"🧠  NODE: generate_sql  (fallback_level={state.get('fallback_level', 0)})")
        print("─" * 70)

        sql_agent = self._get_sql_agent()

        few_shots = list(state["few_shot_examples"])
        # On fallback retry after empty generation, inject a format reminder so the
        # LLM knows it must wrap SQL in ```sql ... ``` fences.
        if state.get("fallback_level", 0) >= 1 and not state.get("generated_sqls"):
            few_shots = [
                "IMPORTANT: Your previous response did not contain a SQL code block. "
                "You MUST wrap your SQL in ```sql ... ``` fences."
            ] + few_shots

        # Wrap LLM calls so an outage (bad key, network, rate limit) degrades to
        # the graceful fallback path instead of bubbling up a 500. An empty
        # best_sql is treated as invalid by validate_sql → fallback_handler.
        try:
            sqls = sql_agent.generate(
                query=state["user_query"],
                schema_context=state["schema_context"],
                few_shots=few_shots,
                history=state["conversation_history"][-4:],
            )
            print(f"\n  ✅ {len(sqls)} SQL candidate(s) generated:")
            for i, sql in enumerate(sqls, 1):
                print(f"\n  ── Candidate {i} ──")
                for line in sql.splitlines():
                    print(f"     {line}")
            logger.info(f"[SQL_GEN] {len(sqls)} candidate(s) generated")

            best = sql_agent.rerank(sqls, state["user_query"], state["schema_context"])
            print(f"\n  🏆 Reranker selected:")
            for line in best.splitlines():
                print(f"     {line}")
            logger.info(f"[SQL_GEN] best={best[:80]}")
        except Exception as exc:
            print(f"  ❌ SQL generation failed: {exc}")
            logger.warning(f"[SQL_GEN] generation failed, degrading gracefully: {exc}")
            return {**state, "generated_sqls": [], "best_sql": ""}

        return {**state, "generated_sqls": sqls, "best_sql": best}

    def validate_sql(self, state: AgentState) -> AgentState:
        """Validation: syntax → safety → semantic checks."""
        print("\n" + "─" * 70)
        print("✔️   NODE: validate_sql")
        print("─" * 70)
        result = self.validator.validate(state["best_sql"], state["schema_context"])
        status = "✅ VALID" if result.get("valid") else "❌ INVALID"
        print(f"  {status}")
        print(f"    syntax_ok   : {result.get('syntax_ok')}")
        print(f"    safety_ok   : {result.get('safety_ok')}")
        print(f"    semantic_ok : {result.get('semantic_ok')}")
        if result.get("error"):
            print(f"    error       : {result['error']}")
        logger.info(f"[VALIDATE] {result}")
        return {**state, "validation": result}

    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute validated SELECT-only query against PostgreSQL."""
        print("\n" + "─" * 70)
        print("⚡  NODE: execute_sql")
        print("─" * 70)
        sql = enforce_limit(state["best_sql"])
        print("  SQL to execute:")
        for line in sql.splitlines():
            print(f"    {line}")
        try:
            rows = execute_query(sql)
            row_count = len(rows) if isinstance(rows, list) else "?"
            print(f"  ✅ Returned {row_count} row(s)")
            if isinstance(rows, list) and rows:
                preview = rows[:3]
                print(f"  Preview (first {len(preview)}):")
                for r in preview:
                    print(f"    {dict(r)}")
            return {**state, "best_sql": sql, "execution_result": rows, "error": None}
        except Exception as e:
            print(f"  ❌ Execution error: {e}")
            logger.error(f"[EXECUTE] {e}")
            return {**state, "execution_result": None, "error": str(e)}

    # Search results are usable when non-empty and not an error/diagnostic string.
    _SEARCH_RESULT_PREFIXES = ("Web search results for:", "1.")

    def handle_general(self, state: AgentState) -> AgentState:
        """Handle non-SQL questions via web search + LLM synthesis."""
        query = state["user_query"]

        # 1. Web search for context
        search_context = ""
        try:
            raw = self.tool_registry.run("web_search", {"query": query})
            # Accept the result only if it looks like real search results, not
            # an error/diagnostic string (e.g. "Web search failed: ...",
            # "No web results found ...").
            if raw and any(raw.startswith(p) for p in self._SEARCH_RESULT_PREFIXES):
                search_context = raw
                logger.info(f"[GENERAL] web_search returned {len(search_context)} chars")
            else:
                logger.info(f"[GENERAL] web_search returned non-result: {raw[:80]}")
        except Exception as exc:
            logger.warning(f"[GENERAL] web_search failed: {exc}")

        # 2. LLM synthesis with search results
        system_prompt = (
            "You are a helpful database assistant. "
            "Answer the user's question clearly and concisely. "
            + (
                "Use the web search results below as context. Cite sources where relevant. "
                if search_context
                else ""
            )
            + "Do NOT generate SQL unless explicitly asked."
        )
        user_content = query
        if search_context:
            user_content = (
                f"Question: {query}\n\n"
                f"Web search results:\n{search_context}\n\n"
                "Please synthesize the above into a clear, concise answer."
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        try:
            answer = self._get_llm().invoke(messages).content
        except Exception as e:
            logger.warning(f"[GENERAL] LLM synthesis failed: {e}")
            answer = "I'm temporarily unable to answer. Please try again in a moment."

        # 3. Guardrail: summarize over-long answers + output safety check
        answer = self._get_optimizer().truncate_general(answer)
        answer = self._get_optimizer()._guardrail(answer)

        turn_id = self.memory.add(
            session_id=state["session_id"],
            user_query=query,
            response=answer,
            sql=None,
            valid=None,
        )
        return {**state, "response": answer, "turn_id": turn_id}

    def handle_write_blocked(self, state: AgentState) -> AgentState:
        response = (
            "⚠️ This system is read-only. Data modification queries (INSERT/UPDATE/DELETE/DROP) "
            "are not permitted. Please contact your database administrator."
        )
        turn_id = self.memory.add(
            session_id=state["session_id"],
            user_query=state["user_query"],
            response=response,
            sql=None,
            valid=None,
        )
        return {**state, "response": response, "turn_id": turn_id}

    def format_response(self, state: AgentState) -> AgentState:
        """Format execution results into natural language response."""
        print("\n" + "─" * 70)
        print("📝  NODE: format_response")
        print("─" * 70)
        if state.get("error"):
            response = f"I encountered an error executing the query: {state['error']}"
            csv_path = None
        else:
            response, csv_path = self._get_optimizer().format(
                query=state["user_query"],
                sql=state["best_sql"],
                results=state["execution_result"],
            )

        # Only record the SQL when it was actually valid and executed — not
        # when we arrived here via the graceful-fallback path (validation failed
        # twice) or when execution errored. Writing an invalid SQL would poison
        # get_last_sql() for follow-up reference resolution.
        sql_valid = state["validation"].get("valid", False) and not state.get("error")
        turn_id = self.memory.add(
            session_id=state["session_id"],
            user_query=state["user_query"],
            sql=state["best_sql"] if sql_valid else None,
            response=response,
            valid=bool(sql_valid),
        )
        return {**state, "response": response, "csv_path": csv_path, "turn_id": turn_id}

    def fallback_handler(self, state: AgentState) -> AgentState:
        """Tiered fallback: broader schema search → ask for clarification."""
        print("\n" + "─" * 70)
        level = state.get("fallback_level", 0) + 1
        print(f"⚠️   NODE: fallback_handler → level {level}")
        if state.get("validation", {}).get("error"):
            print(f"  Reason: {state['validation']['error']}")
        print("─" * 70)

        if level == 1:
            # Broaden schema scan
            schema_chunks = retrieve(state["user_query"], "FYP", top_k_tables=10, store="pgvector")
            schema_context = schema_chunks["prompt_context"]
            return {**state, "schema_context": schema_context, "fallback_level": 1}

        if level == 2:
            response = (
                "I'm having trouble generating a valid SQL query for your question. "
                "Could you clarify which tables or columns you're referring to? "
                "For example: 'Show me players from the **players** table with rating above 80...'"
            )
            # Persist clarification so it appears in conversation history
            self.memory.add(
                session_id=state["session_id"],
                user_query=state["user_query"],
                response=response,
                sql=None,
                valid=None,
            )
            return {**state, "response": response, "fallback_level": 2}

        response = "Unable to process the query. Please rephrase."
        self.memory.add(
            session_id=state["session_id"],
            user_query=state["user_query"],
            response=response,
            sql=None,
            valid=None,
        )
        return {**state, "response": response, "fallback_level": 3}

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
    def run(self, session_id: str, user_query: str, api_key: str | None = None) -> Dict[str, Any]:
        """
        Run the full pipeline for one user turn.

        If ``api_key`` is provided it is used in place of the system default
        OpenAI key for all LLM calls in this request (thread-safe).

        Returns:
            {
                "response":   str,
                "sql":        str | None,
                "intent":     str,
                "validation": dict,
                "row_count":  int | None,
                "csv_path":   str | None,
            }
        """
        # Set up per-request LLM overrides in thread-local storage.
        # Only branch when a custom key is actually provided; otherwise leave
        # _tls untouched and read via getattr(..., self.attr) fallback.
        use_custom_key = bool(api_key and api_key.strip())
        if use_custom_key:
            _tls.llm = ChatOpenAI(model=self._model, temperature=0, api_key=api_key)
            _tls.sql_agent = SQLCodeAgent(self._config, _tls.llm)
            _tls.optimizer = ResponseOptimizer(_tls.llm)

        history = self.memory.get_history(session_id)
        initial_state: AgentState = {
            "session_id": session_id,
            "user_query": user_query,
            "conversation_history": history,
            "intent": "",
            "schema_context": "",
            "few_shot_examples": [],
            "query_embedding": None,
            "generated_sqls": [],
            "best_sql": "",
            "validation": {},
            "execution_result": None,
            "response": "",
            "turn_id": None,
            "csv_path": None,
            "fallback_level": 0,
            "error": None,
        }
        t0 = time.perf_counter()
        try:
            final = self.graph.invoke(initial_state)
        finally:
            # Clear the per-request overrides so the thread can be reused cleanly.
            # Only needed when we set them.
            if use_custom_key:
                _tls.llm = self.llm
                _tls.sql_agent = self.sql_agent
                _tls.optimizer = self.optimizer
        latency_ms = int((time.perf_counter() - t0) * 1000)
        print("\n" + "━" * 70)
        print(f"✅  PIPELINE DONE  |  latency={latency_ms}ms  |  intent={final.get('intent')}")
        print("━" * 70 + "\n")

        rows = final.get("execution_result")
        row_count = len(rows) if isinstance(rows, list) else None

        # Send the raw rows to the frontend (capped at 500 to bound payload size).
        # The frontend RecordsPanel handles display — first 10 visible, rest toggleable.
        serialised_rows: list[dict] | None = None
        if isinstance(rows, list) and rows:
            serialised_rows = [dict(r) for r in rows[:500]]

        result = {
            "response":   final.get("response", ""),
            "sql":        final.get("best_sql") or None,
            "intent":     final.get("intent", ""),
            "validation": final.get("validation", {}),
            "row_count":  row_count,
            "rows":       serialised_rows,
            "csv_path":   final.get("csv_path"),
            "turn_id":    final.get("turn_id"),
        }

        # Semantic cache write-path + interaction log (best-effort, never fatal).
        self._update_cache(session_id, user_query, final, row_count, latency_ms)
        return result

    def _update_cache(
        self,
        session_id: str,
        user_query: str,
        final: Dict[str, Any],
        row_count: Optional[int],
        latency_ms: int,
    ) -> None:
        """Persist a successful query→SQL pair and log the interaction."""
        validation = final.get("validation", {})
        best_sql = final.get("best_sql")
        success = (
            validation.get("valid")
            and not final.get("error")
            and bool(best_sql)
            and isinstance(final.get("execution_result"), list)
        )
        try:
            if success:
                self.semantic_cache.save_pair(
                    user_query=user_query,
                    sql=best_sql,
                    embedding=final.get("query_embedding"),
                )
            self.semantic_cache.log_interaction(
                session_id=session_id,
                user_query=user_query,
                sql=best_sql,
                response=final.get("response", ""),
                intent=final.get("intent", ""),
                valid=validation.get("valid"),
                row_count=row_count,
                latency_ms=latency_ms,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[CACHE] update failed: {e}")

    # ── Helpers ─────────────────────────────────

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "No prior context."
        lines = []
        for m in history:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "summary":
                lines.append(f"[Earlier conversation summary]: {content}")
            else:
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def _summarize_history(self, evicted_text: str, prior_summary: Optional[str]) -> str:
        """LLM summarizer injected into SessionMemory for rolling context."""
        prompt = (
            "Compress the following earlier conversation turns into a concise "
            "context note (max 4 sentences) that preserves entities, filters, "
            "numbers, and any SQL intent needed to understand follow-up questions.\n\n"
        )
        if prior_summary:
            prompt += f"Existing summary:\n{prior_summary}\n\n"
        prompt += f"New turns to fold in:\n{evicted_text}\n\nUpdated summary:"
        return self.llm.invoke([
            SystemMessage(content="You faithfully compress conversation history."),
            HumanMessage(content=prompt),
        ]).content.strip()

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
    import logging
    logging.basicConfig(level=logging.INFO)

    from backend.config.config_loader import load_config
    config = load_config()

    agent = OrchestrationAgent(config)
    result = agent.run(
        session_id="user-123",
        user_query="How many distinct players have overall rating of 67 and above and show me the top 5 names of those players?",
    )
    import json
    print(json.dumps(result, indent=2, default=str))
