"""
main_agent.py — LangGraph Orchestration Agent
Coordinates: Intent Classification → RAG Schema Retrieval → SQL Generation → Validation → Execution → Response
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent, referenced_tables
from backend.agents.schema_linking_agent import SchemaLinkingAgent
from backend.agents.safety import enforce_limit
from backend.agents.trace import trace, set_pipeline_trace
from backend.agents.usage import invoke_tracked
from backend.agents import usage as usage_tracking
from backend.agents.llm_factory import build_chat_llm, describe_target, resolve_base_url
from backend.agents.utils import PromptHarness, strip_reasoning

from backend.memory.session_memory import SessionMemory
from backend.memory.semantic_cache import SemanticCache, literals_conflict
from backend.retrieval.indexing import (
    retrieve, execute_query, expand_fk_tables, fetch_table_chunks,
)
from backend.retrieval.value_index import ValueIndex, render_value_hints
from backend.pipelines.postprocessing import ResponseOptimizer
from backend.tools.registry import ToolRegistry
from backend.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

# Thread-local storage for per-request LLM/agent overrides (api_key support)
_tls: threading.local = threading.local()

# Small pool for prefetching the query embedding concurrently with the intent
# LLM call (GH#38): both are I/O-bound and independent.
from concurrent.futures import ThreadPoolExecutor
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=4)


# WRITE_BLOCKED always wins (safety-critical, never demoted). Among the rest,
# actionable labels are checked before AMBIGUOUS: a verbose/reasoning-style
# reply that hedges ("this could be ambiguous, but it's asking to retrieve
# data, so READ") mentions "ambiguous" incidentally on the way to a real
# conclusion — AMBIGUOUS is the catch-all, not a label worth pre-empting an
# actionable one just because it happens to appear earlier in the text.
_INTENTS = ("WRITE_BLOCKED", "READ", "GENERAL", "AMBIGUOUS")


def _intent_pattern(intent: str) -> str:
    # Tolerate "WRITE BLOCKED" / "WRITE-BLOCKED" alongside the underscored form.
    return intent.replace("_", r"[\s_-]")


# GH#76: markers that a question is harder than a one-table plan suggests.
# Deliberately narrow — this can only ever RAISE the effort budget, but a
# marker on every question would erase the simple tier's cost saving.
_HARD_QUESTION_RE = re.compile(
    r"\b(percentage|percent|proportion|ratio|average|"          # derived values
    r"highest|lowest|most|least|peak|maximum|minimum|top|"      # superlatives
    r"per\s+\w+|each\s+\w+|monthly|yearly|annual|"              # per-group grain
    r"compared|difference|more\s+than|versus|vs\b|"             # comparisons
    r"how\s+much\s+.*\bthan\b)\b",
    re.IGNORECASE)


def _question_looks_hard(question: Optional[str]) -> bool:
    """
    True when the question itself shows complexity a thin plan would hide.

    Two independent signals: recognised markers, or more than one question
    mark (a compound question — the linker has no sub-question field, so it
    collapses both parts into one plan and the result looks deceptively
    simple).
    """
    if not question:
        return False
    if question.count("?") > 1:
        return True
    return bool(_HARD_QUESTION_RE.search(question))


def _prepend_caution(response: str, verification: Optional[Dict[str, Any]]) -> str:
    """
    Lead a doubtful answer with the doubt (GH#75).

    Only HIGH-severity flags qualify. A caveat on every zero-row or
    arity-mismatch turn would appear on roughly a fifth of all answers and be
    tuned out, which is the failure mode the whole risk feature exists to
    avoid. The structured flags still render in the UI either way; this is
    about the sentence the user reads first.
    """
    risks = (verification or {}).get("risks") or []
    high = [r for r in risks if r.get("severity") == "high"]
    if not high or not response:
        return response
    note = high[0].get("message") or "This answer may not be reliable."
    return f"⚠️ {note}\n\n{response}"


def _normalize_intent(raw: str) -> str:
    """
    Reduce the classifier's reply to exactly one canonical intent.

    The model is asked for a bare label, but a chatty reply ("Classification:
    READ — the user wants to retrieve…") must not become the intent: every
    router compares by equality, so an unnormalized string silently falls
    through to the READ path, and the UI renders it as a badge.

    Unrecognized text → AMBIGUOUS, matching the prompt's own safety-first rule
    (better to ask than to guess).
    """
    # GH#65: a reasoning model weighs several labels aloud before deciding —
    # strip the chain of thought so only the verdict is scanned.
    stripped = strip_reasoning(raw or "").strip()
    text = stripped.upper()

    # Fast path: a fully-compliant reply is exactly one label (gpt-4o-mini
    # reliably does this) — trivially correct, no scanning needed.
    if text in _INTENTS:
        return text

    # Fast path: a reasoning-style reply that explains itself before
    # concluding typically states its verdict on the final line.
    lines = [l.strip().rstrip(".:,") for l in text.splitlines() if l.strip()]
    if lines and lines[-1] in _INTENTS:
        return lines[-1]

    if len(stripped) > 60:
        logger.warning(
            f"[INTENT] classifier reply is unusually long for a bare label "
            f"({len(stripped)} chars) — model may not be complying with the "
            f"output-format instruction: {stripped[:120]!r}"
        )

    # Last resort: scan the whole reply. WRITE_BLOCKED first (safety-critical);
    # among the rest, an actionable label beats an incidental "ambiguous"
    # mention in hedging text (see _INTENTS ordering above).
    for intent in _INTENTS:
        if re.search(rf"\b{_intent_pattern(intent)}\b", text):
            return intent
    return "AMBIGUOUS"


# Shared between ask_clarification (which generates this exact opening) and
# classify_intent/_replying_to_own_clarification (which detects it in
# history) — kept as one constant so the two can't silently drift apart.
_CLARIFICATION_PREFIX = "I want to make sure I query the right data"


def _replying_to_own_clarification(history: List[Dict]) -> bool:
    """
    True when the most recent assistant turn was our own ask_clarification
    response — i.e. this turn is (probably) a direct answer to it, not a
    fresh question. `history` is chronological; SessionMemory.add appends a
    user+assistant pair per turn, so the last entry is the prior assistant
    turn once this new user message hasn't been stored yet.
    """
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return str(m.get("content") or "").startswith(_CLARIFICATION_PREFIX)
        if m.get("role") == "user":
            return False
    return False


def _original_question_before_clarification(history: List[Dict]) -> Optional[str]:
    """The user turn that triggered our own clarifying question, if any."""
    seen_clarification = False
    for m in reversed(history or []):
        if m.get("role") == "assistant" and not seen_clarification:
            if not str(m.get("content") or "").startswith(_CLARIFICATION_PREFIX):
                return None
            seen_clarification = True
            continue
        if seen_clarification and m.get("role") == "user":
            return str(m.get("content") or "") or None
    return None


def rows_fingerprint(rows: List[Dict[str, Any]]) -> frozenset:
    """Order-insensitive identity of a result set (execution voting groups by this)."""
    return frozenset(
        tuple("" if v is None else str(v) for v in r.values()) for r in rows
    )


def rows_digest(rows: Any) -> Optional[str]:
    """
    Same identity as ``rows_fingerprint`` but as a stable hex string, so it can be
    stored alongside a cached answer and compared on a later run (GH#57).
    None for anything that isn't a row list.
    """
    if not isinstance(rows, list):
        return None
    h = hashlib.sha256()
    for row in sorted(rows_fingerprint(rows)):
        h.update(repr(row).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _emit(event: str, payload: Any) -> None:
    """Send a pipeline event to the per-request stage callback (GH#39 SSE)."""
    cb = getattr(_tls, "stage_cb", None)
    if cb:
        try:
            cb(event, payload)
        except Exception:
            pass


def _staged(name: str, fn):
    """
    Wrap a graph node: its entry (and the selected SQL) stream as events
    (GH#39), and its wall-clock duration lands in state["timings"] (GH#42) so
    a slow turn can be attributed to a specific stage.
    """
    def inner(state):
        _emit("stage", name)
        t0 = time.perf_counter()
        result = fn(state)
        ms = int((time.perf_counter() - t0) * 1000)
        if isinstance(result, dict):
            timings = dict(state.get("timings") or {})
            timings[name] = timings.get(name, 0) + ms   # accumulate on re-entry
            result = {**result, "timings": timings}
            if name in ("generate_sql", "refine_sql") and result.get("best_sql"):
                _emit("sql", result["best_sql"])
        return result
    inner.__name__ = name
    return inner


# Follow-ups that need condensation before retrieval (GH#40): pronouns /
# comparative refs / short continuations embed to garbage on their own.
_ANAPHORIC_RE = re.compile(
    r"\b(it|that|those|them|these|same|previous|ones?)\b"
    r"|what about|how about|^and\b|^also\b|^only\b|^now\b",
    re.IGNORECASE,
)

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
    query_plan: Optional[Dict[str, Any]]   # structured plan from SchemaLinkingAgent (#17)
    few_shot_examples: List[str]
    query_embedding: Optional[List[float]]   # cached once per turn, reused for cache R/W
    generated_sqls: List[str]
    best_sql: str
    validation: Dict[str, Any]       # {valid, syntax_ok, safety_ok, semantic_ok, error}
    execution_result: Any
    response: str
    turn_id: Optional[int]           # DB id of the assistant turn (for feedback)
    csv_path: Optional[str]          # set when results exceed TABLE_LIMIT
    standalone_query: Optional[str]  # condensed follow-up (GH#40); None = use user_query
    previous_sql: Optional[str]      # last validated SQL for this session (GH#40)
    cache_hit: bool                  # answered via semantic-cache shortcut (GH#38)
    cached_response: Optional[str]      # GH#57 answer reuse
    cached_fingerprint: Optional[str]   # GH#57 answer reuse
    rows_fingerprint: Optional[str]     # GH#57 hash of THIS turn's result rows
    cache_rejected: bool                # cached SQL failed validation this turn
    cache_probed: bool                  # run() already searched the cache
    cache_pair: Optional[dict]          # what that search found (None = miss)
    vote_rows: Optional[List[Dict[str, Any]]]   # winner rows from execution voting
    vote_rows_sql: Optional[str]                # the exact SQL those rows came from
    # GH#67: how the winning SQL was chosen — the system's own evidence about
    # its answer. {candidates, executed, agreement, distinct_result_sets,
    # decided_by}; assembled for the response in run().
    verification: Optional[Dict[str, Any]]
    # GH#75: question spans that match nothing in the retrieved tables — the
    # question asks about data that does not exist, so no SQL can answer it.
    unsatisfiable: List[str]
    # Tables actually retrieved this turn — used to suggest real values when a
    # question names one that does not exist.
    retrieved_tables: List[str]
    fallback_level: int              # 0=normal, 1=broader_search, 2=clarification
    refine_level: int                # ReAct execute→refine rounds taken (0=none)
    tier: Optional[str]              # inferred difficulty tier (GH#49 refine depth)
    timings: Dict[str, int]          # per-stage wall-clock ms (GH#42)
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

    # ReAct: max execute→refine rounds before we accept the result as-is.
    MAX_REFINE_ROUNDS: int = 1

    def __init__(self, config: dict):
        self._config = config
        self._model = config["llm"]["model"]
        self._default_api_key = config["llm"]["api_key"]
        # GH#52: models the frontend's per-request picker may choose from.
        # The configured default is always present, even if omitted from the list.
        self._selectable_models = list(config["llm"].get("selectable_models") or [self._model])
        if self._model not in self._selectable_models:
            self._selectable_models.insert(0, self._model)
        # GH#65: optional OpenAI-COMPATIBLE endpoint (OpenRouter, Groq,
        # DeepSeek, Together, Ollama…), which is what makes open-weight models
        # from other vendors usable. Absent → OpenAI, unchanged. Query stage
        # only: embeddings stay on OpenAI (pgvector columns are 1536-dim).
        self._base_url = resolve_base_url(config)
        if self._base_url:
            logger.info("[LLM] query stage → %s",
                        describe_target(self._model, self._base_url))
        self.llm = self._chat_llm(self._model, self._default_api_key)
        self.sql_agent = SQLCodeAgent(config, self.llm)
        self.validator = SQLValidatorAgent(config)
        self.linker = SchemaLinkingAgent(config, self.llm)
        # Schema linking is on by default; toggle via config `sql.schema_linking`.
        self._schema_linking = bool(config.get("sql", {}).get("schema_linking", True))
        # UI safety net: append LIMIT to executed queries. Disable when
        # benchmarking (config `sql.enforce_limit: false`) so the recorded SQL
        # matches what the model actually produced.
        self._enforce_limit = bool(config.get("sql", {}).get("enforce_limit", True))
        # Execution-guided candidate selection in the LIVE pipeline: run valid
        # candidates against the DB and plurality-vote on result fingerprints;
        # the text reranker only breaks ties (config `sql.execution_voting`).
        self._exec_voting = bool(config.get("sql", {}).get("execution_voting", True))
        # GH#69: gold-free risk checks on the executed result. On by default;
        # off gives a clean A/B against the labelled benchmark data.
        self._risk_checks = bool(config.get("sql", {}).get("risk_checks", True))
        # FK-expansion / context-budget knobs (config `retrieval.*`).
        _r = config.get("retrieval", {}) or {}
        self._fk_max_hops = int(_r.get("fk_max_hops", 2))
        self._fk_max_expansion = int(_r.get("fk_max_expansion", 5))
        self._max_schema_tokens = int(_r.get("max_schema_tokens", 4000))
        # GH#49 P3: deterministic WHERE-literal grounding against real values.
        self._value_grounding = bool(_r.get("value_grounding", True))
        if self._value_grounding:
            from backend.retrieval.pgvector_store import get_query_conn
            self.value_index = ValueIndex(get_query_conn)
        else:
            self.value_index = None
        # GH#49 P4: RSL-SQL-style backward linking — recover tables the plan
        # names but forward retrieval missed.
        self._backward_linking = bool(_r.get("backward_linking", True))
        # GH#38 latency + GH#41 hardening knobs.
        _sql = config.get("sql", {}) or {}
        self._db_name = config.get("db", {}).get("name", "FYP")
        self._stmt_timeout_ms = int(config.get("db", {}).get("statement_timeout_ms", 15000))
        self._vote_max = max(1, int(_sql.get("vote_max_candidates", 5)))
        # GH#48: when execution voting is inconclusive, resolve with a
        # unit-test screen + pairwise LLM comparison instead of the text
        # reranker ("pairwise"); "reranker" restores the old behavior.
        self._selection = str(_sql.get("selection", "pairwise")).lower()
        self._unit_tests = bool(_sql.get("unit_tests", True))
        # GH#49 quick win: ReAct refine depth, deeper for the challenging tier
        # (RSL-SQL/CHESS both use multi-turn correction).
        self._max_refine = int(_sql.get("max_refine_rounds", self.MAX_REFINE_ROUNDS))
        self._max_refine_challenging = int(
            _sql.get("max_refine_rounds_challenging", 2))
        self._cache_shortcut = bool(_sql.get("cache_shortcut", True))
        self._cache_hit_threshold = float(_sql.get("cache_hit_threshold", 0.03))
        # DAIL-SQL-style few-shot selection: blend structural-shape overlap
        # into embedding-distance ranking (see backend.memory.skeleton).
        self._fewshot_skeleton_blend = bool(_sql.get("fewshot_skeleton_blend", True))
        self._fewshot_skeleton_weight = float(_sql.get("fewshot_skeleton_weight", 0.3))
        # GH#40 multi-turn: condense anaphoric follow-ups before retrieval, and
        # ask ONE clarifying question on AMBIGUOUS instead of burning the full
        # generation pipeline first.
        self._condense = bool(_sql.get("condense_followups", True))
        self._clarify = bool(_sql.get("clarify_on_ambiguous", True))
        # GH#42: console banner tracing is config-gated (default on).
        set_pipeline_trace(config.get("logging", {}).get("pipeline_trace", True))
        self._dialect = config.get("db", {}).get("dialect", "postgres")
        # GH#57: repeat-query fast path + reuse of a stored answer when the
        # re-executed rows are unchanged. Both false → pre-GH#57 behavior.
        self._fast_path = bool(_sql.get("fast_path", True))
        self._answer_reuse = bool(_sql.get("answer_reuse", True))
        # GH#57: per-1M-token prices for cost accounting.
        usage_tracking.set_pricing(config["llm"].get("pricing"))
        # Model routing: intent classification is a one-token decision — allow a
        # cheaper/faster model; same for the response formatter. Absent → the
        # main LLM object is REUSED (identical behavior, no extra client).
        intent_model = config["llm"].get("intent_model")
        self.intent_llm = (
            self._chat_llm(intent_model, self._default_api_key)
            if intent_model and intent_model != self._model else self.llm
        )
        format_model = config["llm"].get("format_model")
        self._format_llm = (
            self._chat_llm(format_model, self._default_api_key)
            if format_model and format_model != self._model else self.llm
        )
        self.optimizer = ResponseOptimizer(self._format_llm)
        self.memory = SessionMemory(
            db_config=config.get("db"),
            summarizer=self._summarize_history,
        )
        self.semantic_cache = SemanticCache(db_config=config.get("db"))
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(WebSearchTool())
        self.graph = self._build_graph()

    # ── Public read-only config surface for the API layer (GH#52) ──────────

    @property
    def selectable_models(self) -> List[str]:
        """Models the frontend's per-request picker may choose from."""
        return list(self._selectable_models)

    @property
    def default_model(self) -> str:
        """The configured default LLM model."""
        return self._model

    # ── Per-request LLM helpers (thread-safe via threading.local) ──────────

    def _get_llm(self) -> ChatOpenAI:
        """Return the per-request LLM if an api_key override is active, else the default."""
        return getattr(_tls, "llm", self.llm)

    def _chat_llm(self, model: str, api_key: Optional[str]) -> ChatOpenAI:
        """Build a query-stage client honouring this agent's base_url (GH#65)."""
        return build_chat_llm(model, api_key, getattr(self, "_base_url", None))

    def _get_sql_agent(self) -> SQLCodeAgent:
        return getattr(_tls, "sql_agent", self.sql_agent)

    def _get_linker(self) -> SchemaLinkingAgent:
        return getattr(_tls, "linker", self.linker)

    def _get_intent_llm(self) -> ChatOpenAI:
        tls_llm = getattr(_tls, "intent_llm", None)
        if tls_llm is not None:
            return tls_llm
        own = getattr(self, "intent_llm", None)
        return own if own is not None else self.llm

    def _get_optimizer(self) -> ResponseOptimizer:
        return getattr(_tls, "optimizer", self.optimizer)

    # ── Nodes ──────────────────────────────────

    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify query as READ / GENERAL / WRITE_BLOCKED / AMBIGUOUS."""
        history = state.get("conversation_history", [])
        history_str = self._format_history(history[-6:])

        trace("\n" + "━" * 70)
        trace("🔍  PIPELINE START — classify_intent")
        trace("━" * 70)
        trace(f"  Query   : {state['user_query']}")
        trace(f"  Session : {state.get('session_id', 'n/a')}")
        trace(f"  History : {history_str[:200]}{'…' if len(history_str) > 200 else ''}")

        error: Optional[str] = None

        # Break the AMBIGUOUS clarification loop: a short reply to our own
        # last clarifying question ("frpm", "schools") is, out of context, not
        # obviously a query — the classifier has no instruction telling it to
        # treat a terse but on-topic answer as a continuation, and safety-first
        # bias ("when in doubt, AMBIGUOUS") keeps re-asking forever. Forcing
        # READ here hands off to condense_query (already reachable from READ),
        # which merges this reply with the original question via history —
        # the mechanism built for exactly this, just never reached before.
        if self._looks_anaphoric(state["user_query"], history) and \
                _replying_to_own_clarification(history):
            result = "READ"
            trace("  ⚡ Short reply to our own clarifying question — forcing READ")
            logger.info(f"[INTENT] short-circuited to READ (clarification reply): "
                       f"{state['user_query'][:60]}")
        else:
            messages = [
                SystemMessage(content=self.INTENT_PROMPT),
                HumanMessage(content=(
                    f"Conversation so far:\n{history_str}\n\n"
                    f"New query: {state['user_query']}"
                )),
            ]
            try:
                raw = invoke_tracked(self._get_intent_llm(), messages,
                                     "classify_intent").content
                result = _normalize_intent(raw)
                if result != raw.strip().upper():
                    logger.info(f"[INTENT] normalized {raw.strip()[:80]!r} → {result}")
            except Exception as e:
                # A call failure (invalid model, auth, rate limit, network) is
                # NOT the same as a genuinely ambiguous question — surface it
                # as a real error instead of a misleading "your question is
                # ambiguous" clarification prompt (GH#58: a broken custom
                # model id should fail loudly, not masquerade as normal flow).
                logger.warning(f"[INTENT] LLM call failed: {e}")
                result = "AMBIGUOUS"
                error = f"The selected model could not be reached: {e}"

        # Hard safety: detect DML keywords regardless of LLM output
        dml_keywords = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"}
        if any(kw in state["user_query"].upper() for kw in dml_keywords):
            result = "WRITE_BLOCKED"
            error = None

        trace(f"  ✅ Intent: {result}")
        logger.info(f"[INTENT] {result} | query={state['user_query'][:60]}")
        return {**state, "intent": result, "error": error}

    
    @staticmethod
    def _looks_anaphoric(query: str, history: List[Dict]) -> bool:
        if not history:
            return False
        return len(query.strip()) < 30 or bool(_ANAPHORIC_RE.search(query))

    def condense_query(self, state: AgentState) -> AgentState:
        """
        GH#40: rewrite an anaphoric follow-up ("what about 2013?") into a
        standalone question BEFORE retrieval — embedding the raw follow-up
        retrieves garbage. Cheap gate: first questions pay nothing.
        """
        if not getattr(self, "_condense", True):
            return state
        q = state["user_query"]
        history = state.get("conversation_history") or []
        if not self._looks_anaphoric(q, history):
            return state

        trace("\n" + "─" * 70)
        trace("🔗  NODE: condense_query (anaphoric follow-up detected)")
        trace("─" * 70)
        try:
            template = PromptHarness.get_file("condense_query.md")
        except Exception:
            template = (
                "Rewrite the user's follow-up as ONE self-contained question that "
                "needs no conversation context. Preserve every constraint from the "
                "history that the follow-up implies. Output ONLY the question."
            )
        messages = [
            SystemMessage(content=template),
            HumanMessage(content=(
                f"Conversation so far:\n{self._format_history(history[-6:])}\n\n"
                f"Previous SQL:\n{state.get('previous_sql') or '(none)'}\n\n"
                f"Follow-up: {q}\n\nStandalone question:"
            )),
        ]
        try:
            raw = invoke_tracked(self._get_intent_llm(), messages,
                                 "condense_query").content
            # GH#65: without stripping, a reasoning model's think block pushes
            # the reply past the 500-char guard below, the rewrite is silently
            # dropped, and multi-turn regresses to the GH#63 symptom.
            standalone = strip_reasoning(raw or "").strip()
        except Exception as exc:
            logger.warning(f"[CONDENSE] failed: {exc}")
            return state
        if standalone and len(standalone) < 500:
            trace(f"  ↪ standalone: {standalone}")
            logger.info(f"[CONDENSE] '{q[:40]}' → '{standalone[:80]}'")
            return {**state, "standalone_query": standalone}
        return state

    def ask_clarification(self, state: AgentState) -> AgentState:
        """
        GH#40: AMBIGUOUS questions get ONE targeted clarifying question up
        front instead of burning the full generation pipeline + two fallback
        rounds before asking.
        """
        trace("\n" + "─" * 70)
        trace("❓  NODE: ask_clarification (AMBIGUOUS intent)")
        trace("─" * 70)
        tables: List[str] = []
        try:
            r = retrieve(state["user_query"], getattr(self, "_db_name", "FYP"),
                         top_k_tables=3, store="pgvector",
                         embedding=state.get("query_embedding"))
            tables = [t["table_name"] for t in r.get("tables", [])][:3]
        except Exception:
            pass
        hint = (
            f" Are you asking about data in **{', '.join(tables)}**?"
            if tables else ""
        )
        response = (
            "I want to make sure I query the right data — your question is a "
            "little ambiguous. Could you name the table, entity, or metric you "
            f"mean?{hint}"
        )
        turn_id = self.memory.add(
            session_id=state["session_id"], user_query=state["user_query"],
            response=response, sql=None, valid=None,
        )
        return {**state, "response": response, "turn_id": turn_id}

    def retrieve_schema(self, state: AgentState) -> AgentState:
        """Agentic RAG: retrieve relevant schema chunks + few-shot SQL pairs."""
        trace("\n" + "─" * 70)
        trace("📚  NODE: retrieve_schema")
        trace("─" * 70)

        # Embed the query once and reuse it for both cache read (now) and cache
        # write (after execution). run() prefetches it concurrently with the
        # intent LLM call (GH#38) and hands it over in state — but only the RAW
        # query was embedded, so a condensed follow-up still needs its own.
        if state.get("cache_rejected"):
            # Re-entered after a stale cached SQL failed validation: drop the
            # hit before anything downstream can route on it again.
            state = {**state, "cache_hit": False, "best_sql": "",
                     "cached_response": None, "cached_fingerprint": None}

        search_q = state.get("standalone_query") or state["user_query"]
        query_embedding = state.get("query_embedding")
        if query_embedding is None or search_q != state["user_query"]:
            query_embedding = self.semantic_cache.embed(search_q)

        # GH#57: look for the cache hit BEFORE the RAG stage. Validation only
        # needs the schema of the tables the cached SQL actually reads, and
        # those can be fetched by name — no embedding search, no FK expansion,
        # no value grounding, no few-shot query. If the by-name fetch comes back
        # empty we fall through to full retrieval and re-apply the hit below,
        # so a hit is never *lost*, only made cheaper when possible.
        hit = self._probed_or_lookup(state, search_q, query_embedding)
        if hit:
            narrow_ctx = self._schema_for_sql(hit["sql"])
            if narrow_ctx:
                trace(f"  ⚡ Cache hit (distance={hit['distance']:.4f}) — "
                      f"skipping retrieval, linking and generation")
                logger.info(f"[CACHE] shortcut hit d={hit['distance']:.4f} (pre-RAG)")
                return {**state, "schema_context": narrow_ctx,
                        "few_shot_examples": [],
                        "query_embedding": query_embedding,
                        "best_sql": hit["sql"], "cache_hit": True,
                        "cached_response": hit.get("response"),
                        "cached_fingerprint": hit.get("rows_fingerprint")}

        few_shots = self._retrieve_few_shots(search_q, query_embedding)

        if few_shots:
            trace(f"  💡 Few-shot examples injected ({len(few_shots)}):")
            for i, ex in enumerate(few_shots, 1):
                trace(f"     [{i}] {ex[:120]}{'…' if len(ex) > 120 else ''}")
        else:
            trace("  💡 Few-shot examples: none")

        try:
            db_name = getattr(self, "_db_name", "FYP")
            schema_chunks = retrieve(search_q, db_name, top_k_tables=5,
                                     store="pgvector", embedding=query_embedding)
            # FK-aware expansion: pull in tables linked by FK to the retrieved set
            schema_chunks = expand_fk_tables(
                schema_chunks, db_name,
                max_hops=self._fk_max_hops,
                max_expansion=self._fk_max_expansion,
                max_schema_tokens=self._max_schema_tokens,
            )
            schema_context = schema_chunks["prompt_context"]

            primary = [t["table_name"] for t in schema_chunks["tables"]]
            fk_extra = [c["table_name"] for c in schema_chunks.get("fk_tables", [])]
            trace(f"  📋 Primary tables  : {primary}")
            if fk_extra:
                trace(f"  🔗 FK-expanded     : {fk_extra}")

            # GH#49 P3: ground literal-looking spans against ACTUAL stored
            # values; a hint block tells the generator the exact casing/format.
            unsatisfiable: List[str] = []
            if getattr(self, "value_index", None) is not None:
                hints = self.value_index.ground(search_q, primary + fk_extra)
                if hints:
                    trace(f"  🔤 Value hints     : {hints}")
                    schema_context = (
                        schema_context + "\n\n" + render_value_hints(hints)
                    )
                # GH#75: ground() is silent for a span that matches NOTHING —
                # there is no casing to correct — so the strongest signal
                # available was computed here and thrown away. Asked for
                # "customers from the country Wakanda", the pipeline knew
                # before writing a line of SQL that no such value exists, and
                # generated `segment='LAM' AND currency='EUR'` anyway. Once a
                # question is classified READ there is no exit that produces
                # no SQL, so the model is obliged to invent something.
                unsatisfiable = self._unsatisfiable_spans(search_q, primary + fk_extra)

            # Print similarity distances for debugging retrieval quality
            trace("  📊 Retrieval distances (lower = more relevant):")
            for t in schema_chunks["tables"]:
                trace(f"       {t['distance']:.4f}  {t['table_name']}")

            trace("\n  📄 Schema context sent to LLM:")
            trace("  " + "·" * 60)
            for line in schema_context.splitlines():
                trace(f"  {line}")
            trace("  " + "·" * 60)

            logger.info(
                f"[RAG] Retrieved {len(primary)} table(s): {primary}"
                + (f" | FK-expanded: {fk_extra}" if fk_extra else "")
            )

            # Cache-hit shortcut (GH#38): a near-identical past question with a
            # validated SQL skips linking + generation entirely. The cached SQL
            # still re-validates against the CURRENT schema (drift-safe), and a
            # validation failure falls back to normal generation. Reached only
            # when the cheap by-name schema fetch above came back empty.
            if hit:
                trace(f"  ⚡ Cache hit (distance={hit['distance']:.4f}) — "
                      f"skipping linking + generation")
                logger.info(f"[CACHE] shortcut hit d={hit['distance']:.4f}")
                return {**state, "schema_context": schema_context,
                        "few_shot_examples": few_shots,
                        "query_embedding": query_embedding,
                        "best_sql": hit["sql"], "cache_hit": True,
                        "cached_response": hit.get("response"),
                        "cached_fingerprint": hit.get("rows_fingerprint")}

            return {**state, "schema_context": schema_context,
                    "few_shot_examples": few_shots, "query_embedding": query_embedding,
                    "unsatisfiable": unsatisfiable,
                    "retrieved_tables": primary + fk_extra}
        except Exception as e:
            trace(f"  ❌ RAG retrieval failed: {e}")
            logger.warning(f"[RAG] retrieval failed: {e}")
            return {**state, "schema_context": "", "few_shot_examples": few_shots,
                    "query_embedding": query_embedding, "fallback_level": 1}

    def _unsatisfiable_spans(self, question: str, tables: List[str]) -> List[str]:
        """
        Value-like spans in the question that match NOTHING in the retrieved
        tables (GH#75) — i.e. the question asks about data that does not exist.

        Conservative by construction, because a wrong refusal is worse than a
        wrong answer: it needs the value index to have actually loaded
        (``resolve_spans`` returns {} when it could not check, never a wall of
        Nones), and it skips spans whose absence proves nothing — dates, month
        names and capitalised sentence openers. Those exclusions were not
        guesses: without them the equivalent post-hoc check produced a 54 %
        false-alarm rate on the labelled benchmark.
        """
        idx = getattr(self, "value_index", None)
        if idx is None or not tables or not question:
            return []
        try:
            from backend.agents.risk import _span_is_untestable
            resolved = idx.resolve_spans(question, tables) or {}
            return [s for s, v in resolved.items()
                    if v is None and not _span_is_untestable(s)]
        except Exception as exc:
            logger.warning(f"[RAG] unsatisfiability check skipped: {exc}")
            return []

    def handle_unanswerable(self, state: AgentState) -> AgentState:
        """
        Terminal node for a question naming data that does not exist (GH#75).

        Before this, the graph had no exit for a READ question that could not
        be answered: WRITE_BLOCKED, GENERAL, AMBIGUOUS and intent-error all
        terminate, but a well-formed READ question that happens to be
        unsatisfiable had to produce SQL. That obligation is what made the
        model drop the user's filter and substitute one that "worked".

        Says which value is missing rather than refusing flatly, so the user
        can correct it in one turn.
        """
        spans = state.get("unsatisfiable") or []
        trace("\n" + "─" * 70)
        trace(f"🚫  NODE: handle_unanswerable — no such value(s): {spans}")
        trace("─" * 70)
        logger.info(f"[UNANSWERABLE] {spans}")

        quoted = ", ".join(f"**{s}**" for s in spans[:3])
        nearest = self._nearest_values(spans[:1], state)
        response = (
            f"I could not find {quoted} anywhere in this database, so there is "
            f"no query I can run that would filter on it."
        )
        if nearest:
            response += f"\n\nThe closest values I do have are: {nearest}."
        response += (
            "\n\nIf you meant something else, tell me the value as it is stored "
            "and I will run the query."
        )
        turn_id = self.memory.add(
            session_id=state["session_id"], user_query=state["user_query"],
            response=response, valid=False,
        )
        return {**state, "response": response, "turn_id": turn_id, "best_sql": ""}

    def _nearest_values(self, spans: List[str], state: AgentState) -> str:
        """A few real values from the retrieved tables, to make the refusal
        actionable. Best-effort — an empty string just omits the suggestion."""
        idx = getattr(self, "value_index", None)
        if not spans or idx is None:
            return ""
        try:
            samples: List[str] = []
            for table in (state.get("retrieved_tables") or [])[:2]:
                for _col, values in idx._load_table(table).items():
                    samples.extend(str(v) for v in values[:3])
                    if len(samples) >= 6:
                        break
                if len(samples) >= 6:
                    break
            return ", ".join(f"`{v}`" for v in samples[:6])
        except Exception:
            return ""

    def _fast_path_state(self, hit: dict) -> Optional[Dict[str, Any]]:
        """
        Turn a pre-graph cache hit into seed state that enters at validate_sql,
        skipping intent classification and follow-up condensation on top of the
        linking and generation the in-graph cache shortcut already skipped
        (GH#57).

        Skipping classification is safe because pairs are only ever stored for
        successful, validated READ turns (see _update_cache) — a near-exact
        match to one cannot be a write attempt or a general-knowledge question.
        (run() does not even look the pair up for an anaphoric follow-up:
        "what about last year?" means nothing until condensed against history.)

        None when the cheap by-name schema fetch can't cover the cached SQL —
        the graph then runs normally and re-applies the hit after full retrieval.
        """
        schema_context = self._schema_for_sql(hit["sql"])
        if not schema_context:
            # Can't validate cheaply — let the graph run and re-apply the hit
            # after full retrieval (retrieve_schema reuses this embedding).
            return None
        trace(f"  ⚡ Fast path (distance={hit['distance']:.4f}) — "
              f"skipping intent classification and generation")
        logger.info(f"[CACHE] fast path d={hit['distance']:.4f}")
        return {
            "best_sql": hit["sql"],
            "schema_context": schema_context,
            "cached_response": hit.get("response"),
            "cached_fingerprint": hit.get("rows_fingerprint"),
        }

    @staticmethod
    def _consume_prefetched_embedding() -> Optional[List[float]]:
        """Take the run()-prefetched embedding, leaving nothing for a later reader."""
        future = getattr(_tls, "embed_future", None)
        _tls.embed_future = None
        if future is None:
            return None
        try:
            return future.result(timeout=10)
        except Exception:
            return None

    def _probed_or_lookup(self, state: AgentState, search_q: str,
                          embedding: Optional[List[float]]) -> Optional[dict]:
        """
        The cached pair for this turn, without paying for a second vector search.
        run() already looked one up for the raw question; that answer is reused
        unless the query has since been condensed (different text, different
        vector) or the cached SQL just failed validation.
        """
        if state.get("cache_rejected"):
            return None
        if state.get("cache_probed") and search_q == state["user_query"]:
            return state.get("cache_pair")
        return self._cache_lookup(embedding, search_q)

    def _cache_lookup(self, embedding: Optional[List[float]],
                      query_text: Optional[str] = None) -> Optional[dict]:
        """
        The cached pair for this query, or None when the shortcut is disabled,
        the cache is empty, or nothing is close enough. Never raises.

        ``query_text`` enables the GH#63 literal guard: a pair that is close in
        vector space but names DIFFERENT concrete values (2013 vs 2014,
        customer 6 vs 7, August vs September) is rejected. Without it a
        follow-up would be answered from the previous turn's SQL — the bug that
        made multi-turn conversation appear broken. Omit it only where no
        question text is available; distance alone is not a safe match.
        """
        if not getattr(self, "_cache_shortcut", False) or embedding is None:
            return None
        try:
            hit = self.semantic_cache.best_match(embedding)
        except Exception as exc:
            logger.warning(f"[CACHE] lookup failed: {exc}")
            return None
        if not isinstance(hit, dict) or not isinstance(hit.get("sql"), str):
            return None
        distance = hit.get("distance")
        if not isinstance(distance, (int, float)):
            return None
        threshold = getattr(self, "_cache_hit_threshold", 0.03)
        if distance > threshold:
            return None
        cached_q = hit.get("user_query") or ""
        if query_text is not None and literals_conflict(query_text, cached_q):
            trace(f"  🚫 Cache candidate rejected (d={distance:.4f}) — different "
                  f"literals: {cached_q[:70]!r}")
            logger.info(f"[CACHE] literal-guard rejected d={distance:.4f} | "
                        f"asked={query_text[:60]!r} cached={cached_q[:60]!r}")
            return None
        logger.info(f"[CACHE] hit d={distance:.4f} | asked={(query_text or '')[:60]!r} "
                    f"cached={cached_q[:60]!r}")
        return hit

    def _schema_for_sql(self, sql: str) -> str:
        """
        Schema context covering just the tables ``sql`` reads, fetched by name.
        "" when the SQL can't be parsed or the tables aren't indexed — the
        caller then falls back to full retrieval.
        """
        tables = referenced_tables(sql, getattr(self, "_dialect", "postgres"))
        if not tables:
            return ""
        chunks = fetch_table_chunks(tables, getattr(self, "_db_name", "FYP"))
        # A partial fetch would hand strict-semantic validation an incomplete
        # schema and fail a perfectly good cached query — take all or nothing.
        if len(chunks) < len(tables):
            return ""
        return "\n\n".join(c["content"] for c in chunks)

    def _retrieve_few_shots(
        self, user_query: str, embedding: Optional[List[float]] = None
    ) -> List[str]:
        """Pull similar cached pairs as few-shot examples (never raises).

        When ``sql.fewshot_skeleton_blend`` is on (default), candidates are
        reranked by structural-shape overlap against the question in
        addition to embedding distance (GH#48-50 companion — see
        backend.memory.skeleton / SemanticCache.retrieve_few_shots_blended).
        """
        try:
            if getattr(self, "_fewshot_skeleton_blend", True):
                from backend.memory.skeleton import question_shape_signature
                few_shots = self.semantic_cache.retrieve_few_shots_blended(
                    user_query, question_shape_signature(user_query), top_k=3,
                    embedding=embedding,
                    blend_weight=getattr(self, "_fewshot_skeleton_weight", 0.3),
                )
            else:
                few_shots = self.semantic_cache.retrieve_few_shots(
                    user_query, top_k=3, embedding=embedding
                )
            if few_shots:
                logger.info(f"[CACHE] Injected {len(few_shots)} few-shot example(s).")
            return few_shots
        except Exception as e:
            logger.warning(f"[CACHE] few-shot retrieval failed: {e}")
            return []

    def link_schema(self, state: AgentState) -> AgentState:
        """
        Schema-Linking & Question-Understanding (Issue #17). Produces a structured
        query plan (entities→columns, FK join-path, filters, aggregation/exclusion)
        that generate_sql follows. Degrades to query_plan=None on any failure.
        """
        trace("\n" + "─" * 70)
        trace("🧩  NODE: link_schema")
        trace("─" * 70)

        # Reaching this node means no shortcut applies (or a stale cache hit
        # fell back here) — clear the flag so the response isn't mislabeled.
        if state.get("cache_hit"):
            state = {**state, "cache_hit": False, "best_sql": ""}

        if not self._schema_linking:
            trace("  (schema linking disabled via config sql.schema_linking)")
            return {**state, "query_plan": None}

        try:
            plan = self._get_linker().analyze(
                query=state.get("standalone_query") or state["user_query"],
                schema_context=state["schema_context"],
                history=state["conversation_history"][-4:],
            )
        except Exception as exc:  # pragma: no cover - defensive; analyze already guards
            trace(f"  ❌ schema linking failed: {exc}")
            logger.warning(f"[LINK] schema linking failed: {exc}")
            return {**state, "query_plan": None}

        if not plan:
            trace("  ⚠️  no plan produced — generation proceeds without one")
            return {**state, "query_plan": None}

        trace("  ✅ Query plan:")
        for key in ("target", "tables", "join_path", "filters",
                    "evidence_predicates", "evidence_formula", "aggregation",
                    "group_by", "having", "order_by", "set_op"):
            val = plan.get(key)
            if val not in (None, "", [], {}, "null", "none"):
                trace(f"     {key}: {val}")

        # GH#49 P4 (backward linking): the plan may name tables forward
        # retrieval missed — fetch their chunks by name and append, so the
        # generator can actually join them.
        schema_context = state["schema_context"]
        if getattr(self, "_backward_linking", True):
            planned = [t for t in (plan.get("tables") or [])
                       if isinstance(t, str) and t.strip()]
            ctx_low = schema_context.casefold()
            missing = [t for t in planned if t.casefold() not in ctx_low]
            if missing:
                chunks = fetch_table_chunks(
                    missing, getattr(self, "_db_name", "FYP"))
                if chunks:
                    recovered = [c["table_name"] for c in chunks]
                    trace(f"  🔙 Backward-linked tables recovered: {recovered}")
                    logger.info(f"[LINK] backward-linked: {recovered}")
                    schema_context = (
                        schema_context + "\n\n"
                        + "\n\n".join(c["content"] for c in chunks)
                    )

        return {**state, "query_plan": plan, "schema_context": schema_context}

    def generate_sql(self, state: AgentState) -> AgentState:
        """SQL Code Agent: generate top-K candidates then rerank."""
        trace("\n" + "─" * 70)
        trace(f"🧠  NODE: generate_sql  (fallback_level={state.get('fallback_level', 0)})")
        trace("─" * 70)

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
            tier = self._infer_tier(
                state.get("query_plan"),
                state.get("standalone_query") or state.get("user_query"))
            if tier:
                trace(f"  🎚️  Inferred difficulty tier: {tier}")
            sqls = sql_agent.generate(
                query=state["user_query"],
                schema_context=state["schema_context"],
                few_shots=few_shots,
                history=state["conversation_history"][-4:],
                query_plan=state.get("query_plan"),
                tier=tier,
                previous_sql=state.get("previous_sql"),
            )
            trace(f"\n  ✅ {len(sqls)} SQL candidate(s) generated:")
            for i, sql in enumerate(sqls, 1):
                trace(f"\n  ── Candidate {i} ──")
                for line in sql.splitlines():
                    trace(f"     {line}")
            logger.info(f"[SQL_GEN] {len(sqls)} candidate(s) generated")

            # Execution-guided selection first (plurality over result sets);
            # the text reranker only decides when voting is inconclusive.
            best, vote_rows, vote_rows_sql = "", None, None
            # GH#67: how the winner was chosen is the system's own evidence
            # about its answer — recorded here rather than only traced, so the
            # API and UI can show it alongside the result.
            ev: Dict[str, Any] = {
                "candidates": len(sqls), "executed": 0, "agreement": 0,
                "distinct_result_sets": 0,
                "decided_by": "single_candidate" if len(sqls) <= 1
                              else "voting_disabled",
            }
            if getattr(self, "_exec_voting", True) and len(sqls) > 1:
                best, vote_rows, vote_rows_sql = self._vote_candidates(
                    sqls, state["schema_context"],
                    query=state.get("standalone_query") or state["user_query"],
                    evidence=ev)
            if not best:
                best = sql_agent.rerank(sqls, state["user_query"], state["schema_context"])
                trace(f"\n  🏆 Reranker selected:")
                # The reranker reads text only, never result sets — so a turn
                # decided here carries strictly weaker evidence than a
                # plurality, and must not be reported as agreement.
                if ev["decided_by"] in ("inconclusive", "none_executed"):
                    ev["decided_by"] = "reranker"
                    ev["agreement"] = 0
            else:
                trace(f"\n  🗳️  Execution vote selected:")
            for line in best.splitlines():
                trace(f"     {line}")
            logger.info(f"[SQL_GEN] best={best[:80]}")
        except Exception as exc:
            trace(f"  ❌ SQL generation failed: {exc}")
            logger.warning(f"[SQL_GEN] generation failed, degrading gracefully: {exc}")
            return {**state, "generated_sqls": [], "best_sql": "",
                    "verification": {"candidates": 0, "executed": 0,
                                     "agreement": 0, "distinct_result_sets": 0,
                                     "decided_by": "generation_failed"}}

        return {**state, "generated_sqls": sqls, "best_sql": best,
                "vote_rows": vote_rows, "vote_rows_sql": vote_rows_sql,
                "verification": ev, "tier": tier}

    @staticmethod
    def _infer_tier(plan: Optional[Dict[str, Any]],
                    question: Optional[str] = None) -> Optional[str]:
        """
        Conservative difficulty tier from the linker plan (Issue #27). The BIRD
        harness passes the real label; live queries get this heuristic. None
        (no plan / unclear) → the generator's default (moderate) path.

        GH#76: the tier is read from the plan, and the plan is the artefact
        that fails. A linker that collapses a hard question to one table
        yields "simple" → 2 candidates and no path diversity, so the question
        is under-resourced precisely because the reasoning about it already
        went wrong, and the two near-identical candidates then agree. The
        QUESTION is independent evidence, so complexity markers in it veto a
        downgrade to "simple" — never a downgrade of the plan's own verdict.
        """
        if not plan or not isinstance(plan, dict):
            return None

        def _set(v: Any) -> bool:
            return v is not None and str(v).strip().lower() not in ("", "null", "none")

        tables = [t for t in (plan.get("tables") or []) if t]
        if _set(plan.get("set_op")) or (_set(plan.get("having")) and len(tables) >= 3):
            return "challenging"
        if len(tables) <= 1 and not _set(plan.get("group_by")) and not _set(plan.get("having")):
            return "moderate" if _question_looks_hard(question) else "simple"
        return "moderate"

    def _vote_candidates(
        self, sqls: List[str], schema_context: str, query: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[list], Optional[str]]:
        """
        Execution-guided selection for the live pipeline: validate each
        candidate (SELECT-only safety — invalid/DML candidates are never run),
        execute the survivors with the safety LIMIT applied, and plurality-vote
        on an order-insensitive fingerprint of their result sets.

        Returns (winner_sql, winner_rows, executed_sql) — winner rows are reused by
        execute_sql so the winner runs ONCE per turn (GH#38). ("", None, None)
        when inconclusive (caller falls back to the reranker). Never raises.

        GH#67: when an ``evidence`` dict is passed it is filled in place with
        how the winner was chosen (agreement, distinct result sets, decided_by)
        so the answer can be shown with the reasoning behind it. An out-param
        rather than a fourth return value purely to leave the existing
        three-tuple unpacking — and its tests — alone.
        """
        def _record(**kw: Any) -> None:
            if evidence is not None:
                evidence.update(kw)

        _record(candidates=len(sqls), executed=0, agreement=0,
                distinct_result_sets=0, decided_by="none_executed")
        groups: Dict[frozenset, List[str]] = {}
        group_rows: Dict[frozenset, list] = {}   # first (rows, run_sql) per fingerprint
        group_run: Dict[frozenset, str] = {}
        executed = 0
        timeout_ms = getattr(self, "_stmt_timeout_ms", 15000)
        capped = sqls[: getattr(self, "_vote_max", 5)]
        for idx, sql in enumerate(capped):
            try:
                if not self.validator.validate(sql, schema_context).get("valid"):
                    continue
                run_sql = enforce_limit(sql) if getattr(self, "_enforce_limit", True) else sql
                rows = execute_query(run_sql, timeout_ms=timeout_ms)
                if not isinstance(rows, list):
                    continue
                fp = rows_fingerprint(rows)
                executed += 1
                groups.setdefault(fp, []).append(sql)
                if fp not in group_rows:
                    group_rows[fp], group_run[fp] = rows, run_sql
                # Early plurality (GH#41 cost bound): stop ONLY when the current
                # leader's lead is mathematically unassailable. Each remaining
                # candidate can add at most one vote, and in the worst case they
                # all pile onto the current runner-up, so a competitor can reach
                # at most (second_best + remaining). Breaking merely at 2 votes
                # would wrongly return a minority result set when later
                # candidates agree on a larger plurality (e.g. [A,B,A,B,B]→B).
                sizes = sorted((len(s) for s in groups.values()), reverse=True)
                leader = sizes[0]
                second = sizes[1] if len(sizes) > 1 else 0
                remaining = len(capped) - (idx + 1)
                if leader > second + remaining:
                    trace("  🗳️  Execution vote: unassailable plurality — "
                          "remaining candidates skipped")
                    break
            except Exception as exc:
                logger.debug(f"[VOTE] candidate failed: {exc}")
                continue

        def _win(fp) -> Tuple[str, Optional[list], Optional[str]]:
            return groups[fp][0], group_rows.get(fp), group_run.get(fp)

        if not groups:
            trace("  🗳️  Execution vote: no candidate executed — falling back to reranker")
            _record(executed=0, decided_by="none_executed")
            return "", None, None

        best_size = max(len(s) for s in groups.values())
        _record(executed=executed, distinct_result_sets=len(groups),
                agreement=best_size)
        tied = [fp for fp, s in groups.items() if len(s) == best_size]
        if len(tied) == 1 and (best_size >= 2 or len(groups) == 1):
            trace(f"  🗳️  Execution vote: {best_size}/{executed} candidates agree "
                  f"({len(groups)} distinct result sets)")
            _record(decided_by="plurality")
            return _win(tied[0])
        non_empty = [fp for fp in tied if len(fp) > 0]
        if len(non_empty) == 1:
            trace(f"  🗳️  Execution vote: tie broken toward non-empty result "
                  f"({len(groups[non_empty[0]])}/{executed})")
            _record(decided_by="non_empty_tiebreak",
                    agreement=len(groups[non_empty[0]]))
            return _win(non_empty[0])

        # GH#48: genuinely inconclusive — unit-test screen + pairwise
        # comparison beat both plurality and a rows-blind text reranker
        # (CHASE-SQL ablation: pairwise > self-consistency by ~4 EX).
        if query and getattr(self, "_selection", "pairwise") == "pairwise" and len(groups) >= 2:
            fps = sorted(groups, key=lambda f: len(groups[f]), reverse=True)[:3]
            if getattr(self, "_unit_tests", True) and len(fps) >= 2:
                passing = self._unit_test_screen(query, fps, groups, group_rows)
                if len(passing) == 1:
                    trace("  🧪 Unit-test screen: single surviving result set")
                    _record(decided_by="unit_test",
                            agreement=len(groups[passing[0]]))
                    return _win(passing[0])
                if len(passing) >= 2:
                    fps = passing
            winner_fp = self._pairwise_select(query, fps, groups, group_rows)
            if winner_fp is not None:
                trace("  ⚖️  Pairwise comparison selected the winner")
                _record(decided_by="pairwise", agreement=len(groups[winner_fp]))
                return _win(winner_fp)

        trace("  🗳️  Execution vote: tie between distinct result sets — reranker decides")
        _record(decided_by="inconclusive")
        return "", None, None

    # ── GH#48: unit-test screen + pairwise selection ───────────────────────

    @staticmethod
    def _describe_group(idx: int, fp, groups: Dict, group_rows: Dict) -> str:
        sql = groups[fp][0]
        rows = group_rows.get(fp) or []
        sample = "\n".join(str(dict(r)) for r in rows[:5]) or "(0 rows)"
        return (f"CANDIDATE {idx}:\nSQL:\n{sql}\n"
                f"Result: {len(rows)} row(s), first {min(5, len(rows))}:\n{sample}")

    def _unit_test_screen(
        self, query: str, fps: list, groups: Dict, group_rows: Dict
    ) -> list:
        """
        CHESS-style screen: derive the properties a correct result must have
        (row grain, shape, plausibility) and drop candidate groups whose actual
        rows violate them. One LLM call; on any failure returns all fps.
        """
        blocks = "\n\n".join(
            self._describe_group(i + 1, fp, groups, group_rows)
            for i, fp in enumerate(fps)
        )
        messages = [
            SystemMessage(content=(
                "You check SQL results against the question they must answer. "
                "First derive 3-4 checks a correct result must satisfy: the row "
                "GRAIN (one row per what? a scalar question has exactly one row), "
                "the expected columns, entity values vs prose, plausible "
                "magnitude. Then evaluate each candidate's ACTUAL rows against "
                "the checks. Output ONLY a line 'PASS: <candidate numbers "
                "comma-separated>' naming every candidate consistent with all "
                "checks."
            )),
            HumanMessage(content=f"Question: {query}\n\n{blocks}"),
        ]
        try:
            out = invoke_tracked(self._get_llm(), messages, "unit_test_screen").content
            m = re.search(r"PASS:\s*([\d,\s]+)", out)
            if not m:
                return fps
            picked = {int(t) for t in re.findall(r"\d+", m.group(1))}
            passing = [fp for i, fp in enumerate(fps) if (i + 1) in picked]
            trace(f"  🧪 Unit-test screen: {len(passing)}/{len(fps)} group(s) pass")
            return passing or fps
        except Exception as exc:
            logger.warning(f"[SELECT] unit-test screen failed: {exc}")
            return fps

    def _pairwise_select(
        self, query: str, fps: list, groups: Dict, group_rows: Dict
    ):
        """
        CHASE-SQL-style round-robin: for each pair, ask which candidate answers
        the question, judging SQL *and* actual result rows. Most wins takes it;
        an unresolved tie returns None (caller falls back to the reranker).
        """
        if len(fps) < 2:
            return fps[0] if fps else None
        wins = {fp: 0 for fp in fps}
        try:
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    a, b = fps[i], fps[j]
                    messages = [
                        SystemMessage(content=(
                            "Two SQL candidates answer the same question. Judge "
                            "which one is correct using BOTH the SQL logic and "
                            "the actual result rows (grain, filters, evidence "
                            "adherence). Answer with ONLY the single digit 1 "
                            "or 2."
                        )),
                        HumanMessage(content=(
                            f"Question: {query}\n\n"
                            + self._describe_group(1, a, groups, group_rows)
                            + "\n\n"
                            + self._describe_group(2, b, groups, group_rows)
                        )),
                    ]
                    out = invoke_tracked(self._get_llm(), messages,
                                         "pairwise_select").content.strip()
                    m = re.search(r"[12]", out)
                    if m:
                        wins[a if m.group() == "1" else b] += 1
        except Exception as exc:
            logger.warning(f"[SELECT] pairwise comparison failed: {exc}")
            return None
        best = max(wins.values())
        top = [fp for fp, w in wins.items() if w == best]
        return top[0] if len(top) == 1 else None

    def validate_sql(self, state: AgentState) -> AgentState:
        """Validation: syntax → safety → semantic checks."""
        trace("\n" + "─" * 70)
        trace("✔️   NODE: validate_sql")
        trace("─" * 70)
        result = self.validator.validate(state["best_sql"], state["schema_context"])
        status = "✅ VALID" if result.get("valid") else "❌ INVALID"
        trace(f"  {status}")
        trace(f"    syntax_ok   : {result.get('syntax_ok')}")
        trace(f"    safety_ok   : {result.get('safety_ok')}")
        trace(f"    semantic_ok : {result.get('semantic_ok')}")
        if result.get("error"):
            trace(f"    error       : {result['error']}")
        logger.info(f"[VALIDATE] {result}")
        if state.get("cache_hit") and not result.get("valid"):
            # Stale cached SQL: the turn now needs the FULL schema context that
            # the cheap by-name fetch deliberately skipped, so it unwinds to
            # retrieve_schema — which must not hand back the same bad pair.
            return {**state, "validation": result, "cache_rejected": True}
        return {**state, "validation": result}

    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute validated SELECT-only query against PostgreSQL."""
        trace("\n" + "─" * 70)
        trace("⚡  NODE: execute_sql")
        trace("─" * 70)
        # Safety LIMIT is a UI guardrail; skip it when benchmarking so the
        # recorded SQL is exactly what the model produced (config sql.enforce_limit).
        sql = (
            enforce_limit(state["best_sql"])
            if getattr(self, "_enforce_limit", True)
            else state["best_sql"]
        )
        trace("  SQL to execute:")
        for line in sql.splitlines():
            trace(f"    {line}")
        try:
            if state.get("vote_rows") is not None and state.get("vote_rows_sql") == sql:
                trace("  ♻️  Reusing execution-vote result (winner already ran once)")
                rows = state["vote_rows"]
            else:
                rows = execute_query(
                    sql, timeout_ms=getattr(self, "_stmt_timeout_ms", 15000))
            row_count = len(rows) if isinstance(rows, list) else "?"
            trace(f"  ✅ Returned {row_count} row(s)")
            if isinstance(rows, list) and rows:
                preview = rows[:3]
                trace(f"  Preview (first {len(preview)}):")
                for r in preview:
                    trace(f"    {dict(r)}")
            return {**state, "best_sql": sql, "execution_result": rows, "error": None}
        except Exception as e:
            trace(f"  ❌ Execution error: {e}")
            logger.error(f"[EXECUTE] {e}")
            return {**state, "execution_result": None, "error": str(e)}

    def refine_sql(self, state: AgentState) -> AgentState:
        """
        ReAct refine step: the last query errored or returned 0 rows. Feed the
        failed SQL + the observation back through the regeneration prompt so the
        LLM can fix the specific problem, then re-validate and re-execute.
        """
        level = state.get("refine_level", 0) + 1
        trace("\n" + "─" * 70)
        trace(f"🔁  NODE: refine_sql  (round {level}/{self._refine_limit(state)})")
        trace("─" * 70)

        # Build the observation the model must react to.
        if state.get("error"):
            observation = f"The query raised an execution error: {state['error']}"
        else:
            observation = (
                "The query is valid and executed successfully but returned 0 rows. "
                "If a filter, JOIN, or value/format looks too strict or wrong, fix it; "
                "if the query already looks correct, return it unchanged."
            )
        trace(f"  Observation: {observation}")
        trace("  Failed SQL:")
        for line in (state.get("best_sql") or "").splitlines():
            trace(f"    {line}")

        sql_agent = self._get_sql_agent()
        try:
            sqls = sql_agent.regenerate(
                query=state["user_query"],
                schema_context=state["schema_context"],
                original_sql=state["best_sql"],
                error=observation,
                history=state["conversation_history"][-4:],
            )
            best = sql_agent.rerank(sqls, state["user_query"], state["schema_context"]) if sqls else ""
        except Exception as exc:
            trace(f"  ❌ Refine failed: {exc}")
            logger.warning(f"[REFINE] regeneration failed: {exc}")
            return {**state, "refine_level": level}

        if not best:
            trace("  ⚠️  Refine produced no SQL — giving up, will format prior result.")
            return {**state, "refine_level": level}

        trace("  ✏️  Refined SQL:")
        for line in best.splitlines():
            trace(f"    {line}")
        # Clear the prior error so the refined SQL gets a clean validate→execute pass.
        return {**state, "best_sql": best, "generated_sqls": sqls,
                "refine_level": level, "error": None}

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
            answer = invoke_tracked(self._get_llm(), messages, "handle_general").content
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

    def handle_intent_error(self, state: AgentState) -> AgentState:
        """
        classify_intent's LLM call failed outright (invalid model, auth, rate
        limit, network) — surface that plainly instead of silently presenting
        it as a normal "your question is ambiguous" clarification request,
        which would hide a broken model/connection behind confusing UI text.
        """
        response = (
            f"⚠️ I couldn't process that: {state.get('error') or 'the model call failed.'} "
            "If you picked a custom model, double-check it's a valid, "
            "accessible OpenAI model id."
        )
        turn_id = self.memory.add(
            session_id=state["session_id"],
            user_query=state["user_query"],
            response=response,
            sql=None,
            valid=None,
        )
        return {**state, "response": response, "turn_id": turn_id}

    def assess_risk(self, state: AgentState) -> AgentState:
        """
        GH#69: gold-free checks on the executed result — the signals that do
        NOT depend on candidate agreement.

        Phase 1's agreement signal cannot see the dominant failure: 41 of
        gpt-5-mini's 57 BIRD errors happened on turns where every candidate
        agreed, wrong together. These checks look at the query and its rows
        directly instead.

        Annotates only. Nothing here suppresses, rewrites or retries the
        answer, and a failure inside the checks costs nothing but the flags.
        """
        if not getattr(self, "_risk_checks", True):
            return state
        try:
            from backend.agents import risk as risk_mod
            sql = state.get("best_sql") or ""
            tables = referenced_tables(sql, getattr(self, "_dialect", "postgres"))
            out = risk_mod.assess(
                sql=sql,
                rows=state.get("execution_result"),
                question=state.get("standalone_query") or state.get("user_query") or "",
                schema_context=state.get("schema_context") or "",
                query_plan=state.get("query_plan"),
                tables=tables,
                value_index=getattr(self, "value_index", None),
            )
        except Exception as exc:
            logger.warning(f"[RISK] assessment skipped: {exc}")
            return state

        risks = out.get("risks") or []
        if risks:
            trace("\n" + "─" * 70)
            trace(f"⚠️   NODE: assess_risk — {len(risks)} flag(s)")
            for r in risks:
                trace(f"     [{r['severity']}] {r['code']}: {r['message']}")
            logger.info("[RISK] %s flag(s): %s", len(risks),
                        ", ".join(r["code"] for r in risks))

        ev = dict(state.get("verification") or {})
        ev["risks"] = risks
        ev["risk_score"] = out.get("risk_score", 0)
        return {**state, "verification": ev}

    def format_response(self, state: AgentState) -> AgentState:
        """Format execution results into natural language response."""
        trace("\n" + "─" * 70)
        trace("📝  NODE: format_response")
        trace("─" * 70)
        rows_hash = rows_digest(state.get("execution_result"))
        if state.get("error"):
            response = f"I encountered an error executing the query: {state['error']}"
            csv_path = None
        elif self._can_reuse_answer(state, rows_hash):
            # GH#57: the cached SQL was re-validated and re-executed above and
            # returned exactly the same rows, so the stored wording still holds
            # — the whole turn costs zero LLM calls.
            trace("  ♻️  Rows unchanged since the cached answer — reusing it")
            logger.info("[CACHE] answer reused (rows unchanged)")
            response, csv_path = state["cached_response"], None
        else:
            response, csv_path = self._get_optimizer().format(
                query=state["user_query"],
                sql=state["best_sql"],
                results=state["execution_result"],
            )
            # GH#75: the formatter is given the rows but not the doubts, so it
            # wrote "Here are the customers from Wakanda:" directly above a
            # warning saying Wakanda does not exist. Prepending the caveat
            # deterministically rather than asking the LLM to hedge — a model
            # told to add a caveat can decline; a string cannot.
            response = _prepend_caution(response, state.get("verification"))

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
        return {**state, "response": response, "csv_path": csv_path,
                "turn_id": turn_id, "rows_fingerprint": rows_hash}

    @staticmethod
    def _verification_block(final: Dict[str, Any]) -> Dict[str, Any]:
        """
        GH#67: the evidence behind this turn's answer, assembled for the API.

        Always returns a well-formed block so no consumer has to special-case a
        missing one. Turns that never generate — a cache hit, a blocked write,
        a clarifying question — are labelled by what actually happened rather
        than reported as zero agreement, which would read as "5 candidates
        disagreed" when nothing was voted on at all.
        """
        stored = final.get("verification")
        # Defaults are merged UNDER whatever was recorded, per key rather than
        # all-or-nothing: assess_risk (GH#69) writes risks into this block on
        # paths where generate_sql never ran — a cache hit — leaving it a
        # partial dict that consumers would otherwise find missing the Phase 1
        # counts entirely.
        ev = {"candidates": 0, "executed": 0, "agreement": 0,
              "distinct_result_sets": 0,
              "decided_by": "cache_hit" if final.get("cache_hit") else "not_generated",
              "risks": [], "risk_score": 0}
        if isinstance(stored, dict):
            ev.update(stored)
            # A cached SQL that got re-validated and re-executed was not chosen
            # by this turn's voting, whatever generate_sql last recorded.
            if final.get("cache_hit"):
                ev["decided_by"] = "cache_hit"
            elif "decided_by" not in stored:
                ev["decided_by"] = "not_generated"
        validation = final.get("validation") or {}
        ev["refine_rounds"] = int(final.get("refine_level") or 0)
        ev["validated"] = bool(validation.get("valid"))
        ev["validation_checks"] = {
            k: bool(v) for k, v in validation.items()
            if k in ("syntax_ok", "safety_ok", "semantic_ok")
        }
        return ev

    def _can_reuse_answer(self, state: AgentState, rows_hash: Optional[str]) -> bool:
        """A stored answer is reusable only if THESE rows are byte-identical to it."""
        return bool(
            getattr(self, "_answer_reuse", True)
            and state.get("cached_response")
            and rows_hash
            and state.get("cached_fingerprint") == rows_hash
        )

    def fallback_handler(self, state: AgentState) -> AgentState:
        """Tiered fallback: broader schema search → ask for clarification."""
        trace("\n" + "─" * 70)
        level = state.get("fallback_level", 0) + 1
        trace(f"⚠️   NODE: fallback_handler → level {level}")
        if state.get("validation", {}).get("error"):
            trace(f"  Reason: {state['validation']['error']}")
        trace("─" * 70)

        if level == 1:
            # Broaden schema scan
            # Same search text retrieval used, so the reused vector matches it.
            search_q = state.get("standalone_query") or state["user_query"]
            schema_chunks = retrieve(search_q, getattr(self, "_db_name", "FYP"),
                                     top_k_tables=10, store="pgvector",
                                     embedding=state.get("query_embedding"))
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
        # A classify_intent call FAILURE (not a genuinely unclear question)
        # must never be presented as an ordinary clarification request —
        # checked before intent, since a failed call still defaults intent to
        # AMBIGUOUS as a safe routing value.
        if state.get("error"):
            return "handle_intent_error"
        intent = state.get("intent", "READ")
        if intent == "WRITE_BLOCKED":
            return "handle_write_blocked"
        if intent == "GENERAL":
            return "handle_general"
        if intent == "AMBIGUOUS" and getattr(self, "_clarify", True):
            return "ask_clarification"   # GH#40: ask first, don't burn the pipeline
        return "retrieve_schema"  # READ (and AMBIGUOUS when clarification is off)

    def route_after_validation(self, state: AgentState) -> str:
        v = state.get("validation", {})
        if v.get("valid"):
            return "execute_sql"
        if state.get("cache_hit"):
            return "retrieve_schema"  # stale cached SQL → full retrieval, then generation
        if state.get("fallback_level", 0) < 2:
            return "fallback_handler"
        return "format_response"  # give up gracefully

    def route_after_retrieval(self, state: AgentState) -> str:
        if state.get("cache_hit") and state.get("best_sql"):
            return "validate_sql"     # GH#38 shortcut: skip linking + generation
        if not state.get("schema_context") and state.get("fallback_level", 0) == 0:
            return "fallback_handler"
        # GH#75: the question names a value that does not exist. Generating
        # here is what produced the substituted filter — the model cannot
        # satisfy the request, so it satisfies a different one.
        if state.get("unsatisfiable"):
            return "handle_unanswerable"
        return "generate_sql"

    def _refine_limit(self, state: AgentState) -> int:
        """Max ReAct refine rounds — deeper for the challenging tier (GH#49)."""
        if state.get("tier") == "challenging":
            return getattr(self, "_max_refine_challenging", 2)
        return getattr(self, "_max_refine", self.MAX_REFINE_ROUNDS)

    def route_after_execution(self, state: AgentState) -> str:
        """ReAct: refine on an execution error or an empty result set."""
        if state.get("refine_level", 0) >= self._refine_limit(state):
            return "format_response"
        rows = state.get("execution_result")
        has_error = bool(state.get("error"))
        zero_rows = isinstance(rows, list) and len(rows) == 0
        if has_error or zero_rows:
            return "refine_sql"
        return "format_response"

    def route_entry(self, state: AgentState) -> str:
        """Enter at validation when run() already resolved the SQL from cache."""
        return "validate_sql" if state.get("cache_hit") and state.get("best_sql") \
            else "classify_intent"

    # ── Graph Builder ───────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("classify_intent", _staged("classify_intent", self.classify_intent))
        g.add_node("condense_query", _staged("condense_query", self.condense_query))
        g.add_node("ask_clarification", _staged("ask_clarification", self.ask_clarification))
        g.add_node("retrieve_schema", _staged("retrieve_schema", self.retrieve_schema))
        g.add_node("link_schema", _staged("link_schema", self.link_schema))
        g.add_node("generate_sql", _staged("generate_sql", self.generate_sql))
        g.add_node("validate_sql", _staged("validate_sql", self.validate_sql))
        g.add_node("execute_sql", _staged("execute_sql", self.execute_sql))
        g.add_node("refine_sql", _staged("refine_sql", self.refine_sql))
        g.add_node("handle_unanswerable",
                   _staged("handle_unanswerable", self.handle_unanswerable))
        g.add_node("assess_risk", _staged("assess_risk", self.assess_risk))
        g.add_node("format_response", _staged("format_response", self.format_response))
        g.add_node("handle_general", _staged("handle_general", self.handle_general))
        g.add_node("handle_write_blocked", _staged("handle_write_blocked", self.handle_write_blocked))
        g.add_node("handle_intent_error", _staged("handle_intent_error", self.handle_intent_error))
        g.add_node("fallback_handler", _staged("fallback_handler", self.fallback_handler))

        # GH#57: run() pre-resolves a repeat question against the semantic cache
        # and marks the state; that skips intent classification and condensation
        # on top of the linking/generation the cache shortcut already skipped.
        g.set_conditional_entry_point(self.route_entry, {
            "classify_intent": "classify_intent",
            "validate_sql": "validate_sql",
        })

        g.add_conditional_edges("classify_intent", self.route_by_intent, {
            "retrieve_schema": "condense_query",   # READ path condenses follow-ups first
            "handle_general": "handle_general",
            "handle_write_blocked": "handle_write_blocked",
            "ask_clarification": "ask_clarification",
            "handle_intent_error": "handle_intent_error",
        })
        g.add_edge("condense_query", "retrieve_schema")
        g.add_edge("ask_clarification", END)
        g.add_edge("handle_intent_error", END)

        # Happy path routes through schema-linking (#17) before generation;
        # a cache hit (GH#38) jumps straight to validation.
        g.add_conditional_edges("retrieve_schema", self.route_after_retrieval, {
            "generate_sql": "link_schema",
            "fallback_handler": "fallback_handler",
            "validate_sql": "validate_sql",
            "handle_unanswerable": "handle_unanswerable",
        })
        g.add_edge("handle_unanswerable", END)

        g.add_edge("link_schema", "generate_sql")
        g.add_edge("generate_sql", "validate_sql")

        g.add_conditional_edges("validate_sql", self.route_after_validation, {
            "execute_sql": "execute_sql",
            "fallback_handler": "fallback_handler",
            "format_response": "format_response",
            "retrieve_schema": "retrieve_schema",   # stale cache hit → regenerate
        })

        # ReAct: after execution, refine once on error/empty result, else format.
        # GH#69: risk assessment sits on the answer path, AFTER any refine
        # rounds, so it judges the result the user will actually be shown.
        g.add_conditional_edges("execute_sql", self.route_after_execution, {
            "refine_sql": "refine_sql",
            "format_response": "assess_risk",
        })
        g.add_edge("assess_risk", "format_response")
        # Refined SQL re-enters validation (safety must run on it) → execute.
        g.add_edge("refine_sql", "validate_sql")

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
    def run(self, session_id: str, user_query: str, api_key: str | None = None,
            model: str | None = None, stage_callback=None) -> Dict[str, Any]:
        """
        Run the full pipeline for one user turn.

        If ``api_key`` is provided it is used in place of the system default
        OpenAI key for all LLM calls in this request (thread-safe). If
        ``model`` is provided and differs from the configured default
        (GH#52), it overrides the model used for all LLM calls in this
        request the same way. Selecting the already-default model is a no-op
        (no extra ``ChatOpenAI`` construction).

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
        # Only branch when a custom key/model is actually provided; otherwise
        # leave _tls untouched and read via getattr(..., self.attr) fallback.
        # Selecting the already-default model is treated as no override (GH#52)
        # so the common case never pays for an extra ChatOpenAI construction.
        has_override = bool(
            (api_key and api_key.strip())
            or (model and model.strip() and model != self._model)
        )
        # The model actually used for this request's LLM calls — distinct from
        # self._model (the static configured default) whenever a per-request
        # override is active. Threaded through to _update_cache so the
        # query_interactions audit row's `model` column matches the model the
        # tokens/cost in that same row were actually billed against.
        effective_model = model.strip() if (model and model.strip()) else self._model
        if has_override:
            # Must go through _chat_llm too (GH#65): a bare ChatOpenAI here
            # would silently send a bring-your-own-key request to OpenAI while
            # every other stage talks to the configured provider.
            _tls.llm = self._chat_llm(
                effective_model, api_key or self._default_api_key)
            _tls.intent_llm = _tls.llm
            _tls.sql_agent = SQLCodeAgent(self._config, _tls.llm)
            _tls.linker = SchemaLinkingAgent(self._config, _tls.llm)
            _tls.optimizer = ResponseOptimizer(_tls.llm)

        # Prefetch the query embedding concurrently with the intent LLM call.
        try:
            _tls.embed_future = _PREFETCH_POOL.submit(
                self.semantic_cache.embed, user_query)
        except Exception:
            _tls.embed_future = None

        history = self.memory.get_history(session_id)
        try:
            previous_sql = self.memory.get_last_sql(session_id)
        except Exception:
            previous_sql = None

        # Consumed here (not in retrieve_schema) so the fast-path probe can use
        # it; the value rides into the graph via state so nothing re-embeds.
        query_embedding = self._consume_prefetched_embedding()
        cache_probed = bool(getattr(self, "_fast_path", True)) and \
            not self._looks_anaphoric(user_query, history)
        cache_pair = self._cache_lookup(query_embedding, user_query) if cache_probed else None
        fast = self._fast_path_state(cache_pair) if cache_pair else None
        initial_state: AgentState = {
            "session_id": session_id,
            "user_query": user_query,
            "conversation_history": history,
            "intent": "READ" if fast else "",
            "schema_context": fast["schema_context"] if fast else "",
            "query_plan": None,
            "few_shot_examples": [],
            "query_embedding": query_embedding,
            "generated_sqls": [],
            "best_sql": fast["best_sql"] if fast else "",
            "validation": {},
            "execution_result": None,
            "response": "",
            "turn_id": None,
            "csv_path": None,
            "standalone_query": None,
            "previous_sql": previous_sql,
            "cache_hit": bool(fast),
            "cached_response": fast["cached_response"] if fast else None,
            "cached_fingerprint": fast["cached_fingerprint"] if fast else None,
            "rows_fingerprint": None,
            "cache_rejected": False,
            "cache_probed": cache_probed,
            "cache_pair": cache_pair,
            "vote_rows": None,
            "vote_rows_sql": None,
            "verification": None,
            "unsatisfiable": [],
            "retrieved_tables": [],
            "fallback_level": 0,
            "refine_level": 0,
            "tier": None,
            "timings": {},
            "error": None,
        }
        _tls.stage_cb = stage_callback
        tracker = usage_tracking.start_request()
        t0 = time.perf_counter()
        try:
            final = self.graph.invoke(initial_state)
        finally:
            # Clear the per-request overrides so the thread can be reused cleanly.
            # Only needed when we set them.
            _tls.stage_cb = None
            if has_override:
                _tls.llm = self.llm
                _tls.intent_llm = self.intent_llm
                _tls.sql_agent = self.sql_agent
                _tls.linker = self.linker
                _tls.optimizer = self.optimizer
        latency_ms = int((time.perf_counter() - t0) * 1000)
        trace("\n" + "━" * 70)
        trace(f"✅  PIPELINE DONE  |  latency={latency_ms}ms  |  intent={final.get('intent')}")
        trace("━" * 70 + "\n")

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
            "cache_hit":  bool(final.get("cache_hit")),
            # GH#63: what the follow-up was condensed into, so a bad rewrite is
            # visible rather than silently producing a confusing answer.
            "standalone_query": final.get("standalone_query"),
            # GH#67: the evidence behind this answer, not just the answer.
            "verification": self._verification_block(final),
            "latency_ms": latency_ms,
            "timings":    final.get("timings") or {},
            "usage":      tracker.as_dict(),
        }

        # Semantic cache write-path + interaction log (best-effort, never fatal).
        self._update_cache(session_id, user_query, final, row_count, latency_ms,
                           usage=result["usage"], model=effective_model)
        return result

    def _update_cache(
        self,
        session_id: str,
        user_query: str,
        final: Dict[str, Any],
        row_count: Optional[int],
        latency_ms: int,
        usage: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> None:
        """Persist a successful query→SQL pair and log the interaction."""
        validation = final.get("validation", {})
        best_sql = final.get("best_sql")
        fingerprint = final.get("rows_fingerprint")
        success = (
            validation.get("valid")
            and not final.get("error")
            and bool(best_sql)
            and isinstance(final.get("execution_result"), list)
            # A reused pair is already stored — rewrite it only when the rows
            # (and therefore the answer) have moved on, so the next hit can
            # reuse the fresh wording instead of re-formatting forever.
            and (not final.get("cache_hit")
                 or fingerprint != final.get("cached_fingerprint"))
            # GH#75: never memorise an answer the system itself flagged. Without
            # this the "customers from Wakanda" answer was stored and replayed
            # from cache with zero LLM calls — the pipeline detected its own
            # error and then learned it, so every later ask got the same wrong
            # rows faster and the model never got another chance to do better.
            and not (final.get("verification") or {}).get("risk_score")
        )
        # GH#63: store the CONDENSED question when there is one. The embedding
        # written alongside it is the condensed text's (retrieve_schema
        # re-embeds after condense_query), so storing the raw follow-up here
        # left text and vector describing different questions — and made the
        # pair useless as a few-shot ("Q: what about 2014?"). The raw text is
        # still the audit record, so log_interaction below keeps it verbatim.
        cache_query = final.get("standalone_query") or user_query
        try:
            if success:
                self.semantic_cache.save_pair(
                    user_query=cache_query,
                    sql=best_sql,
                    embedding=final.get("query_embedding"),
                    rows_fingerprint=fingerprint,
                    response=final.get("response") or None,
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
                model=model or getattr(self, "_model", None),
                tokens_in=(usage or {}).get("input_tokens"),
                tokens_out=(usage or {}).get("output_tokens"),
                cost_usd=(usage or {}).get("est_cost_usd"),
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
        """LLM summarizer injected into SessionMemory for rolling context.

        Known limitation (GH#52, pre-existing for api_key too): this calls
        self.llm directly rather than self._get_llm(), so a per-request
        api_key/model override never reaches the rolling summarizer — it
        always runs on the configured default. Accepted, not fixed here: this
        runs off an eviction callback, not the per-turn request path.
        """
        prompt = (
            "Compress the following earlier conversation turns into a concise "
            "context note (max 4 sentences) that preserves entities, filters, "
            "numbers, and any SQL intent needed to understand follow-up questions.\n\n"
        )
        if prior_summary:
            prompt += f"Existing summary:\n{prior_summary}\n\n"
        prompt += f"New turns to fold in:\n{evicted_text}\n\nUpdated summary:"
        return invoke_tracked(self.llm, [
            SystemMessage(content="You faithfully compress conversation history."),
            HumanMessage(content=prompt),
        ], "summarize_history").content.strip()

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
    trace(json.dumps(result, indent=2, default=str))
