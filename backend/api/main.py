"""
backend/api/main.py — FastAPI HTTP server
-----------------------------------------
Exposes OrchestrationAgent over HTTP. The agent is instantiated ONCE at
startup via FastAPI lifespan and reused across all requests.

Run:
    uvicorn backend.api.main:app --reload --port 8000

Endpoints:
    GET  /health                    → service + DB liveness check
    POST /session                   → create a new session_id
    POST /chat                      → send a message, get a structured answer
    GET  /history/{session_id}      → return turn history for a session
    GET  /download/{filename}       → serve a CSV from outputs/
    GET  /schema                    → return table + column schema for the DB
"""

from __future__ import annotations

import contextvars
import json
import logging
import queue
import re
import threading
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.agents.db_agent import OrchestrationAgent
from backend.api.models import (
    ChatRequest, ChatResponse, FeedbackRequest, HealthResponse, SessionResponse,
    SchemaResponse, SessionSummary, SchemaDiffResponse, SchemaSyncRequest,
    SchemaSyncResponse, ModelsResponse,
)
from backend.config.config_loader import load_config
from backend.pipelines.postprocessing import ResponseOptimizer
from backend.retrieval.data_processing import get_schema_info

# ── request correlation (GH#42) ───────────────────────────────────────────────
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Stamp every log record with the current request id (or '-')."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s [%(request_id)s] — %(message)s",
)
for _h in logging.getLogger().handlers:
    _h.addFilter(RequestIdFilter())
logger = logging.getLogger(__name__)

# ── rolling in-memory metrics (GH#42) — reset on restart, demo-grade ──────────
_METRICS_LOCK = threading.Lock()
_METRICS = {
    "turns": 0,
    "errors": 0,
    "cache_hits": 0,
    "latencies_ms": deque(maxlen=200),
    "stage_ms": defaultdict(int),
    "stage_n": defaultdict(int),
    # GH#57 cost accounting
    "llm_calls": 0,
    "tokens_in": 0,
    "tokens_out": 0,
    "cost_usd": 0.0,
}


def _record_turn(result: dict) -> None:
    with _METRICS_LOCK:
        _METRICS["turns"] += 1
        if result.get("cache_hit"):
            _METRICS["cache_hits"] += 1
        if result.get("latency_ms") is not None:
            _METRICS["latencies_ms"].append(int(result["latency_ms"]))
        for stage, ms in (result.get("timings") or {}).items():
            _METRICS["stage_ms"][stage] += int(ms)
            _METRICS["stage_n"][stage] += 1
        usage = result.get("usage") or {}
        _METRICS["llm_calls"] += int(usage.get("calls") or 0)
        _METRICS["tokens_in"] += int(usage.get("input_tokens") or 0)
        _METRICS["tokens_out"] += int(usage.get("output_tokens") or 0)
        _METRICS["cost_usd"] += float(usage.get("est_cost_usd") or 0.0)

# ── outputs directory — single source of truth shared with ResponseOptimizer ───
_OUTPUTS_DIR = ResponseOptimizer._DEFAULT_OUTPUT_DIR

# ── app-level state ────────────────────────────────────────────────────────────
_agent: OrchestrationAgent | None = None
_db_config: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and initialise the agent once at startup."""
    global _agent, _db_config
    config = load_config()
    _db_config = config.get("db", {})
    _agent = OrchestrationAgent(config)
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("OrchestrationAgent ready.")
    yield
    logger.info("Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Conversational Database Assistant",
    description="Read-only Text-to-SQL assistant powered by LangGraph + pgvector.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _check_db() -> bool:
    """Quick liveness check: can we connect to Postgres?"""
    if not _db_config:
        return False
    try:
        conn = psycopg2.connect(
            dbname=_db_config.get("name", "FYP"),
            user=_db_config.get("user", "postgres"),
            password=_db_config.get("password"),
            host=_db_config.get("host", "127.0.0.1"),
            port=_db_config.get("port", 5432),
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


def _check_vector_index() -> bool:
    """Is the schema vector index populated? False on any error (degraded mode)."""
    if not _db_config:
        return False
    try:
        conn = psycopg2.connect(
            dbname=_db_config.get("name", "FYP"),
            user=_db_config.get("user", "postgres"),
            password=_db_config.get("password"),
            host=_db_config.get("host", "127.0.0.1"),
            port=_db_config.get("port", 5432),
            connect_timeout=2,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM schema_embeddings;")
                return (cur.fetchone()[0] or 0) > 0
        finally:
            conn.close()
    except Exception:
        return False


def _get_agent() -> OrchestrationAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")
    return _agent


# GH#58: OpenAI model ids are lowercase-alnum with '.', '_', '-', ':' separators
# (e.g. "gpt-4o-mini", "ft:gpt-4o-mini-2024-07-18:org::abc123"). This is a shape
# check, not an allow-list — it exists to reject garbage/injection attempts, not
# to restrict which real models a user may type.
# `/` is allowed (GH#65) because open-weight ids are vendor-namespaced —
# meta-llama/Llama-3.3-70B-Instruct, deepseek-ai/…, Qwen/…, google/… — and
# rejecting them here blocked OpenRouter and Together entirely. Still anchored
# and charset-limited: the id is only ever a request payload field, never a
# path segment or filename, so no traversal surface is opened.
_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{1,79}$")


def _validate_model(req_model: Optional[str], agent: OrchestrationAgent) -> Optional[str]:
    """
    Accept a model from the curated `selectable_models` list, OR any
    well-formed model id (GH#58) — the frontend's picker offers a free-text
    "Custom…" entry, matching the posture the eval CLIs already take
    (backend/eval/bird.py::resolve_model warns on an unlisted model but never
    blocks it). Reaching OrchestrationAgent.run() with an unusable model id
    still fails safely there — it becomes an OpenAI API error surfaced to the
    caller, not a silent misroute.
    """
    if not req_model:
        return None
    req_model = req_model.strip()
    if req_model in agent.selectable_models:
        return req_model
    if not _MODEL_ID_RE.match(req_model):
        raise HTTPException(status_code=400, detail=f"Invalid model id: {req_model!r}")
    return req_model


@app.middleware("http")
async def _request_id_middleware(request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = _request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        _request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Service and DB liveness check including optional component status."""
    agent = _agent  # may be None during startup
    cache_enabled = bool(agent and agent.semantic_cache.is_enabled)
    memory_persistent = bool(agent and agent.memory.is_persistent)
    return HealthResponse(
        status="ok",
        db_connected=_check_db(),
        vector_index=_check_vector_index(),
        cache_enabled=cache_enabled,
        memory_persistent=memory_persistent,
    )


@app.post("/session", response_model=SessionResponse, tags=["session"])
def create_session() -> SessionResponse:
    """Create a new session and return its ID."""
    return SessionResponse(session_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """
    Process one user message and return a structured response.

    The agent classifies intent, retrieves schema, generates + validates SQL,
    executes it, and formats the answer.

    If ``api_key`` is supplied in the request body it overrides the system
    default OpenAI key for this request only — it is never stored server-side.
    ``model`` (GH#52) similarly overrides the LLM model for this request only;
    it must be one of the models returned by ``GET /models``.
    """
    agent = _get_agent()
    model = _validate_model(req.model, agent)
    try:
        result = agent.run(
            session_id=req.session_id,
            user_query=req.message,
            api_key=req.api_key or None,
            model=model,
        )
    except Exception as exc:
        logger.exception("Agent error for session %s", req.session_id)
        with _METRICS_LOCK:
            _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    _record_turn(result)
    return ChatResponse(**result)


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Streaming variant of /chat (GH#39): Server-Sent Events with live pipeline
    progress so the user isn't staring at a spinner through 4–6 LLM calls.

    Event stream (each line `data: <json>\\n\\n`):
        {"type": "stage", "stage": "<node name>"}   — pipeline stage entered
        {"type": "sql",   "sql": "..."}            — winning SQL selected
        {"type": "done",  ...ChatResponse fields}  — terminal success event
        {"type": "error", "message": "..."}        — terminal failure event
    Always ends with a terminal event — never a silently dropped connection.
    """
    agent = _get_agent()
    # Validated before the worker thread starts (GH#52) so an unlisted model
    # returns a synchronous HTTP 400 rather than a mid-stream `error` event.
    model = _validate_model(req.model, agent)
    q: "queue.Queue[Optional[dict]]" = queue.Queue()

    def _cb(event_type: str, payload: Any) -> None:
        if event_type == "stage":
            q.put({"type": "stage", "stage": str(payload)})
        elif event_type == "sql":
            q.put({"type": "sql", "sql": str(payload)})

    rid = _request_id_var.get()   # contextvars don't cross manual threads

    def _worker() -> None:
        token = _request_id_var.set(rid)
        try:
            result = agent.run(
                session_id=req.session_id,
                user_query=req.message,
                api_key=req.api_key or None,
                model=model,
                stage_callback=_cb,
            )
            _record_turn(result)
            q.put({"type": "done", **result})
        except Exception as exc:  # terminal error event, then close
            logger.exception("Agent stream error for session %s", req.session_id)
            with _METRICS_LOCK:
                _METRICS["errors"] += 1
            q.put({"type": "error", "message": str(exc)})
        finally:
            _request_id_var.reset(token)
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    def _gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/models", response_model=ModelsResponse, tags=["chat"])
def list_models() -> ModelsResponse:
    """Models the frontend's per-request picker (GH#52) may choose from."""
    agent = _get_agent()
    return ModelsResponse(models=agent.selectable_models, default=agent.default_model)


@app.get("/metrics-lite", tags=["ops"])
def metrics_lite() -> Dict[str, Any]:
    """Rolling in-memory pipeline metrics (GH#42). Reset on restart."""
    with _METRICS_LOCK:
        lats = sorted(_METRICS["latencies_ms"])
        n = len(lats)
        avg_stage = {
            s: round(_METRICS["stage_ms"][s] / _METRICS["stage_n"][s])
            for s in _METRICS["stage_ms"] if _METRICS["stage_n"][s]
        }
        return {
            "turns": _METRICS["turns"],
            "errors": _METRICS["errors"],
            "cache_hits": _METRICS["cache_hits"],
            "cache_hit_rate": round(_METRICS["cache_hits"] / _METRICS["turns"], 3)
            if _METRICS["turns"] else 0.0,
            "latency_avg_ms": round(sum(lats) / n) if n else None,
            "latency_p95_ms": lats[max(0, int(n * 0.95) - 1)] if n else None,
            "avg_stage_ms": avg_stage,
            "llm_calls": _METRICS["llm_calls"],
            "llm_calls_per_turn": round(_METRICS["llm_calls"] / _METRICS["turns"], 2)
            if _METRICS["turns"] else 0.0,
            "tokens_in": _METRICS["tokens_in"],
            "tokens_out": _METRICS["tokens_out"],
            "cost_usd": round(_METRICS["cost_usd"], 6),
            "cost_usd_per_turn": round(_METRICS["cost_usd"] / _METRICS["turns"], 6)
            if _METRICS["turns"] else 0.0,
        }


@app.get("/sessions", response_model=List[SessionSummary], tags=["session"])
def list_sessions() -> List[SessionSummary]:
    """Return all known sessions (newest first) for the sidebar."""
    agent = _get_agent()
    return [SessionSummary(**s) for s in agent.memory.list_sessions()]


@app.get("/history/{session_id}", response_model=List[Dict[str, Any]], tags=["session"])
def get_history(session_id: str) -> List[Dict[str, Any]]:
    """Return the full turn history for a session."""
    agent = _get_agent()
    return agent.memory.get_history(session_id)


@app.get("/download/{filename}", tags=["files"])
def download_csv(filename: str) -> FileResponse:
    """
    Serve a CSV file from the outputs/ directory.

    Path-traversal guard: only files directly inside outputs/ are served.
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = (_OUTPUTS_DIR / filename).resolve()

    if not str(filepath).startswith(str(_OUTPUTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=str(filepath),
        media_type="text/csv",
        filename=filename,
    )


@app.post("/feedback", tags=["feedback"])
def submit_feedback(req: FeedbackRequest) -> Dict[str, str]:
    """
    Record thumbs-up / thumbs-down feedback for an assistant turn.

    On a down-vote, the SQL pair that was used is marked ``low_conf`` in the
    semantic cache so it is no longer injected as a few-shot example.
    """
    agent = _get_agent()
    try:
        agent.semantic_cache.add_feedback(
            turn_id=req.turn_id,
            session_id=req.session_id,
            rating=req.rating,
            comment=req.comment,
        )
    except Exception as exc:
        logger.exception("Feedback error for turn %s", req.turn_id)
        raise HTTPException(status_code=500, detail=f"Feedback error: {exc}") from exc
    return {"status": "ok"}


@app.get("/schema", response_model=SchemaResponse, tags=["schema"])
def get_db_schema() -> SchemaResponse:
    """
    Return the table and column schema for the connected PostgreSQL database.

    Used by the frontend Database tab to show the live schema browser.
    """
    try:
        info = get_schema_info()
    except Exception as exc:
        logger.exception("Schema fetch failed")
        raise HTTPException(status_code=500, detail=f"Schema fetch failed: {exc}") from exc
    return SchemaResponse(**info)


def _versioner() -> "SchemaVersioningAgent":
    from backend.agents.schema_versioning_agent import SchemaVersioningAgent
    return SchemaVersioningAgent(
        db_name=(_db_config or {}).get("name"), db_config=_db_config,
    )


@app.get("/schema/diff", response_model=SchemaDiffResponse, tags=["schema"])
def schema_diff() -> SchemaDiffResponse:
    """Report schema changes + unindexed tables vs the last synced snapshot (Issue #7)."""
    try:
        return SchemaDiffResponse(**_versioner().check())
    except Exception as exc:
        logger.exception("Schema diff failed")
        raise HTTPException(status_code=500, detail=f"Schema diff failed: {exc}") from exc


@app.post("/schema/sync", response_model=SchemaSyncResponse, tags=["schema"])
def schema_sync(req: SchemaSyncRequest) -> SchemaSyncResponse:
    """
    Re-index the RAG store to match the live database (Issue #7). Surgical by
    default (only changed/unindexed tables); `full=true` re-embeds everything —
    the refresh for DATA changes. NOTE: no admin-auth guard yet (Issue #6/GH#7).
    """
    try:
        result = _versioner().sync(full=req.full, tables=req.tables)
    except Exception as exc:
        logger.exception("Schema sync failed")
        raise HTTPException(status_code=500, detail=f"Schema sync failed: {exc}") from exc
    return SchemaSyncResponse(**result)
