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

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.agents.db_agent import OrchestrationAgent
from backend.api.models import (
    ChatRequest, ChatResponse, HealthResponse, SessionResponse,
    SchemaResponse, SessionSummary,
)
from backend.config.config_loader import load_config
from backend.pipelines.postprocessing import ResponseOptimizer
from backend.retrieval.data_processing import get_schema_info

logger = logging.getLogger(__name__)

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


def _get_agent() -> OrchestrationAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")
    return _agent


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
    """
    agent = _get_agent()
    try:
        result = agent.run(
            session_id=req.session_id,
            user_query=req.message,
            api_key=req.api_key or None,
        )
    except Exception as exc:
        logger.exception("Agent error for session %s", req.session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(**result)


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
