"""Pydantic request/response models for the FastAPI server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)
    api_key: Optional[str] = Field(None, description="Override the default OpenAI key for this request")
    model: Optional[str] = Field(None, description="Override the default LLM model for this request")
    db_scope: Optional[str] = Field(
        None,
        description="Answer this question against ONE logical database "
                    "(a name from GET /databases). Omitted/empty searches "
                    "every indexed scope.",
    )


class ModelsResponse(BaseModel):
    models: List[str]
    default: str
    # {model id: provider label}. The picker shows which endpoint a
    # choice actually reaches, since one deployment now serves OpenAI models
    # directly and others through a gateway.
    providers: Dict[str, str] = Field(default_factory=dict)


class DatabaseInfo(BaseModel):
    """One logical database (BIRD db_id) present in the schema vector index."""
    name: str
    tables: int
    active: bool


class DatabasesResponse(BaseModel):
    """Retrieval scopes available, and which one is in force."""
    databases: List[DatabaseInfo]
    active: str


class ChatResponse(BaseModel):
    response: str
    sql: Optional[str] = None
    intent: str
    validation: Dict[str, Any] = Field(default_factory=dict)
    row_count: Optional[int] = None
    rows: Optional[List[Dict[str, Any]]] = None   # actual DB rows sent to frontend
    csv_path: Optional[str] = None
    turn_id: Optional[int] = None
    cache_hit: bool = False   # answered from the semantic cache (no generation)
    # The endpoint that served this turn ("openai", "openrouter", …),
    # shown beside the model so a routed answer is identifiable at a glance.
    provider: Optional[str] = None
    # How this answer was arrived at — candidate agreement, how the
    # winner was decided, refine rounds, validation flags. Facts only; no
    # confidence verdict until the calibration study establishes one.
    verification: Optional[Dict[str, Any]] = None
    # The follow-up rewritten as a standalone question, surfaced so a
    # bad condensation is visible instead of silent. None on a first
    # turn or any question that needed no context.
    standalone_query: Optional[str] = None
    latency_ms: Optional[int] = None            # total turn wall-clock
    timings: Optional[Dict[str, int]] = None    # per-stage ms
    usage: Optional[Dict[str, Any]] = None      # tokens + est. cost


class FeedbackRequest(BaseModel):
    turn_id: int
    session_id: str
    rating: str = Field(..., pattern=r"^(up|down)$")
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    vector_index: bool = False
    cache_enabled: bool = False
    memory_persistent: bool = False


class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Schema browser models ──────────────────────────────────────────────────

class SchemaColumn(BaseModel):
    name: str
    type: str
    not_null: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    description: Optional[str] = None        # what the column represents
    value_description: Optional[str] = None  # what its values mean (optional)


class SchemaTable(BaseModel):
    table_name: str
    column_count: int
    columns: List[SchemaColumn]
    # The logical database (BIRD db_id) this table belongs to, so the
    # schema browser can group 75 tables from 11 domains instead of listing
    # them flat. None when no table catalog is configured.
    domain: Optional[str] = None


class SchemaResponse(BaseModel):
    db_name: str
    table_count: int
    tables: List[SchemaTable]


# ── Schema versioning / index sync ──────────────────────────────

class SchemaChange(BaseModel):
    action: str            # add_table|drop_table|add_column|drop_column|modify_column
    table: str
    column: Optional[str] = None
    detail: Optional[str] = None


class SchemaDiffResponse(BaseModel):
    db_name: str
    in_sync: bool
    last_synced_at: Optional[str] = None
    changes: List[SchemaChange] = Field(default_factory=list)
    unindexed_tables: List[str] = Field(default_factory=list)


class SchemaSyncRequest(BaseModel):
    full: bool = False                       # re-embed every table (data changed)
    tables: Optional[List[str]] = None       # restrict to these tables


class SchemaSyncResponse(BaseModel):
    db_name: str
    schema_hash: str
    changes_applied: List[SchemaChange] = Field(default_factory=list)
    reembedded_tables: List[str] = Field(default_factory=list)
    removed_tables: List[str] = Field(default_factory=list)
    # Semantic-cache cascade: dropped tables archive their
    # cached pairs, modified columns demote confidence — {table: count}.
    archived_pairs: Dict[str, int] = Field(default_factory=dict)
    demoted_pairs: Dict[str, int] = Field(default_factory=dict)
    full: bool = False
