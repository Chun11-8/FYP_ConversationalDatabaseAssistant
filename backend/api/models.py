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


class ModelsResponse(BaseModel):
    models: List[str]
    default: str


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
    latency_ms: Optional[int] = None            # total turn wall-clock (GH#42)
    timings: Optional[Dict[str, int]] = None    # per-stage ms (GH#42)


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


class SchemaResponse(BaseModel):
    db_name: str
    table_count: int
    tables: List[SchemaTable]


# ── Schema versioning / index sync (Issue #7) ──────────────────────────────

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
    full: bool = False
