"""Pydantic request/response models for the FastAPI server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    sql: Optional[str] = None
    intent: str
    validation: Dict[str, Any] = Field(default_factory=dict)
    row_count: Optional[int] = None
    csv_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
