from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Schema Versioning Trigger (Admin Hook)
# ─────────────────────────────────────────────

class SchemaVersioningAgent:
    """
    Lightweight hash-based schema drift detector.
    Called by admin hook or auto-triggered on mismatch.
    Full re-embedding only for affected tables (surgical update).
    """

    def __init__(self, config: dict, retriever):
        self.retriever = retriever
        self.config = config

    def check_and_update(self, current_schema_hash: str, cached_hash: str) -> Dict[str, Any]:
        if current_schema_hash == cached_hash:
            return {"updated": False, "reason": "No schema drift detected"}

        logger.info("[SCHEMA_VERSION] Drift detected — running surgical update")
        diff = self.retriever.db.get_schema_diff(cached_hash)

        results = {"updated": True, "changes": []}
        for change in diff:
            action = change["action"]  # add | modify | drop
            table = change["table"]

            if action == "add":
                self.retriever.embed_table(table)
                results["changes"].append(f"Added embeddings for: {table}")

            elif action == "modify":
                self.retriever.update_table_embedding(table)
                results["changes"].append(f"Updated embeddings for: {table}")

            elif action == "drop":
                self.retriever.remove_table_embedding(table)
                results["changes"].append(f"Removed embeddings for: {table}")

        self.retriever.persist_hash(current_schema_hash)
        return results