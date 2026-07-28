"""
schema_versioning_agent.py — Schema Versioning Layer (Issue #7 / GH#8).

Detects live-database schema changes (new/dropped tables, added/removed/
retyped columns) against the last synced snapshot, and SURGICALLY updates the
RAG knowledge store: only affected tables are re-embedded (chunk upserts are
keyed by chunk_id, so re-embedding updates in place), dropped tables have their
chunks deleted. A `full` sync re-embeds every table — the explicit answer for
DATA changes (sample rows and `values: {…}` lists refresh on re-embed; row-level
change detection would require hashing all data).

Replaces the previous non-functional stub. NOTE: the admin-auth guard
(Issue #6/GH#7) does not exist yet — the sync endpoint is currently open.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

Snapshot = Dict[str, Dict[str, str]]   # {table: {column: type}}


# ── pure functions ────────────────────────────────────────────────────────────

def hash_schema(snapshot: Snapshot) -> str:
    """Deterministic sha256 of a normalized schema snapshot."""
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()


def diff_schemas(old: Snapshot, new: Snapshot) -> List[Dict[str, str]]:
    """
    Typed change list between two snapshots:
        {action: add_table|drop_table|add_column|drop_column|modify_column,
         table, column?, detail?}
    """
    changes: List[Dict[str, str]] = []
    old_t, new_t = set(old), set(new)
    for t in sorted(new_t - old_t):
        changes.append({"action": "add_table", "table": t})
    for t in sorted(old_t - new_t):
        changes.append({"action": "drop_table", "table": t})
    for t in sorted(old_t & new_t):
        oc, nc = old[t], new[t]
        for c in sorted(set(nc) - set(oc)):
            changes.append({"action": "add_column", "table": t, "column": c})
        for c in sorted(set(oc) - set(nc)):
            changes.append({"action": "drop_column", "table": t, "column": c})
        for c in sorted(set(oc) & set(nc)):
            if oc[c] != nc[c]:
                changes.append({"action": "modify_column", "table": t, "column": c,
                                "detail": f"{oc[c]} → {nc[c]}"})
    return changes


def snapshot_from_tables(tables: List[dict]) -> Snapshot:
    """Normalize extract_schema_pg() output into a Snapshot."""
    return {
        t["table_name"]: {c["name"]: str(c["type"]) for c in t["columns"]}
        for t in tables
    }


# ── the agent ─────────────────────────────────────────────────────────────────

class SchemaVersioningAgent:
    """Snapshot store + diff + surgical re-embedding for one database."""

    def __init__(self, db_name: Optional[str] = None):
        self.db_name = db_name or os.getenv("PG_DB", "FYP")

    # ── infrastructure (thin wrappers — mocked in tests) ──────────────────────

    def _connect(self):
        from backend.retrieval.data_processing import connect_postgres
        conn = connect_postgres(self.db_name)
        if conn is None:
            raise RuntimeError(f"Cannot connect to database '{self.db_name}'")
        return conn

    def _init_store(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version_store (
                    id          SERIAL PRIMARY KEY,
                    schema_hash TEXT        NOT NULL,
                    snapshot    JSONB       NOT NULL,
                    diff_log    JSONB       NOT NULL DEFAULT '[]',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
        conn.commit()

    def current_snapshot(self) -> Snapshot:
        from backend.retrieval.data_processing import extract_schema_pg
        conn = self._connect()
        try:
            return snapshot_from_tables(extract_schema_pg(conn, self.db_name))
        finally:
            conn.close()

    def last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Latest stored {hash, snapshot, created_at} or None."""
        conn = self._connect()
        try:
            self._init_store(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT schema_hash, snapshot, created_at FROM schema_version_store "
                    "ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
            if not row:
                return None
            snap = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            return {"hash": row[0], "snapshot": snap,
                    "created_at": str(row[2]) if row[2] else None}
        finally:
            conn.close()

    def _persist(self, snapshot: Snapshot, changes: List[dict]) -> str:
        h = hash_schema(snapshot)
        conn = self._connect()
        try:
            self._init_store(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO schema_version_store (schema_hash, snapshot, diff_log) "
                    "VALUES (%s, %s, %s)",
                    (h, json.dumps(snapshot), json.dumps(changes)),
                )
            conn.commit()
        finally:
            conn.close()
        return h

    def _indexed_tables(self) -> List[str]:
        import backend.retrieval.pgvector_store as pgstore
        conn = pgstore.get_store()
        try:
            return pgstore.list_tables(conn, self.db_name)
        finally:
            conn.close()

    def _reembed_tables(self, table_names: List[str]) -> List[str]:
        """Re-process → re-embed → upsert chunks for ONLY the given tables."""
        if not table_names:
            return []
        from backend.retrieval.data_processing import (
            extract_schema_pg, extract_sample_rows_pg, extract_distinct_values_pg,
            load_column_meanings, resolve_column_meaning,
        )
        from backend.retrieval.embedding import (
            build_chunks, embed_chunks, get_openai_client,
        )
        import backend.retrieval.pgvector_store as pgstore

        wanted = set(table_names)
        conn = self._connect()
        try:
            tables = [t for t in extract_schema_pg(conn, self.db_name)
                      if t["table_name"] in wanted]
            meanings = load_column_meanings()
            for t in tables:
                t["sample_rows"] = extract_sample_rows_pg(conn, t["table_name"])
                for col in t["columns"]:
                    col["description"], col["value_description"] = resolve_column_meaning(
                        meanings, t["table_name"], col["name"])
                    col["distinct_values"] = (
                        None if col.get("primary_key")
                        else extract_distinct_values_pg(
                            conn, t["table_name"], col["name"], col["type"])
                    )
        finally:
            conn.close()

        if not tables:
            return []
        chunks = build_chunks([{"db_name": self.db_name, "tables": tables}])
        chunks = embed_chunks(chunks, client=get_openai_client())
        store_conn = pgstore.get_store()
        try:
            pgstore.save_chunks(chunks, conn=store_conn)
        finally:
            store_conn.close()
        return [t["table_name"] for t in tables]

    def _remove_tables(self, table_names: List[str]) -> List[str]:
        if not table_names:
            return []
        import backend.retrieval.pgvector_store as pgstore
        conn = pgstore.get_store()
        try:
            for t in table_names:
                pgstore.delete_table_chunks(conn, self.db_name, t)
        finally:
            conn.close()
        return list(table_names)

    # ── public API ─────────────────────────────────────────────────────────────

    def check(self) -> Dict[str, Any]:
        """Change report: snapshot diff + tables missing from the vector index."""
        current = self.current_snapshot()
        current_hash = hash_schema(current)
        last = self.last_snapshot()
        if last:
            changes = diff_schemas(last["snapshot"], current)
        else:
            changes = [{"action": "add_table", "table": t} for t in sorted(current)]
        try:
            indexed = set(self._indexed_tables())
        except Exception as exc:   # vector store down → report, don't crash
            logger.warning(f"[SCHEMA_SYNC] index listing failed: {exc}")
            indexed = set()
        unindexed = sorted(set(current) - indexed)
        return {
            "db_name": self.db_name,
            "current_hash": current_hash,
            "last_hash": last["hash"] if last else None,
            "last_synced_at": last["created_at"] if last else None,
            "in_sync": bool(last) and not changes and not unindexed,
            "changes": changes,
            "unindexed_tables": unindexed,
        }

    def sync(self, full: bool = False,
             tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply changes to the RAG store. Default: surgical (only changed +
        unindexed tables). `tables` restricts further; `full` re-embeds every
        live table (the "data changed" refresh).
        """
        current = self.current_snapshot()
        last = self.last_snapshot()
        changes = (diff_schemas(last["snapshot"], current) if last
                   else [{"action": "add_table", "table": t} for t in sorted(current)])

        dropped = [c["table"] for c in changes if c["action"] == "drop_table"]
        touched = {c["table"] for c in changes if c["action"] != "drop_table"}
        try:
            indexed = set(self._indexed_tables())
        except Exception:
            indexed = set()
        touched |= set(current) - indexed          # never-indexed tables

        if full:
            to_embed = sorted(current)
        elif tables:
            to_embed = sorted(set(tables) & set(current))
        else:
            to_embed = sorted(touched)

        reembedded = self._reembed_tables(to_embed)
        removed = self._remove_tables(dropped)
        new_hash = self._persist(current, changes)
        logger.info(f"[SCHEMA_SYNC] re-embedded={reembedded} removed={removed}")
        return {
            "db_name": self.db_name,
            "schema_hash": new_hash,
            "changes_applied": changes,
            "reembedded_tables": reembedded,
            "removed_tables": removed,
            "full": full,
        }
