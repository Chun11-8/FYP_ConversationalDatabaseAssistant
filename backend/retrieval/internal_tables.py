"""
internal_tables.py — the assistant's OWN tables, and why they must never be
indexed as user data (GH#79).

The assistant stores its memory, semantic cache, telemetry and — recursively —
its schema vectors in the SAME PostgreSQL database it answers questions about.
`extract_schema_pg()` reads `information_schema` with no filter, so those
housekeeping tables were embedded alongside the real domain tables and could
be retrieved as "relevant schema" for a user question. A question that
mentions a session, a turn, feedback or a query then pulls plumbing into the
top-K instead of the tables that hold the answer, and the value index scans
cached SQL text looking for data literals.

This list is the single source of truth. Both the indexer and the value index
consult it, so the two cannot drift apart.

NOTE: this excludes the assistant's tables ONLY. User/domain tables are never
filtered here — including the BIRD benchmark tables, which must stay indexed.
"""

from __future__ import annotations

# Created by: memory/session_memory.py (sessions, turns),
# memory/semantic_cache.py (query_sql_pairs), agents/db_agent.py
# (query_interactions, feedback), retrieval/pgvector_store.py
# (schema_embeddings, kb_embeddings), agents/schema_versioning_agent.py
# (schema_version_store).
INTERNAL_TABLES: frozenset[str] = frozenset({
    "sessions",
    "turns",
    "query_sql_pairs",
    "query_interactions",
    "feedback",
    "schema_embeddings",
    "kb_embeddings",
    "schema_version_store",
})


def is_internal_table(table_name: str | None) -> bool:
    """True when *table_name* is one of the assistant's own tables."""
    if not table_name:
        return False
    return table_name.strip().strip('"').lower() in INTERNAL_TABLES


def filter_internal(table_names) -> list[str]:
    """Drop the assistant's own tables from an iterable of table names."""
    return [t for t in (table_names or []) if not is_internal_table(t)]
