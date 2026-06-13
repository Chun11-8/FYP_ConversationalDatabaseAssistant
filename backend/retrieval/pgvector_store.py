"""
store_pgvector.py
-----------------
Persists embedded chunks into PostgreSQL using pgvector extension.
Reuses the SAME PostgreSQL connection from data_processing.py.

Setup (run once in your PostgreSQL):
    CREATE EXTENSION IF NOT EXISTS vector;

Table created automatically on first run:
    schema_embeddings  → table schema + column meanings + sample rows
    
[FUTURE] kb_embeddings → knowledge base entries (commented out)

Usage:
    from store_pgvector import get_store, save_chunks, retrieve
"""

import os
import json
import psycopg2
import psycopg2.extras
from typing import Optional


# ── connection (reuses data_processing.py config) ──────────────────────────────

DB_CONFIG = {
    "dbname":   os.getenv("PG_DB",       "FYP"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", "Toyo20001108!"),
    # On Windows, `localhost` may resolve to IPv6 (`::1`) and hit a different
    # Postgres instance than IPv4 (`127.0.0.1`) if both are bound on :5432.
    "host":     os.getenv("PG_HOST",     "127.0.0.1"),
    "port":     int(os.getenv("PG_PORT", 5432)),
}

# text-embedding-3-small outputs 1536 dimensions
VECTOR_DIM = 1536


def get_store(config: dict = None) -> psycopg2.extensions.connection:
    """
    Connect to PostgreSQL and ensure pgvector extension + tables exist.
    Returns an open psycopg2 connection.
    """
    cfg = config or DB_CONFIG
    try:
        print(f"[INFO] Connecting to PostgreSQL with config: {cfg}")
        conn = psycopg2.connect(**cfg)
        conn.autocommit = False
        _init_tables(conn)
        print(f"[INFO] pgvector store connected → {cfg['dbname']}@{cfg['host']}")
        return conn
    except Exception as e:
        raise ConnectionError(f"[ERROR] Cannot connect to PostgreSQL: {e}")


def _collect_conn_diagnostics(conn: psycopg2.extensions.connection) -> dict:
    """
    Best-effort diagnostics to help identify which PostgreSQL instance we are
    connected to, and whether pgvector is available on that server.
    """
    diagnostics: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            diagnostics["server_version"] = cur.fetchone()[0]

            cur.execute("SELECT current_database(), current_user;")
            current_db, current_user = cur.fetchone()
            diagnostics["current_database"] = current_db
            diagnostics["current_user"] = current_user

            cur.execute("SELECT inet_server_addr(), inet_server_port();")
            addr, port = cur.fetchone()
            diagnostics["inet_server_addr"] = str(addr) if addr is not None else None
            diagnostics["inet_server_port"] = port

            cur.execute("SHOW data_directory;")
            diagnostics["data_directory"] = cur.fetchone()[0]

            cur.execute(
                "SELECT name, default_version, installed_version "
                "FROM pg_available_extensions WHERE name = 'vector';"
            )
            row = cur.fetchone()
            if row:
                diagnostics["pgvector_available"] = True
                diagnostics["pgvector_default_version"] = row[1]
                diagnostics["pgvector_installed_version"] = row[2]
            else:
                diagnostics["pgvector_available"] = False
    except Exception as e:
        diagnostics["diagnostics_error"] = str(e)
    return diagnostics


def _init_tables(conn: psycopg2.extensions.connection) -> None:
    """Create pgvector extension and embedding tables if they don't exist."""
    with conn.cursor() as cur:
        # Enable pgvector
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except psycopg2.errors.FeatureNotSupported as e:
            # Transaction is now aborted; rollback before any further queries.
            conn.rollback()
            diagnostics = _collect_conn_diagnostics(conn)
            raise RuntimeError(
                "pgvector extension not available on the connected PostgreSQL server. "
                "This usually means you're connected to a different PostgreSQL instance "
                "than the one you checked in `psql`.\n\n"
                f"Original error: {e}\n"
                f"Diagnostics: {json.dumps(diagnostics, indent=2)}\n\n"
                "Next checks:\n"
                "  - In `psql`, run: `\\conninfo` and ensure host/port/db match your Python config.\n"
                "  - In Python, ensure `PG_HOST`/`PG_PORT`/`PG_DB`/`PG_USER` point to the same instance.\n"
                "  - If you have multiple Postgres installs (Docker/WSL/Windows service), "
                "verify which one is listening on the configured port."
            ) from e

        # Table schema embeddings
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS schema_embeddings (
                id          SERIAL PRIMARY KEY,
                chunk_id    TEXT UNIQUE NOT NULL,
                chunk_type  TEXT        NOT NULL DEFAULT 'table_schema',
                db_name     TEXT        NOT NULL,
                table_name  TEXT        NOT NULL,
                content     TEXT        NOT NULL,
                embedding   vector({VECTOR_DIM})
            );
        """)

        # IVFFlat index for cosine similarity search
        # (Run AFTER inserting data for best performance)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_schema_embeddings_cosine
            ON schema_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50);
        """)

        # ── [FUTURE] KB embeddings table ───────────────────────────────────
        # Uncomment when knowledge base support is added
        #
        # cur.execute(f"""
        #     CREATE TABLE IF NOT EXISTS kb_embeddings (
        #         id          SERIAL PRIMARY KEY,
        #         chunk_id    TEXT UNIQUE NOT NULL,
        #         chunk_type  TEXT        NOT NULL DEFAULT 'knowledge_base',
        #         db_name     TEXT        NOT NULL,
        #         kb_id       TEXT,
        #         kb_name     TEXT,
        #         content     TEXT        NOT NULL,
        #         embedding   vector({VECTOR_DIM})
        #     );
        # """)
        # cur.execute("""
        #     CREATE INDEX IF NOT EXISTS idx_kb_embeddings_cosine
        #     ON kb_embeddings
        #     USING ivfflat (embedding vector_cosine_ops)
        #     WITH (lists = 50);
        # """)

    conn.commit()
    print("[INFO] Tables and indexes verified.")


# ── save chunks ────────────────────────────────────────────────────────────────

def save_chunks(
    chunks: list[dict],
    conn: Optional[psycopg2.extensions.connection] = None,
    batch_size: int = 100,
) -> None:
    """
    Upsert all embedded chunks into PostgreSQL.
    Uses INSERT ... ON CONFLICT DO UPDATE so it's safe to re-run.

    Args:
        chunks:     List of chunk dicts with 'embedding' field.
        conn:       Existing psycopg2 connection, or None to create one.
        batch_size: Number of rows per batch insert.
    """
    if conn is None:
        conn = get_store()

    table_chunks = [c for c in chunks if c["chunk_type"] == "table_schema"]

    # ── [FUTURE] KB chunks ─────────────────────────────────────────────────
    # kb_chunks = [c for c in chunks if c["chunk_type"] == "knowledge_base"]

    _upsert_schema_chunks(conn, table_chunks, batch_size)

    # ── [FUTURE] uncomment to save KB chunks ──────────────────────────────
    # _upsert_kb_chunks(conn, kb_chunks, batch_size)

    conn.commit()
    print("[INFO] All chunks committed to PostgreSQL.")


def _upsert_schema_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[dict],
    batch_size: int,
) -> None:
    """Batch upsert table schema chunks into schema_embeddings."""
    if not chunks:
        return

    sql = """
        INSERT INTO schema_embeddings
            (chunk_id, chunk_type, db_name, table_name, content, embedding)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id)
        DO UPDATE SET
            content   = EXCLUDED.content,
            embedding = EXCLUDED.embedding;
    """

    total  = len(chunks)
    saved  = 0

    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            rows  = [
                (
                    c["chunk_id"],
                    c["chunk_type"],
                    c["db_name"],
                    c["table_name"],
                    c["text"],
                    c["embedding"],        # psycopg2 + pgvector accepts list[float]
                )
                for c in batch
            ]
            psycopg2.extras.execute_batch(cur, sql, rows)
            saved += len(batch)
            print(f"  → [schema_embeddings] {saved}/{total} upserted")


# ── [FUTURE] KB upsert ─────────────────────────────────────────────────────────
# def _upsert_kb_chunks(conn, chunks, batch_size):
#     if not chunks:
#         return
#     sql = """
#         INSERT INTO kb_embeddings
#             (chunk_id, chunk_type, db_name, kb_id, kb_name, content, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s)
#         ON CONFLICT (chunk_id)
#         DO UPDATE SET
#             content   = EXCLUDED.content,
#             embedding = EXCLUDED.embedding;
#     """
#     total = len(chunks)
#     saved = 0
#     with conn.cursor() as cur:
#         for i in range(0, total, batch_size):
#             batch = chunks[i : i + batch_size]
#             rows = [
#                 (c["chunk_id"], c["chunk_type"], c["db_name"],
#                  c.get("kb_id",""), c.get("kb_name",""), c["text"], c["embedding"])
#                 for c in batch
#             ]
#             psycopg2.extras.execute_batch(cur, sql, rows)
#             saved += len(batch)
#             print(f"  → [kb_embeddings] {saved}/{total} upserted")


# ── retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query_embedding: list[float],
    db_name: Optional[str] = None,
    top_k_tables: int = 3,
    conn: Optional[psycopg2.extensions.connection] = None,
) -> dict:
    """
    Find the most relevant table chunks for a query embedding.

    Args:
        query_embedding: The embedded user query (list of 1536 floats).
        db_name:         Filter results to this database only (optional).
        top_k_tables:    Number of table chunks to return.
        conn:            Existing connection, or None to create one.

    Returns:
        {
            "tables":         [ { chunk_id, db_name, table_name, content, distance }, ... ],
            "knowledge_base": [],   # empty until KB is implemented
        }
    """
    if conn is None:
        conn = get_store()

    results = {"tables": [], "knowledge_base": []}

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

        # Cosine distance: <=> operator (lower = more similar)
        if db_name:
            cur.execute("""
                SELECT chunk_id, db_name, table_name, content,
                       embedding <=> %s::vector AS distance
                FROM schema_embeddings
                WHERE db_name = %s
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, db_name, top_k_tables))
        else:
            cur.execute("""
                SELECT chunk_id, db_name, table_name, content,
                       embedding <=> %s::vector AS distance
                FROM schema_embeddings
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, top_k_tables))

        rows = cur.fetchall()
        results["tables"] = [dict(r) for r in rows]

    # ── [FUTURE] KB retrieval ──────────────────────────────────────────────
    # with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    #     if db_name:
    #         cur.execute("""
    #             SELECT chunk_id, db_name, kb_id, kb_name, content,
    #                    embedding <=> %s::vector AS distance
    #             FROM kb_embeddings
    #             WHERE db_name = %s
    #             ORDER BY distance ASC
    #             LIMIT %s;
    #         """, (query_embedding, db_name, top_k_kb))
    #     else:
    #         cur.execute("""
    #             SELECT chunk_id, db_name, kb_id, kb_name, content,
    #                    embedding <=> %s::vector AS distance
    #             FROM kb_embeddings
    #             ORDER BY distance ASC
    #             LIMIT %s;
    #         """, (query_embedding, top_k_kb))
    #     results["knowledge_base"] = [dict(r) for r in cur.fetchall()]

    return results


# ── inspection helpers ─────────────────────────────────────────────────────────
def run_query(conn: psycopg2.extensions.connection, sql: str) -> list[dict]:
    """Run a raw SQL query and return results as list of dicts."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]
    
def list_databases(conn: psycopg2.extensions.connection) -> list[str]:
    """Return all unique db_names stored in schema_embeddings."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT db_name FROM schema_embeddings ORDER BY db_name;")
        return [row[0] for row in cur.fetchall()]


def list_tables(conn: psycopg2.extensions.connection, db_name: str) -> list[str]:
    """Return all table names for a given database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM schema_embeddings WHERE db_name = %s ORDER BY table_name;",
            (db_name,)
        )
        return [row[0] for row in cur.fetchall()]


def delete_database(conn: psycopg2.extensions.connection, db_name: str) -> None:
    """Remove all chunks for a specific database (useful for re-indexing)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM schema_embeddings WHERE db_name = %s;", (db_name,))
        # cur.execute("DELETE FROM kb_embeddings WHERE db_name = %s;", (db_name,))  # [FUTURE]
    conn.commit()
    print(f"[INFO] Deleted all chunks for database '{db_name}'")


def get_stats(conn: psycopg2.extensions.connection) -> dict:
    """Return row counts for all embedding tables."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_embeddings;")
        schema_count = cur.fetchone()[0]
        # cur.execute("SELECT COUNT(*) FROM kb_embeddings;")  # [FUTURE]
        # kb_count = cur.fetchone()[0]
    return {
        "schema_embeddings": schema_count,
        # "kb_embeddings": kb_count,  # [FUTURE]
    }


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_store()

    stats = get_stats(conn)
    print(f"\nStore stats: {stats}")

    dbs = list_databases(conn)
    if dbs:
        print(f"Stored databases: {dbs}")
        for db in dbs:
            tables = list_tables(conn, db)
            print(f"  {db}: {tables}")
    else:
        print("Store is empty. Run data_processing.py to extract and save schema chunks first.")

    conn.close()
