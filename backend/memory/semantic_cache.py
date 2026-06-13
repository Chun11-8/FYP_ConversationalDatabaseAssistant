"""
semantic_cache.py
-----------------
Semantic cache of successful (user_query → SQL) pairs (Issue #4).

After a query is validated and executed successfully, the orchestrator stores the
(question, SQL) pair with an embedding of the question. On the next turn the most
similar past pairs are retrieved and injected into the SQL generator as few-shot
examples — the spec's "semantic retrieval / semantic cache" feature.

Two tables (created on first use, in the same Postgres as everything else):

    query_sql_pairs       embedded question→SQL pairs (the cache)
    query_interactions    append-only log of every turn (query/sql/valid/latency)

Design notes:
  * Connect-per-operation, like SessionMemory, for thread safety.
  * Embeddings are bound as pgvector text literals ('[..]'::vector) so no extra
    psycopg2 adapter registration is required.
  * If no db_config is given, pgvector is unavailable, or embedding fails, the
    cache degrades to a no-op (retrieval returns [], writes are skipped) so the
    pipeline never breaks.
  * The embedder is injectable, which keeps the class testable against a real
    database without calling OpenAI.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

VECTOR_DIM = 1536

EmbedFn = Callable[[str], List[float]]


def _to_vector_literal(vec: List[float]) -> str:
    """Render a float list as a pgvector text literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _format_few_shot(user_query: str, sql: str) -> str:
    """Format one cached pair for the <few_shot_examples> prompt block."""
    return f"Q: {user_query}\nSQL:\n{sql.strip()}"


class SemanticCache:
    """
    Embedded question→SQL cache backed by pgvector.

    Args:
        db_config:      host/port/name/user/password dict. None → disabled.
        embed_fn:       callable(text) -> list[float]. None → lazy OpenAI client.
        dedup_distance: cosine distance below which a new pair is treated as a
                        near-duplicate and updates the existing row instead of
                        inserting a second copy.
    """

    def __init__(
        self,
        db_config: Optional[dict] = None,
        embed_fn: Optional[EmbedFn] = None,
        dedup_distance: float = 0.05,
    ) -> None:
        self.dedup_distance = dedup_distance
        self._embed_fn = embed_fn
        self._enabled = False
        self._cfg = None

        if not db_config:
            logger.info("SemanticCache: disabled (no db_config).")
            return

        try:
            import psycopg2
            self._psycopg2 = psycopg2
            self._cfg = {
                "dbname":   db_config.get("name", "FYP"),
                "user":     db_config.get("user", "postgres"),
                "password": db_config.get("password"),
                "host":     db_config.get("host", "127.0.0.1"),
                "port":     db_config.get("port", 5432),
            }
            self._init_tables()
            self._enabled = True
            logger.info("SemanticCache: enabled (pgvector).")
        except Exception as exc:  # pragma: no cover - depends on live DB
            logger.warning("SemanticCache: disabled (%s).", exc)
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ── embedding ─────────────────────────────────────────────────────────────
    #
    # Embeddings intentionally use the SYSTEM OpenAI key (not any per-request
    # api_key override). Embedding vectors are model-deterministic regardless of
    # which key produced them, so a single key keeps the shared cache consistent
    # across users. If the system key is unavailable the cache degrades to a
    # no-op rather than failing the request.

    def _embed(self, text: str) -> List[float]:
        if self._embed_fn is None:
            # Lazily build a default OpenAI embedder on first use.
            from backend.retrieval.embedding import get_openai_client, EMBED_MODEL
            client = get_openai_client()

            def _default(t: str) -> List[float]:
                return client.embeddings.create(model=EMBED_MODEL, input=[t]).data[0].embedding

            self._embed_fn = _default
        return self._embed_fn(text)

    def embed(self, text: str) -> Optional[List[float]]:
        """Public, non-raising embed used to compute a query vector once per turn."""
        if not self._enabled:
            return None
        try:
            return self._embed(text)
        except Exception as exc:
            logger.warning("SemanticCache.embed failed: %s", exc)
            return None

    @staticmethod
    def _group_filter(group_id: Optional[str]):
        """Return (sql_clause, params_tuple) for an optional group filter."""
        if group_id is not None:
            return "group_id = %s", (group_id,)
        return "group_id IS NULL", ()

    # ── connection / schema ───────────────────────────────────────────────────

    def _connect(self):
        return self._psycopg2.connect(**self._cfg)

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS query_sql_pairs (
                            id          SERIAL PRIMARY KEY,
                            group_id    TEXT,
                            user_query  TEXT NOT NULL,
                            sql         TEXT NOT NULL,
                            embedding   vector({VECTOR_DIM}),
                            confidence  REAL NOT NULL DEFAULT 1.0,
                            status      TEXT NOT NULL DEFAULT 'active',
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_query_sql_pairs_cosine
                        ON query_sql_pairs
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 50);
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS query_interactions (
                            id          SERIAL PRIMARY KEY,
                            session_id  TEXT,
                            group_id    TEXT,
                            user_query  TEXT,
                            sql         TEXT,
                            response    TEXT,
                            intent      TEXT,
                            valid       BOOLEAN,
                            row_count   INTEGER,
                            latency_ms  INTEGER,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS feedback (
                            id          SERIAL PRIMARY KEY,
                            turn_id     INTEGER,
                            session_id  TEXT,
                            rating      TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                            comment     TEXT,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
        finally:
            conn.close()

    # ── write path ────────────────────────────────────────────────────────────

    def save_pair(
        self,
        user_query: str,
        sql: str,
        group_id: Optional[str] = None,
        confidence: float = 1.0,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Store a (question → SQL) pair, deduplicating near-identical questions.

        If an active pair within ``dedup_distance`` already exists for this group,
        it is refreshed in place rather than inserting a second near-copy.

        ``embedding`` may be supplied to reuse a vector already computed this turn
        and avoid a redundant embedding call.
        """
        if not self._enabled:
            return
        if embedding is None:
            try:
                embedding = self._embed(user_query)
            except Exception as exc:
                logger.warning("SemanticCache.save_pair embedding failed: %s", exc)
                return

        lit = _to_vector_literal(embedding)
        group_clause, params_filter = self._group_filter(group_id)

        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, embedding <=> %s::vector AS distance
                        FROM query_sql_pairs
                        WHERE status = 'active' AND {group_clause}
                        ORDER BY distance ASC
                        LIMIT 1;
                        """,
                        (lit, *params_filter),
                    )
                    row = cur.fetchone()
                    if row is not None and row[1] is not None and row[1] < self.dedup_distance:
                        cur.execute(
                            """
                            UPDATE query_sql_pairs
                            SET sql = %s, embedding = %s::vector, user_query = %s,
                                confidence = %s, created_at = now()
                            WHERE id = %s;
                            """,
                            (sql, lit, user_query, confidence, row[0]),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO query_sql_pairs
                                (group_id, user_query, sql, embedding, confidence, status)
                            VALUES (%s, %s, %s, %s::vector, %s, 'active');
                            """,
                            (group_id, user_query, sql, lit, confidence),
                        )
        except Exception as exc:
            logger.warning("SemanticCache.save_pair failed: %s", exc)
        finally:
            conn.close()

    def log_interaction(
        self,
        session_id: Optional[str],
        user_query: str,
        sql: Optional[str],
        response: str,
        intent: str,
        valid: Optional[bool],
        row_count: Optional[int],
        latency_ms: Optional[int],
        group_id: Optional[str] = None,
    ) -> None:
        """Append a row to the query_interactions audit log (best-effort)."""
        if not self._enabled:
            return
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO query_interactions
                            (session_id, group_id, user_query, sql, response,
                             intent, valid, row_count, latency_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (session_id, group_id, user_query, sql, response,
                         intent, valid, row_count, latency_ms),
                    )
        except Exception as exc:
            logger.warning("SemanticCache.log_interaction failed: %s", exc)
        finally:
            conn.close()

    def add_feedback(
        self,
        turn_id: Optional[int],
        session_id: Optional[str],
        rating: str,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record user feedback (thumbs up / thumbs down) for an assistant turn.

        On a down-vote, finds the SQL executed in that turn (via the ``turns``
        table) and marks any matching ``query_sql_pairs`` row as ``'low_conf'``
        so it is no longer injected as a few-shot example.
        """
        if not self._enabled:
            return

        # Two separate transactions so a missing/inaccessible `turns` table
        # (e.g. SessionMemory is in RAM mode) never rolls back the feedback row.
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO feedback (turn_id, session_id, rating, comment)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (turn_id, session_id, rating, comment),
                    )
        except Exception as exc:
            logger.warning("SemanticCache.add_feedback insert failed: %s", exc)
            return
        finally:
            conn.close()

        if rating != "down" or turn_id is None:
            return

        # Best-effort downvote cascade: look up the SQL for this turn, then
        # mark any active cache pair with that SQL as low_conf.
        conn2 = self._connect()
        try:
            with conn2:
                with conn2.cursor() as cur:
                    cur.execute(
                        "SELECT sql FROM turns WHERE id = %s;", (turn_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        cur.execute(
                            """
                            UPDATE query_sql_pairs
                            SET status = 'low_conf'
                            WHERE sql = %s AND status = 'active';
                            """,
                            (row[0],),
                        )
                        logger.info(
                            "[FEEDBACK] Marked pair low_conf for turn_id=%s", turn_id
                        )
        except Exception as exc:
            logger.warning("SemanticCache.add_feedback downvote cascade failed: %s", exc)
        finally:
            conn2.close()

    # ── read path ─────────────────────────────────────────────────────────────

    def retrieve_pairs(
        self,
        query: str,
        top_k: int = 3,
        group_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[dict]:
        """Return the top-k most similar active pairs as dicts with distances."""
        if not self._enabled:
            return []
        if embedding is None:
            try:
                embedding = self._embed(query)
            except Exception as exc:
                logger.warning("SemanticCache.retrieve_pairs embedding failed: %s", exc)
                return []

        lit = _to_vector_literal(embedding)
        group_clause, params_filter = self._group_filter(group_id)

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT user_query, sql, confidence,
                           embedding <=> %s::vector AS distance
                    FROM query_sql_pairs
                    WHERE status = 'active' AND {group_clause}
                    ORDER BY distance ASC
                    LIMIT %s;
                    """,
                    (lit, *params_filter, top_k),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("SemanticCache.retrieve_pairs failed: %s", exc)
            return []
        finally:
            conn.close()

        return [
            {"user_query": r[0], "sql": r[1], "confidence": r[2], "distance": r[3]}
            for r in rows
        ]

    def retrieve_few_shots(
        self,
        query: str,
        top_k: int = 3,
        group_id: Optional[str] = None,
        max_distance: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Return formatted few-shot example strings for the SQL generator prompt.

        ``max_distance`` optionally drops weak matches (cosine distance above the
        threshold) so only genuinely relevant examples are injected.
        """
        pairs = self.retrieve_pairs(query, top_k=top_k, group_id=group_id, embedding=embedding)
        if max_distance is not None:
            pairs = [p for p in pairs if p["distance"] is not None and p["distance"] <= max_distance]
        return [_format_few_shot(p["user_query"], p["sql"]) for p in pairs]
