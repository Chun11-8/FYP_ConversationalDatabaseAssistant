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
import re
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


# ── literal-aware cache matching (GH#63) ──────────────────────────────────────
#
# Embeddings are structurally near-blind to exact numeric/date tokens: "…in
# 2013?" and "…in 2014?" differ by one character and land well inside the 0.03
# hit threshold, so a follow-up would silently be answered from the previous
# turn's SQL. Distance alone cannot express "same shape, different value", so
# the concrete values get a symbolic equality check alongside it.

_MONTHS = {
    "jan": "jan", "january": "jan", "feb": "feb", "february": "feb",
    "mar": "mar", "march": "mar", "apr": "apr", "april": "apr",
    "may": "may", "jun": "jun", "june": "jun", "jul": "jul", "july": "jul",
    "aug": "aug", "august": "aug", "sep": "sep", "sept": "sep",
    "september": "sep", "oct": "oct", "october": "oct",
    "nov": "nov", "november": "nov", "dec": "dec", "december": "dec",
}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_WORD_RE = re.compile(r"[A-Za-z]+")


def discriminative_literals(text: str) -> frozenset:
    """
    The concrete values a question pins down: numbers, quoted strings and
    month names, normalized so only a REAL difference registers ("Aug" ==
    "August", "06" == "6", "'Brazil'" == "'brazil'").

    Deliberately narrow. Every token type included here changes which rows a
    correct answer returns, so two questions differing in any of them cannot
    share one cached SQL. Ordinary words are excluded — distinguishing those
    is what the embedding distance is for.
    """
    if not text:
        return frozenset()
    out = set()
    for m in _NUMBER_RE.finditer(text):
        out.add(("num", m.group(0).lstrip("0") or "0"))
    for m in _QUOTED_RE.finditer(text):
        val = (m.group(1) or m.group(2) or "").strip().casefold()
        if val:
            out.add(("str", val))
    for m in _WORD_RE.finditer(text):
        canon = _MONTHS.get(m.group(0).casefold())
        if canon:
            out.add(("month", canon))
    return frozenset(out)


def literals_conflict(a: Optional[str], b: Optional[str]) -> bool:
    """True when two questions name different concrete values — i.e. one's
    cached SQL cannot answer the other no matter how close their vectors are."""
    return discriminative_literals(a or "") != discriminative_literals(b or "")


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
                    # GH#57 in-place migrations for installs created before
                    # cost accounting (query_interactions) and answer reuse
                    # (query_sql_pairs) existed.
                    for stmt in (
                        "ALTER TABLE query_interactions "
                        "ADD COLUMN IF NOT EXISTS model TEXT",
                        "ALTER TABLE query_interactions "
                        "ADD COLUMN IF NOT EXISTS tokens_in INTEGER",
                        "ALTER TABLE query_interactions "
                        "ADD COLUMN IF NOT EXISTS tokens_out INTEGER",
                        "ALTER TABLE query_interactions "
                        "ADD COLUMN IF NOT EXISTS cost_usd REAL",
                        "ALTER TABLE query_sql_pairs "
                        "ADD COLUMN IF NOT EXISTS rows_fingerprint TEXT",
                        "ALTER TABLE query_sql_pairs "
                        "ADD COLUMN IF NOT EXISTS response TEXT",
                    ):
                        cur.execute(stmt + ";")
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
        rows_fingerprint: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        """
        Store a (question → SQL) pair, deduplicating near-identical questions.

        If an active pair within ``dedup_distance`` already exists for this group,
        it is refreshed in place rather than inserting a second near-copy.

        ``embedding`` may be supplied to reuse a vector already computed this turn
        and avoid a redundant embedding call.

        ``rows_fingerprint``/``response`` back the GH#57 answer-reuse fast path:
        a later hit whose re-executed rows hash identically can return the stored
        wording without an LLM call.
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
                        SELECT id, embedding <=> %s::vector AS distance, user_query
                        FROM query_sql_pairs
                        WHERE status = 'active' AND {group_clause}
                        ORDER BY distance ASC
                        LIMIT 1;
                        """,
                        (lit, *params_filter),
                    )
                    row = cur.fetchone()
                    # GH#63: dedup_distance (0.05) is WIDER than the read-side
                    # hit threshold (0.03), so without this guard saving "…in
                    # 2014?" would overwrite the near-identical "…in 2013?" row
                    # and the cache could only ever hold one of the two. Two
                    # questions naming different concrete values are different
                    # questions — insert, don't merge.
                    near_dupe = (row is not None and row[1] is not None
                                 and row[1] < self.dedup_distance)
                    if near_dupe and literals_conflict(user_query, row[2]):
                        logger.info(
                            "SemanticCache: near-duplicate (d=%.4f) kept separate "
                            "— different literals: %r vs %r",
                            row[1], user_query[:60], (row[2] or "")[:60])
                        near_dupe = False
                    if near_dupe:
                        cur.execute(
                            """
                            UPDATE query_sql_pairs
                            SET sql = %s, embedding = %s::vector, user_query = %s,
                                confidence = %s, created_at = now(),
                                rows_fingerprint = %s, response = %s
                            WHERE id = %s;
                            """,
                            (sql, lit, user_query, confidence,
                             rows_fingerprint, response, row[0]),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO query_sql_pairs
                                (group_id, user_query, sql, embedding, confidence,
                                 status, rows_fingerprint, response)
                            VALUES (%s, %s, %s, %s::vector, %s, 'active', %s, %s);
                            """,
                            (group_id, user_query, sql, lit, confidence,
                             rows_fingerprint, response),
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
        model: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: Optional[float] = None,
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
                             intent, valid, row_count, latency_ms,
                             model, tokens_in, tokens_out, cost_usd)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s);
                        """,
                        (session_id, group_id, user_query, sql, response,
                         intent, valid, row_count, latency_ms,
                         model, tokens_in, tokens_out, cost_usd),
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

    # ── schema-change cascade (Issue #7/GH#8) ────────────────────────────────
    #
    # A schema change affects every session's cache, not just one group, so
    # these intentionally sweep all groups rather than using _group_filter.

    def _update_pairs_for_table(self, table: str, sql: str, extra_params: tuple = ()) -> int:
        """Run an UPDATE against active pairs whose SQL references `table`
        (case-insensitive, word-bounded). Best-effort: 0 on any failure."""
        if not self._enabled:
            return 0
        pattern = r"\y" + re.escape(table) + r"\y"   # Postgres word-boundary metachar
        try:
            conn = self._connect()
        except Exception as exc:
            logger.warning("SemanticCache table-cascade for %r failed: %s", table, exc)
            return 0
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (*extra_params, pattern))
                    return cur.rowcount
        except Exception as exc:
            logger.warning("SemanticCache table-cascade for %r failed: %s", table, exc)
            return 0
        finally:
            conn.close()

    def archive_pairs_referencing_table(self, table: str) -> int:
        """
        DROPPED-table cascade: mark active pairs whose SQL references `table`
        as 'archived' rather than deleting them (recoverable, and consistent
        with the existing 'low_conf' down-vote pattern in add_feedback).
        Archived pairs are excluded from retrieve_pairs/best_match (status
        != 'active'). Returns the number of pairs archived.
        """
        return self._update_pairs_for_table(
            table,
            "UPDATE query_sql_pairs SET status = 'archived' "
            "WHERE status = 'active' AND sql ~* %s;",
        )

    def demote_pairs_referencing_table(self, table: str, factor: float = 0.5) -> int:
        """
        MODIFIED-column cascade: halve (by default) the confidence of active
        pairs whose SQL references `table`, rather than deleting them — they
        stay available but surface less often until re-validated. Returns the
        number of pairs demoted.
        """
        return self._update_pairs_for_table(
            table,
            "UPDATE query_sql_pairs SET confidence = confidence * %s "
            "WHERE status = 'active' AND sql ~* %s;",
            (factor,),
        )

    # ── read path ─────────────────────────────────────────────────────────────

    def retrieve_pairs(
        self,
        query: str,
        top_k: int = 3,
        group_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        skeleton_pool_multiplier: int = 1,
    ) -> List[dict]:
        """
        Return the top-k most similar active pairs as dicts with distances.

        ``skeleton_pool_multiplier`` > 1 fetches a wider candidate pool
        (top_k * multiplier) instead of exactly top_k — used by
        retrieve_few_shots_blended() to have a real pool to rerank by
        structural-shape overlap rather than reshuffling an
        already-truncated top-k.
        """
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
        pool_size = max(1, top_k) * max(1, skeleton_pool_multiplier)

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT user_query, sql, confidence,
                           embedding <=> %s::vector AS distance,
                           rows_fingerprint, response
                    FROM query_sql_pairs
                    WHERE status = 'active' AND {group_clause}
                    ORDER BY distance ASC
                    LIMIT %s;
                    """,
                    (lit, *params_filter, pool_size),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("SemanticCache.retrieve_pairs failed: %s", exc)
            return []
        finally:
            conn.close()

        return [
            {"user_query": r[0], "sql": r[1], "confidence": r[2], "distance": r[3],
             # GH#57 answer-reuse columns; tolerate short rows so tests and any
             # pre-migration row still yield a usable pair.
             "rows_fingerprint": r[4] if len(r) > 4 else None,
             "response": r[5] if len(r) > 5 else None}
            for r in rows
        ]

    def best_match(
        self,
        embedding: Optional[List[float]],
        group_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Top-1 ACTIVE cached pair for the cache-hit shortcut (Issue #22/GH#38):
        {"user_query", "sql", "confidence", "distance"}. Down-voted pairs never
        surface (retrieve_pairs filters status='active'). None when the cache
        is disabled, the embedding is missing, or the cache is empty.
        """
        if not self._enabled or embedding is None:
            return None
        pairs = self.retrieve_pairs("", top_k=1, group_id=group_id, embedding=embedding)
        return pairs[0] if pairs else None

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

    def retrieve_few_shots_blended(
        self,
        query: str,
        question_shape,
        top_k: int = 3,
        group_id: Optional[str] = None,
        max_distance: Optional[float] = None,
        embedding: Optional[List[float]] = None,
        blend_weight: float = 0.3,
        pool_multiplier: int = 3,
    ) -> List[str]:
        """
        DAIL-SQL-style blended few-shot selection (GH#48-50 companion):
        rerank a wider embedding-distance candidate pool by structural-shape
        overlap (backend.memory.skeleton) against ``question_shape`` — a
        frozenset from skeleton.question_shape_signature(user_query),
        computed by the caller since it has no SQL dependency here.

        ``blend_weight=0`` degrades to exactly retrieve_few_shots()'s
        ranking (embedding distance only) — the config-off / regression-safe
        path. Never raises; any failure here should fall back to the plain
        method at the call site.
        """
        from backend.memory.skeleton import sql_skeleton, skeleton_overlap

        pairs = self.retrieve_pairs(
            query, top_k=top_k, group_id=group_id, embedding=embedding,
            skeleton_pool_multiplier=pool_multiplier,
        )
        if max_distance is not None:
            pairs = [p for p in pairs if p["distance"] is not None and p["distance"] <= max_distance]
        if not pairs:
            return []

        def _score(p: dict) -> float:
            distance = p["distance"] if p["distance"] is not None else 1.0
            embed_sim = 1.0 - min(max(distance, 0.0), 1.0)
            shape_sim = skeleton_overlap(question_shape, sql_skeleton(p["sql"]))
            return (1 - blend_weight) * embed_sim + blend_weight * shape_sim

        ranked = sorted(pairs, key=_score, reverse=True)[:top_k]
        return [_format_few_shot(p["user_query"], p["sql"]) for p in ranked]
