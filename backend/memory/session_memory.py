"""
session_memory.py
-----------------
Conversation memory for the orchestration agent.

Two interchangeable storage backends sit behind one public API:

  * ``_RamBackend``  — in-process dict; used for tests / when no DB is configured.
  * ``_PgBackend``   — PostgreSQL (`sessions` + `turns` tables); durable across
                       restarts.

`SessionMemory` owns the *policy* (title generation, rolling summarization,
trimming) so both backends behave identically. The public method signatures are
unchanged from the original in-RAM version, so ``db_agent.py`` keeps working:

    add(session_id, user_query, response, sql=None, valid=None)
    get_history(session_id)            -> list[dict]
    get_last_sql(session_id)           -> str | None
    clear_session(session_id)          -> None
    active_sessions()                  -> list[str]

Multi-turn support:

    list_sessions(user_id=None)        -> list[dict]   (sidebar)
    get_summary(session_id)            -> str | None

If ``db_config`` is given but the database can't be reached, the store
transparently falls back to RAM so the app still runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# A summarizer condenses evicted messages into a rolling summary.
#   summarizer(evicted_text, prior_summary) -> new_summary
Summarizer = Callable[[str, Optional[str]], str]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_title(first_user_message: str, max_len: int = 60) -> str:
    """Derive a short session title from the first user message."""
    title = " ".join((first_user_message or "").split()).strip()
    if not title:
        return "New conversation"
    return title if len(title) <= max_len else title[: max_len - 1].rstrip() + "…"


def _format_messages_for_summary(messages: List[dict]) -> str:
    """Render evicted messages as plain text for the summarizer."""
    lines = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _fallback_summarize(evicted_text: str, prior_summary: Optional[str]) -> str:
    """Cheap, deterministic summary used when no LLM summarizer is supplied."""
    combined = (f"{prior_summary}\n" if prior_summary else "") + evicted_text
    combined = combined.strip()
    # Bound the rolling summary so it can't grow without limit.
    return combined[-1200:]


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

class _RamBackend:
    """In-process storage. Not durable; intended for tests / no-DB mode."""

    def __init__(self) -> None:
        # session_id -> {"user_id", "title", "summary", "created_at", "updated_at"}
        self._sessions: Dict[str, dict] = {}
        # session_id -> [ {role, content, sql, valid, timestamp}, ... ]
        self._turns: Dict[str, List[dict]] = {}
        # Global counter for synthetic turn IDs (consistent within a process lifetime)
        self._next_id: int = 1

    def ensure_session(self, session_id: str, title: str, user_id: Optional[str]) -> None:
        if session_id not in self._sessions:
            now = _utcnow_iso()
            self._sessions[session_id] = {
                "user_id": user_id,
                "title": title,
                "summary": None,
                "created_at": now,
                "updated_at": now,
            }
            self._turns[session_id] = []

    def append_messages(self, session_id: str, messages: List[dict]) -> Optional[int]:
        """Append messages and return the synthetic id of the last inserted message."""
        turns = self._turns.setdefault(session_id, [])
        last_id: Optional[int] = None
        for m in messages:
            turns.append(m)
            last_id = self._next_id
            self._next_id += 1
        if session_id in self._sessions:
            self._sessions[session_id]["updated_at"] = _utcnow_iso()
        return last_id

    def fetch_messages(self, session_id: str) -> List[dict]:
        return list(self._turns.get(session_id, []))

    def fetch_summary(self, session_id: str) -> Optional[str]:
        return (self._sessions.get(session_id) or {}).get("summary")

    def replace_summary_and_trim(
        self, session_id: str, summary: str, keep_last_n: int
    ) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["summary"] = summary
        turns = self._turns.get(session_id, [])
        self._turns[session_id] = turns[-keep_last_n:] if keep_last_n > 0 else []

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._turns.pop(session_id, None)

    def list_session_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def list_sessions(self, user_id: Optional[str]) -> List[dict]:
        out = []
        for sid, row in self._sessions.items():
            if user_id is not None and row.get("user_id") != user_id:
                continue
            out.append({
                "session_id": sid,
                "title": row.get("title"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            })
        out.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return out


class _PgBackend:
    """PostgreSQL storage. Connect-per-operation for thread safety."""

    def __init__(self, db_config: dict) -> None:
        import psycopg2  # local import so RAM-only installs don't need it

        self._psycopg2 = psycopg2
        self._cfg = {
            "dbname":   db_config.get("name", "FYP"),
            "user":     db_config.get("user", "postgres"),
            "password": db_config.get("password"),
            "host":     db_config.get("host", "127.0.0.1"),
            "port":     db_config.get("port", 5432),
        }
        self._init_tables()

    def _connect(self):
        return self._psycopg2.connect(**self._cfg)

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id  TEXT PRIMARY KEY,
                            user_id     TEXT,
                            title       TEXT,
                            summary     TEXT,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS turns (
                            id          SERIAL PRIMARY KEY,
                            session_id  TEXT NOT NULL
                                        REFERENCES sessions(session_id) ON DELETE CASCADE,
                            role        TEXT NOT NULL,
                            content     TEXT NOT NULL,
                            sql         TEXT,
                            valid       BOOLEAN,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_turns_session
                        ON turns(session_id, id);
                    """)
        finally:
            conn.close()

    def ensure_session(self, session_id: str, title: str, user_id: Optional[str]) -> None:
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sessions (session_id, user_id, title)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO NOTHING;
                        """,
                        (session_id, user_id, title),
                    )
        finally:
            conn.close()

    def append_messages(self, session_id: str, messages: List[dict]) -> Optional[int]:
        """Append messages and return the DB id of the last inserted turn."""
        conn = self._connect()
        last_id: Optional[int] = None
        try:
            with conn:
                with conn.cursor() as cur:
                    for m in messages:
                        cur.execute(
                            """
                            INSERT INTO turns (session_id, role, content, sql, valid)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id;
                            """,
                            (session_id, m["role"], m["content"],
                             m.get("sql"), m.get("valid")),
                        )
                        last_id = cur.fetchone()[0]
                    cur.execute(
                        "UPDATE sessions SET updated_at = now() WHERE session_id = %s;",
                        (session_id,),
                    )
        finally:
            conn.close()
        return last_id

    def fetch_messages(self, session_id: str) -> List[dict]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, sql, valid, created_at
                    FROM turns
                    WHERE session_id = %s
                    ORDER BY id ASC;
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "role": r[0],
                "content": r[1],
                "sql": r[2],
                "valid": r[3],
                "timestamp": r[4].isoformat() if r[4] else None,
            }
            for r in rows
        ]

    def fetch_summary(self, session_id: str) -> Optional[str]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT summary FROM sessions WHERE session_id = %s;",
                    (session_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def replace_summary_and_trim(
        self, session_id: str, summary: str, keep_last_n: int
    ) -> None:
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET summary = %s WHERE session_id = %s;",
                        (summary, session_id),
                    )
                    # Delete everything except the most recent keep_last_n rows.
                    cur.execute(
                        """
                        DELETE FROM turns
                        WHERE session_id = %s
                          AND id NOT IN (
                              SELECT id FROM turns
                              WHERE session_id = %s
                              ORDER BY id DESC
                              LIMIT %s
                          );
                        """,
                        (session_id, session_id, keep_last_n),
                    )
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE session_id = %s;", (session_id,)
                    )
        finally:
            conn.close()

    def list_session_ids(self) -> List[str]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC;")
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    def list_sessions(self, user_id: Optional[str]) -> List[dict]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if user_id is None:
                    cur.execute(
                        """
                        SELECT session_id, title, created_at, updated_at
                        FROM sessions ORDER BY updated_at DESC;
                        """
                    )
                else:
                    cur.execute(
                        """
                        SELECT session_id, title, created_at, updated_at
                        FROM sessions WHERE user_id = %s ORDER BY updated_at DESC;
                        """,
                        (user_id,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "session_id": r[0],
                "title": r[1],
                "created_at": r[2].isoformat() if r[2] else None,
                "updated_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Public store
# ─────────────────────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Durable conversation memory with rolling summarization.

    Args:
        db_config:  dict with host/port/name/user/password. If None (or the DB
                    is unreachable) an in-RAM backend is used instead.
        max_turns:  number of user/assistant *pairs* to keep verbatim. When the
                    stored message count exceeds ``2 * max_turns`` the oldest
                    overflow is summarized into the session's rolling summary and
                    trimmed.
        summarizer: optional callable(evicted_text, prior_summary) -> str. When
                    omitted a deterministic truncation summary is used.
    """

    def __init__(
        self,
        db_config: Optional[dict] = None,
        max_turns: int = 10,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.max_turns = max_turns
        self._summarizer = summarizer
        self._backend = self._make_backend(db_config)

    def _make_backend(self, db_config: Optional[dict]):
        if not db_config:
            logger.info("SessionMemory: using in-RAM backend (no db_config).")
            return _RamBackend()
        try:
            backend = _PgBackend(db_config)
            logger.info("SessionMemory: using PostgreSQL backend.")
            return backend
        except Exception as exc:  # pragma: no cover - depends on live DB
            logger.warning(
                "SessionMemory: PostgreSQL unavailable (%s) — falling back to RAM.",
                exc,
            )
            return _RamBackend()

    @property
    def is_persistent(self) -> bool:
        return isinstance(self._backend, _PgBackend)

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        session_id: str,
        user_query: str,
        response: str,
        sql: Optional[str] = None,
        valid: Optional[bool] = None,
    ) -> Optional[int]:
        """
        Append a user turn and an assistant turn, then maybe summarize.

        Returns the turn id of the assistant message (int in PG mode, synthetic
        int in RAM mode). Callers can store this and pass it to ``/feedback``.
        """
        self._backend.ensure_session(
            session_id, title=_make_title(user_query), user_id=None
        )
        last_id = self._backend.append_messages(session_id, [
            {"role": "user", "content": user_query,
             "sql": None, "valid": None, "timestamp": _utcnow_iso()},
            {"role": "assistant", "content": response,
             "sql": sql, "valid": valid, "timestamp": _utcnow_iso()},
        ])
        self._maybe_summarize(session_id)
        return last_id  # id of the assistant turn

    def _maybe_summarize(self, session_id: str) -> None:
        """Roll the oldest messages into the summary once over the limit."""
        keep = self.max_turns * 2
        # NB: use fetch_messages (raw turns), NOT get_history, which prepends a
        # synthetic summary entry — that would double-count the summary here.
        messages = self._backend.fetch_messages(session_id)
        if len(messages) <= keep:
            return

        evicted = messages[: len(messages) - keep]
        prior = self._backend.fetch_summary(session_id)
        summarize = self._summarizer or _fallback_summarize
        try:
            new_summary = summarize(_format_messages_for_summary(evicted), prior)
        except Exception as exc:  # never let summarization break a turn
            logger.warning("Summarization failed (%s); using fallback.", exc)
            new_summary = _fallback_summarize(
                _format_messages_for_summary(evicted), prior
            )
        self._backend.replace_summary_and_trim(session_id, new_summary, keep)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[Dict]:
        """
        Return conversation history for the graph and the /history endpoint.

        If a rolling summary exists it is prepended as a single entry with
        role ``"summary"`` so the agent can fold earlier context into its prompt.
        """
        messages = self._backend.fetch_messages(session_id)
        summary = self._backend.fetch_summary(session_id)
        if summary:
            return [{"role": "summary", "content": summary,
                     "sql": None, "valid": None, "timestamp": None}] + messages
        return messages

    def get_last_sql(self, session_id: str) -> Optional[str]:
        """Return the most recently generated SQL for a session."""
        for turn in reversed(self._backend.fetch_messages(session_id)):
            if turn.get("sql"):
                return turn["sql"]
        return None

    def get_summary(self, session_id: str) -> Optional[str]:
        return self._backend.fetch_summary(session_id)

    # ── Management ───────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        self._backend.delete_session(session_id)

    def active_sessions(self) -> List[str]:
        return self._backend.list_session_ids()

    def list_sessions(self, user_id: Optional[str] = None) -> List[dict]:
        """Return session metadata for the sidebar (newest first)."""
        return self._backend.list_sessions(user_id)

    def __repr__(self) -> str:
        kind = "pg" if self.is_persistent else "ram"
        return f"SessionMemory(backend={kind}, max_turns={self.max_turns})"
