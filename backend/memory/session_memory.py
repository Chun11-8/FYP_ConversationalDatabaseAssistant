from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional


class SessionMemory:
    """
    In-memory session store. Holds per-session conversation history.
    Drop-in for your commented-out memory calls in main_agent.py.
    """

    def __init__(self, max_turns_per_session: int = 20):
        # { session_id: [ { role, content, sql, valid, timestamp }, ... ] }
        self._store: Dict[str, List[dict]] = defaultdict(list)
        self.max_turns = max_turns_per_session

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        session_id: str,
        user_query: str,
        response: str,
        sql: Optional[str] = None,
        valid: Optional[bool] = None,
    ) -> None:
        """Append a user/assistant turn to the session."""
        history = self._store[session_id]

        history.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.utcnow().isoformat(),
        })
        history.append({
            "role": "assistant",
            "content": response,
            "sql": sql,
            "valid": valid,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Trim oldest turns if over limit (keep pairs intact)
        if len(history) > self.max_turns * 2:
            self._store[session_id] = history[-(self.max_turns * 2):]

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[Dict]:
        """Return full turn history for a session (empty list if new session)."""
        return list(self._store.get(session_id, []))

    def get_last_sql(self, session_id: str) -> Optional[str]:
        """Convenience: return the most recently generated SQL for a session."""
        for turn in reversed(self._store.get(session_id, [])):
            if turn.get("sql"):
                return turn["sql"]
        return None

    # ── Management ───────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Wipe a single session (e.g. user clicks 'New Chat')."""
        self._store.pop(session_id, None)

    def active_sessions(self) -> List[str]:
        """Return all session IDs currently in memory."""
        return list(self._store.keys())

    def __repr__(self) -> str:
        return (
            f"SessionMemory("
            f"sessions={len(self._store)}, "
            f"turns={sum(len(v) for v in self._store.values())})"
        )