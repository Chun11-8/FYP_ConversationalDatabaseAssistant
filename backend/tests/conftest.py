"""
Shared pytest fixtures/helpers for the backend test-suite.

`live_db_config()` returns the DB config when a real PostgreSQL is reachable, or
None otherwise, so integration tests can be skipped cleanly in environments
without a database.
"""

from __future__ import annotations

from typing import Optional


def live_db_config() -> Optional[dict]:
    """Return the DB config if PostgreSQL is reachable, else None."""
    try:
        from backend.config.config_loader import load_config
        cfg = load_config()
    except Exception:
        return None
    db = cfg.get("db")
    if not db:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=db["name"], user=db["user"], password=db["password"],
            host=db["host"], port=db["port"], connect_timeout=2,
        )
        conn.close()
        return db
    except Exception:
        return None
