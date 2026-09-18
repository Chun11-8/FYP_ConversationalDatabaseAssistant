"""
db_backends.py — where an evaluation run executes its SQL (GH#129).

BIRD publishes two shapes of the same benchmark: SQLite database files, and a
PostgreSQL dump with `mini_dev_postgresql.json` carrying gold SQL in Postgres
dialect. The harness supported only the first.

Two engines, one interface, because everything downstream — schema context,
gold execution, execution voting, refine, VES timing — needs exactly three
things from a database: extract a schema, run a query with a deadline, and say
which SQL dialect the generator should target.

The engine is a property of the RUN, not of a question: a run must not mix
them, or EX compares rows produced by two different engines.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

ENGINES = ("postgres", "sqlite")

# Value grounding: distinct values are appended for a text column with at most
# this many, so the model can map a question phrase to an exact stored value.
MAX_DISTINCT_VALUES = 20
MAX_VALUE_LEN = 40


class Backend(Protocol):
    """What the evaluation loop needs from a database."""

    dialect: str

    def schema(self, db_id: str) -> str:
        """Plain-text schema context for one BIRD database."""

    def execute(self, sql: str, db_id: str,
                timeout: float) -> Tuple[List[tuple], Optional[str], float]:
        """``(rows, error, elapsed_seconds)``. Never raises for SQL errors."""


def _render(tables: List[Tuple[str, List[Tuple[str, str, bool, bool, Optional[List[str]]]]]],
            fks: List[str]) -> str:
    """
    One schema string, identical in shape for both engines.

    Shared deliberately: a Postgres run and a SQLite run of the same question
    must differ only in the DATA and the dialect, never in how the schema was
    described — otherwise a score difference between engines could be a
    prompting difference instead.
    """
    sections: List[str] = []
    for table, cols in tables:
        lines = []
        for name, ctype, is_pk, not_null, values in cols:
            markers = []
            if is_pk:
                markers.append("PK")
            if not_null and not is_pk:
                markers.append("NOT NULL")
            suffix = f" [{', '.join(markers)}]" if markers else ""
            line = f"  - {name}: {ctype}{suffix}"
            if values:
                line += " | values: {" + ", ".join(values) + "}"
            lines.append(line)
        sections.append(f"Table: {table}\nColumns:\n" + "\n".join(lines))
    schema = "\n\n".join(sections)
    if fks:
        schema += "\n\nForeign Key Relationships:\n" + "\n".join(fks)
    return schema


def _usable_values(rows: List[Any]) -> Optional[List[str]]:
    """Distinct values worth showing, or None for free text / high cardinality."""
    if not rows or len(rows) > MAX_DISTINCT_VALUES:
        return None
    out: List[str] = []
    for v in rows:
        s = str(v).strip()
        if s == "":
            continue
        if len(s) > MAX_VALUE_LEN:      # a long value → free text, not a category
            return None
        out.append(s)
    return out or None


# ── SQLite ────────────────────────────────────────────────────────────────────

class SqliteBackend:
    """One file per BIRD database, as published."""

    dialect = "sqlite"

    def __init__(self, db_dir: str) -> None:
        self.db_dir = db_dir
        self._schemas: Dict[str, str] = {}

    def path_for(self, db_id: str) -> str:
        # Official mini-dev ships {db_id}.sqlite; older/self-built sets use .db.
        p = Path(self.db_dir) / db_id / f"{db_id}.sqlite"
        if not p.exists():
            p = Path(self.db_dir) / db_id / f"{db_id}.db"
        return str(p)

    def schema(self, db_id: str) -> str:
        """The schema of one database file, cached — it cannot change mid-run."""
        if db_id in self._schemas:
            return self._schemas[db_id]
        db_path = self.path_for(db_id)
        if not Path(db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name")
            tables: List[Any] = []
            fks: List[str] = []
            for (table,) in cur.fetchall():
                cur.execute(f'PRAGMA table_info("{table}")')
                cols = []
                for _cid, name, ctype, notnull, _dflt, pk in cur.fetchall():
                    values = None
                    if not pk and _is_text(ctype or "TEXT"):
                        values = self._distinct(cur, table, name)
                    cols.append((name, ctype or "TEXT", bool(pk), bool(notnull), values))
                tables.append((table, cols))
                cur.execute(f'PRAGMA foreign_key_list("{table}")')
                for fk in cur.fetchall():
                    fks.append(f"  {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            schema = _render(tables, fks)
        finally:
            conn.close()
        self._schemas[db_id] = schema
        return schema

    @staticmethod
    def _distinct(cur, table: str, col: str) -> Optional[List[str]]:
        try:
            cur.execute(f'SELECT DISTINCT "{col}" FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL LIMIT {MAX_DISTINCT_VALUES + 1}')
            return _usable_values([r[0] for r in cur.fetchall()])
        except Exception:
            return None

    def execute(self, sql: str, db_id: str,
                timeout: float) -> Tuple[List[tuple], Optional[str], float]:
        return execute_sqlite_path(sql, self.path_for(db_id), timeout)


def execute_sqlite_path(sql: str, db_path: str, timeout: float = 30.0,
                        ) -> Tuple[List[tuple], Optional[str], float]:
    """
    Execute against a SQLite file, bounded by a real deadline.

    GH#128: the deadline is enforced by a progress handler INSIDE SQLite.
    Abandoning a worker thread only pretended to bound the query — the
    executor's shutdown joined it again, and sqlite3 cannot be interrupted
    from Python.
    """
    conn = sqlite3.connect(db_path)
    t0 = time.perf_counter()
    deadline = t0 + max(timeout, 0.0)
    try:
        if timeout and timeout > 0:
            conn.set_progress_handler(
                lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None, time.perf_counter() - t0
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            return [], f"Query timed out after {timeout}s", timeout
        return [], str(exc), 0.0
    except Exception as exc:
        return [], str(exc), 0.0
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()



def _is_text(sql_type: str) -> bool:
    t = (sql_type or "").upper()
    return any(k in t for k in ("CHAR", "TEXT", "CLOB", "STRING", "NAME"))


# ── PostgreSQL ────────────────────────────────────────────────────────────────

class PostgresBackend:
    """
    All BIRD tables in ONE PostgreSQL database.

    The db_id therefore selects nothing at connection time; it only labels the
    question. That is only sound while table names are unique across the eleven
    databases — `verify_layout()` checks it rather than assuming, because a
    silent collision would make a question read another database's rows and
    score wrong for a reason nothing in the output would reveal.
    """

    dialect = "postgres"

    def __init__(self, dsn: Dict[str, Any], schema_name: str = "public") -> None:
        self.dsn = dsn
        self.schema_name = schema_name
        self._schema: Optional[str] = None
        self._conn = None

    def _connect(self):
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.dsn)
            self._conn.set_session(readonly=True, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def table_names(self) -> List[str]:
        cur = self._connect().cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
            "ORDER BY table_name", (self.schema_name,))
        return [r[0] for r in cur.fetchall()]

    def schema(self, db_id: str) -> str:
        """
        The whole database's schema, cached.

        Every question sees every table because they share one namespace. This
        is the real cost of the flat layout and it is not hidden: the generator
        gets a larger context than the SQLite path gives it, which is a
        difference worth remembering when comparing scores across engines.
        """
        if self._schema is not None:
            return self._schema
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT c.table_name, c.column_name, c.data_type, c.is_nullable "
            "FROM information_schema.columns c "
            "JOIN information_schema.tables t "
            "  ON t.table_schema = c.table_schema AND t.table_name = c.table_name "
            "WHERE c.table_schema = %s AND t.table_type = 'BASE TABLE' "
            "ORDER BY c.table_name, c.ordinal_position", (self.schema_name,))
        raw = cur.fetchall()

        cur.execute(
            "SELECT tc.table_name, kcu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema = kcu.table_schema "
            "WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY'",
            (self.schema_name,))
        pks = {(t, c) for t, c in cur.fetchall()}

        cur.execute(
            "SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "  ON ccu.constraint_name = tc.constraint_name "
            " AND ccu.table_schema = tc.table_schema "
            "WHERE tc.table_schema = %s AND tc.constraint_type = 'FOREIGN KEY'",
            (self.schema_name,))
        fks = [f"  {a}.{b} -> {c}.{d}" for a, b, c, d in cur.fetchall()]

        tables: List[Tuple[str, List[Any]]] = []
        current, cols = None, []
        for table, col, ctype, nullable in raw:
            if table != current:
                if current is not None:
                    tables.append((current, cols))
                current, cols = table, []
            is_pk = (table, col) in pks
            values = None
            if not is_pk and _is_text(ctype):
                values = self._distinct(cur, table, col)
            cols.append((col, ctype, is_pk, nullable == "NO", values))
        if current is not None:
            tables.append((current, cols))

        self._schema = _render(tables, fks)
        return self._schema

    def _distinct(self, cur, table: str, col: str) -> Optional[List[str]]:
        try:
            cur.execute(
                f'SELECT DISTINCT "{col}" FROM "{self.schema_name}"."{table}" '
                f'WHERE "{col}" IS NOT NULL LIMIT {MAX_DISTINCT_VALUES + 1}')
            return _usable_values([r[0] for r in cur.fetchall()])
        except Exception:
            # A read-only transaction that errored must be cleared, or every
            # later statement on this connection fails with InFailedSqlTransaction.
            try:
                cur.connection.rollback()
            except Exception:
                pass
            return None

    def execute(self, sql: str, db_id: str,
                timeout: float) -> Tuple[List[tuple], Optional[str], float]:
        """
        Run one query under a server-side deadline.

        `statement_timeout` is enforced by PostgreSQL itself, so a runaway query
        is cancelled rather than merely abandoned — the same property GH#128
        established for SQLite, and the reason neither engine needs a worker
        thread.
        """
        import psycopg2
        conn = self._connect()
        t0 = time.perf_counter()
        try:
            cur = conn.cursor()
            ms = int(max(timeout, 0.0) * 1000)
            cur.execute(f"SET statement_timeout = {ms}")   # 0 disables it
            cur.execute(sql)
            rows = cur.fetchall() if cur.description else []
            return rows, None, time.perf_counter() - t0
        except psycopg2.errors.QueryCanceled:
            self._reset()
            return [], f"Query timed out after {timeout}s", timeout
        except Exception as exc:
            self._reset()
            return [], str(exc).strip(), 0.0

    def _reset(self) -> None:
        """Clear a failed transaction so the next query is not rejected."""
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.rollback()
        except Exception:
            self.close()

    def verify_layout(self, expected_tables: Optional[List[str]] = None) -> List[str]:
        """
        Problems with the flat layout, as human-readable strings.

        Checked rather than assumed: with all eleven databases sharing one
        namespace, a duplicated table name would silently point a question at
        another database's rows.
        """
        problems: List[str] = []
        present = {t.lower() for t in self.table_names()}
        if not present:
            problems.append(
                f"schema '{self.schema_name}' contains no tables — is the BIRD "
                f"PostgreSQL dump loaded into this database?")
        for want in expected_tables or []:
            if want.lower() not in present:
                problems.append(f"table '{want}' is missing from the database")
        return problems


class SingleFileBackend:
    """
    A backend bound to ONE SQLite file, ignoring db_id.

    For callers that already hold a path rather than a directory-and-id — the
    execution-voting helper and its tests. Same interface, so nothing
    downstream needs to know which kind it was handed.
    """

    dialect = "sqlite"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def schema(self, db_id: str = "") -> str:
        raise NotImplementedError(
            "SingleFileBackend executes only; use SqliteBackend for schemas")

    def execute(self, sql: str, db_id: str = "",
                timeout: float = 30.0) -> Tuple[List[tuple], Optional[str], float]:
        return execute_sqlite_path(sql, self.db_path, timeout)


def as_backend(target: Any, db_id: str = "") -> Tuple[Any, str]:
    """
    ``(backend, db_id)`` from either a Backend or a plain SQLite path.

    Accepting both keeps `select_by_execution` usable by callers that hold a
    file path, without the engine-agnostic loop having to care.
    """
    if isinstance(target, (str, Path)):
        return SingleFileBackend(str(target)), db_id
    return target, db_id


def make_backend(engine: str, db_dir: Optional[str] = None,
                 dsn: Optional[Dict[str, Any]] = None,
                 schema_name: str = "public") -> Backend:
    """The backend for `engine`. Raises on an unknown name rather than
    defaulting — silently falling back would run the wrong engine and report
    the other one's dialect."""
    engine = (engine or "").strip().lower()
    if engine == "sqlite":
        if not db_dir:
            raise ValueError("engine 'sqlite' needs --db-dir")
        return SqliteBackend(db_dir)
    if engine == "postgres":
        if not dsn:
            raise ValueError("engine 'postgres' needs connection settings")
        return PostgresBackend(dsn, schema_name)
    raise ValueError(f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}")
