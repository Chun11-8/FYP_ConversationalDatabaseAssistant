"""
Tests for backend/eval/db_backends.py (GH#129).

BIRD publishes the same benchmark twice: SQLite files, and a PostgreSQL dump
with gold SQL in Postgres dialect. The harness now runs either.

The SQLite tests run everywhere. The PostgreSQL tests need a live server and
SKIP without one -- they are not mocked, because what is being checked is that
real psycopg2 behaviour (server-side statement_timeout, transaction recovery,
information_schema shapes) is handled correctly, and a mock would assert only
that the code calls the functions it calls.

    export BIRD_TEST_PG_DSN='host=127.0.0.1 port=5432 user=postgres dbname=bird_test'
    pytest backend/tests/test_db_backends.py -v
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from backend.eval.db_backends import (
    ENGINES,
    SingleFileBackend,
    SqliteBackend,
    as_backend,
    make_backend,
)


@pytest.fixture(scope="module")
def sqlite_dir(tmp_path_factory) -> str:
    """A BIRD-layout directory: {db_dir}/{db_id}/{db_id}.sqlite."""
    root = tmp_path_factory.mktemp("dbs")
    d = root / "toxicology"
    d.mkdir()
    conn = sqlite3.connect(d / "toxicology.sqlite")
    conn.execute("CREATE TABLE molecule (molecule_id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE bond (bond_id TEXT PRIMARY KEY, "
                 "molecule_id TEXT REFERENCES molecule(molecule_id), bond_type TEXT)")
    conn.executemany("INSERT INTO molecule VALUES (?,?)",
                     [("TR000", "+"), ("TR001", "-")])
    conn.executemany("INSERT INTO bond VALUES (?,?,?)",
                     [("b1", "TR000", "#"), ("b2", "TR000", "-")])
    conn.commit()
    conn.close()
    return str(root)


class TestMakeBackend:

    def test_an_unknown_engine_raises_rather_than_defaulting(self):
        """Silently falling back would run one engine and report the other's
        dialect, and EX would compare rows from two different databases."""
        with pytest.raises(ValueError, match="unknown engine"):
            make_backend("mysql", db_dir="x")

    def test_each_engine_requires_what_it_needs(self):
        with pytest.raises(ValueError, match="db-dir"):
            make_backend("sqlite")
        with pytest.raises(ValueError, match="connection settings"):
            make_backend("postgres")

    def test_the_engine_names_are_the_advertised_ones(self):
        assert set(ENGINES) == {"postgres", "sqlite"}


class TestSqliteBackend:

    def test_the_dialect_is_reported(self, sqlite_dir):
        assert make_backend("sqlite", db_dir=sqlite_dir).dialect == "sqlite"

    def test_schema_carries_columns_keys_and_values(self, sqlite_dir):
        schema = SqliteBackend(sqlite_dir).schema("toxicology")
        assert "Table: molecule" in schema and "Table: bond" in schema
        assert "molecule_id: TEXT [PK]" in schema
        assert "bond.molecule_id -> molecule.molecule_id" in schema
        assert "values: {" in schema, "value grounding did not run"

    def test_the_schema_is_cached(self, sqlite_dir):
        be = SqliteBackend(sqlite_dir)
        assert be.schema("toxicology") is be.schema("toxicology")

    def test_a_missing_database_raises(self, sqlite_dir):
        with pytest.raises(FileNotFoundError):
            SqliteBackend(sqlite_dir).schema("not_a_db")

    def test_execute_returns_rows(self, sqlite_dir):
        rows, err, _ = SqliteBackend(sqlite_dir).execute(
            "SELECT COUNT(*) FROM bond", "toxicology", 30)
        assert err is None and rows == [(2,)]

    def test_a_sql_error_is_returned_not_raised(self, sqlite_dir):
        rows, err, _ = SqliteBackend(sqlite_dir).execute(
            "SELECT nope FROM bond", "toxicology", 30)
        assert rows == [] and err and "no such column" in err


class TestSingleFileBackend:
    """The shim that lets a caller holding a path use the same interface."""

    def test_as_backend_accepts_a_path(self, sqlite_dir):
        path = Path(sqlite_dir) / "toxicology" / "toxicology.sqlite"
        be, db_id = as_backend(str(path))
        assert isinstance(be, SingleFileBackend)
        rows, err, _ = be.execute("SELECT COUNT(*) FROM bond", db_id, 30)
        assert err is None and rows == [(2,)]

    def test_as_backend_passes_a_backend_through_unchanged(self, sqlite_dir):
        original = SqliteBackend(sqlite_dir)
        be, _ = as_backend(original, "toxicology")
        assert be is original

    def test_it_refuses_to_pretend_it_has_a_schema(self, sqlite_dir):
        """It knows one file, not a db_id map — returning an empty schema would
        make every semantic check silently pass."""
        path = Path(sqlite_dir) / "toxicology" / "toxicology.sqlite"
        with pytest.raises(NotImplementedError):
            SingleFileBackend(str(path)).schema("toxicology")


# ── PostgreSQL: real server, or skip ──────────────────────────────────────────

def _dsn():
    raw = os.getenv("BIRD_TEST_PG_DSN")
    if not raw:
        return None
    out = {}
    for part in raw.split():
        k, _, v = part.partition("=")
        out[k] = int(v) if k == "port" else v
    return out


pg = pytest.mark.skipif(_dsn() is None,
                        reason="set BIRD_TEST_PG_DSN to run the PostgreSQL tests")


TEST_SCHEMA = "bird_eval_selftest"


@pytest.fixture(scope="module")
def pg_backend():
    """
    Fixture data in a DEDICATED schema, created and dropped here.

    Never in `public`. BIRD_TEST_PG_DSN will often point at the developer's real
    BIRD database, and a fixture that dropped tables by name there would destroy
    the loaded benchmark -- `molecule` and `bond` are real BIRD tables.
    """
    import psycopg2
    dsn = _dsn()
    conn = psycopg2.connect(**dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    cur.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
    cur.execute(f"CREATE TABLE {TEST_SCHEMA}.molecule "
                f"(molecule_id TEXT PRIMARY KEY, label TEXT)")
    cur.execute(f"CREATE TABLE {TEST_SCHEMA}.bond (bond_id TEXT PRIMARY KEY, "
                f"molecule_id TEXT REFERENCES {TEST_SCHEMA}.molecule(molecule_id), "
                f"bond_type TEXT)")
    cur.execute(f"INSERT INTO {TEST_SCHEMA}.molecule VALUES ('TR000','+'),('TR001','-')")
    cur.execute(f"INSERT INTO {TEST_SCHEMA}.bond "
                f"VALUES ('b1','TR000','#'),('b2','TR000','-')")
    conn.close()

    be = make_backend("postgres", dsn=dsn, schema_name=TEST_SCHEMA)
    yield be
    be.close()

    conn = psycopg2.connect(**dsn)
    conn.autocommit = True
    conn.cursor().execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    conn.close()


@pg
class TestPostgresBackend:

    def test_the_dialect_is_postgres(self, pg_backend):
        assert pg_backend.dialect == "postgres"

    def test_schema_carries_columns_keys_and_foreign_keys(self, pg_backend):
        schema = pg_backend.schema("toxicology")
        assert "Table: molecule" in schema and "Table: bond" in schema
        assert "[PK]" in schema
        assert "bond.molecule_id -> molecule.molecule_id" in schema
        assert "values: {" in schema

    def test_execute_returns_rows(self, pg_backend):
        rows, err, _ = pg_backend.execute(f"SELECT COUNT(*) FROM {TEST_SCHEMA}.bond", "toxicology", 30)
        assert err is None and rows == [(2,)]

    def test_a_sql_error_is_returned_and_the_connection_recovers(self, pg_backend):
        """A failed statement leaves the transaction aborted; without a rollback
        every later query fails with InFailedSqlTransaction and the whole run
        would report nonsense from the first bad candidate onward."""
        _, err, _ = pg_backend.execute(f"SELECT nope FROM {TEST_SCHEMA}.bond", "toxicology", 30)
        assert err and "does not exist" in err
        rows, err2, _ = pg_backend.execute(f"SELECT COUNT(*) FROM {TEST_SCHEMA}.bond", "toxicology", 30)
        assert err2 is None and rows == [(2,)]

    def test_the_timeout_really_cancels(self, pg_backend):
        t0 = time.perf_counter()
        _, err, _ = pg_backend.execute(
            "SELECT COUNT(*) FROM generate_series(1,200000000)", "toxicology", 1.0)
        elapsed = time.perf_counter() - t0
        assert err and "timed out" in err
        assert elapsed < 5.0, f"took {elapsed:.1f}s for a 1s timeout"

    def test_it_is_usable_after_a_timeout(self, pg_backend):
        pg_backend.execute("SELECT COUNT(*) FROM generate_series(1,200000000)",
                           "toxicology", 1.0)
        rows, err, _ = pg_backend.execute(f"SELECT COUNT(*) FROM {TEST_SCHEMA}.bond", "toxicology", 30)
        assert err is None and rows == [(2,)]

    def test_verify_layout_reports_an_empty_or_incomplete_database(self, pg_backend):
        assert pg_backend.verify_layout(["molecule", "bond"]) == []
        problems = pg_backend.verify_layout(["absent_table"])
        assert problems and "absent_table" in problems[0]


# ── GH#130/131: dialect correctness ───────────────────────────────────────────

class TestDialectNotes:
    """
    "Target SQL dialect: POSTGRES" was the only dialect signal in a ~4,000-token
    prompt, and a model writes SQLite by habit — BIRD is overwhelmingly a SQLite
    benchmark in training data, and `strftime('%Y', Birthday)` appears verbatim
    in its thrombosis gold. Naming the equivalents is what actually helps.

    Every rule asserted here was verified against a live PostgreSQL 16 server:
    the SQLite forms raise "function ... does not exist", the replacements run.
    """

    def test_postgres_notes_name_the_functions_that_do_not_exist(self):
        from backend.agents.sql_code_agent import DIALECT_NOTES
        notes = DIALECT_NOTES["postgres"]
        for absent in ("strftime", "IIF", "julianday"):
            assert absent in notes, f"{absent} is not flagged"

    def test_postgres_notes_give_a_replacement_for_each(self):
        from backend.agents.sql_code_agent import DIALECT_NOTES
        notes = DIALECT_NOTES["postgres"]
        for replacement in ("EXTRACT(YEAR FROM", "CASE WHEN", "::date"):
            assert replacement in notes

    def test_the_boolean_sum_trap_is_covered(self):
        """`SUM(col = 'x')` is valid SQLite and a type error in PostgreSQL —
        and it is what the generator actually produced."""
        from backend.agents.sql_code_agent import DIALECT_NOTES
        notes = DIALECT_NOTES["postgres"]
        assert "SUM(col = 'x')" in notes
        assert "SUM(CASE WHEN" in notes or "FILTER (WHERE" in notes

    def test_identifier_folding_is_covered(self):
        """Unquoted identifiers fold to lowercase, so BIRD's mixed-case columns
        must be quoted or they are simply not found."""
        from backend.agents.sql_code_agent import DIALECT_NOTES
        assert "LOWERCASE" in DIALECT_NOTES["postgres"].upper()

    def test_an_unknown_dialect_contributes_nothing(self):
        """Wrong dialect advice is worse than none."""
        from backend.agents.sql_code_agent import DIALECT_NOTES
        assert DIALECT_NOTES.get("mysql") is None

    def test_the_notes_reach_the_generation_prompt(self):
        from unittest.mock import MagicMock
        from backend.agents.sql_code_agent import SQLCodeAgent
        for dialect, expect in (("postgres", "NO `strftime`"),
                                ("sqlite", "strftime('%Y', col)")):
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
            agent = SQLCodeAgent({"db": {"dialect": dialect},
                                  "logging": {"pipeline_trace": False}}, llm)
            agent.generate(query="q", schema_context="s", few_shots=[], history=[])
            prompt = llm.invoke.call_args[0][0][1].content
            assert f"Target SQL dialect: {dialect.upper()}" in prompt
            assert expect in prompt


class TestGoldDialectPreflight:
    """
    Pairing --engine postgres with mini_dev_sqlite.json fails QUIETLY: gold that
    cannot execute is recorded as gold_error, and compute_bird_metrics drops
    gold_error questions from the EX denominator. The run reports a plausible
    score over whatever survived. Hence a hard stop, not a warning.
    """

    SQLITE_GOLD = [
        {"question_id": 7, "db_id": "thrombosis_prediction", "SQL":
         "SELECT strftime('%Y', CURRENT_TIMESTAMP) - strftime('%Y', Birthday) FROM Patient"},
        {"question_id": 8, "db_id": "x", "SQL": "SELECT IIF(a=1,'y','n') FROM t"},
        {"question_id": 9, "db_id": "y", "SQL": "SELECT julianday(d) FROM t"},
    ]
    PG_GOLD = [
        {"question_id": 7, "db_id": "thrombosis_prediction", "SQL":
         'SELECT EXTRACT(YEAR FROM AGE(CURRENT_DATE, "Birthday")) FROM "Patient"'},
    ]

    def test_sqlite_gold_on_postgres_is_reported(self):
        from backend.eval.bird import gold_dialect_problems
        problems = gold_dialect_problems(self.SQLITE_GOLD, "postgres")
        assert len(problems) == 3
        joined = " ".join(problems)
        for name in ("strftime", "IIF", "julianday"):
            assert name in joined

    def test_the_message_names_a_concrete_question(self):
        """A count alone is not actionable — the first offender lets the file
        be identified immediately."""
        from backend.eval.bird import gold_dialect_problems
        problems = gold_dialect_problems(self.SQLITE_GOLD, "postgres")
        assert "question_id=7" in problems[0]
        assert "thrombosis_prediction" in problems[0]

    def test_postgres_gold_passes(self):
        from backend.eval.bird import gold_dialect_problems
        assert gold_dialect_problems(self.PG_GOLD, "postgres") == []

    def test_sqlite_engine_is_never_blocked_by_its_own_gold(self):
        from backend.eval.bird import gold_dialect_problems
        assert gold_dialect_problems(self.SQLITE_GOLD, "sqlite") == []

    def test_detection_is_case_insensitive(self):
        from backend.eval.bird import gold_dialect_problems
        upper = [{"question_id": 1, "db_id": "d",
                  "SQL": "SELECT STRFTIME('%Y', x) FROM t"}]
        assert gold_dialect_problems(upper, "postgres")

    def test_a_column_merely_named_like_a_function_is_not_flagged(self):
        """The pattern requires a call, so a column called `strftime_result`
        does not trip it."""
        from backend.eval.bird import gold_dialect_problems
        ok = [{"question_id": 1, "db_id": "d",
               "SQL": "SELECT strftime_result FROM t"}]
        assert gold_dialect_problems(ok, "postgres") == []


# ── GH#133: superlatives and NULL ordering ────────────────────────────────────

class TestSuperlativeRules:
    """
    Measured, not assumed. In the SQLite benchmark runs 18-25% of predictions
    emitted `NULLS FIRST` and 44% of those scored 0: DESC already sorts NULLs
    last in SQLite, so the clause inverted the intent and returned the NULL row.

    The direction is DIALECT-SPECIFIC, which is the part that is easy to get
    backwards. BIRD's PostgreSQL gold explicitly pins PostgreSQL back to SQLite
    ordering — `DESC NULLS LAST` 21x and `ASC NULLS FIRST` 14x across the gold
    in outputs/ — so matching gold means WRITING the clause on PostgreSQL and
    OMITTING it on SQLite.
    """

    def test_sqlite_is_told_never_to_write_the_clause(self):
        from backend.agents.sql_code_agent import DIALECT_NOTES
        notes = DIALECT_NOTES["sqlite"]
        assert "NEVER write `NULLS FIRST`" in notes
        assert "INVERTS" in notes

    def test_postgres_is_told_to_write_it_in_both_directions(self):
        from backend.agents.sql_code_agent import DIALECT_NOTES
        notes = DIALECT_NOTES["postgres"]
        assert "DESC NULLS LAST" in notes
        assert "ASC NULLS FIRST" in notes

    def test_the_two_dialects_give_OPPOSITE_advice(self):
        """The whole point. Identical advice would be wrong for one of them."""
        from backend.agents.sql_code_agent import DIALECT_NOTES
        assert "NEVER write `NULLS FIRST`" not in DIALECT_NOTES["postgres"]
        assert "DESC NULLS LAST" not in DIALECT_NOTES["sqlite"]

    def test_both_warn_against_the_MIN_MAX_subquery(self):
        """Gold uses ORDER BY ... LIMIT; a `WHERE col = (SELECT MAX(col))`
        returns every tied row instead of one, and those golds already score
        30% against 52% for the rest on the challenging tier."""
        from backend.agents.sql_code_agent import DIALECT_NOTES
        for dialect in ("postgres", "sqlite"):
            notes = DIALECT_NOTES[dialect]
            assert "SELECT MAX(col)" in notes
            assert "LIMIT 1" in notes

    def test_the_rules_reach_the_generation_prompt(self):
        from unittest.mock import MagicMock
        from backend.agents.sql_code_agent import SQLCodeAgent
        for dialect, expect in (("postgres", "DESC NULLS LAST"),
                                ("sqlite", "NEVER write `NULLS FIRST`")):
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
            agent = SQLCodeAgent({"db": {"dialect": dialect},
                                  "logging": {"pipeline_trace": False}}, llm)
            agent.generate(query="q", schema_context="s", few_shots=[], history=[])
            assert expect in llm.invoke.call_args[0][0][1].content


class TestNullOrderingIsReallyInverted:
    """
    The advice above is only correct if the engines really do differ. Asserted
    against real engines so a future edit cannot quietly flip the guidance.
    """

    def test_sqlite_sorts_nulls_last_on_desc_and_first_on_asc(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "n.db")
        conn.execute("CREATE TABLE t (id INT, v INT)")
        conn.executemany("INSERT INTO t VALUES (?,?)", [(1, 10), (2, None), (3, 30)])
        conn.commit()
        assert conn.execute("SELECT id FROM t ORDER BY v DESC LIMIT 1").fetchone() == (3,)
        assert conn.execute("SELECT id FROM t ORDER BY v ASC LIMIT 1").fetchone() == (2,)
        conn.close()

    @pg
    def test_postgres_sorts_nulls_first_on_desc_and_last_on_asc(self, pg_backend):
        t = f"{TEST_SCHEMA}.nullorder"
        pg_backend.execute(f"DROP TABLE IF EXISTS {t}", "", 30)
        import psycopg2
        conn = psycopg2.connect(**_dsn())
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(f"CREATE TABLE {t} (id INT, v INT)")
        cur.execute(f"INSERT INTO {t} VALUES (1,10),(2,NULL),(3,30)")
        conn.close()

        desc, err, _ = pg_backend.execute(
            f"SELECT id FROM {t} ORDER BY v DESC LIMIT 1", "", 30)
        asc, _, _ = pg_backend.execute(
            f"SELECT id FROM {t} ORDER BY v ASC LIMIT 1", "", 30)
        fixed, _, _ = pg_backend.execute(
            f"SELECT id FROM {t} ORDER BY v DESC NULLS LAST LIMIT 1", "", 30)
        assert err is None
        assert desc == [(2,)], "PostgreSQL DESC must put the NULL first"
        assert asc == [(1,)], "PostgreSQL ASC must put the NULL last"
        assert fixed == [(3,)], "NULLS LAST must recover the real maximum"
