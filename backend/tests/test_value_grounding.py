"""
Tests for value grounding in the schema context (Issue #19).

extract_sqlite_schema() appends the actual distinct values of low-cardinality
TEXT columns so the model can map question phrases to the exact stored value
('medium' -> 'Medium'). High-cardinality / free-text / id columns are skipped.

Run from repo root:
    pytest backend/tests/test_value_grounding.py -v
"""

from __future__ import annotations

import sqlite3

from backend.eval.bird import (
    extract_sqlite_schema,
    _distinct_values,
    _is_text_type,
    MAX_DISTINCT_VALUES,
)


def _build_db(path: str) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE member ("
        "  member_id INTEGER PRIMARY KEY,"
        "  t_shirt_size TEXT,"   # low-card → values shown
        "  full_name TEXT,"      # high-card → omitted
        "  bio TEXT"             # long free-text → omitted
        ")"
    )
    rows = []
    sizes = ["Small", "Medium", "Large"]
    for i in range(25):  # 25 distinct full_names → high cardinality
        rows.append((sizes[i % 3], f"Person Number {i}", "x" * 60))
    cur.executemany(
        "INSERT INTO member (t_shirt_size, full_name, bio) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    conn.close()


def _column_line(schema: str, col: str) -> str:
    for line in schema.splitlines():
        if line.strip().startswith(f"- {col}"):
            return line
    return ""


class TestIsTextType:
    def test_text_like(self):
        assert _is_text_type("TEXT") and _is_text_type("VARCHAR(20)") and _is_text_type("")
    def test_non_text(self):
        assert not _is_text_type("INTEGER") and not _is_text_type("REAL")


class TestExtractSchemaValues:

    def test_low_card_text_shows_values(self, tmp_path):
        db = tmp_path / "club.sqlite"
        _build_db(str(db))
        schema = extract_sqlite_schema(str(db))
        line = _column_line(schema, "t_shirt_size")
        assert "values: {" in line
        for v in ("Small", "Medium", "Large"):
            assert v in line

    def test_high_card_text_omits_values(self, tmp_path):
        db = tmp_path / "club.sqlite"
        _build_db(str(db))
        schema = extract_sqlite_schema(str(db))
        assert "values:" not in _column_line(schema, "full_name")

    def test_long_freetext_omits_values(self, tmp_path):
        db = tmp_path / "club.sqlite"
        _build_db(str(db))
        schema = extract_sqlite_schema(str(db))
        assert "values:" not in _column_line(schema, "bio")

    def test_pk_id_column_omits_values(self, tmp_path):
        db = tmp_path / "club.sqlite"
        _build_db(str(db))
        schema = extract_sqlite_schema(str(db))
        assert "values:" not in _column_line(schema, "member_id")

    def test_value_block_format_stable(self, tmp_path):
        db = tmp_path / "club.sqlite"
        _build_db(str(db))
        schema = extract_sqlite_schema(str(db))
        line = _column_line(schema, "t_shirt_size")
        # format: "... | values: {Small, Medium, Large}"
        assert "| values: {" in line and line.rstrip().endswith("}")


class TestDistinctValuesHelper:

    def _conn(self, tmp_path, n_distinct):
        db = tmp_path / "d.sqlite"
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cur.execute("CREATE TABLE t (c TEXT)")
        cur.executemany("INSERT INTO t (c) VALUES (?)", [(f"v{i}",) for i in range(n_distinct)])
        conn.commit()
        return conn

    def test_at_limit_included(self, tmp_path):
        conn = self._conn(tmp_path, MAX_DISTINCT_VALUES)
        vals = _distinct_values(conn.cursor(), "t", "c")
        assert vals is not None and len(vals) == MAX_DISTINCT_VALUES
        conn.close()

    def test_over_limit_omitted(self, tmp_path):
        conn = self._conn(tmp_path, MAX_DISTINCT_VALUES + 1)
        assert _distinct_values(conn.cursor(), "t", "c") is None
        conn.close()

    def test_all_null_returns_none(self, tmp_path):
        db = tmp_path / "n.sqlite"
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cur.execute("CREATE TABLE t (c TEXT)")
        cur.executemany("INSERT INTO t (c) VALUES (?)", [(None,), (None,), (None,)])
        conn.commit()
        assert _distinct_values(conn.cursor(), "t", "c") is None
        conn.close()
