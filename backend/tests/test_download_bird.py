"""
Offline tests for backend/eval/download_bird.py.

No network and no `datasets` dependency: the split-resolution and file-writing
logic is exercised against plain lists and dicts, which is all `write_dialect`
iterates over.
"""

from __future__ import annotations

import json

import pytest

from backend.eval.download_bird import (
    DIALECTS,
    _gold_line,
    _normalise_row,
    resolve_split,
    write_dialect,
)


# ── resolve_split ─────────────────────────────────────────────────────────────

def test_postgres_split_is_the_pg_one_not_the_sqlite_one():
    """The whole point of the dialect flag: pg questions, not sqlite ones."""
    available = ["mini_dev_sqlite", "mini_dev_pg", "mini_dev_mysql"]
    assert resolve_split(available, "postgresql") == "mini_dev_pg"
    assert resolve_split(available, "sqlite") == "mini_dev_sqlite"
    assert resolve_split(available, "mysql") == "mini_dev_mysql"


@pytest.mark.parametrize("spelling", ["mini_dev_postgresql", "postgresql", "pg"])
def test_alternate_upstream_spellings_resolve(spelling):
    assert resolve_split([spelling], "postgresql") == spelling


def test_matching_ignores_case_and_separators():
    assert resolve_split(["Mini-Dev-PG"], "postgresql") == "Mini-Dev-PG"


def test_unknown_split_is_none_rather_than_a_wrong_guess():
    """An unresolvable dialect must not fall through to another dialect."""
    assert resolve_split(["mini_dev_sqlite"], "postgresql") is None
    assert resolve_split([], "sqlite") is None


def test_every_dialect_has_candidates():
    assert set(DIALECTS) == {"postgresql", "sqlite", "mysql"}
    assert all(names for names in DIALECTS.values())


# ── row normalisation ─────────────────────────────────────────────────────────

def test_alternate_field_names_are_mapped():
    row = {"db_id": "superhero", "question": "how many?",
           "query": "SELECT 1", "hard_level": "moderate"}
    out = _normalise_row(row, 7)
    assert out["SQL"] == "SELECT 1"
    assert out["difficulty"] == "moderate"
    assert out["question_id"] == 7


def test_canonical_field_wins_over_its_alias():
    row = {"SQL": "SELECT canonical", "query": "SELECT alias"}
    assert _normalise_row(row, 0)["SQL"] == "SELECT canonical"


def test_missing_fields_get_defaults_so_bird_can_read_the_file():
    out = _normalise_row({}, 3)
    assert set(out) >= {"question_id", "db_id", "question",
                        "evidence", "SQL", "difficulty"}


def test_gold_line_is_one_line_with_a_tab():
    line = _gold_line("SELECT a\n  FROM t\n WHERE b = 1", "mydb")
    assert "\n" not in line
    assert line == "SELECT a FROM t WHERE b = 1\tmydb"


# ── write_dialect ─────────────────────────────────────────────────────────────

ROWS = [
    {"question_id": 0, "db_id": "superhero", "question": "q0",
     "SQL": "SELECT 1", "difficulty": "simple"},
    {"question_id": 1, "db_id": "formula_1", "question": "q1",
     "SQL": "SELECT\n 2", "difficulty": "challenging"},
]


def test_filenames_carry_the_dialect(tmp_path):
    """bird.py pairs --engine with a file; same name for both would defeat it."""
    path = write_dialect(ROWS, tmp_path, "postgresql")
    assert path.name == "mini_dev_postgresql.json"
    assert (tmp_path / "mini_dev_postgresql_gold.sql").exists()

    write_dialect(ROWS, tmp_path, "sqlite")
    assert (tmp_path / "mini_dev_sqlite.json").exists()
    assert (tmp_path / "mini_dev_sqlite_gold.sql").exists()


def test_written_questions_round_trip(tmp_path):
    path = write_dialect(ROWS, tmp_path, "sqlite")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert [q["db_id"] for q in loaded] == ["superhero", "formula_1"]
    assert [q["difficulty"] for q in loaded] == ["simple", "challenging"]


def test_gold_file_has_one_line_per_question(tmp_path):
    write_dialect(ROWS, tmp_path, "sqlite")
    gold = (tmp_path / "mini_dev_sqlite_gold.sql").read_text(encoding="utf-8")
    lines = [ln for ln in gold.splitlines() if ln.strip()]
    assert len(lines) == len(ROWS)
    assert lines[1] == "SELECT 2\tformula_1"


def test_files_are_utf8_regardless_of_platform_default(tmp_path):
    """CLAUDE.md §6.4: cp1252 mangles or crashes on non-ASCII."""
    rows = [{"question_id": 0, "db_id": "d", "question": "café ☕ naïve",
             "SQL": "SELECT 1", "difficulty": "simple"}]
    path = write_dialect(rows, tmp_path, "sqlite")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["question"] == "café ☕ naïve"
