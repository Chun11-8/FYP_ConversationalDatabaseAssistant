"""
Tests for backend/eval/bird_predict.py — official-format prediction generation.

All agent calls are mocked; SQLite databases are built in tmp_path. Offline.

Run from repo root:
    pytest backend/tests/test_bird_predict.py -v
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.eval.bird_predict import (
    BIRD_SEP,
    _to_single_line,
    generate_predictions,
    write_predictions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def questions_file(tmp_path: Path) -> str:
    data = [
        {"db_id": "test_db", "question": "How many players?",
         "evidence": "", "SQL": "SELECT COUNT(*) FROM players", "difficulty": "simple"},
        {"db_id": "test_db", "question": "Highest rating?",
         "evidence": "higher is better", "SQL": "SELECT MAX(rating) FROM players",
         "difficulty": "moderate"},
    ]
    p = tmp_path / "mini_dev_sqlite.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture()
def db_dir(tmp_path: Path) -> str:
    db_id_dir = tmp_path / "dev_databases" / "test_db"
    db_id_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_id_dir / "test_db.sqlite"))
    conn.execute("CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT, rating INTEGER)")
    conn.commit()
    conn.close()
    return str(tmp_path / "dev_databases")


def _agent(sql: str = "SELECT COUNT(*) FROM players LIMIT 100"):
    agent = MagicMock()
    agent.generate.return_value = [sql]
    agent.rerank.return_value = sql
    return agent


# ── _to_single_line ───────────────────────────────────────────────────────────

class TestToSingleLine:

    def test_collapses_newlines(self):
        assert _to_single_line("SELECT id\nFROM t\n  WHERE x = 1") == "SELECT id FROM t WHERE x = 1"

    def test_strips_trailing_semicolon(self):
        assert _to_single_line("SELECT 1;") == "SELECT 1"

    def test_empty_returns_empty(self):
        assert _to_single_line("") == ""

    def test_tabs_collapsed(self):
        assert "\t" not in _to_single_line("SELECT\tid\tFROM\tt")


# ── generate_predictions ──────────────────────────────────────────────────────

class TestGeneratePredictions:

    def test_official_format_and_keys(self, questions_file, db_dir):
        preds = generate_predictions(questions_file, db_dir, _agent())
        # Keyed by positional index as strings
        assert set(preds.keys()) == {"0", "1"}
        # Each value uses the official separator and ends with the db_id
        for v in preds.values():
            assert BIRD_SEP in v
            assert v.split(BIRD_SEP)[1] == "test_db"

    def test_sql_is_single_line(self, questions_file, db_dir):
        preds = generate_predictions(questions_file, db_dir, _agent("SELECT id\nFROM players"))
        sql = preds["0"].split(BIRD_SEP)[0]
        assert "\n" not in sql

    def test_evidence_promoted_to_domain_context(self, questions_file, db_dir):
        agent = _agent()
        generate_predictions(questions_file, db_dir, agent)
        # Q index 1 has evidence — must be wrapped in <domain_context>
        second_query = agent.generate.call_args_list[1][1]["query"]
        assert "<domain_context>higher is better</domain_context>" in second_query

    def test_generation_failure_yields_placeholder(self, questions_file, db_dir):
        agent = MagicMock()
        agent.generate.side_effect = RuntimeError("LLM down")
        preds = generate_predictions(questions_file, db_dir, agent)
        # Still produces a syntactically valid, db-tagged placeholder (no crash)
        assert preds["0"] == "SELECT 1" + BIRD_SEP + "test_db"

    def test_missing_db_yields_placeholder(self, questions_file, tmp_path):
        # db-dir with no matching .sqlite file
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        preds = generate_predictions(questions_file, str(empty_dir), _agent())
        assert preds["0"].startswith("SELECT 1")
        assert preds["0"].endswith("test_db")

    def test_subset_limits(self, questions_file, db_dir):
        preds = generate_predictions(questions_file, db_dir, _agent(), subset=1)
        assert set(preds.keys()) == {"0"}

    def test_difficulty_filter(self, questions_file, db_dir):
        preds = generate_predictions(questions_file, db_dir, _agent(), difficulty_filter="moderate")
        assert set(preds.keys()) == {"1"}


# ── write_predictions ─────────────────────────────────────────────────────────

class TestWritePredictions:

    def test_writes_valid_json(self, tmp_path):
        out = str(tmp_path / "nested" / "predict.json")
        write_predictions({"0": "SELECT 1\t----- bird -----\tdb"}, out)
        loaded = json.loads(Path(out).read_text(encoding="utf-8"))
        assert loaded["0"].endswith("db")
