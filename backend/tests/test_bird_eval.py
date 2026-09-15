"""
Tests for backend/eval/bird.py.

All LLM and agent calls are mocked — no live services or BIRD data required.
SQLite databases are created in-memory or in tmp_path fixtures.

Run from repo root:
    pytest backend/tests/test_bird_eval.py -v
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.eval.bird import (
    _error_record,
    _rows_to_set,
    compute_bird_metrics,
    compute_ex,
    compute_f1,
    compute_ves_reward,
    execute_sqlite,
    extract_sqlite_schema,
    load_dev_json,
    run_bird_eval,
    write_results,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_db(tmp_path: Path) -> str:
    """Create a minimal SQLite database with one table for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT NOT NULL, rating INTEGER)"
    )
    conn.execute("INSERT INTO players VALUES (1, 'Alice', 85)")
    conn.execute("INSERT INTO players VALUES (2, 'Bob', 72)")
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture()
def fk_db(tmp_path: Path) -> str:
    """SQLite database with two tables linked by a foreign key."""
    db_path = tmp_path / "fk.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE players "
        "(id INTEGER PRIMARY KEY, name TEXT NOT NULL, team_id INTEGER, "
        "FOREIGN KEY (team_id) REFERENCES teams(id))"
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture()
def dev_json(tmp_path: Path) -> str:
    """Write a minimal BIRD dev.json to a temp file and return its path."""
    data = [
        {
            "question_id": 0,
            "question":    "How many players are there?",
            "evidence":    "",
            "SQL":         "SELECT COUNT(*) FROM players",
            "db_id":       "test_db",
            "difficulty":  "simple",
        },
        {
            "question_id": 1,
            "question":    "What is the highest rating?",
            "evidence":    "higher is better",
            "SQL":         "SELECT MAX(rating) FROM players",
            "db_id":       "test_db",
            "difficulty":  "moderate",
        },
    ]
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture()
def db_dir_with_test_db(tmp_path: Path, simple_db: str) -> str:
    """
    Create a BIRD-layout db directory:
        {tmp_path}/test_db/test_db.db
    """
    db_id_dir = tmp_path / "test_db"
    db_id_dir.mkdir()
    dest = db_id_dir / "test_db.db"
    # copy simple_db contents
    src_conn = sqlite3.connect(simple_db)
    dst_conn = sqlite3.connect(str(dest))
    for line in src_conn.iterdump():
        dst_conn.execute(line)
    dst_conn.commit()
    src_conn.close()
    dst_conn.close()
    return str(tmp_path)


# ── extract_sqlite_schema ───────────────────────────────────────────────────────

class TestExtractSqliteSchema:

    def test_single_table(self, simple_db: str):
        schema = extract_sqlite_schema(simple_db)
        assert "players" in schema
        assert "id" in schema
        assert "name" in schema
        assert "rating" in schema

    def test_pk_and_notnull_annotated(self, simple_db: str):
        schema = extract_sqlite_schema(simple_db)
        assert "PK" in schema
        assert "NOT NULL" in schema

    def test_fk_relationship_listed(self, fk_db: str):
        schema = extract_sqlite_schema(fk_db)
        assert "Foreign Key Relationships" in schema
        assert "team_id" in schema
        assert "teams" in schema

    def test_missing_db_raises(self, tmp_path: Path):
        with pytest.raises(Exception):
            extract_sqlite_schema(str(tmp_path / "nonexistent.db"))


# ── execute_sqlite ──────────────────────────────────────────────────────────────

class TestExecuteSqlite:

    def test_basic_select(self, simple_db: str):
        rows, error, elapsed = execute_sqlite("SELECT COUNT(*) FROM players", simple_db)
        assert error is None
        assert rows == [(2,)]
        assert elapsed >= 0

    def test_returns_empty_on_no_rows(self, simple_db: str):
        rows, error, elapsed = execute_sqlite(
            "SELECT * FROM players WHERE rating > 9999", simple_db
        )
        assert error is None
        assert rows == []

    def test_error_on_bad_sql(self, simple_db: str):
        rows, error, elapsed = execute_sqlite("SELECT * FROM ghost_table", simple_db)
        assert error is not None
        assert rows == []

    def test_timeout_fires(self, simple_db: str):
        # Simulate a very short timeout; execution should still return something
        # (the query finishes faster than 0.001s or returns a timeout error)
        rows, error, elapsed = execute_sqlite(
            "SELECT COUNT(*) FROM players", simple_db, timeout=0.001
        )
        # Either it finished or it timed out — both are valid; no exception should propagate
        assert isinstance(rows, list)


# ── _rows_to_set ────────────────────────────────────────────────────────────────

class TestRowsToSet:

    def test_none_values_become_empty_string(self):
        s = _rows_to_set([(None, "a")])
        assert ("", "a") in s

    def test_string_coercion(self):
        s1 = _rows_to_set([(1, 2)])
        s2 = _rows_to_set([("1", "2")])
        assert s1 == s2

    def test_order_insensitive(self):
        s1 = _rows_to_set([(1,), (2,)])
        s2 = _rows_to_set([(2,), (1,)])
        assert s1 == s2


# ── compute_ex ─────────────────────────────────────────────────────────────────

class TestComputeEx:

    def test_identical_single_row(self):
        assert compute_ex([(42,)], [(42,)]) == 1.0

    def test_order_independent(self):
        assert compute_ex([(1,), (2,)], [(2,), (1,)]) == 1.0

    def test_different_values(self):
        assert compute_ex([(1,)], [(2,)]) == 0.0

    def test_different_counts(self):
        assert compute_ex([(1,), (2,)], [(1,)]) == 0.0

    def test_both_empty(self):
        assert compute_ex([], []) == 1.0

    def test_pred_empty_gold_not(self):
        assert compute_ex([], [(1,)]) == 0.0

    def test_string_int_equivalence(self):
        assert compute_ex([("42",)], [(42,)]) == 1.0


# ── compute_f1 ─────────────────────────────────────────────────────────────────

class TestComputeF1:

    def test_perfect_match(self):
        assert compute_f1([(1,), (2,)], [(1,), (2,)]) == 1.0

    def test_both_empty(self):
        assert compute_f1([], []) == 1.0

    def test_pred_empty(self):
        assert compute_f1([], [(1,)]) == 0.0

    def test_gold_empty(self):
        assert compute_f1([(1,)], []) == 0.0

    def test_partial_overlap(self):
        # pred has (1,) and (2,); gold has (1,) and (3,)
        # TP=1, precision=0.5, recall=0.5, F1=0.5
        f1 = compute_f1([(1,), (2,)], [(1,), (3,)])
        assert abs(f1 - 0.5) < 1e-9

    def test_no_overlap(self):
        assert compute_f1([(1,)], [(2,)]) == 0.0


# ── compute_ves_reward ──────────────────────────────────────────────────────────

class TestComputeVesReward:
    # ratio = gold_elapsed / pred_elapsed (how many times faster pred is)

    def test_zero_gold_time_returns_full(self):
        assert compute_ves_reward(0.1, 0.0) == 1.0

    def test_faster_pred_gets_bonus(self):
        # pred=0.04s, gold=0.1s → ratio=0.1/0.04=2.5 → reward=1.25
        assert compute_ves_reward(0.04, 0.1) == 1.25

    def test_equal_time(self):
        # pred=1.0s, gold=1.0s → ratio=1.0 → reward=1.0
        assert compute_ves_reward(1.0, 1.0) == 1.0

    def test_slightly_slower(self):
        # pred=1.2s, gold=1.0s → ratio=1.0/1.2≈0.83 → 0.5≤ratio<1 → reward=0.75
        assert compute_ves_reward(1.2, 1.0) == 0.75

    def test_pred_faster_than_gold(self):
        # pred=1.0s, gold=2.0s → ratio=2.0/1.0=2.0 → ≥2.0 → reward=1.25
        assert compute_ves_reward(1.0, 2.0) == 1.25

    def test_pred_moderately_slower(self):
        # pred=3.0s, gold=1.0s → ratio=1.0/3.0≈0.33 → 0.25≤ratio<0.5 → reward=0.5
        assert compute_ves_reward(3.0, 1.0) == 0.5

    def test_pred_much_slower(self):
        # pred=10.0s, gold=1.0s → ratio=1.0/10=0.1 → <0.25 → reward=0.25
        assert compute_ves_reward(10.0, 1.0) == 0.25


# ── load_dev_json ───────────────────────────────────────────────────────────────

class TestLoadDevJson:

    def test_loads_valid_json(self, dev_json: str):
        data = load_dev_json(dev_json)
        assert len(data) == 2
        assert data[0]["question_id"] == 0
        assert data[0]["SQL"] == "SELECT COUNT(*) FROM players"

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dev_json(str(tmp_path / "missing.json"))


# ── compute_bird_metrics ────────────────────────────────────────────────────────

class TestComputeBirdMetrics:

    def _make_results(self, ex_list, f1_list, ves_list, diffs):
        return [
            {"ex": ex, "f1": f1, "ves_reward": ves, "difficulty": d}
            for ex, f1, ves, d in zip(ex_list, f1_list, ves_list, diffs)
        ]

    def test_empty_results(self):
        m = compute_bird_metrics([])
        assert m["total"] == 0
        assert m["EX"] == 0.0

    def test_perfect_score(self):
        results = self._make_results([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], ["simple", "simple"])
        m = compute_bird_metrics(results)
        assert m["EX"] == 100.0
        assert m["F1"] == 100.0
        assert m["total"] == 2

    def test_zero_score(self):
        results = self._make_results([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], ["simple", "simple"])
        m = compute_bird_metrics(results)
        assert m["EX"] == 0.0

    def test_mixed_score(self):
        results = self._make_results(
            [1.0, 0.0], [1.0, 0.5], [1.0, 0.0], ["simple", "moderate"]
        )
        m = compute_bird_metrics(results)
        assert m["EX"] == 50.0
        assert abs(m["F1"] - 75.0) < 0.01

    def test_ves_formula(self):
        # VES = mean(sqrt(reward)) * 100
        # reward=1.0 for both → VES = mean(1.0, 1.0) * 100 = 100.0
        results = self._make_results([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], ["simple", "simple"])
        m = compute_bird_metrics(results)
        assert m["VES"] == 100.0

    def test_by_difficulty_split(self):
        results = self._make_results(
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            ["simple", "moderate", "challenging"],
        )
        m = compute_bird_metrics(results)
        assert m["by_difficulty"]["simple"]["count"] == 1
        assert m["by_difficulty"]["simple"]["EX"] == 100.0
        assert m["by_difficulty"]["moderate"]["EX"] == 0.0


# ── write_results ───────────────────────────────────────────────────────────────

class TestWriteResults:

    def test_creates_json_file(self, tmp_path: Path):
        results = [{"question_id": 0, "ex": 1.0, "f1": 1.0}]
        metrics = {"EX": 100.0, "total": 1}
        out = str(tmp_path / "out" / "results.json")
        write_results(results, metrics, out)
        assert Path(out).exists()
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        assert payload["metrics"]["EX"] == 100.0
        assert len(payload["questions"]) == 1

    def test_creates_parent_directories(self, tmp_path: Path):
        out = str(tmp_path / "deep" / "nested" / "bird.json")
        write_results([], {}, out)
        assert Path(out).exists()


# ── run_bird_eval integration ───────────────────────────────────────────────────

class TestRunBirdEval:
    """Integration-style tests — mocks LLM agents, uses real SQLite databases."""

    def _make_agents(self, pred_sql: str = "SELECT COUNT(*) FROM players"):
        sql_agent = MagicMock()
        sql_agent.generate.return_value = [pred_sql]
        sql_agent.rerank.return_value = pred_sql

        validator = MagicMock()
        validator.validate.return_value = {
            "valid": True,
            "syntax_ok": True,
            "safety_ok": True,
            "semantic_ok": True,
            "error": None,
        }
        return sql_agent, validator

    def test_correct_prediction(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents("SELECT COUNT(*) FROM players")
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert len(results) == 1
        assert results[0]["ex"] == 1.0
        assert results[0]["pred_sql"] == "SELECT COUNT(*) FROM players"

    def test_wrong_prediction(self, dev_json: str, db_dir_with_test_db: str):
        # Pred returns wrong result (empty set)
        sql_agent, validator = self._make_agents(
            "SELECT COUNT(*) FROM players WHERE rating > 9999"
        )
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert results[0]["ex"] == 0.0

    def test_no_sql_generated(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent = MagicMock()
        sql_agent.generate.return_value = []
        sql_agent.rerank.return_value = ""
        validator = MagicMock()

        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert results[0]["pred_sql"] == ""
        assert results[0]["ex"] == 0.0

    def test_invalid_sql_fails_gracefully(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, _ = self._make_agents("THIS IS NOT SQL")
        validator = MagicMock()
        validator.validate.return_value = {
            "valid": False, "error": "Syntax: invalid", "syntax_ok": False,
            "safety_ok": False, "semantic_ok": False,
        }
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert results[0]["valid"] is False
        assert results[0]["ex"] == 0.0

    def test_subset_limits_questions(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents()
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert len(results) == 1

    def test_difficulty_filter(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents()
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            difficulty_filter="moderate",
        )
        assert all(r["difficulty"] == "moderate" for r in results)

    def test_missing_db_file_yields_error_record(self, tmp_path: Path):
        # Write a dev.json referencing a nonexistent db
        data = [{"question_id": 0, "question": "Q", "evidence": "",
                 "SQL": "SELECT 1", "db_id": "no_such_db", "difficulty": "simple"}]
        dev_path = tmp_path / "dev.json"
        dev_path.write_text(json.dumps(data))
        sql_agent, validator = self._make_agents()
        results = run_bird_eval(
            dev_json_path=str(dev_path),
            db_dir=str(tmp_path),
            sql_agent=sql_agent,
            validator=validator,
        )
        assert len(results) == 1
        assert results[0]["ex"] == 0.0
        assert results[0]["gold_error"] is not None

    def test_generation_exception_caught(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent = MagicMock()
        sql_agent.generate.side_effect = RuntimeError("LLM failure")
        validator = MagicMock()
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        assert results[0]["pred_sql"] == ""
        assert results[0]["ex"] == 0.0

    def test_evidence_wrapped_in_domain_context(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents("SELECT MAX(rating) FROM players")
        run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=2,
        )
        # The second question has evidence="higher is better"; it must be promoted
        # into a <domain_context> block prepended to the question.
        call_args_list = sql_agent.generate.call_args_list
        second_call_query = call_args_list[1][1]["query"]  # keyword arg
        assert "<domain_context>higher is better</domain_context>" in second_call_query
        assert second_call_query.strip().startswith("<domain_context>")

    def test_no_evidence_omits_domain_context(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents("SELECT COUNT(*) FROM players")
        run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        # The first question has evidence=""; no empty block should be added.
        first_call_query = sql_agent.generate.call_args_list[0][1]["query"]
        assert "<domain_context>" not in first_call_query

    def test_result_record_has_all_fields(self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._make_agents()
        results = run_bird_eval(
            dev_json_path=dev_json,
            db_dir=db_dir_with_test_db,
            sql_agent=sql_agent,
            validator=validator,
            subset=1,
        )
        expected_keys = {
            "question_id", "db_id", "difficulty", "question", "evidence",
            "gold_sql", "pred_sql", "valid", "gold_error", "pred_error",
            "ex", "f1", "ves_reward", "gold_row_count", "pred_row_count",
            "gen_ms", "gold_elapsed_ms", "pred_elapsed_ms",
        }
        assert expected_keys.issubset(results[0].keys())


# ── _error_record helper ─────────────────────────────────────────────────────────

class TestErrorRecord:

    def test_zeros_out_metrics(self):
        r = _error_record(99, "my_db", "simple", "Q?", "", "SELECT 1", "some error")
        assert r["ex"] == 0.0
        assert r["f1"] == 0.0
        assert r["ves_reward"] == 0.0
        assert r["pred_sql"] == ""
        assert r["gold_error"] == "some error"


# ── GH#116: restrict a run to one database ────────────────────────────────────

class TestDbIdFilter:
    """
    `--db-id` exists so a run can target one domain. The filter ORDER is the
    part worth pinning: dev.json is grouped by database, so applying `subset`
    first would take the first N questions and then narrow them, leaving
    `--db-id X --subset 20` evaluating however few of those 20 belong to X —
    usually zero.
    """

    QUESTIONS = [
        {"question_id": i, "db_id": db, "difficulty": diff,
         "question": "q", "SQL": "SELECT 1", "evidence": ""}
        for i, (db, diff) in enumerate([
            ("california_schools", "simple"),
            ("california_schools", "moderate"),
            ("california_schools", "moderate"),
            ("debit_card_specializing", "simple"),
            ("debit_card_specializing", "moderate"),
            ("student_club", "challenging"),
        ])
    ]

    def _selected(self, tmp_path, **kwargs):
        """The questions run_bird_eval would evaluate, without running one."""
        import json
        from unittest.mock import patch
        import backend.eval.bird as B
        seen = []
        dev = tmp_path / "dev.json"
        dev.write_text(json.dumps(self.QUESTIONS), encoding="utf-8")

        def spy(db_path):
            raise RuntimeError("stop")          # abort before any LLM work

        with patch.object(B, "extract_sqlite_schema", side_effect=spy):
            out = B.run_bird_eval(dev_json_path=str(dev), db_dir=str(tmp_path),
                                  sql_agent=None, validator=None, **kwargs)
        return [r["db_id"] for r in out], [r["question_id"] for r in out]

    def test_one_database_only(self, tmp_path):
        dbs, _ = self._selected(tmp_path, db_filter="debit_card_specializing")
        assert set(dbs) == {"debit_card_specializing"}
        assert len(dbs) == 2

    def test_db_and_difficulty_combine(self, tmp_path):
        dbs, ids = self._selected(tmp_path, db_filter="california_schools",
                                  difficulty_filter="moderate")
        assert ids == [1, 2]

    def test_subset_applies_after_the_filters(self, tmp_path):
        """The ordering bug: with subset first, this would return nothing —
        the first 2 questions are california_schools."""
        dbs, ids = self._selected(tmp_path, db_filter="debit_card_specializing",
                                  subset=2)
        assert ids == [3, 4]

    def test_an_unknown_db_id_selects_nothing_rather_than_everything(self, tmp_path):
        dbs, _ = self._selected(tmp_path, db_filter="typo_not_a_db")
        assert dbs == []

    def test_no_filter_keeps_every_question(self, tmp_path):
        dbs, _ = self._selected(tmp_path)
        assert len(dbs) == 6
