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
import time
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.eval.bird import (
    _error_record,
    select_questions,
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


# ── GH#119: per-question wall-clock budget ──────────────────────────────────────

class TestQuestionBudget:
    """
    A stalled question must not hold the run — and, just as importantly, must
    not quietly vanish from the score. See `_timeout_record`.
    """

    def _slow_agents(self, delay: float, pred_sql: str = "SELECT COUNT(*) FROM players"):
        """A generator that blocks for `delay` seconds before answering."""
        sql_agent = MagicMock()

        def _slow_generate(**_kwargs):
            time.sleep(delay)
            return [pred_sql]

        sql_agent.generate.side_effect = _slow_generate
        sql_agent.rerank.return_value = pred_sql
        validator = MagicMock()
        validator.validate.return_value = {
            "valid": True, "syntax_ok": True, "safety_ok": True,
            "semantic_ok": True, "error": None,
        }
        return sql_agent, validator

    def test_a_stalled_question_is_abandoned_and_the_run_continues(
            self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._slow_agents(delay=5.0)
        t0 = time.perf_counter()
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            question_timeout=0.3, refine=False,
        )
        elapsed = time.perf_counter() - t0
        # Both questions still produce a record, and neither waited 5s.
        assert len(results) == 2
        assert elapsed < 4.0, "the budget did not actually release the loop"
        assert all(r["timed_out"] for r in results)

    def test_an_abandoned_question_stays_in_the_denominator(
            self, dev_json: str, db_dir_with_test_db: str):
        """
        The load-bearing assertion. compute_bird_metrics drops `gold_error`
        records from the denominator because they are unwinnable. If a timeout
        were reported that way, abandoning the slowest questions would RAISE
        the reported EX — a model that stalls would outscore one that answers
        slowly. A timeout is the model's failure, so it scores 0 and counts.
        """
        sql_agent, validator = self._slow_agents(delay=5.0)
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            question_timeout=0.3, refine=False,
        )
        metrics = compute_bird_metrics(results)
        assert metrics["total"] == 2
        assert metrics["EX"] == 0.0
        assert metrics["gold_errors"] == 0          # NOT counted as unwinnable
        assert metrics["EX_winnable"] == 0.0        # denominator intact
        assert metrics["timeouts"] == 2
        assert metrics["error_categories"]["timeout"] == 2

    def test_the_failure_is_recorded_as_a_prediction_error(
            self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._slow_agents(delay=5.0)
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, question_timeout=0.3, refine=False,
        )
        rec = results[0]
        assert rec["gold_error"] is None
        assert "budget" in (rec["pred_error"] or "")
        assert rec["ex"] == 0.0 and rec["f1"] == 0.0 and rec["valid"] is False
        # The question is still identifiable in the output file.
        assert rec["question_id"] == 0 and rec["db_id"] == "test_db"

    def test_a_fast_question_is_untouched(
            self, dev_json: str, db_dir_with_test_db: str):
        sql_agent, validator = self._slow_agents(delay=0.0)
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, question_timeout=30.0,
        )
        assert results[0]["ex"] == 1.0
        assert not results[0].get("timed_out")

    def test_zero_disables_the_budget(
            self, dev_json: str, db_dir_with_test_db: str):
        """Off must mean off — the pre-GH#119 behaviour, for a reproduction
        run that must not have questions abandoned under it."""
        sql_agent, validator = self._slow_agents(delay=0.4)
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, question_timeout=0,
        )
        assert results[0]["ex"] == 1.0

    def test_token_accounting_survives_the_worker_thread(
            self, dev_json: str, db_dir_with_test_db: str):
        """
        The budgeted work runs on another thread and usage tracking is
        thread-local, so without an explicit `usage.bind` every run would
        report zero tokens and zero cost while looking entirely healthy.
        """
        from backend.agents import usage as usage_tracking

        sql_agent = MagicMock()

        def _generate(**_kwargs):
            tracker = usage_tracking.current()
            assert tracker is not None, "tracker not bound in the worker thread"
            tracker.record("generate", "gpt-4o-mini", 100, 20)
            return ["SELECT COUNT(*) FROM players"]

        sql_agent.generate.side_effect = _generate
        sql_agent.rerank.return_value = "SELECT COUNT(*) FROM players"
        validator = MagicMock()
        validator.validate.return_value = {"valid": True, "error": None}

        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, question_timeout=30.0, refine=False,
        )
        assert results[0]["usage"]["input_tokens"] == 100
        assert results[0]["usage"]["output_tokens"] == 20

    def test_a_slow_refine_keeps_the_original_prediction(
            self, dev_json: str, db_dir_with_test_db: str):
        """
        Generation succeeded; refine is an optional improvement. Dropping the
        whole question because the optional step overran would turn a scorable
        answer into a zero.
        """
        sql_agent = MagicMock()
        # Valid SQL that returns 0 rows → the refine trigger.
        sql_agent.generate.return_value = ["SELECT id FROM players WHERE id < 0"]
        sql_agent.rerank.return_value = "SELECT id FROM players WHERE id < 0"

        def _slow_regenerate(**_kwargs):
            time.sleep(5.0)
            return ["SELECT COUNT(*) FROM players"]

        sql_agent.regenerate.side_effect = _slow_regenerate
        validator = MagicMock()
        validator.validate.return_value = {"valid": True, "error": None}

        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, question_timeout=2.0, refine=True, exec_voting=False,
        )
        rec = results[0]
        assert not rec.get("timed_out")
        assert rec["pred_sql"] == "SELECT id FROM players WHERE id < 0"
        assert rec["refined"] is False
        assert "budget" in (rec["timeout_note"] or "")

    def test_an_exception_still_propagates_through_the_budget(self):
        """The budget must not swallow a real error into a fake timeout."""
        from backend.eval.bird import _Budget

        def _boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            _Budget(30.0).run(_boom, "test")

    def test_the_budget_is_shared_across_phases_not_reset(self):
        """A 60s budget means 60s per QUESTION, not 60s per call — otherwise
        ten calls could legitimately take ten minutes."""
        from backend.eval.bird import _Budget

        b = _Budget(1.0)
        first = b.remaining()
        time.sleep(0.3)
        second = b.remaining()
        assert second < first
        assert b.expired() is False
        time.sleep(0.8)
        assert b.expired() is True


# ── GH#120: cross-domain question selection ─────────────────────────────────────

class TestSelectQuestions:
    """
    mini_dev_sqlite.json stores each database as one CONTIGUOUS block, so
    slicing the front of the file returns a single database. A 20-question run
    with no --db-id therefore reads as an overall score while measuring one
    domain. These tests pin the strategies that fix that.
    """

    @staticmethod
    def _corpus():
        """11 databases in contiguous blocks, as the real file is laid out."""
        qs = []
        qid = 0
        for i, db in enumerate(f"db{n:02d}" for n in range(11)):
            for j in range(10 + i):          # uneven sizes, like the real set
                qs.append({
                    "question_id": qid, "db_id": db,
                    "difficulty": ["simple", "moderate", "challenging"][j % 3],
                    "question": f"q{qid}", "SQL": "SELECT 1", "evidence": "",
                })
                qid += 1
        return qs

    def test_head_reproduces_the_one_database_problem(self):
        """Not a bug to fix but the documented default — asserted so the
        motivation for the other strategies stays visible."""
        picked = select_questions(self._corpus(), subset=10, strategy="head")
        assert len({q["db_id"] for q in picked}) == 1

    def test_head_is_unchanged_from_the_pre_gh120_behaviour(self):
        """Existing result files must stay reproducible."""
        corpus = self._corpus()
        assert select_questions(corpus, subset=7, strategy="head") == corpus[:7]

    def test_stratified_covers_every_database(self):
        corpus = self._corpus()
        picked = select_questions(corpus, subset=22, strategy="stratified")
        assert len(picked) == 22
        assert len({q["db_id"] for q in picked}) == 11

    def test_stratified_gives_each_database_a_slot_when_it_can(self):
        """A 'cross-domain' sample that silently omits a domain is the failure
        this strategy exists to prevent, so the small databases get a slot
        before the large ones get a second."""
        picked = select_questions(self._corpus(), subset=11, strategy="stratified")
        assert sorted(q["db_id"] for q in picked) == sorted(
            f"db{n:02d}" for n in range(11))

    def test_stratified_cannot_over_or_under_deliver(self):
        corpus = self._corpus()
        for n in (1, 5, 11, 12, 30, len(corpus) - 1, len(corpus)):
            assert len(select_questions(corpus, subset=n,
                                        strategy="stratified")) == n

    def test_stratified_is_proportional_not_equal(self):
        """Equal-per-database would misrepresent the benchmark's own mix; the
        allocation tracks each database's share."""
        corpus = self._corpus()
        picked = select_questions(corpus, subset=100, strategy="stratified")
        counts = Counter(q["db_id"] for q in picked)
        assert counts["db10"] > counts["db00"]      # 20 questions vs 10

    def test_random_mixes_databases(self):
        picked = select_questions(self._corpus(), subset=20, strategy="random")
        assert len(picked) == 20
        assert len({q["db_id"] for q in picked}) > 1

    def test_every_strategy_is_deterministic_under_one_seed(self):
        """A benchmark number that cannot be re-derived is not a measurement."""
        corpus = self._corpus()
        for strategy in ("random", "stratified"):
            a = select_questions(corpus, subset=15, strategy=strategy, seed=1)
            b = select_questions(corpus, subset=15, strategy=strategy, seed=1)
            assert [q["question_id"] for q in a] == [q["question_id"] for q in b]

    def test_a_different_seed_gives_a_different_sample(self):
        corpus = self._corpus()
        a = select_questions(corpus, subset=15, strategy="random", seed=1)
        b = select_questions(corpus, subset=15, strategy="random", seed=2)
        assert [q["question_id"] for q in a] != [q["question_id"] for q in b]

    def test_no_question_is_duplicated(self):
        for strategy in ("head", "random", "stratified"):
            picked = select_questions(self._corpus(), subset=30, strategy=strategy)
            ids = [q["question_id"] for q in picked]
            assert len(ids) == len(set(ids))

    def test_filters_still_run_before_subset(self):
        """GH#116's ordering must survive the new sampling — the other order
        would narrow whatever the sample happened to pick."""
        corpus = self._corpus()
        picked = select_questions(corpus, db_filter="db05", subset=5,
                                  strategy="stratified")
        assert len(picked) == 5
        assert {q["db_id"] for q in picked} == {"db05"}

    def test_both_filters_combine_with_sampling(self):
        picked = select_questions(self._corpus(), difficulty_filter="simple",
                                  subset=4, strategy="random")
        assert len(picked) == 4
        assert {q["difficulty"] for q in picked} == {"simple"}

    def test_a_subset_larger_than_the_pool_returns_everything(self):
        corpus = self._corpus()
        for strategy in ("head", "random", "stratified"):
            assert len(select_questions(corpus, subset=10_000,
                                        strategy=strategy)) == len(corpus)

    def test_no_subset_returns_everything_unshuffled(self):
        corpus = self._corpus()
        assert select_questions(corpus, strategy="stratified") == corpus

    def test_an_unknown_strategy_is_rejected_not_silently_ignored(self):
        """Falling back to 'head' on a typo would hand back a single-database
        sample while the run_config claims it was stratified."""
        with pytest.raises(ValueError, match="unknown sample strategy"):
            select_questions(self._corpus(), subset=5, strategy="startified")

    def test_run_bird_eval_threads_the_strategy_through(
            self, dev_json: str, db_dir_with_test_db: str):
        sql_agent = MagicMock()
        sql_agent.generate.return_value = ["SELECT COUNT(*) FROM players"]
        sql_agent.rerank.return_value = "SELECT COUNT(*) FROM players"
        validator = MagicMock()
        validator.validate.return_value = {"valid": True, "error": None}
        results = run_bird_eval(
            dev_json_path=dev_json, db_dir=db_dir_with_test_db,
            sql_agent=sql_agent, validator=validator,
            subset=1, sample="stratified", seed=3, refine=False,
        )
        assert len(results) == 1


# ── GH#126: schema caching and progress visibility ──────────────────────────────

class TestSchemaCacheAndProgress:
    """
    A run that prints nothing between result lines cannot be told apart from a
    hung one. These pin the per-question announcements and the schema cache.
    """

    def _agents(self):
        sql_agent = MagicMock()
        sql_agent.generate.return_value = ["SELECT COUNT(*) FROM players"]
        sql_agent.rerank.return_value = "SELECT COUNT(*) FROM players"
        sql_agent.top_k = 7
        validator = MagicMock()
        validator.validate.return_value = {"valid": True, "error": None}
        return sql_agent, validator

    def test_the_schema_is_extracted_once_per_database(
            self, dev_json: str, db_dir_with_test_db: str):
        """Both questions in the fixture share a database; the second must hit
        the cache rather than re-scanning every text column."""
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        sql_agent, validator = self._agents()
        with patch.object(B, "extract_sqlite_schema",
                          wraps=B.extract_sqlite_schema) as spy:
            results = run_bird_eval(
                dev_json_path=dev_json, db_dir=db_dir_with_test_db,
                sql_agent=sql_agent, validator=validator, refine=False)
        assert len(results) == 2
        assert spy.call_count == 1, "the schema was re-extracted per question"

    def test_the_cache_is_keyed_on_the_path_not_the_db_name(self, tmp_path: Path):
        """Two copies of a database under different paths are different
        schemas as far as a run is concerned."""
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        made = []
        for name in ("a", "b"):
            p = tmp_path / f"{name}.sqlite"
            conn = sqlite3.connect(p)
            conn.execute(f"CREATE TABLE t_{name} (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()
            made.append(str(p))
        first = B.cached_sqlite_schema(made[0])
        second = B.cached_sqlite_schema(made[1])
        assert first != second
        assert len(B._SCHEMA_CACHE) == 2

    def test_a_cache_hit_returns_the_identical_string(self, tmp_path: Path):
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        p = tmp_path / "one.sqlite"
        conn = sqlite3.connect(p)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()
        assert B.cached_sqlite_schema(str(p)) == B.cached_sqlite_schema(str(p))

    def test_extraction_failure_is_not_cached_as_success(self, tmp_path: Path):
        """A missing database must keep raising, not poison the cache with an
        empty schema that every later question would silently accept."""
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        missing = str(tmp_path / "nope.sqlite")
        with pytest.raises(FileNotFoundError):
            B.cached_sqlite_schema(missing)
        assert missing not in B._SCHEMA_CACHE

    def test_each_question_announces_itself_before_the_work(
            self, dev_json: str, db_dir_with_test_db: str, caplog):
        """The line that tells a slow question from a hung one."""
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        sql_agent, validator = self._agents()
        with caplog.at_level("INFO", logger="backend.eval.bird"):
            run_bird_eval(
                dev_json_path=dev_json, db_dir=db_dir_with_test_db,
                sql_agent=sql_agent, validator=validator, subset=1, refine=False)
        text = caplog.text
        assert "START qid=" in text
        assert "tier=simple" in text
        assert "generating" in text

    def test_the_start_line_names_the_tier(
            self, dev_json: str, db_dir_with_test_db: str, caplog):
        """Tier drives the candidate count (challenging is 7 against simple's
        2), which is the main reason a challenging run feels stalled."""
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        sql_agent, validator = self._agents()
        with caplog.at_level("INFO", logger="backend.eval.bird"):
            run_bird_eval(
                dev_json_path=dev_json, db_dir=db_dir_with_test_db,
                sql_agent=sql_agent, validator=validator,
                difficulty_filter="moderate", refine=False)
        assert "tier=moderate" in caplog.text

    def test_the_budget_is_shown_when_one_is_set(
            self, dev_json: str, db_dir_with_test_db: str, caplog):
        import backend.eval.bird as B
        B._SCHEMA_CACHE.clear()
        sql_agent, validator = self._agents()
        with caplog.at_level("INFO", logger="backend.eval.bird"):
            run_bird_eval(
                dev_json_path=dev_json, db_dir=db_dir_with_test_db,
                sql_agent=sql_agent, validator=validator, subset=1,
                question_timeout=90.0, refine=False)
        assert "budget=90s" in caplog.text
