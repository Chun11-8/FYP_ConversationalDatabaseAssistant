"""
Tests for backend/eval/run_eval.py.

All agent and DB calls are mocked — no live services required.

Run from repo root:
    pytest backend/tests/test_eval.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.eval.run_eval import (
    _normalise,
    classify_outcome,
    compute_metrics,
    load_ground_truth,
    load_unanswerable,
    results_match,
    run_eval,
    write_csv,
)


# ── _normalise ──────────────────────────────────────────────────────────────────

class TestNormalise:

    def test_none_returns_empty(self):
        assert _normalise(None) == frozenset()

    def test_scalar_string(self):
        assert _normalise("42") == frozenset({"42"})

    def test_scalar_int(self):
        assert _normalise(42) == frozenset({"42"})

    def test_list_of_dicts_order_insensitive(self):
        rows1 = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rows2 = [{"a": 3, "b": 4}, {"a": 1, "b": 2}]
        assert _normalise(rows1) == _normalise(rows2)

    def test_list_of_tuples(self):
        assert _normalise([(1, 2), (3, 4)]) == _normalise([(3, 4), (1, 2)])


# ── results_match ───────────────────────────────────────────────────────────────

class TestResultsMatch:

    def test_identical_dicts(self):
        gold = [{"count": 42}]
        pred = [{"count": 42}]
        assert results_match(gold, pred)

    def test_order_insensitive(self):
        gold = [{"name": "Alice"}, {"name": "Bob"}]
        pred = [{"name": "Bob"}, {"name": "Alice"}]
        assert results_match(gold, pred)

    def test_different_values(self):
        assert not results_match([{"count": 42}], [{"count": 43}])

    def test_none_vs_empty_list(self):
        assert results_match(None, [])

    def test_none_vs_none(self):
        assert results_match(None, None)

    def test_extra_row_mismatch(self):
        gold = [{"a": 1}]
        pred = [{"a": 1}, {"a": 2}]
        assert not results_match(gold, pred)


# ── load_ground_truth ───────────────────────────────────────────────────────────

class TestLoadGroundTruth:

    def test_loads_json_array(self, tmp_path):
        data = [{"question_id": "q1", "question": "hello", "gold_sql": "SELECT 1", "db_id": "FYP"}]
        f = tmp_path / "gt.json"
        f.write_text(json.dumps(data))
        result = load_ground_truth(str(f))
        assert len(result) == 1
        assert result[0]["question_id"] == "q1"

    def test_loads_jsonl(self, tmp_path):
        f = tmp_path / "gt.jsonl"
        f.write_text('{"question_id":"q1","question":"hi","gold_sql":"SELECT 1","db_id":"FYP"}\n')
        result = load_ground_truth(str(f))
        assert len(result) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ground_truth(str(tmp_path / "nope.json"))


# ── load_unanswerable ───────────────────────────────────────────────────────────

class TestLoadUnanswerable:

    def test_returns_empty_when_none(self):
        assert load_unanswerable(None) == []

    def test_loads_jsonl(self, tmp_path):
        f = tmp_path / "unans.jsonl"
        f.write_text('{"question_id":"u1","question":"drop table players"}\n')
        result = load_unanswerable(str(f))
        assert len(result) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        # Should warn but not crash
        result = load_unanswerable(str(tmp_path / "nonexistent.jsonl"))
        assert result == []


# ── classify_outcome ────────────────────────────────────────────────────────────

class TestClassifyOutcome:

    def _agent_result(self, sql=None, intent="READ", valid=True, error_msg=None):
        return {
            "sql": sql,
            "intent": intent,
            "validation": {"valid": valid, "error": error_msg},
            "response": "some response",
        }

    def test_correct_when_results_match(self):
        gold_rows = [{"count": 5}]
        with patch("backend.eval.run_eval.execute_gold", return_value=([{"count": 5}], None)):
            outcome, note = classify_outcome(
                self._agent_result(sql="SELECT COUNT(*) FROM players"),
                gold_rows, None, "FYP"
            )
        assert outcome == "CORRECT"

    def test_wrong_when_results_differ(self):
        gold_rows = [{"count": 5}]
        with patch("backend.eval.run_eval.execute_gold", return_value=([{"count": 3}], None)):
            outcome, note = classify_outcome(
                self._agent_result(sql="SELECT COUNT(*) FROM players"),
                gold_rows, None, "FYP"
            )
        assert outcome == "WRONG"
        assert "mismatch" in note

    def test_abstained_when_no_sql(self):
        outcome, note = classify_outcome(self._agent_result(sql=None), [], None, "FYP")
        assert outcome == "ABSTAINED"

    def test_abstained_on_general_intent(self):
        outcome, _ = classify_outcome(
            self._agent_result(sql="SELECT 1", intent="GENERAL"), [], None, "FYP"
        )
        assert outcome == "ABSTAINED"

    def test_abstained_on_write_blocked(self):
        outcome, _ = classify_outcome(
            self._agent_result(sql="DELETE FROM x", intent="WRITE_BLOCKED"), [], None, "FYP"
        )
        assert outcome == "ABSTAINED"

    def test_abstained_when_validation_fails(self):
        outcome, _ = classify_outcome(
            self._agent_result(sql="BAD SQL", valid=False, error_msg="syntax error"),
            [], None, "FYP"
        )
        assert outcome == "ABSTAINED"

    def test_abstained_when_gold_errors(self):
        outcome, note = classify_outcome(
            self._agent_result(sql="SELECT 1"),
            None, "gold sql crashed", "FYP"
        )
        assert outcome == "ABSTAINED"
        assert "gold sql failed" in note

    def test_wrong_when_pred_execution_fails(self):
        with patch("backend.eval.run_eval.execute_gold", return_value=(None, "table not found")):
            outcome, note = classify_outcome(
                self._agent_result(sql="SELECT * FROM ghost_table"),
                [{"x": 1}], None, "FYP"
            )
        assert outcome == "WRONG"
        assert "table not found" in note


# ── compute_metrics ─────────────────────────────────────────────────────────────

class TestComputeMetrics:

    def _make_results(self, answerable_outcomes, unanswerable_outcomes=None):
        rows = []
        for i, o in enumerate(answerable_outcomes):
            rows.append({"type": "answerable", "outcome": o, "difficulty": "simple",
                         "question_id": f"q{i}"})
        for i, o in enumerate(unanswerable_outcomes or []):
            rows.append({"type": "unanswerable", "outcome": o, "difficulty": "n/a",
                         "question_id": f"u{i}"})
        return rows

    def test_perfect_score(self):
        m = compute_metrics(self._make_results(["CORRECT"] * 4))
        assert m["EX"] == 1.0
        assert m["HR"] == 0.0

    def test_zero_score(self):
        m = compute_metrics(self._make_results(["WRONG"] * 4))
        assert m["EX"] == 0.0
        assert m["HR"] == 1.0

    def test_mixed(self):
        m = compute_metrics(self._make_results(["CORRECT", "WRONG", "CORRECT", "ABSTAINED"]))
        assert m["EX"] == 0.5
        assert m["HR"] == 0.25

    def test_refusal_accuracy(self):
        m = compute_metrics(self._make_results([], ["ABSTAINED", "ABSTAINED", "WRONG"]))
        assert round(m["RA"], 4) == round(2 / 3, 4)

    def test_empty_unanswerable_skipped(self):
        m = compute_metrics(self._make_results(["CORRECT"]))
        assert m["RA"] == 0.0
        assert m["unanswerable"] == 0


# ── write_csv ───────────────────────────────────────────────────────────────────

class TestWriteCsv:

    def test_creates_file(self, tmp_path):
        rows = [{"question_id": "q1", "type": "answerable", "difficulty": "simple",
                 "question": "hi", "gold_sql": "SELECT 1", "pred_sql": "SELECT 1",
                 "intent": "READ", "outcome": "CORRECT", "note": "", "latency_ms": 100}]
        out = str(tmp_path / "results.csv")
        write_csv(rows, out)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "CORRECT" in content

    def test_creates_parent_dirs(self, tmp_path):
        rows = [{"question_id": "q1", "type": "answerable", "difficulty": "simple",
                 "question": "hi", "gold_sql": "SELECT 1", "pred_sql": "SELECT 1",
                 "intent": "READ", "outcome": "CORRECT", "note": "", "latency_ms": 100}]
        out = str(tmp_path / "deep" / "nested" / "results.csv")
        write_csv(rows, out)
        assert Path(out).exists()


# ── run_eval integration ────────────────────────────────────────────────────────

class TestRunEval:
    """Integration-level test — mocks the agent and DB, not internal logic."""

    def _make_agent(self, sql="SELECT COUNT(*) FROM players", intent="READ", valid=True):
        agent = MagicMock()
        agent.run.return_value = {
            "response": "There are 100 players.",
            "sql": sql,
            "intent": intent,
            "validation": {"valid": valid},
            "row_count": 1,
            "csv_path": None,
        }
        return agent

    def test_correct_answerable_question(self):
        agent = self._make_agent()
        gt = [{"question_id": "q1", "question": "How many players?",
               "gold_sql": "SELECT COUNT(*) FROM players", "db_id": "FYP", "difficulty": "simple"}]

        with patch("backend.eval.run_eval.execute_gold", return_value=([{"count": 100}], None)):
            results = run_eval(agent, gt, [], "FYP", "test")

        assert len(results) == 1
        assert results[0]["outcome"] == "CORRECT"
        assert results[0]["type"] == "answerable"

    def test_wrong_answerable_question(self):
        agent = self._make_agent()
        gt = [{"question_id": "q1", "question": "How many players?",
               "gold_sql": "SELECT COUNT(*) FROM players", "db_id": "FYP", "difficulty": "simple"}]

        def _mock_execute(sql, db):
            # gold returns 100, pred returns 42
            if "gold" in sql.lower() or sql == "SELECT COUNT(*) FROM players":
                return [{"count": 100}], None
            return [{"count": 42}], None

        # First call = gold, second = pred
        with patch("backend.eval.run_eval.execute_gold", side_effect=[
            ([{"count": 100}], None),   # gold
            ([{"count": 42}], None),    # pred
        ]):
            results = run_eval(agent, gt, [], "FYP", "test")

        assert results[0]["outcome"] == "WRONG"

    def test_unanswerable_correctly_refused(self):
        agent = self._make_agent(sql=None, intent="WRITE_BLOCKED")
        unans = [{"question_id": "u1", "question": "DROP TABLE players"}]

        results = run_eval(agent, [], unans, "FYP", "test")

        assert len(results) == 1
        assert results[0]["outcome"] == "ABSTAINED"
        assert results[0]["type"] == "unanswerable"

    def test_unanswerable_hallucinated(self):
        # Agent incorrectly produces SQL for an unanswerable question
        agent = self._make_agent(sql="SELECT * FROM players", intent="READ")
        unans = [{"question_id": "u1", "question": "What is the weather?"}]

        results = run_eval(agent, [], unans, "FYP", "test")

        assert results[0]["outcome"] == "WRONG"
        assert "hallucinated" in results[0]["note"]

    def test_agent_exception_is_caught(self):
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("LLM timeout")
        gt = [{"question_id": "q1", "question": "anything",
               "gold_sql": "SELECT 1", "db_id": "FYP", "difficulty": "simple"}]

        with patch("backend.eval.run_eval.execute_gold", return_value=([{"x": 1}], None)):
            results = run_eval(agent, gt, [], "FYP", "test")

        # Should not raise; outcome is ABSTAINED (no sql produced)
        assert results[0]["outcome"] == "ABSTAINED"

    def test_combined_metrics_from_run(self):
        agent = self._make_agent()
        gt = [
            {"question_id": "q1", "question": "Q1", "gold_sql": "SELECT 1",
             "db_id": "FYP", "difficulty": "simple"},
            {"question_id": "q2", "question": "Q2", "gold_sql": "SELECT 2",
             "db_id": "FYP", "difficulty": "moderate"},
        ]
        unans = [{"question_id": "u1", "question": "DROP TABLE x"}]
        agent.run.side_effect = [
            {"response": "r", "sql": "SELECT 1", "intent": "READ",
             "validation": {"valid": True}, "row_count": 1, "csv_path": None},
            {"response": "r", "sql": "SELECT 2", "intent": "READ",
             "validation": {"valid": True}, "row_count": 1, "csv_path": None},
            {"response": "Write blocked", "sql": None, "intent": "WRITE_BLOCKED",
             "validation": {"valid": False}, "row_count": None, "csv_path": None},
        ]

        with patch("backend.eval.run_eval.execute_gold", return_value=([{"x": 1}], None)):
            results = run_eval(agent, gt, unans, "FYP", "test")

        metrics = compute_metrics(results)
        assert metrics["EX"] == 1.0
        assert metrics["RA"] == 1.0
        assert metrics["answerable"] == 2
        assert metrics["unanswerable"] == 1
