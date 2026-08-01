"""
Tests for backend/finetune/prepare_data.py — fine-tuning JSONL construction.

All offline: fake bird_results.json fixtures + a real tmp SQLite db for
schema extraction (mirrors test_bird_predict.py's fixture style), no
network/API key needed.

Run from repo root:
    pytest backend/tests/test_finetune_prepare.py -v
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from unittest.mock import patch

from backend.agents.sql_code_agent import SQLCodeAgent
from backend.finetune.prepare_data import (
    build_example,
    examples_from_bird_results,
    render_postgres_schema,
    split_train_eval,
    write_jsonl,
)


@pytest.fixture()
def sql_agent() -> SQLCodeAgent:
    return SQLCodeAgent({"db": {"dialect": "sqlite"}}, llm=None)


@pytest.fixture()
def db_dir(tmp_path: Path) -> str:
    db_id_dir = tmp_path / "dev_databases" / "test_db"
    db_id_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_id_dir / "test_db.sqlite"))
    conn.execute("CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT, rating INTEGER)")
    conn.commit()
    conn.close()
    return str(tmp_path / "dev_databases")


@pytest.fixture()
def bird_results_file(tmp_path: Path) -> str:
    data = {
        "run_config": {"model": "gpt-4o-mini"},
        "metrics": {},
        "questions": [
            {"question_id": 1, "db_id": "test_db", "difficulty": "simple",
             "question": "How many players?", "evidence": "",
             "gold_sql": "SELECT COUNT(*) FROM players", "gold_error": None},
            {"question_id": 2, "db_id": "test_db", "difficulty": "moderate",
             "question": "Highest rating?", "evidence": "higher is better",
             "gold_sql": "SELECT MAX(rating) FROM players", "gold_error": None},
            {"question_id": 3, "db_id": "test_db", "difficulty": "challenging",
             "question": "Broken gold", "evidence": "",
             "gold_sql": "SELECT nonsense", "gold_error": "no such column: nonsense"},
        ],
    }
    p = tmp_path / "bird_results.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ── build_example ─────────────────────────────────────────────────────────────

class TestBuildExample:

    def test_message_shape(self, sql_agent):
        ex = build_example(sql_agent, "How many players?", "", "Table: players", "SELECT 1")
        assert list(ex.keys()) == ["messages"]
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_assistant_content_is_fenced_gold_sql(self, sql_agent):
        ex = build_example(sql_agent, "q", "", "schema", "SELECT COUNT(*) FROM players")
        assistant = ex["messages"][-1]["content"]
        assert assistant == "```sql\nSELECT COUNT(*) FROM players\n```"

    def test_evidence_wrapped_in_domain_context(self, sql_agent):
        ex = build_example(sql_agent, "q", "higher is better", "schema", "SELECT 1")
        user_msg = ex["messages"][1]["content"]
        assert "<domain_context>higher is better</domain_context>" in user_msg

    def test_no_evidence_omits_domain_context_wrapper(self, sql_agent):
        # The prompt's generic instructions legitimately mention
        # "<domain_context>" as boilerplate guidance regardless of whether
        # evidence was given — what must NOT happen is the actual
        # <user_question> block being wrapped in an empty tag.
        ex = build_example(sql_agent, "q", "", "schema", "SELECT 1")
        user_msg = ex["messages"][1]["content"]
        assert "<user_question>\nq\n</user_question>" in user_msg
        assert "<domain_context></domain_context>" not in user_msg

    def test_system_message_is_the_real_production_prompt(self, sql_agent):
        ex = build_example(sql_agent, "q", "", "schema", "SELECT 1")
        assert ex["messages"][0]["content"] == sql_agent._system_prompt()

    def test_gold_sql_whitespace_stripped(self, sql_agent):
        ex = build_example(sql_agent, "q", "", "schema", "  SELECT 1  \n\n")
        assert ex["messages"][-1]["content"] == "```sql\nSELECT 1\n```"


# ── examples_from_bird_results ────────────────────────────────────────────────

class TestExamplesFromBirdResults:

    def test_builds_one_example_per_winnable_question(self, sql_agent, bird_results_file, db_dir):
        examples = examples_from_bird_results(sql_agent, [bird_results_file], db_dir)
        # 3 questions total, 1 has gold_error -> 2 winnable
        assert len(examples) == 2

    def test_gold_error_question_skipped(self, sql_agent, bird_results_file, db_dir):
        examples = examples_from_bird_results(sql_agent, [bird_results_file], db_dir)
        golds = [ex["messages"][-1]["content"] for ex in examples]
        assert not any("nonsense" in g for g in golds)

    def test_difficulty_filter(self, sql_agent, bird_results_file, db_dir):
        examples = examples_from_bird_results(
            sql_agent, [bird_results_file], db_dir, min_difficulty=["simple"])
        assert len(examples) == 1
        assert "COUNT(*)" in examples[0]["messages"][-1]["content"]

    def test_no_db_dir_skips_all(self, sql_agent, bird_results_file):
        examples = examples_from_bird_results(sql_agent, [bird_results_file], db_dir=None)
        assert examples == []

    def test_multiple_result_files_combined(self, sql_agent, bird_results_file, db_dir):
        examples = examples_from_bird_results(
            sql_agent, [bird_results_file, bird_results_file], db_dir)
        assert len(examples) == 4  # 2 winnable x 2 files

    def test_schema_reused_across_questions_same_db(self, sql_agent, bird_results_file, db_dir):
        # Both winnable questions share db_id "test_db" -> same schema context
        examples = examples_from_bird_results(sql_agent, [bird_results_file], db_dir)
        user_msgs = [ex["messages"][1]["content"] for ex in examples]
        assert "players" in user_msgs[0] and "players" in user_msgs[1]


# ── write_jsonl / split_train_eval ──────────────────────────────────────────────

class TestWriteJsonl:

    def test_writes_one_line_per_example_valid_json(self, tmp_path):
        out = str(tmp_path / "nested" / "train.jsonl")
        examples = [{"messages": [{"role": "user", "content": "a"}]},
                    {"messages": [{"role": "user", "content": "b"}]}]
        write_jsonl(examples, out)
        lines = Path(out).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(l)["messages"] for l in lines)


class TestSplitTrainEval:

    def test_split_sizes_sum_to_input(self):
        examples = [{"i": i} for i in range(20)]
        train, eval_ = split_train_eval(examples, 0.25)
        assert len(train) + len(eval_) == 20
        assert len(eval_) == 5

    def test_zero_fraction_keeps_everything_in_train(self):
        examples = [{"i": i} for i in range(10)]
        train, eval_ = split_train_eval(examples, 0.0)
        assert len(train) == 10 and len(eval_) == 0

    def test_deterministic_for_fixed_seed(self):
        examples = [{"i": i} for i in range(20)]
        a = split_train_eval(examples, 0.3, seed=42)
        b = split_train_eval(examples, 0.3, seed=42)
        assert a == b

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            split_train_eval([{"i": 1}], 1.0)
        with pytest.raises(ValueError):
            split_train_eval([{"i": 1}], -0.1)


# ── render_postgres_schema ────────────────────────────────────────────────────

class TestRenderPostgresSchema:

    def test_renders_table_columns_and_pk(self):
        fake_info = {
            "db_name": "FYP", "table_count": 1,
            "tables": [{
                "table_name": "players", "column_count": 2,
                "columns": [
                    {"name": "id", "type": "INTEGER", "not_null": True,
                     "primary_key": True, "foreign_key": None},
                    {"name": "team_id", "type": "INTEGER", "not_null": False,
                     "primary_key": False, "foreign_key": "teams.id"},
                ],
            }],
        }
        with patch("backend.retrieval.data_processing.get_schema_info", return_value=fake_info):
            text = render_postgres_schema("FYP")
        assert "Table: players" in text
        assert "- id: INTEGER [PK, NOT NULL]" in text
        assert "- team_id: INTEGER" in text
        assert "players.team_id → teams.id" in text

    def test_no_foreign_keys_omits_section(self):
        fake_info = {
            "db_name": "FYP", "table_count": 1,
            "tables": [{
                "table_name": "t", "column_count": 1,
                "columns": [{"name": "id", "type": "INTEGER", "not_null": True,
                            "primary_key": True, "foreign_key": None}],
            }],
        }
        with patch("backend.retrieval.data_processing.get_schema_info", return_value=fake_info):
            text = render_postgres_schema("FYP")
        assert "Foreign Keys" not in text


# ── CLI argparse smoke test ───────────────────────────────────────────────────

class TestCliArgparse:

    def test_requires_a_data_source(self, capsys):
        from backend.finetune.prepare_data import main
        with pytest.raises(SystemExit):
            main(["--output", "out.jsonl"])
