"""
Tests for the failure-cluster fixes derived from the moderate-50 analysis:

  - gold-dialect repair (MySQL gold on SQLite made 8/50 unwinnable)
  - evidence pseudo-function translation (SUBTRACT/DIVIDE/Total → operators)
  - identifier reconciliation (space/hyphen columns un-mangled + quoted)
  - output-shape + conditional-aggregation prompt coverage

All LLM calls are mocked; SQLite DBs are built in tmp dirs. Runs offline.

Run from repo root:
    pytest backend/tests/test_failure_clusters.py -v
"""

from __future__ import annotations

import json
import re
import sqlite3
from unittest.mock import MagicMock

import pytest
import sqlglot

from backend.agents.sql_code_agent import (
    SQLCodeAgent,
    DEFAULT_FEWSHOTS,
    translate_pseudo_functions,
    reconcile_identifiers,
)
from backend.agents.sql_validator_agent import SQLValidatorAgent
from backend.agents.utils import PromptHarness
from backend.eval.bird import gold_to_sqlite, compute_bird_metrics, run_bird_eval


# ── gold-dialect repair ───────────────────────────────────────────────────────

class TestGoldDialectRepair:

    def test_date_format_transpiles_and_executes_correctly(self):
        gold = "SELECT DATE_FORMAT(CAST(`Birthday` AS DATETIME), '%Y') FROM `Patient`"
        fixed = gold_to_sqlite(gold)
        assert fixed and "STRFTIME" in fixed.upper()
        # The CAST(... AS DATETIME) must be STRIPPED — in SQLite it numerically
        # coerces an ISO date and silently corrupts STRFTIME results.
        assert "DATETIME" not in fixed.upper()
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE Patient (Birthday TEXT); INSERT INTO Patient VALUES ('1980-05-02');"
        )
        assert conn.execute(fixed).fetchall() == [("1980",)]

    def test_untranspilable_returns_none_or_original(self):
        assert gold_to_sqlite("SELECT ((((") in (None, "SELECT ((((")

    def test_run_bird_eval_repairs_gold(self, tmp_path):
        db_dir = tmp_path / "dbs"
        (db_dir / "d1").mkdir(parents=True)
        conn = sqlite3.connect(db_dir / "d1" / "d1.sqlite")
        conn.executescript(
            "CREATE TABLE p (b TEXT); INSERT INTO p VALUES ('1980-05-02');"
        )
        conn.commit(); conn.close()
        dev = tmp_path / "dev.json"
        dev.write_text(json.dumps([{
            "question_id": 1, "db_id": "d1", "difficulty": "moderate",
            "question": "year?", "evidence": "",
            # MySQL-dialect gold — errors on SQLite without the repair
            "SQL": "SELECT DATE_FORMAT(CAST(`b` AS DATETIME), '%Y') FROM `p`",
        }]))
        agent = MagicMock()
        agent.generate.return_value = ["SELECT STRFTIME('%Y', b) FROM p"]
        agent.rerank.side_effect = lambda sqls, q, s: sqls[0]
        validator = SQLValidatorAgent({"db": {"dialect": "sqlite"}})
        results = run_bird_eval(str(dev), str(db_dir), agent, validator, refine=False)
        r = results[0]
        assert r["gold_error"] is None
        assert r["gold_transpiled"] is True
        assert r["ex"] == 1.0   # previously unwinnable

    def test_metrics_report_gold_errors_and_winnable_ex(self):
        results = [
            {"ex": 1.0, "f1": 1.0, "ves_reward": 1.0, "difficulty": "moderate",
             "gold_error": None},
            {"ex": 0.0, "f1": 0.0, "ves_reward": 0.0, "difficulty": "moderate",
             "gold_error": "no such function: DATE_FORMAT"},
        ]
        m = compute_bird_metrics(results)
        assert m["gold_errors"] == 1
        assert m["EX_winnable"] == 100.0   # 1/1 on the winnable subset
        assert m["EX"] == 50.0


# ── pseudo-function translation ───────────────────────────────────────────────

class TestPseudoFunctions:

    def test_nested_divide_subtract(self):
        sql = ("SELECT DIVIDE(SUM(SUBTRACT(YEAR(CURRENT_TIMESTAMP), YEAR(Birthday))), "
               "COUNT(ID)) FROM Patient WHERE SEX = 'M'")
        out = translate_pseudo_functions(sql)
        assert "DIVIDE" not in out.upper() and "SUBTRACT" not in out.upper()
        sqlglot.parse_one(out, dialect="sqlite")   # parses cleanly

    def test_total_becomes_sum(self):
        out = translate_pseudo_functions("SELECT Total(price) / Total(amount) FROM t")
        assert "SUM(price)" in out and "SUM(amount)" in out

    def test_plain_sql_unchanged(self):
        sql = "SELECT AVG(c) / 12 FROM t WHERE d = '201208'"
        assert translate_pseudo_functions(sql) == sql

    def test_idempotent(self):
        once = translate_pseudo_functions("SELECT SUBTRACT(a, b) FROM t")
        assert translate_pseudo_functions(once) == once

    def test_divide_casts_numerator_to_real(self):
        # Integer/integer division truncates in SQLite/Postgres; a REAL cast
        # on the numerator preserves the fractional ratio BIRD evidence means.
        out = translate_pseudo_functions("SELECT DIVIDE(male, total) FROM t")
        assert out == "SELECT (CAST(male AS REAL) / total) FROM t"
        # And it must remain idempotent (no double DIVIDE / re-cast pass).
        assert translate_pseudo_functions(out) == out

    def test_string_literal_with_pseudo_name_is_untouched(self):
        # A value that merely contains 'Total(' / 'DIVIDE(' must NOT be rewritten.
        sql = "SELECT name FROM t WHERE label = 'Grand Total(2020)'"
        assert translate_pseudo_functions(sql) == sql
        sql2 = "SELECT name FROM t WHERE note = 'use DIVIDE(a, b)'"
        assert translate_pseudo_functions(sql2) == sql2

    def test_unbalanced_parens_left_untouched(self):
        sql = "SELECT SUBTRACT(a, b FROM t"
        assert translate_pseudo_functions(sql) == sql

    def test_applied_via_normalize_for_all_dialects(self):
        for dialect in ("postgres", "sqlite"):
            agent = SQLCodeAgent({"db": {"dialect": dialect}}, MagicMock())
            out = agent.normalize_dialect("SELECT DIVIDE(SUM(x), COUNT(*)) FROM t")
            assert "DIVIDE" not in out.upper()


# ── identifier reconciliation ─────────────────────────────────────────────────

BIRD_SCHEMA = (
    "Table: Examination\n"
    "Columns:\n"
    "  - ID: INTEGER\n"
    "  - Examination Date: DATE\n"
    "  - aCL IgM: REAL\n\n"
    "Table: Laboratory\n"
    "Columns:\n"
    "  - T-CHO: INTEGER\n"
    "  - GPT: INTEGER"
)


class TestIdentifierReconciliation:

    def test_underscore_variant_rewritten(self):
        out = reconcile_identifiers(
            "SELECT Examination_Date FROM Examination", BIRD_SCHEMA
        )
        assert '"Examination Date"' in out and "Examination_Date" not in out

    def test_spaced_hyphen_rewritten(self):
        out = reconcile_identifiers(
            "SELECT * FROM Laboratory WHERE T - CHO >= 250", BIRD_SCHEMA
        )
        assert '"T-CHO"' in out

    def test_bare_real_name_gets_quoted(self):
        out = reconcile_identifiers(
            "SELECT aCL_IgM FROM Examination", BIRD_SCHEMA
        )
        assert '"aCL IgM"' in out

    def test_already_quoted_untouched(self):
        sql = 'SELECT "Examination Date" FROM Examination'
        assert reconcile_identifiers(sql, BIRD_SCHEMA) == sql

    def test_string_literal_never_touched(self):
        sql = "SELECT ID FROM Examination WHERE note = 'saw Examination Date today'"
        assert reconcile_identifiers(sql, BIRD_SCHEMA) == sql

    def test_normal_columns_untouched(self):
        sql = "SELECT GPT, ID FROM Laboratory"
        assert reconcile_identifiers(sql, BIRD_SCHEMA) == sql

    def test_result_parses_and_executes(self):
        out = reconcile_identifiers(
            "SELECT Examination_Date FROM Examination WHERE aCL_IgM > 1", BIRD_SCHEMA
        )
        sqlglot.parse_one(out, dialect="sqlite")
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            'CREATE TABLE Examination (ID INTEGER, "Examination Date" DATE, "aCL IgM" REAL);'
        )
        conn.execute(out)   # must not raise

    def test_generate_pipeline_applies_reconciliation(self):
        agent = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT Examination_Date FROM Examination\n```"
        )
        sqls = agent.generate(query="q", schema_context=BIRD_SCHEMA,
                              few_shots=["x"], history=[])
        assert '"Examination Date"' in sqls[0]


# ── prompt coverage ───────────────────────────────────────────────────────────

class TestPromptPacks:

    def test_output_conventions_in_prompt(self):
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "first_name, last_name" in p          # two-column full name
        assert "THEN 1 ELSE 0" in p                  # boolean convention
        assert "concatenate" in p.lower()

    def test_percentage_pattern_in_prompt(self):
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "CASE WHEN" in p
        assert "100%" in p                            # the WHERE-filter warning
        assert "unfiltered" in p.lower()

    def test_pseudo_code_and_quoting_rules_in_prompt(self):
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "PSEUDO-CODE" in p
        assert '"T-CHO"' in p or "T-CHO" in p

    def test_reranker_covers_new_classes(self):
        p = PromptHarness.get_file("sql_reranker.md")
        low = p.lower()
        assert "concatenate" in low
        assert "1/0" in p or "case when" in low
        assert "100%" in p

    def test_new_fewshots_present_and_valid_sqlite(self):
        joined = "\n".join(DEFAULT_FEWSHOTS)
        # The neutral schema renames these, but the LESSON is the same —
        # two columns selected separately, never concatenated into one.
        assert "t2.given_name, t2.family_name" in joined
        assert "CASE WHEN" in joined
        for fs in DEFAULT_FEWSHOTS:
            sql = re.search(r"```sql\n(.*?)```", fs, re.DOTALL).group(1).strip()
            sqlglot.parse_one(sql, dialect="sqlite")
