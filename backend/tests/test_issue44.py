"""
Tests for Issue #27/GH#44: error telemetry, experiment tracking, F1-overlap
tie-break, and difficulty-adaptive prompting. All offline (mocked LLM/tmp SQLite).

Run from repo root:
    pytest backend/tests/test_issue44.py -v
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from backend.agents.db_agent import OrchestrationAgent
from backend.agents.sql_code_agent import SQLCodeAgent, TIER_STRATEGIES
from backend.agents.sql_validator_agent import SQLValidatorAgent
from backend.eval.bird import select_by_execution, write_results
from backend.eval.error_analysis import categorize, summarize


# ── error categorization ──────────────────────────────────────────────────────

class TestCategorize:

    def _cat(self, **kw):
        base = dict(ex=0.0, gold_error=None, pred_sql="SELECT 1", valid=True,
                    pred_error=None, pred_rows=[(1,)], gold_rows=[(2,)])
        base.update(kw)
        return categorize(**base)

    def test_correct(self):
        assert self._cat(ex=1.0) == "correct"

    def test_gold_error_wins_over_everything_else(self):
        assert self._cat(gold_error="no such function") == "gold_error"

    def test_no_pred(self):
        assert self._cat(pred_sql="") == "no_pred"

    def test_pred_invalid(self):
        assert self._cat(valid=False) == "pred_invalid"

    def test_pred_exec_error(self):
        assert self._cat(pred_error="timeout") == "pred_exec_error"

    def test_zero_rows(self):
        assert self._cat(pred_rows=[], gold_rows=[(1,)]) == "zero_rows"

    def test_arity_mismatch(self):
        # concatenated-name fingerprint: 1 column returned vs gold's 2
        assert self._cat(pred_rows=[("A B",)], gold_rows=[("A", "B")]) == "arity_mismatch"

    def test_superset_subset_partial_disjoint(self):
        g = [("a",), ("b",)]
        assert self._cat(pred_rows=[("a",), ("b",), ("c",)], gold_rows=g) == "superset"
        assert self._cat(pred_rows=[("a",)], gold_rows=g) == "subset"
        assert self._cat(pred_rows=[("a",), ("x",)], gold_rows=g) == "partial_overlap"
        assert self._cat(pred_rows=[("x",)], gold_rows=g) == "disjoint"

    def test_summarize_counts(self):
        results = [{"error_category": "correct"}, {"error_category": "disjoint"},
                   {"error_category": "correct"}, {}]
        assert summarize(results) == {"correct": 2, "disjoint": 1}


# ── experiment tracking ───────────────────────────────────────────────────────

class TestWriteResults:

    def test_versioned_copy_and_run_config(self, tmp_path):
        out = tmp_path / "bird_results.json"
        metrics = {"EX": 51.3}
        write_results([{"q": 1}], metrics, str(out), run_config={"model": "m"})
        payload = json.loads(out.read_text())
        assert payload["run_config"] == {"model": "m"}
        versioned = [p for p in tmp_path.iterdir() if "EX51.3" in p.name and p != out]
        assert len(versioned) == 1
        assert json.loads(versioned[0].read_text()) == payload


# ── F1-overlap tie-break ──────────────────────────────────────────────────────

class TestOverlapTieBreak:

    SCHEMA = "Table: t\nColumns:\n  - id: INTEGER [PK]\n  - name: TEXT\n  - grp: TEXT"

    @pytest.fixture()
    def db_path(self, tmp_path):
        p = tmp_path / "v.sqlite"
        conn = sqlite3.connect(p)
        conn.executescript(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, grp TEXT);"
            "INSERT INTO t VALUES (1,'a','g1'),(2,'b','g1'),(3,'c','g2');"
        )
        conn.commit(); conn.close()
        return str(p)

    @pytest.fixture()
    def validator(self):
        return SQLValidatorAgent({"db": {"dialect": "sqlite"}})

    def test_overlap_resolves_all_distinct_tie(self, db_path, validator):
        # Three distinct results: {a,b}, {a,b,c}, {c}. The first two overlap
        # heavily; {c} is the outlier. Overlap consensus must pick a/b family.
        cands = [
            "SELECT name FROM t WHERE grp = 'g1'",           # {a,b}
            "SELECT name FROM t",                            # {a,b,c}
            "SELECT name FROM t WHERE grp = 'g2'",           # {c}
        ]
        winner, info = select_by_execution(cands, db_path, validator, self.SCHEMA)
        assert winner in cands[:2]
        assert info.get("tie_break") == "overlap"

    def test_fully_disjoint_tie_still_falls_back(self, db_path, validator):
        cands = [
            "SELECT name FROM t WHERE grp = 'g1'",   # {a,b}
            "SELECT name FROM t WHERE grp = 'g2'",   # {c} — disjoint
        ]
        # both non-empty, disjoint → overlap 0 for both → text-judge fallback
        winner, info = select_by_execution(cands, db_path, validator, self.SCHEMA)
        assert winner is None
        assert info["fallback_reason"] == "tie between distinct result sets"

    def test_plurality_still_beats_overlap_path(self, db_path, validator):
        cands = [
            "SELECT name FROM t WHERE grp = 'g1'",
            "SELECT name FROM t WHERE grp IN ('g1')",   # same rows → 2 votes
            "SELECT name FROM t",
        ]
        winner, info = select_by_execution(cands, db_path, validator, self.SCHEMA)
        assert winner in cands[:2]
        assert "tie_break" not in info or info.get("tie_break") is None


# ── difficulty-adaptive prompting ─────────────────────────────────────────────

class TestAdaptivePrompting:

    def _agent(self, adaptive=True):
        # path_diversity off: these tests pin the TIER block / candidate count
        # of the direct prompt; GH#48 fan-out is covered in
        # test_path_diversity_selection.py.
        cfg = {"db": {"dialect": "sqlite"},
               "sql": {"n_candidates": 5, "adaptive_prompting": adaptive,
                       "path_diversity": False}}
        a = SQLCodeAgent(cfg, MagicMock())
        a.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        return a

    def _sent(self, agent):
        return agent.llm.invoke.call_args[0][0][1].content

    def test_simple_tier_prompt_and_count(self):
        a = self._agent()
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[],
                   tier="simple")
        sent = self._sent(a)
        assert "STRATEGY (SIMPLE" in sent
        assert "Generate 2 candidate queries" in sent

    def test_challenging_tier_prompt_and_count(self):
        a = self._agent()
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[],
                   tier="challenging")
        sent = self._sent(a)
        assert "STRATEGY (CHALLENGING" in sent
        assert "Generate 7 candidate queries" in sent
        assert "GRAIN" in sent  # the anti-AVG-of-AVG caveat survives in the block

    def test_moderate_and_none_identical_to_default(self):
        a = self._agent()
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[])
        default = self._sent(a)
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[],
                   tier="moderate")
        assert "STRATEGY (" not in self._sent(a)
        assert "Generate 5 candidate queries" in default

    def test_adaptive_disabled_ignores_tier(self):
        a = self._agent(adaptive=False)
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[],
                   tier="simple")
        sent = self._sent(a)
        assert "STRATEGY (" not in sent
        assert "Generate 5 candidate queries" in sent

    def test_unknown_tier_falls_back_to_default(self):
        a = self._agent()
        a.generate(query="q", schema_context="s", few_shots=["x"], history=[],
                   tier="weird")
        assert "STRATEGY (" not in self._sent(a)


# ── live tier heuristic ───────────────────────────────────────────────────────

class TestInferTier:

    def _tier(self, plan):
        return OrchestrationAgent._infer_tier(plan)

    def test_no_plan_none(self):
        assert self._tier(None) is None
        assert self._tier({}) is None

    def test_single_table_simple(self):
        assert self._tier({"tables": ["t"], "group_by": None}) == "simple"

    def test_set_op_challenging(self):
        assert self._tier({"tables": ["a", "b"], "set_op": "EXCEPT ..."}) == "challenging"

    def test_having_three_tables_challenging(self):
        plan = {"tables": ["a", "b", "c"], "having": "COUNT(*) > 20"}
        assert self._tier(plan) == "challenging"

    def test_two_table_join_moderate(self):
        assert self._tier({"tables": ["a", "b"]}) == "moderate"

    def test_null_strings_treated_unset(self):
        assert self._tier({"tables": ["t"], "group_by": "null", "having": "None"}) == "simple"


# ── round-2 quick wins (from the 3-tier run analysis) ─────────────────────────

class TestDateFuncRewrite:

    def _norm(self, sql):
        a = SQLCodeAgent({"db": {"dialect": "sqlite"}}, MagicMock())
        return a.normalize_dialect(sql)

    def test_year_comparison_rewritten(self):
        out = self._norm("SELECT COUNT(*) FROM p WHERE YEAR(Description) = '1997'")
        assert "STRFTIME('%Y', Description)" in out and "YEAR(" not in out.upper().replace("STRFTIME", "")

    def test_nested_year_subtraction(self):
        out = self._norm("SELECT YEAR(CURRENT_TIMESTAMP) - YEAR(Birthday) FROM p")
        assert out.count("STRFTIME('%Y'") == 2

    def test_month_rewritten(self):
        assert "STRFTIME('%m', d)" in self._norm("SELECT MONTH(d) FROM t")

    def test_postgres_dialect_untouched(self):
        a = SQLCodeAgent({"db": {"dialect": "postgres"}}, MagicMock())
        sql = "SELECT YEAR(d) FROM t"
        assert "STRFTIME" not in a.normalize_dialect(sql)


class TestRound2PromptRules:

    def test_ratio_cast_rule(self):
        from backend.agents.utils import PromptHarness
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "EVERY division of aggregates" in p
        assert "integer division" in " ".join(p.lower().split())

    def test_grouped_superlative_rule(self):
        from backend.agents.utils import PromptHarness
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "GROUP KEY" in p and "peak month" in p.lower()

    def test_entity_not_prose_rule(self):
        from backend.agents.utils import PromptHarness
        p = PromptHarness.get("SQL_CODE_AGENT")
        assert "never a constructed sentence" in p
