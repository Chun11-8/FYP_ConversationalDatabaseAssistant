"""
Tests for the multi-turn evaluation harness (GH#71).

The harness itself has to be trustworthy before anyone spends tokens on it, so
scoring, stale-repeat detection and aggregation are all verified here with
canned predictions — no LLM, no key, no PostgreSQL. The SQLite fixtures are
built from code, so the execution path is exercised for real.

Run from repo root:
    pytest backend/tests/test_multiturn_eval.py -v
"""

from __future__ import annotations

import pytest

from backend.eval import multiturn as MT


# ── scoring ───────────────────────────────────────────────────────────────────

class TestScoreTurn:

    def test_exact_match_is_correct(self):
        assert MT.score_turn([(1,), (2,)], [(2,), (1,)]) == MT.CORRECT

    def test_order_does_not_matter(self):
        """Row order is not part of a result's meaning unless ORDER BY says so,
        and BIRD scores it the same way."""
        assert MT.score_turn([(2,), (1,)], [(1,), (2,)]) == MT.CORRECT

    def test_wrong_rows_are_wrong(self):
        assert MT.score_turn([(9,)], [(1,)]) == MT.WRONG

    def test_empty_prediction_against_non_empty_gold(self):
        assert MT.score_turn([], [(1,)]) == MT.ZERO_ROWS

    def test_execution_error(self):
        assert MT.score_turn(None, [(1,)], pred_error="no such table") == MT.EXEC_ERROR

    def test_no_sql_generated(self):
        assert MT.score_turn(None, [(1,)], had_sql=False) == MT.NO_SQL


class TestStaleRepeat:
    """The GH#63 signature: a follow-up that returns the PREVIOUS turn's rows."""

    def test_replaying_the_previous_answer_is_stale_repeat_not_generic_wrong(self):
        v = MT.score_turn(pred_rows=[(100.0,)], gold_rows=[(50.0,)],
                          prev_pred_rows=[(100.0,)],
                          expect_differs_from_previous=True)
        assert v == MT.STALE_REPEAT

    def test_a_different_wrong_answer_is_just_wrong(self):
        """Only an exact replay of the previous rows earns the special verdict —
        otherwise every wrong follow-up would be mislabelled as a cache bug."""
        v = MT.score_turn(pred_rows=[(77.0,)], gold_rows=[(50.0,)],
                          prev_pred_rows=[(100.0,)],
                          expect_differs_from_previous=True)
        assert v == MT.WRONG

    def test_a_correct_answer_is_never_stale(self):
        v = MT.score_turn(pred_rows=[(50.0,)], gold_rows=[(50.0,)],
                          prev_pred_rows=[(100.0,)],
                          expect_differs_from_previous=True)
        assert v == MT.CORRECT

    def test_not_flagged_when_the_case_does_not_expect_a_change(self):
        """A legitimate repeat SHOULD return the same rows."""
        v = MT.score_turn(pred_rows=[(3,)], gold_rows=[(3,)],
                          prev_pred_rows=[(3,)],
                          expect_differs_from_previous=False)
        assert v == MT.CORRECT

    def test_no_previous_turn_cannot_be_stale(self):
        assert MT.score_turn([(1,)], [(2,)], prev_pred_rows=None,
                             expect_differs_from_previous=True) == MT.WRONG


# ── aggregation ───────────────────────────────────────────────────────────────

def _turn(case="c", idx=0, verdict=MT.CORRECT, category="literal_swap", **kw):
    base = {"case_id": case, "category": category, "turn_index": idx,
            "verdict": verdict, "cache_hit": False, "expect_cache_hit": False}
    base.update(kw)
    return base


class TestSummarize:

    def test_splits_first_turn_from_followups(self):
        """The headline number: a system can look fine on an aggregate
        dominated by opening questions while being useless at follow-ups."""
        m = MT.summarize([
            _turn(idx=0, verdict=MT.CORRECT),
            _turn(idx=1, verdict=MT.WRONG),
            _turn(case="d", idx=0, verdict=MT.CORRECT),
            _turn(case="d", idx=1, verdict=MT.WRONG),
        ])
        assert m["accuracy_first_turn"] == 100.0
        assert m["accuracy_followup"] == 0.0
        assert m["accuracy_overall"] == 50.0

    def test_counts_stale_repeats(self):
        m = MT.summarize([_turn(idx=1, verdict=MT.STALE_REPEAT),
                          _turn(idx=1, verdict=MT.CORRECT)])
        assert m["stale_repeats"] == 1

    def test_skipped_turns_are_excluded_from_accuracy(self):
        """A skipped turn is a missing measurement, not a failure — counting it
        as wrong would punish the run for an absent dependency."""
        m = MT.summarize([_turn(verdict=MT.CORRECT), _turn(verdict=MT.SKIPPED)])
        assert m["turns_scored"] == 1
        assert m["turns_skipped"] == 1
        assert m["accuracy_overall"] == 100.0

    def test_cache_expectations_tracked_apart_from_accuracy(self):
        m = MT.summarize([
            _turn(idx=1, verdict=MT.CORRECT, expect_cache_hit=True, cache_hit=True),
            _turn(idx=1, verdict=MT.CORRECT, expect_cache_hit=True, cache_hit=False),
        ])
        assert m["accuracy_followup"] == 100.0        # both answers were right
        assert m["cache_expectations_met"] == 1
        assert m["cache_expectations_missed"] == 1    # …but one cost a regeneration

    def test_per_category_breakdown(self):
        m = MT.summarize([
            _turn(idx=1, category="literal_swap", verdict=MT.STALE_REPEAT),
            _turn(idx=1, category="ellipsis", verdict=MT.CORRECT),
        ])
        assert m["by_category"]["literal_swap"]["accuracy"] == 0.0
        assert m["by_category"]["ellipsis"]["accuracy"] == 100.0

    def test_empty_input_does_not_divide_by_zero(self):
        m = MT.summarize([])
        assert m["accuracy_overall"] is None
        assert m["turns_scored"] == 0


# ── the case set itself ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db_root(tmp_path_factory):
    return MT.prepare_databases(tmp_path_factory.mktemp("mt_dbs"))


class TestCaseSet:

    def test_loads(self):
        cases = MT.load_cases()
        assert len(cases) >= 10

    def test_every_case_is_well_formed(self):
        for c in MT.load_cases():
            assert c["id"] and c["db_id"] and c["category"]
            assert c.get("why"), f"{c['id']} must document what it catches"
            assert len(c["turns"]) >= 2, f"{c['id']} is not multi-turn"
            for t in c["turns"]:
                assert t["question"]

    def test_case_ids_are_unique(self):
        ids = [c["id"] for c in MT.load_cases()]
        assert len(ids) == len(set(ids))

    def test_covers_every_intended_category(self):
        cats = {c["category"] for c in MT.load_cases()}
        assert {"literal_swap", "ellipsis", "anaphora", "refinement",
                "topic_switch", "legitimate_repeat",
                "clarification_reply"} <= cats

    def test_includes_a_legitimate_repeat_control(self):
        """Without this the metric would reward disabling the cache entirely —
        the wrong lesson to draw from GH#63."""
        assert any(c["category"] == "legitimate_repeat" for c in MT.load_cases())

    def test_the_gh63_scenario_is_present(self):
        c = next(c for c in MT.load_cases() if c["id"] == "literal-swap-month")
        assert c["turns"][1]["expect_differs_from_previous"] is True

    def test_single_case_filter(self):
        assert len(MT.load_cases(only="literal-swap-month")) == 1
        with pytest.raises(SystemExit):
            MT.load_cases(only="does-not-exist")


class TestGoldSqlAgainstRealFixtures:
    """Every gold query must run and return what the case implies. A wrong
    fixture would silently mark correct predictions as failures."""

    def test_every_gold_sql_executes(self, db_root):
        for c in MT.load_cases():
            path = MT.db_path_for(db_root, c["db_id"])
            for i, t in enumerate(c["turns"]):
                if not t.get("gold_sql"):
                    continue
                # GH#87: a turn may move the conversation to another database.
                turn_path = (MT.db_path_for(db_root, t["db_id"])
                             if t.get("db_id") else path)
                rows, err = MT.run_sqlite(turn_path, t["gold_sql"])
                assert err is None, f"{c['id']} T{i}: {err}"
                assert rows is not None

    def test_turns_marked_as_differing_really_do(self, db_root):
        for c in MT.load_cases():
            path = MT.db_path_for(db_root, c["db_id"])
            prev = None
            for i, t in enumerate(c["turns"]):
                if not t.get("gold_sql"):
                    prev = None
                    continue
                turn_path = (MT.db_path_for(db_root, t["db_id"])
                             if t.get("db_id") else path)
                rows, _ = MT.run_sqlite(turn_path, t["gold_sql"])
                if t.get("expect_differs_from_previous") and prev is not None:
                    assert MT._row_set(rows) != MT._row_set(prev), (
                        f"{c['id']} T{i} claims to differ but gold is identical — "
                        f"a stale answer would score as CORRECT")
                prev = rows


class TestSqliteExecution:

    def test_reads_rows(self, db_root):
        path = MT.db_path_for(db_root, "debit_card_specializing")
        rows, err = MT.run_sqlite(path, "SELECT COUNT(*) FROM customers")
        assert err is None and rows == [(3,)]

    def test_reports_errors_instead_of_raising(self, db_root):
        rows, err = MT.run_sqlite(
            MT.db_path_for(db_root, "student_club"), "SELECT * FROM nope")
        assert rows is None and err

    def test_rejects_writes(self, db_root):
        """The fixtures open immutable — an eval run must not mutate them,
        or later cases would score against a changed database."""
        path = MT.db_path_for(db_root, "student_club")
        rows, err = MT.run_sqlite(path, "DELETE FROM member")
        assert err is not None
        after, _ = MT.run_sqlite(path, "SELECT COUNT(*) FROM member")
        assert after == [(30,)]

    def test_empty_sql(self, db_root):
        rows, err = MT.run_sqlite(MT.db_path_for(db_root, "student_club"), "")
        assert rows is None and err


# ── end-to-end with a canned agent (no LLM) ───────────────────────────────────

class FakeAgent:
    """Replays scripted SQL per turn, so run_case can be exercised whole."""

    def __init__(self, sqls, cache_hits=None):
        self.sqls = list(sqls)
        self.cache_hits = list(cache_hits or [False] * len(sqls))
        self.calls = []

    def run(self, session_id, user_query):
        i = len(self.calls)
        self.calls.append((session_id, user_query))
        return {"sql": self.sqls[i], "response": "ok",
                "cache_hit": self.cache_hits[i],
                "standalone_query": None, "verification": {}}


class TestRunCase:

    CASE = {
        "id": "t", "db_id": "debit_card_specializing", "category": "literal_swap",
        "turns": [
            {"question": "consumption of customer 1 in 201208?",
             "gold_sql": "SELECT Consumption FROM yearmonth WHERE CustomerID=1 AND Date='201208'"},
            {"question": "what about 201207?",
             "gold_sql": "SELECT Consumption FROM yearmonth WHERE CustomerID=1 AND Date='201207'",
             "expect_differs_from_previous": True},
        ],
    }

    def _run(self, db_root, sqls, **kw):
        return MT.run_case(FakeAgent(sqls, **kw), self.CASE,
                           MT.db_path_for(db_root, "debit_card_specializing"),
                           cache_enabled=True)

    def test_a_correct_conversation_scores_correct(self, db_root):
        out = self._run(db_root, [
            "SELECT Consumption FROM yearmonth WHERE CustomerID=1 AND Date='201208'",
            "SELECT Consumption FROM yearmonth WHERE CustomerID=1 AND Date='201207'",
        ])
        assert [r["verdict"] for r in out] == [MT.CORRECT, MT.CORRECT]

    def test_the_gh63_bug_is_caught_as_stale_repeat(self, db_root):
        """Turn 2 re-runs turn 1's SQL — exactly what the cache used to do."""
        same = "SELECT Consumption FROM yearmonth WHERE CustomerID=1 AND Date='201208'"
        out = self._run(db_root, [same, same])
        assert [r["verdict"] for r in out] == [MT.CORRECT, MT.STALE_REPEAT]

    def test_all_turns_share_one_session(self, db_root):
        agent = FakeAgent(["SELECT 1", "SELECT 2"])
        MT.run_case(agent, self.CASE,
                    MT.db_path_for(db_root, "debit_card_specializing"), True)
        assert len({s for s, _ in agent.calls}) == 1

    def test_an_agent_exception_becomes_a_verdict_not_a_crash(self, db_root):
        class Boom:
            def run(self, **kw):
                raise RuntimeError("model down")
        out = MT.run_case(Boom(), self.CASE,
                          MT.db_path_for(db_root, "debit_card_specializing"), True)
        assert all(r["verdict"] in (MT.NO_SQL, MT.EXEC_ERROR) for r in out)

    def test_cache_expectation_is_skipped_without_the_cache(self, db_root):
        case = {**self.CASE, "turns": [
            self.CASE["turns"][0],
            {"question": "again", "gold_sql": "SELECT COUNT(*) FROM customers",
             "expect_cache_hit": True},
        ]}
        out = MT.run_case(FakeAgent(["SELECT 1", "SELECT 2"]), case,
                          MT.db_path_for(db_root, "debit_card_specializing"),
                          cache_enabled=False)
        assert out[1]["verdict"] == MT.SKIPPED
        assert "semantic cache" in out[1]["note"]

    def test_a_turn_without_gold_is_skipped_not_failed(self, db_root):
        case = {**self.CASE, "turns": [
            {"question": "vague?", "allow_clarification": True},
            self.CASE["turns"][1],
        ]}
        out = MT.run_case(FakeAgent(["SELECT 1", "SELECT 2"]), case,
                          MT.db_path_for(db_root, "debit_card_specializing"), True)
        assert out[0]["verdict"] == MT.SKIPPED

    def test_records_diagnostic_fields(self, db_root):
        out = self._run(db_root, ["SELECT 1", "SELECT 2"])
        for r in out:
            for k in ("standalone_query", "cache_hit", "pred_sql", "gold_sql",
                      "elapsed_ms", "category"):
                assert k in r


# ── the schema seam (GH#85 interaction) ───────────────────────────────────────

class TestSchemaSeam:
    """
    GH#85 gave the graph a full-schema path that runs BEFORE retrieve()
    whenever a scope is configured. build_agent must stub it: otherwise a
    config.yaml carrying `retrieval.active_db` sends this harness to the live
    pgvector index instead of the SQLite fixture it just built — silently,
    because a failure there degrades to cosine rather than raising.
    """

    def test_build_agent_stubs_every_schema_entry_point(self):
        import inspect
        src = inspect.getsource(MT.build_agent)
        for seam in ("dbm.retrieve", "dbm.execute_query",
                     "dbm.expand_fk_tables", "dbm.fetch_full_schema"):
            assert seam in src, seam

    def test_the_stub_forces_the_fixture_path(self):
        """Whatever a scope is set to, the full-schema lookup must decline so
        fake_retrieve stays the single source of schema."""
        import inspect
        src = inspect.getsource(MT.build_agent)
        assert "dbm.fetch_full_schema = lambda *a, **kw: None" in src
