"""
Tests for the verification evidence surfaced with each answer (GH#67, Phase 1).

The measured motivation (see outputs/): on the gpt-5-mini BIRD run, 0 of 150
predictions failed validation and 0 failed at execution — every single failure
was a semantically wrong query that ran cleanly and returned plausible rows.
The pipeline already computes evidence about which answer to trust; before
this change it printed that evidence to a trace log and discarded it.

These tests pin the evidence to each decision path, and pin the degraded
paths (cache hit, single candidate, voting off, generation failure) so no
consumer has to special-case a missing block.

Run from repo root:
    pytest backend/tests/test_verification_evidence.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent
from backend.agents.sql_validator_agent import SQLValidatorAgent

SCHEMA = "Table t: name TEXT, grp TEXT"


def _agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_exec_voting=True, _enforce_limit=False, _vote_max=5,
                    _stmt_timeout_ms=15000, _selection="pairwise",
                    _unit_tests=False)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    a.validator = SQLValidatorAgent({"db": {"dialect": "postgres"}})
    return a


def _vote(agent, sqls, results_by_sql, query=None):
    """Run _vote_candidates against canned per-candidate result sets,
    returning the evidence dict it fills in."""
    original = dbm.execute_query
    ev: dict = {}

    def fake(sql, **kw):
        for k, rows in results_by_sql.items():
            if k in sql:
                return rows
        raise RuntimeError("boom")

    dbm.execute_query = fake
    try:
        agent._vote_candidates(sqls, SCHEMA, query=query, evidence=ev)
    finally:
        dbm.execute_query = original
    return ev


# ── evidence recorded by each decision path ───────────────────────────────────

class TestVoteEvidence:

    def test_plurality_records_the_real_agreement_ratio(self):
        """The number that was previously only ever printed to a trace log.

        Ordered a / b / aa so the early-exit optimisation cannot fire before
        the last candidate — this is a test of the ratio, not of the exit.
        """
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'b'",
                "SELECT name FROM t WHERE grp = 'aa'"]
        ev = _vote(_agent(), sqls,
                   {"'a'": [{"name": "A"}], "'aa'": [{"name": "A"}],
                    "'b'": [{"name": "B"}]})
        assert ev["decided_by"] == "plurality"
        assert ev["agreement"] == 2
        assert ev["executed"] == 3
        assert ev["candidates"] == 3
        assert ev["distinct_result_sets"] == 2

    def test_executed_counts_only_candidates_actually_run(self):
        """`agreement`/`executed` must describe what really happened. The
        unassailable-plurality early exit (GH#41 cost bound) skips the trailing
        candidates, so reporting 2/2 here is honest — claiming 2/3 would assert
        something about a candidate that was never executed.
        """
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'aa'",
                "SELECT name FROM t WHERE grp = 'b'"]
        ev = _vote(_agent(), sqls,
                   {"'a'": [{"name": "A"}], "'aa'": [{"name": "A"}],
                    "'b'": [{"name": "B"}]})
        assert ev["candidates"] == 3      # three were generated
        assert ev["executed"] == 2        # only two were run
        assert ev["agreement"] == 2

    def test_unanimous_agreement(self):
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'aa'"]
        ev = _vote(_agent(), sqls,
                   {"'a'": [{"name": "A"}], "'aa'": [{"name": "A"}]})
        assert (ev["agreement"], ev["executed"]) == (2, 2)
        assert ev["distinct_result_sets"] == 1

    def test_non_empty_tiebreak_is_labelled_as_such(self):
        """A 1-1 tie resolved by preferring rows over none is weaker evidence
        than a plurality, and must not be reported as one."""
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'z'"]
        ev = _vote(_agent(), sqls, {"'a'": [{"name": "A"}], "'z'": []})
        assert ev["decided_by"] == "non_empty_tiebreak"
        assert ev["executed"] == 2

    def test_no_candidate_executed(self):
        ev = _vote(_agent(), ["SELECT name FROM t WHERE grp = 'x'"], {})
        assert ev["decided_by"] == "none_executed"
        assert ev["executed"] == 0
        assert ev["agreement"] == 0

    def test_inconclusive_three_way_split(self):
        """Three distinct result sets with pairwise disabled — genuinely
        unresolved, and the evidence must say so rather than imply agreement."""
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'b'",
                "SELECT name FROM t WHERE grp = 'c'"]
        ev = _vote(_agent(_selection="none"), sqls,
                   {"'a'": [{"name": "A"}], "'b'": [{"name": "B"}],
                    "'c'": [{"name": "C"}]})
        assert ev["decided_by"] == "inconclusive"
        assert ev["distinct_result_sets"] == 3
        assert ev["agreement"] == 1

    def test_evidence_is_optional(self):
        """Callers that don't want evidence (and the existing tests) still get
        the plain three-tuple with no argument."""
        original = dbm.execute_query
        dbm.execute_query = lambda sql, **kw: [{"name": "A"}]
        try:
            out = _agent()._vote_candidates(
                ["SELECT name FROM t", "SELECT name FROM t2"], SCHEMA)
        finally:
            dbm.execute_query = original
        assert len(out) == 3


# ── the block assembled for the API ───────────────────────────────────────────

class TestVerificationBlock:

    def _block(self, **final):
        return OrchestrationAgent._verification_block(final)

    def test_always_well_formed_even_with_nothing_recorded(self):
        """No consumer should have to guard against a missing block."""
        b = self._block()
        assert b["decided_by"] == "not_generated"
        for k in ("candidates", "executed", "agreement",
                  "distinct_result_sets", "refine_rounds"):
            assert k in b

    def test_cache_hit_is_labelled_not_reported_as_zero_agreement(self):
        """A cached answer wasn't voted on at all; reporting agreement 0 would
        read as '5 candidates disagreed'."""
        b = self._block(cache_hit=True)
        assert b["decided_by"] == "cache_hit"

    def test_cache_hit_overrides_a_stale_generation_record(self):
        b = self._block(cache_hit=True,
                        verification={"decided_by": "plurality", "agreement": 3,
                                      "executed": 3, "candidates": 3,
                                      "distinct_result_sets": 1})
        assert b["decided_by"] == "cache_hit"

    def test_carries_refine_rounds_and_validation(self):
        b = self._block(
            verification={"decided_by": "plurality", "agreement": 2,
                          "executed": 2, "candidates": 2,
                          "distinct_result_sets": 1},
            refine_level=2,
            validation={"valid": True, "syntax_ok": True, "safety_ok": True,
                        "semantic_ok": False, "error": None},
        )
        assert b["refine_rounds"] == 2
        assert b["validated"] is True
        assert b["validation_checks"] == {"syntax_ok": True, "safety_ok": True,
                                          "semantic_ok": False}

    def test_validation_checks_exclude_non_boolean_noise(self):
        b = self._block(validation={"valid": False, "error": "no such table"})
        assert b["validation_checks"] == {}
        assert b["validated"] is False

    def test_does_not_mutate_the_state_dict(self):
        ev = {"decided_by": "plurality", "agreement": 1, "executed": 1,
              "candidates": 1, "distinct_result_sets": 1}
        self._block(verification=ev, refine_level=1)
        assert "refine_rounds" not in ev


# ── generate_sql degraded paths ───────────────────────────────────────────────

class TestGenerateSqlEvidence:
    """Turns that never reach voting must still produce a coherent block."""

    def _agent_for_generate(self, sqls, **attrs):
        a = _agent(**attrs)
        a._schema_linking = False
        a._condense = False
        sql_agent = MagicMock()
        sql_agent.generate.return_value = sqls
        sql_agent.rerank.return_value = sqls[0] if sqls else ""
        a._get_sql_agent = MagicMock(return_value=sql_agent)
        a._get_linker = MagicMock(return_value=MagicMock())
        return a

    def _state(self):
        return {"user_query": "q", "schema_context": SCHEMA,
                "few_shot_examples": [], "conversation_history": [],
                "query_plan": None, "previous_sql": None, "tier": None}

    def test_single_candidate_is_labelled(self):
        a = self._agent_for_generate(["SELECT name FROM t"])
        ev = a.generate_sql(self._state())["verification"]
        assert ev["decided_by"] == "single_candidate"
        assert ev["candidates"] == 1

    def test_voting_disabled_is_labelled(self):
        a = self._agent_for_generate(
            ["SELECT name FROM t", "SELECT grp FROM t"], _exec_voting=False)
        ev = a.generate_sql(self._state())["verification"]
        assert ev["decided_by"] == "voting_disabled"
        assert ev["candidates"] == 2

    def test_generation_failure_is_labelled(self):
        a = self._agent_for_generate(["SELECT name FROM t"])
        a._get_sql_agent.return_value.generate.side_effect = RuntimeError("down")
        out = a.generate_sql(self._state())
        assert out["best_sql"] == ""
        assert out["verification"]["decided_by"] == "generation_failed"

    def test_reranker_fallback_reports_no_agreement(self):
        """The reranker reads text only, never result sets — a turn it decided
        carries strictly weaker evidence than an execution plurality."""
        sqls = ["SELECT name FROM t WHERE grp = 'a'",
                "SELECT name FROM t WHERE grp = 'b'",
                "SELECT name FROM t WHERE grp = 'c'"]
        a = self._agent_for_generate(sqls, _selection="none")
        original = dbm.execute_query
        dbm.execute_query = lambda sql, **kw: (
            [{"name": "A"}] if "'a'" in sql else
            [{"name": "B"}] if "'b'" in sql else [{"name": "C"}])
        try:
            ev = a.generate_sql(self._state())["verification"]
        finally:
            dbm.execute_query = original
        assert ev["decided_by"] == "reranker"
        assert ev["agreement"] == 0
        assert ev["distinct_result_sets"] == 3


# ── API surface ───────────────────────────────────────────────────────────────

class TestChatResponseCarriesIt:

    def test_response_model_accepts_and_defaults(self):
        from backend.api.models import ChatResponse
        assert ChatResponse(response="x", intent="READ").verification is None
        r = ChatResponse(response="x", intent="READ",
                         verification={"decided_by": "plurality", "agreement": 3})
        assert r.verification["agreement"] == 3

    @pytest.mark.parametrize("decided_by", [
        "plurality", "non_empty_tiebreak", "unit_test", "pairwise", "reranker",
        "inconclusive", "single_candidate", "voting_disabled", "none_executed",
        "generation_failed", "cache_hit", "not_generated",
    ])
    def test_every_decided_by_value_survives_serialisation(self, decided_by):
        """The UI maps these to human-readable labels; an unmapped value must
        still round-trip rather than break the response."""
        from backend.api.models import ChatResponse
        r = ChatResponse(response="x", intent="READ",
                         verification={"decided_by": decided_by})
        assert r.model_dump()["verification"]["decided_by"] == decided_by
