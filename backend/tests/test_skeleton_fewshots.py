"""
Tests for the DAIL-SQL-style skeleton-similarity few-shot blend:
backend/memory/skeleton.py + SemanticCache.retrieve_few_shots_blended +
db_agent.py's _retrieve_few_shots wiring.

All offline: no live Postgres needed. SemanticCache's DB layer is mocked at
the retrieve_pairs() boundary (a public method), never touching psycopg2.

Run from repo root:
    pytest backend/tests/test_skeleton_fewshots.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.memory.semantic_cache import SemanticCache
from backend.memory.skeleton import (
    question_shape_signature,
    skeleton_overlap,
    sql_skeleton,
)


# ── sql_skeleton ─────────────────────────────────────────────────────────────

GROUPED_SUPERLATIVE_A = (
    "SELECT team_id, COUNT(*) FROM players GROUP BY team_id "
    "ORDER BY COUNT(*) DESC LIMIT 1"
)
GROUPED_SUPERLATIVE_B = (
    "SELECT dept_id, COUNT(*) FROM staff GROUP BY dept_id "
    "ORDER BY COUNT(*) DESC LIMIT 1"
)
PLAIN_COUNT = "SELECT COUNT(*) FROM players WHERE rating > 80"


class TestSqlSkeleton:

    def test_structurally_identical_different_names_same_skeleton(self):
        assert sql_skeleton(GROUPED_SUPERLATIVE_A) == sql_skeleton(GROUPED_SUPERLATIVE_B)

    def test_different_shape_different_skeleton(self):
        assert sql_skeleton(GROUPED_SUPERLATIVE_A) != sql_skeleton(PLAIN_COUNT)

    def test_grouped_superlative_tags(self):
        tags = sql_skeleton(GROUPED_SUPERLATIVE_A)
        assert {"AGG:COUNT", "GROUP_BY", "SUPERLATIVE"} <= tags

    def test_plain_count_no_group_or_superlative(self):
        tags = sql_skeleton(PLAIN_COUNT)
        assert "GROUP_BY" not in tags
        assert "SUPERLATIVE" not in tags
        assert "AGG:COUNT" in tags

    def test_join_count_bucketed(self):
        two_joins = ("SELECT a.x FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id")
        four_joins = ("SELECT a.x FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id "
                      "JOIN d ON c.id=d.id JOIN e ON d.id=e.id")
        assert "JOINS:2" in sql_skeleton(two_joins)
        assert "JOINS:3+" in sql_skeleton(four_joins)

    def test_unparseable_returns_empty_set(self):
        assert sql_skeleton("not valid sql (((") == frozenset()

    def test_empty_string_returns_empty_set(self):
        assert sql_skeleton("") == frozenset()
        assert sql_skeleton("   ") == frozenset()


# ── question_shape_signature ────────────────────────────────────────────────

class TestQuestionShapeSignature:

    def test_how_many_maps_to_count(self):
        assert "AGG:COUNT" in question_shape_signature("How many players are there?")

    def test_highest_per_maps_to_superlative_and_group(self):
        sig = question_shape_signature("What is the highest rating per team?")
        assert "SUPERLATIVE" in sig
        assert "GROUP_BY" in sig

    def test_average_maps_to_avg(self):
        assert "AGG:AVG" in question_shape_signature("What is the average age?")

    def test_exclusion_cue(self):
        assert "EXCLUSION" in question_shape_signature("List players not from Brazil")

    def test_no_cue_returns_empty_or_minimal(self):
        sig = question_shape_signature("Messi")
        assert sig == frozenset()

    def test_empty_question(self):
        assert question_shape_signature("") == frozenset()


# ── skeleton_overlap ───────────────────────────────────────────────────────────

class TestSkeletonOverlap:

    def test_identical_sets_full_overlap(self):
        s = frozenset({"AGG:COUNT", "GROUP_BY"})
        assert skeleton_overlap(s, s) == 1.0

    def test_disjoint_sets_zero_overlap(self):
        assert skeleton_overlap(frozenset({"AGG:COUNT"}), frozenset({"SUPERLATIVE"})) == 0.0

    def test_both_empty_is_zero_not_one(self):
        # No signal on either side must not be scored as a "perfect match".
        assert skeleton_overlap(frozenset(), frozenset()) == 0.0

    def test_one_empty_is_zero(self):
        assert skeleton_overlap(frozenset({"AGG:COUNT"}), frozenset()) == 0.0

    def test_partial_overlap_jaccard(self):
        a = frozenset({"AGG:COUNT", "GROUP_BY"})
        b = frozenset({"AGG:COUNT", "SUPERLATIVE"})
        # intersection={AGG:COUNT} (1), union=3 -> 1/3
        assert skeleton_overlap(a, b) == pytest.approx(1 / 3)


# ── retrieve_few_shots_blended ────────────────────────────────────────────────

def _cache_with_pairs(pairs):
    cache = SemanticCache(db_config=None)
    cache._enabled = True  # bypass "disabled" short-circuit for this unit test
    cache.retrieve_pairs = MagicMock(return_value=pairs)
    return cache


class TestRetrieveFewShotsBlended:

    def test_structural_match_beats_closer_embedding_distance(self):
        # "closer" pair (lower distance) is structurally mismatched;
        # "farther" pair matches the question's grouped-superlative shape.
        pairs = [
            {"user_query": "closer but wrong shape", "sql": PLAIN_COUNT,
             "confidence": 1.0, "distance": 0.05},
            {"user_query": "farther but right shape", "sql": GROUPED_SUPERLATIVE_A,
             "confidence": 1.0, "distance": 0.4},
        ]
        cache = _cache_with_pairs(pairs)
        question_shape = question_shape_signature("What is the highest rating per team?")
        result = cache.retrieve_few_shots_blended(
            "q", question_shape, top_k=1, blend_weight=0.8)
        assert "farther but right shape" in result[0]

    def test_blend_weight_zero_matches_plain_ranking(self):
        pairs = [
            {"user_query": "closer", "sql": PLAIN_COUNT, "confidence": 1.0, "distance": 0.1},
            {"user_query": "farther", "sql": GROUPED_SUPERLATIVE_A, "confidence": 1.0, "distance": 0.4},
        ]
        cache = _cache_with_pairs(pairs)
        question_shape = question_shape_signature("What is the highest rating per team?")
        result = cache.retrieve_few_shots_blended(
            "q", question_shape, top_k=1, blend_weight=0.0)
        assert "closer" in result[0]   # pure embedding distance wins, shape ignored

    def test_empty_pool_returns_empty_list(self):
        cache = _cache_with_pairs([])
        assert cache.retrieve_few_shots_blended("q", frozenset(), top_k=3) == []

    def test_pool_multiplier_forwarded(self):
        cache = _cache_with_pairs([])
        cache.retrieve_few_shots_blended("q", frozenset(), top_k=3, pool_multiplier=5)
        kwargs = cache.retrieve_pairs.call_args.kwargs
        assert kwargs["skeleton_pool_multiplier"] == 5

    def test_max_distance_filters_before_reranking(self):
        pairs = [
            {"user_query": "far", "sql": GROUPED_SUPERLATIVE_A, "confidence": 1.0, "distance": 0.9},
        ]
        cache = _cache_with_pairs(pairs)
        result = cache.retrieve_few_shots_blended(
            "q", frozenset(), top_k=3, max_distance=0.5)
        assert result == []


# ── retrieve_pairs pool sizing ────────────────────────────────────────────────

class TestRetrievePairsPoolSizing:

    def test_default_multiplier_keeps_top_k_limit(self):
        cache = SemanticCache(db_config=None)
        cache._enabled = True
        cache._embed = MagicMock(return_value=[0.0] * 1536)
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        cache._connect = MagicMock(return_value=conn)

        cache.retrieve_pairs("q", top_k=3)
        limit_param = cur.execute.call_args[0][1][-1]
        assert limit_param == 3

    def test_multiplier_widens_the_pool(self):
        cache = SemanticCache(db_config=None)
        cache._enabled = True
        cache._embed = MagicMock(return_value=[0.0] * 1536)
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        cache._connect = MagicMock(return_value=conn)

        cache.retrieve_pairs("q", top_k=3, skeleton_pool_multiplier=3)
        limit_param = cur.execute.call_args[0][1][-1]
        assert limit_param == 9


# ── db_agent.py wiring ─────────────────────────────────────────────────────────

class TestDbAgentWiring:

    def _agent(self, **attrs):
        import backend.agents.db_agent as dbm
        a = dbm.OrchestrationAgent.__new__(dbm.OrchestrationAgent)
        defaults = dict(_fewshot_skeleton_blend=True, _fewshot_skeleton_weight=0.3)
        defaults.update(attrs)
        for k, v in defaults.items():
            setattr(a, k, v)
        a.semantic_cache = MagicMock()
        return a

    def test_blend_enabled_calls_blended_method(self):
        agent = self._agent(_fewshot_skeleton_blend=True)
        agent.semantic_cache.retrieve_few_shots_blended.return_value = ["ex"]
        result = agent._retrieve_few_shots("how many players?")
        agent.semantic_cache.retrieve_few_shots_blended.assert_called_once()
        agent.semantic_cache.retrieve_few_shots.assert_not_called()
        assert result == ["ex"]

    def test_blend_disabled_calls_plain_method(self):
        agent = self._agent(_fewshot_skeleton_blend=False)
        agent.semantic_cache.retrieve_few_shots.return_value = ["ex"]
        result = agent._retrieve_few_shots("how many players?")
        agent.semantic_cache.retrieve_few_shots.assert_called_once()
        agent.semantic_cache.retrieve_few_shots_blended.assert_not_called()
        assert result == ["ex"]

    def test_failure_degrades_to_empty_list(self):
        agent = self._agent(_fewshot_skeleton_blend=True)
        agent.semantic_cache.retrieve_few_shots_blended.side_effect = RuntimeError("db down")
        assert agent._retrieve_few_shots("q") == []
