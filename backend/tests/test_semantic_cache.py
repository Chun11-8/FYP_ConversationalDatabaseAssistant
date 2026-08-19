"""
Tests for backend/memory/semantic_cache.py (Issue #4).

The disabled-mode tests run everywhere. The pgvector tests run only when a live
PostgreSQL is reachable; they use a deterministic bag-of-words fake embedder so
real cosine search, dedup, and logging are exercised without calling OpenAI.

Run from repo root:
    pytest backend/tests/test_semantic_cache.py -v
"""

from __future__ import annotations

import hashlib
import math
import uuid

import pytest

from backend.memory.semantic_cache import (
    SemanticCache, _to_vector_literal, _format_few_shot,
    discriminative_literals, literals_conflict,
)

VECTOR_DIM = 1536


def fake_embed(text: str):
    """Deterministic bag-of-words embedding: shared words → smaller distance."""
    vec = [0.0] * VECTOR_DIM
    for tok in set(text.lower().split()):
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % VECTOR_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ── Pure helpers / disabled mode ─────────────────────────────────────────────

class TestHelpersAndDisabled:

    def test_vector_literal_format(self):
        assert _to_vector_literal([1.0, 2.5, 3.0]).startswith("[")
        assert _to_vector_literal([1.0, 2.5, 3.0]).endswith("]")

    def test_format_few_shot(self):
        s = _format_few_shot("how many players", "SELECT count(*) FROM players")
        assert "Q: how many players" in s
        assert "SELECT count(*) FROM players" in s

    def test_disabled_without_db_config(self):
        cache = SemanticCache(db_config=None, embed_fn=fake_embed)
        assert cache.is_enabled is False
        # All ops are safe no-ops
        cache.save_pair("q", "SELECT 1")
        cache.log_interaction("s", "q", "SELECT 1", "r", "READ", True, 1, 10)
        cache.add_feedback(turn_id=1, session_id="s", rating="down")
        assert cache.retrieve_pairs("q") == []
        assert cache.retrieve_few_shots("q") == []
        # Issue #7/GH#8 schema-change cascade: also a safe no-op when disabled.
        assert cache.archive_pairs_referencing_table("players") == 0
        assert cache.demote_pairs_referencing_table("players") == 0


# ── literal-aware matching (GH#63) ───────────────────────────────────────────

class TestDiscriminativeLiterals:

    def test_extracts_numbers_and_months(self):
        lits = discriminative_literals("customer 6 between August and November 2013")
        assert ("num", "6") in lits
        assert ("num", "2013") in lits
        assert ("month", "aug") in lits
        assert ("month", "nov") in lits

    def test_extracts_quoted_strings_case_insensitively(self):
        assert ("str", "brazil") in discriminative_literals("players from 'Brazil'")
        assert ("str", "brazil") in discriminative_literals('players from "brazil"')

    def test_ordinary_words_are_not_literals(self):
        # Distinguishing prose is the embedding's job; over-collecting here
        # would reject legitimate rewordings of the same question.
        assert discriminative_literals("how many players are there") == frozenset()

    def test_leading_zeros_normalize(self):
        assert discriminative_literals("month 06") == discriminative_literals("month 6")

    def test_month_abbreviations_normalize(self):
        assert discriminative_literals("in Aug") == discriminative_literals("in August")
        assert discriminative_literals("in Sept") == discriminative_literals("in September")

    def test_empty_and_none_are_safe(self):
        assert discriminative_literals("") == frozenset()
        assert literals_conflict(None, None) is False
        assert literals_conflict(None, "in 2013") is True


class TestLiteralsConflict:

    Q2013 = "How much did customer 6 consume in total between August and November 2013?"

    def test_the_reported_bug_year_change_conflicts(self):
        assert literals_conflict(self.Q2013, self.Q2013.replace("2013", "2014")) is True

    def test_identical_questions_do_not_conflict(self):
        assert literals_conflict(self.Q2013, self.Q2013) is False

    def test_reworded_same_values_do_not_conflict(self):
        assert literals_conflict("how many players are there",
                                 "how many players do we have") is False

    def test_entity_id_change_conflicts(self):
        assert literals_conflict("total for customer 6", "total for customer 7") is True

    def test_word_order_does_not_matter(self):
        assert literals_conflict("6 then 2013", "2013 then 6") is False


class TestSavePairLiteralDedupOffline:
    """
    GH#63 write path: dedup_distance (0.05) is WIDER than the read-side hit
    threshold (0.03), so without a literal guard saving the 2014 pair would
    UPDATE the 2013 row in place and the cache could only ever hold one.
    """

    def _cache(self, nearest_row):
        from unittest.mock import MagicMock
        cache = SemanticCache(db_config={"name": "x"}, embed_fn=fake_embed)
        cache._enabled = True
        cur = MagicMock()
        cur.fetchone.return_value = nearest_row     # (id, distance, user_query)
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cur
        cache._connect = MagicMock(return_value=conn)
        return cache, cur

    @staticmethod
    def _second_statement(cur) -> str:
        return cur.execute.call_args_list[1][0][0]

    def test_different_year_inserts_instead_of_overwriting(self):
        cache, cur = self._cache((7, 0.01, "revenue in 2013"))
        cache.save_pair("revenue in 2014", "SELECT 2014", embedding=[0.1] * 4)
        assert "INSERT INTO query_sql_pairs" in self._second_statement(cur)

    def test_same_question_still_updates_in_place(self):
        cache, cur = self._cache((7, 0.01, "revenue in 2013"))
        cache.save_pair("revenue in 2013", "SELECT 2013b", embedding=[0.1] * 4)
        assert "UPDATE query_sql_pairs" in self._second_statement(cur)

    def test_far_away_pair_inserts_as_before(self):
        cache, cur = self._cache((7, 0.9, "revenue in 2013"))
        cache.save_pair("revenue in 2013", "SELECT x", embedding=[0.1] * 4)
        assert "INSERT INTO query_sql_pairs" in self._second_statement(cur)

    def test_empty_cache_inserts(self):
        cache, cur = self._cache(None)
        cache.save_pair("revenue in 2014", "SELECT 1", embedding=[0.1] * 4)
        assert "INSERT INTO query_sql_pairs" in self._second_statement(cur)


# ── schema-change cascade — offline (mocked connection) ──────────────────────

class TestSchemaChangeCascadeOffline:
    """Issue #7/GH#8: verifies the SQL issued and result plumbing without a
    live database — the pgvector-gated class below exercises it for real."""

    def _cache_with_mock_conn(self, rowcount: int):
        from unittest.mock import MagicMock
        cache = SemanticCache(db_config={"name": "x"}, embed_fn=fake_embed)
        cache._enabled = True  # bypass real connect attempted in __init__
        cur = MagicMock()
        cur.rowcount = rowcount
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cur
        cache._connect = MagicMock(return_value=conn)
        return cache, cur

    def test_archive_issues_status_update_and_returns_rowcount(self):
        cache, cur = self._cache_with_mock_conn(rowcount=2)
        n = cache.archive_pairs_referencing_table("players")
        assert n == 2
        sql, params = cur.execute.call_args[0]
        assert "SET status = 'archived'" in sql
        assert "status = 'active'" in sql
        assert params[-1] == r"\yplayers\y"

    def test_demote_issues_confidence_update_with_factor(self):
        cache, cur = self._cache_with_mock_conn(rowcount=3)
        n = cache.demote_pairs_referencing_table("teams", factor=0.25)
        assert n == 3
        sql, params = cur.execute.call_args[0]
        assert "confidence = confidence * %s" in sql
        assert params == (0.25, r"\yteams\y")

    def test_table_name_is_regex_escaped(self):
        cache, cur = self._cache_with_mock_conn(rowcount=0)
        cache.archive_pairs_referencing_table("weird.table")
        _, params = cur.execute.call_args[0]
        assert params[-1] == r"\yweird\.table\y"

    def test_cascade_survives_db_failure(self):
        cache = SemanticCache(db_config={"name": "x"}, embed_fn=fake_embed)
        cache._enabled = True
        cache._connect = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
        assert cache.archive_pairs_referencing_table("players") == 0
        assert cache.demote_pairs_referencing_table("players") == 0


# ── pgvector integration (skipped without a live DB) ─────────────────────────

from backend.tests.conftest import live_db_config

_DB = live_db_config()
pg = pytest.mark.skipif(_DB is None, reason="no live PostgreSQL")


def _count_pairs(group_id: str) -> int:
    import psycopg2
    conn = psycopg2.connect(
        dbname=_DB["name"], user=_DB["user"], password=_DB["password"],
        host=_DB["host"], port=_DB["port"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM query_sql_pairs WHERE group_id = %s;", (group_id,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def _count_interactions(session_id: str) -> int:
    import psycopg2
    conn = psycopg2.connect(
        dbname=_DB["name"], user=_DB["user"], password=_DB["password"],
        host=_DB["host"], port=_DB["port"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM query_interactions WHERE session_id = %s;", (session_id,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def _cleanup(group_id: str, session_id: str | None = None):
    import psycopg2
    conn = psycopg2.connect(
        dbname=_DB["name"], user=_DB["user"], password=_DB["password"],
        host=_DB["host"], port=_DB["port"],
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM query_sql_pairs WHERE group_id = %s;", (group_id,))
                if session_id:
                    cur.execute("DELETE FROM query_interactions WHERE session_id = %s;", (session_id,))
    finally:
        conn.close()


@pg
class TestSemanticCachePg:

    @pytest.fixture()
    def cache(self):
        return SemanticCache(db_config=_DB, embed_fn=fake_embed, dedup_distance=0.05)

    @pytest.fixture()
    def gid(self):
        g = f"grp-{uuid.uuid4()}"
        yield g
        _cleanup(g)

    def test_enabled(self, cache):
        assert cache.is_enabled

    def test_save_then_retrieve(self, cache, gid):
        cache.save_pair("how many players from brazil",
                        "SELECT count(*) FROM players WHERE country='Brazil'", group_id=gid)
        pairs = cache.retrieve_pairs("how many players from brazil", top_k=3, group_id=gid)
        assert len(pairs) == 1
        assert pairs[0]["sql"].startswith("SELECT count(*)")
        assert pairs[0]["distance"] < 0.01  # exact match → ~0 distance

    def test_dedup_exact_duplicate_updates_in_place(self, cache, gid):
        cache.save_pair("top scorers this season", "SELECT name FROM players LIMIT 5", group_id=gid)
        cache.save_pair("top scorers this season", "SELECT name FROM players LIMIT 10", group_id=gid)
        assert _count_pairs(gid) == 1                      # not duplicated
        pairs = cache.retrieve_pairs("top scorers this season", group_id=gid)
        assert pairs[0]["sql"] == "SELECT name FROM players LIMIT 10"  # refreshed

    def test_distinct_queries_insert_separately(self, cache, gid):
        cache.save_pair("players from brazil", "SELECT 1", group_id=gid)
        cache.save_pair("stadium capacity in london", "SELECT 2", group_id=gid)
        assert _count_pairs(gid) == 2

    def test_relevance_ranking(self, cache, gid):
        cache.save_pair("how many players from brazil",
                        "SELECT count(*) FROM players WHERE country='Brazil'", group_id=gid)
        cache.save_pair("stadium capacity in london stadiums",
                        "SELECT capacity FROM stadiums WHERE city='London'", group_id=gid)
        pairs = cache.retrieve_pairs("players brazil count", top_k=2, group_id=gid)
        # The Brazil/players pair must rank first (shared words)
        assert "Brazil" in pairs[0]["sql"]

    def test_retrieve_few_shots_formatting(self, cache, gid):
        cache.save_pair("how many goalkeepers",
                        "SELECT count(*) FROM players WHERE position='GK'", group_id=gid)
        shots = cache.retrieve_few_shots("how many goalkeepers", top_k=3, group_id=gid)
        assert len(shots) == 1
        assert shots[0].startswith("Q: how many goalkeepers")
        assert "position='GK'" in shots[0]

    def test_retrieve_few_shots_max_distance_filters(self, cache, gid):
        cache.save_pair("how many players from brazil", "SELECT 1", group_id=gid)
        # An orthogonal query (no shared words) should be filtered out
        shots = cache.retrieve_few_shots(
            "completely unrelated zzz qqq", top_k=3, group_id=gid, max_distance=0.1
        )
        assert shots == []

    def test_log_interaction(self, cache, gid):
        sid = f"sess-{uuid.uuid4()}"
        try:
            cache.log_interaction(
                session_id=sid, user_query="q", sql="SELECT 1", response="r",
                intent="READ", valid=True, row_count=3, latency_ms=120, group_id=gid,
            )
            assert _count_interactions(sid) == 1
        finally:
            _cleanup(gid, sid)

    def test_group_isolation(self, cache, gid):
        other = f"grp-{uuid.uuid4()}"
        try:
            cache.save_pair("players from brazil", "SELECT 1", group_id=gid)
            # A different group should not see this pair
            pairs = cache.retrieve_pairs("players from brazil", group_id=other)
            assert pairs == []
        finally:
            _cleanup(gid)

    def test_year_variants_coexist_instead_of_overwriting(self, cache, gid):
        """GH#63 acceptance criterion, against a real pgvector index."""
        q13 = "How much did customer 6 consume between August and November 2013?"
        q14 = q13.replace("2013", "2014")
        cache.save_pair(q13, "SELECT 13", group_id=gid)
        cache.save_pair(q14, "SELECT 14", group_id=gid)
        assert _count_pairs(gid) == 2
        assert {p["sql"] for p in cache.retrieve_pairs(q13, top_k=5, group_id=gid)} \
            == {"SELECT 13", "SELECT 14"}

    def test_repeating_a_question_still_dedupes(self, cache, gid):
        q = "How much did customer 6 consume between August and November 2013?"
        cache.save_pair(q, "SELECT a", group_id=gid)
        cache.save_pair(q, "SELECT b", group_id=gid)
        assert _count_pairs(gid) == 1

    def test_archive_pairs_referencing_table_hides_them_from_retrieval(self, cache, gid):
        cache.save_pair("how many players", "SELECT count(*) FROM players", group_id=gid)
        n = cache.archive_pairs_referencing_table("players")
        assert n >= 1
        assert cache.retrieve_pairs("how many players", group_id=gid) == []

    def test_archive_ignores_pairs_for_unrelated_tables(self, cache, gid):
        cache.save_pair("how many teams", "SELECT count(*) FROM teams", group_id=gid)
        cache.archive_pairs_referencing_table("players")
        assert len(cache.retrieve_pairs("how many teams", group_id=gid)) == 1

    def test_demote_pairs_referencing_table_halves_confidence(self, cache, gid):
        cache.save_pair("how many players", "SELECT count(*) FROM players",
                        group_id=gid, confidence=1.0)
        n = cache.demote_pairs_referencing_table("players", factor=0.5)
        assert n >= 1
        pairs = cache.retrieve_pairs("how many players", group_id=gid)
        assert pairs[0]["confidence"] == pytest.approx(0.5)

    def test_demote_does_not_archive(self, cache, gid):
        cache.save_pair("how many players", "SELECT count(*) FROM players", group_id=gid)
        cache.demote_pairs_referencing_table("players")
        assert len(cache.retrieve_pairs("how many players", group_id=gid)) == 1
