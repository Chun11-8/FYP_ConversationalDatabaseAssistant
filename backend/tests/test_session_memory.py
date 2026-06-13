"""
Tests for backend/memory/session_memory.py (Issue #3).

RAM-mode tests run everywhere. The PostgreSQL integration tests run only when a
live database is reachable (they are skipped otherwise) and prove durability —
the core acceptance criterion that restarting preserves conversations.

Run from repo root:
    pytest backend/tests/test_session_memory.py -v
"""

from __future__ import annotations

import uuid

import pytest

from backend.memory.session_memory import (
    SessionMemory,
    _make_title,
    _fallback_summarize,
)


# ── Pure helpers ─────────────────────────────────────────────────────────────

class TestHelpers:

    def test_make_title_truncates_long_message(self):
        title = _make_title("a" * 200)
        assert len(title) <= 60
        assert title.endswith("…")

    def test_make_title_collapses_whitespace(self):
        assert _make_title("  show   me   players \n") == "show me players"

    def test_make_title_empty_falls_back(self):
        assert _make_title("") == "New conversation"

    def test_fallback_summarize_bounds_length(self):
        out = _fallback_summarize("x" * 5000, prior_summary="y" * 5000)
        assert len(out) <= 1200

    def test_fallback_summarize_includes_prior(self):
        out = _fallback_summarize("new stuff", prior_summary="old summary")
        assert "old summary" in out
        assert "new stuff" in out


# ── RAM-mode behaviour ───────────────────────────────────────────────────────

class TestRamMode:

    def test_add_and_get_history_order(self):
        mem = SessionMemory()
        mem.add("s1", "show 5 players", "Here are 5 players", sql="SELECT 1 LIMIT 5", valid=True)
        hist = mem.get_history("s1")
        assert [m["role"] for m in hist] == ["user", "assistant"]
        assert hist[0]["content"] == "show 5 players"
        assert hist[1]["content"] == "Here are 5 players"
        assert hist[1]["sql"] == "SELECT 1 LIMIT 5"
        assert hist[1]["valid"] is True

    def test_new_session_returns_empty_history(self):
        mem = SessionMemory()
        assert mem.get_history("never-seen") == []

    def test_title_generated_from_first_message(self):
        mem = SessionMemory()
        mem.add("s1", "How many goalkeepers are there?", "42")
        sessions = mem.list_sessions()
        assert sessions[0]["title"] == "How many goalkeepers are there?"

    def test_get_last_sql_returns_most_recent(self):
        mem = SessionMemory()
        mem.add("s1", "q1", "r1", sql="SELECT 1")
        mem.add("s1", "q2", "r2", sql="SELECT 2")
        mem.add("s1", "q3", "r3", sql=None)   # assistant with no SQL
        assert mem.get_last_sql("s1") == "SELECT 2"

    def test_get_last_sql_none_when_no_sql(self):
        mem = SessionMemory()
        mem.add("s1", "q1", "r1")
        assert mem.get_last_sql("s1") is None

    def test_clear_session(self):
        mem = SessionMemory()
        mem.add("s1", "q", "r")
        mem.clear_session("s1")
        assert mem.get_history("s1") == []
        assert "s1" not in mem.active_sessions()

    def test_active_sessions_lists_all(self):
        mem = SessionMemory()
        mem.add("s1", "q", "r")
        mem.add("s2", "q", "r")
        assert set(mem.active_sessions()) == {"s1", "s2"}

    def test_list_sessions_sorted_newest_first(self):
        mem = SessionMemory()
        mem.add("old", "first", "r")
        mem.add("new", "second", "r")
        ids = [s["session_id"] for s in mem.list_sessions()]
        # "new" was updated most recently
        assert ids[0] == "new"

    def test_add_returns_int_turn_id(self):
        mem = SessionMemory()
        tid = mem.add("s1", "q1", "r1")
        assert isinstance(tid, int)

    def test_add_returns_increasing_ids(self):
        mem = SessionMemory()
        t1 = mem.add("s1", "q1", "r1")
        t2 = mem.add("s1", "q2", "r2")
        assert t2 is not None and t1 is not None
        assert t2 > t1

    # ── The Issue #3 acceptance scenario, at the memory layer ──────────────
    def test_two_turn_context_preserved(self):
        """'show 5 players' then 'now only the top 3' — turn 2 must see turn 1."""
        mem = SessionMemory()
        mem.add("s1", "show 5 players", "Here are 5 players",
                sql="SELECT name FROM players LIMIT 5", valid=True)
        history_for_turn2 = mem.get_history("s1")
        # The prior question AND its SQL are available for the follow-up
        assert any("show 5 players" in m["content"] for m in history_for_turn2)
        assert mem.get_last_sql("s1") == "SELECT name FROM players LIMIT 5"


# ── Rolling summarization ────────────────────────────────────────────────────

class TestSummarization:

    def test_summarizer_invoked_and_history_trimmed(self):
        calls = {"n": 0, "last_text": None, "last_prior": "sentinel"}

        def fake_summarizer(evicted_text, prior_summary):
            calls["n"] += 1
            calls["last_text"] = evicted_text
            calls["last_prior"] = prior_summary
            return "SUMMARY-v{}".format(calls["n"])

        # max_turns=2 → keep 4 messages verbatim; summarize beyond that
        mem = SessionMemory(max_turns=2, summarizer=fake_summarizer)
        for i in range(3):  # 3 add() = 6 messages > 4
            mem.add("s1", f"query {i}", f"resp {i}", sql=f"SELECT {i}")

        assert calls["n"] >= 1
        # Oldest evicted messages were passed to the summarizer
        assert "query 0" in calls["last_text"]
        # First summarization had no prior summary
        # (cannot assert exact value after multiple rolls, but it was used)

        hist = mem.get_history("s1")
        assert hist[0]["role"] == "summary"
        assert hist[0]["content"].startswith("SUMMARY")
        # Verbatim window is bounded to 2*max_turns messages
        verbatim = [m for m in hist if m["role"] != "summary"]
        assert len(verbatim) == 4
        # Most recent turn is retained
        assert verbatim[-1]["content"] == "resp 2"

    def test_fallback_summarizer_used_when_none(self):
        mem = SessionMemory(max_turns=1)  # keep 2 messages
        mem.add("s1", "alpha query", "alpha resp")
        mem.add("s1", "beta query", "beta resp")
        summary = mem.get_summary("s1")
        assert summary is not None
        assert "alpha" in summary

    def test_summarizer_exception_does_not_break_add(self):
        def boom(evicted_text, prior_summary):
            raise RuntimeError("LLM down")

        mem = SessionMemory(max_turns=1, summarizer=boom)
        mem.add("s1", "q1", "r1")
        mem.add("s1", "q2", "r2")  # triggers summarize → boom → fallback
        # Did not raise; summary exists via fallback
        assert mem.get_summary("s1") is not None


# ── PostgreSQL integration (skipped without a live DB) ───────────────────────

from backend.tests.conftest import live_db_config

_DB = live_db_config()
pg = pytest.mark.skipif(_DB is None, reason="no live PostgreSQL")


@pg
class TestPostgresPersistence:

    @pytest.fixture()
    def sid(self):
        sid = f"test-{uuid.uuid4()}"
        yield sid
        # cleanup
        SessionMemory(db_config=_DB).clear_session(sid)

    def test_uses_pg_backend(self):
        mem = SessionMemory(db_config=_DB)
        assert mem.is_persistent

    def test_roundtrip_and_durability(self, sid):
        mem = SessionMemory(db_config=_DB)
        mem.add(sid, "show 5 players", "Here are 5 players",
                sql="SELECT name FROM players LIMIT 5", valid=True)

        # Simulate a server restart: a brand-new instance reads the same data.
        mem2 = SessionMemory(db_config=_DB)
        hist = mem2.get_history(sid)
        assert [m["role"] for m in hist] == ["user", "assistant"]
        assert hist[0]["content"] == "show 5 players"
        assert hist[1]["sql"] == "SELECT name FROM players LIMIT 5"
        assert mem2.get_last_sql(sid) == "SELECT name FROM players LIMIT 5"

    def test_title_persisted(self, sid):
        mem = SessionMemory(db_config=_DB)
        mem.add(sid, "How many goalkeepers?", "42")
        rows = SessionMemory(db_config=_DB).list_sessions()
        match = next((r for r in rows if r["session_id"] == sid), None)
        assert match is not None
        assert match["title"] == "How many goalkeepers?"

    def test_summary_persisted_and_trimmed(self, sid):
        mem = SessionMemory(db_config=_DB, max_turns=2,
                            summarizer=lambda t, p: "ROLLED-UP")
        for i in range(4):  # 8 messages > 4 kept
            mem.add(sid, f"q{i}", f"r{i}", sql=f"SELECT {i}")

        mem2 = SessionMemory(db_config=_DB)
        assert mem2.get_summary(sid) == "ROLLED-UP"
        hist = mem2.get_history(sid)
        assert hist[0]["role"] == "summary"
        verbatim = [m for m in hist if m["role"] != "summary"]
        assert len(verbatim) == 4
        assert verbatim[-1]["content"] == "r3"

    def test_clear_session_cascades(self, sid):
        mem = SessionMemory(db_config=_DB)
        mem.add(sid, "q", "r")
        mem.clear_session(sid)
        assert mem.get_history(sid) == []
        assert sid not in mem.active_sessions()
