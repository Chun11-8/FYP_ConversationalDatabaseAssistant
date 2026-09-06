"""
Tests for GH#49 — P3 value index (WHERE-literal grounding), P4 backward schema
linking, P5 verbatim evidence formula, and the per-tier refine-depth quick win.
All offline: DB and LLM boundaries mocked.

Run from repo root:
    pytest backend/tests/test_value_index_backward_link.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent
from backend.agents.schema_linking_agent import render_query_plan
from backend.retrieval.value_index import (
    ValueIndex, extract_value_spans, render_value_hints,
)


# ── P3: value index ───────────────────────────────────────────────────────────

class _FakeCursor:
    """information_schema + DISTINCT queries against a canned column map."""

    def __init__(self, columns_by_table):
        self._cols = columns_by_table
        self._rows = []

    def execute(self, sql, params=None):
        if "information_schema" in sql:
            table = params[0]
            self._rows = [(c,) for c in self._cols.get(table, {})]
        else:
            for table, cols in self._cols.items():
                if f'"{table}"' in sql:
                    for col, values in cols.items():
                        if f'"{col}"' in sql:
                            self._rows = [(v,) for v in values]
                            return
            self._rows = []

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _index(columns_by_table) -> ValueIndex:
    conn = MagicMock()
    conn.cursor.return_value = _FakeCursor(columns_by_table)
    return ValueIndex(lambda: conn)


class TestValueIndex:

    DATA = {"country": {"Name": ["USA", "United Kingdom", "France"]}}

    def test_extract_spans(self):
        spans = extract_value_spans("How many players are from 'usa' or France?")
        assert "usa" in spans and "France" in spans

    def test_casing_mismatch_produces_hint(self):
        hints = _index(self.DATA).ground("players from 'usa'", ["country"])
        assert hints == ["'usa' → use 'USA' (country.Name)"]

    def test_exact_match_produces_no_hint(self):
        assert _index(self.DATA).ground("players from 'France'", ["country"]) == []

    def test_fuzzy_match_close_value(self):
        hints = _index(self.DATA).ground(
            "stats for 'United Kingdon'", ["country"])
        assert hints and "United Kingdom" in hints[0]

    def test_unknown_table_degrades_silently(self):
        assert _index(self.DATA).ground("from 'usa'", ["nope"]) == []

    def test_db_failure_degrades_silently(self):
        def boom():
            raise RuntimeError("db down")
        vi = ValueIndex(boom)
        assert vi.ground("from 'usa'", ["country"]) == []

    def test_table_loaded_once(self):
        calls = {"n": 0}
        conn = MagicMock()
        conn.cursor.return_value = _FakeCursor(self.DATA)

        def factory():
            calls["n"] += 1
            return conn

        vi = ValueIndex(factory)
        vi.ground("from 'usa'", ["country"])
        vi.ground("from 'france'", ["country"])
        assert calls["n"] == 1          # cached after first load

    def test_high_cardinality_column_skipped(self):
        big = {"t": {"noise": [f"v{i}" for i in range(2001)]}}
        assert _index(big).ground("looking for 'V7'", ["t"]) == []

    def test_render_block(self):
        block = render_value_hints(["'usa' → use 'USA' (country.Name)"])
        assert block.startswith("<value_hints>") and "USA" in block
        assert render_value_hints([]) == ""


# ── P4: backward linking in link_schema ───────────────────────────────────────

def _bare_agent(**attrs) -> OrchestrationAgent:
    a = OrchestrationAgent.__new__(OrchestrationAgent)
    defaults = dict(_schema_linking=True, _backward_linking=True,
                    _db_name="FYP")
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def _link_state(**over):
    state = {"user_query": "q", "standalone_query": None,
             "schema_context": "Table: players\n  - id: INTEGER [PK]",
             "conversation_history": [], "cache_hit": False}
    state.update(over)
    return state


class TestBackwardLinking:

    def _agent_with_plan(self, plan):
        a = _bare_agent()
        a.linker = MagicMock()
        a.linker.analyze.return_value = plan
        return a

    def test_missing_planned_table_recovered(self):
        a = self._agent_with_plan({"tables": ["players", "teams"]})
        with patch.object(dbm, "fetch_table_chunks", return_value=[
            {"table_name": "teams", "content": "Table: teams\n  - id: INTEGER"},
        ]) as fetch:
            out = a.link_schema(_link_state())
        fetch.assert_called_once_with(["teams"], "FYP")
        assert "Table: teams" in out["schema_context"]

    def test_present_tables_not_refetched(self):
        a = self._agent_with_plan({"tables": ["players"]})
        with patch.object(dbm, "fetch_table_chunks") as fetch:
            a.link_schema(_link_state())
        fetch.assert_not_called()

    def test_config_off_skips_recovery(self):
        a = self._agent_with_plan({"tables": ["players", "teams"]})
        a._backward_linking = False
        with patch.object(dbm, "fetch_table_chunks") as fetch:
            a.link_schema(_link_state())
        fetch.assert_not_called()

    def test_fetch_failure_keeps_context(self):
        a = self._agent_with_plan({"tables": ["teams"]})
        with patch.object(dbm, "fetch_table_chunks", return_value=[]):
            out = a.link_schema(_link_state())
        assert out["schema_context"].startswith("Table: players")


# ── P5: evidence_formula rendering ────────────────────────────────────────────

class TestEvidenceFormula:

    def test_rendered_verbatim_with_instruction(self):
        block = render_query_plan({
            "tables": ["t"],
            "evidence_formula": "DIVIDE(SUM(x), COUNT(y))",
        })
        assert "evidence_formula (COPY VERBATIM): DIVIDE(SUM(x), COUNT(y))" in block
        assert "VERBATIM" in block

    def test_null_formula_not_rendered(self):
        block = render_query_plan({"tables": ["t"], "evidence_formula": "null"})
        assert "COPY VERBATIM" not in block


# ── Quick win: per-tier refine depth ──────────────────────────────────────────

class TestRefineDepth:

    def _agent(self):
        return _bare_agent(_max_refine=1, _max_refine_challenging=2)

    def test_default_tier_stops_after_one_round(self):
        a = self._agent()
        state = {"refine_level": 1, "tier": "moderate",
                 "execution_result": [], "error": None}
        assert a.route_after_execution(state) == "format_response"

    def test_challenging_tier_gets_second_round(self):
        a = self._agent()
        state = {"refine_level": 1, "tier": "challenging",
                 "execution_result": [], "error": None}
        assert a.route_after_execution(state) == "refine_sql"
        state["refine_level"] = 2
        assert a.route_after_execution(state) == "format_response"

    def test_success_never_refines(self):
        a = self._agent()
        state = {"refine_level": 0, "tier": "challenging",
                 "execution_result": [{"x": 1}], "error": None}
        assert a.route_after_execution(state) == "format_response"


class TestResolveSpans:
    """
    GH#73: ground() returns nothing for a span that matches nothing, because
    there is no casing to correct — but "this value is not in the database" is
    the more useful of the two signals, and silence discards it.
    resolve_spans() exposes both outcomes.
    """

    IDX = {"customers": {"country": ["Brazil", "Spain"], "segment": ["LAM", "SME"]}}

    def test_exact_match_resolves_to_itself(self):
        vi = _index(self.IDX)
        assert vi.resolve_spans("customers in Brazil", ["customers"]) == {"Brazil": "Brazil"}

    def test_near_match_resolves_to_the_stored_spelling(self):
        vi = _index({"customers": {"country": ["BRAZIL"]}})
        assert vi.resolve_spans("customers in Brazil", ["customers"])["Brazil"] == "BRAZIL"

    def test_a_value_absent_from_the_data_resolves_to_None(self):
        """The signal that ground() throws away — and the reported failure."""
        vi = _index(self.IDX)
        assert vi.resolve_spans("customers from Wakanda", ["customers"]) == {"Wakanda": None}

    def test_searches_every_candidate_table(self):
        vi = _index({"a": {"x": ["nope"]}, "customers": {"country": ["Brazil"]}})
        assert vi.resolve_spans("in Brazil", ["a", "customers"])["Brazil"] == "Brazil"

    def test_no_spans_yields_an_empty_map(self):
        assert _index(self.IDX).resolve_spans("how many are there", ["customers"]) == {}

    def test_a_failing_connection_yields_an_empty_map_not_false_negatives(self):
        """An empty map must read as "could not check", never as "nothing
        matched" — callers gate their flags on the difference."""
        def boom():
            raise RuntimeError("db down")
        assert ValueIndex(boom).resolve_spans("customers from Wakanda", ["customers"]) == {}
