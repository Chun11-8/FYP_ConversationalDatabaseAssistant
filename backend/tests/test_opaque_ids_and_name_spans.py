"""
Tests: the value index must not treat machine-generated ids as
values a user could have meant, and a name split across two columns must not
be reported as absent from the database.

Reported: "Calculate the total average cost that Elijah Allen spent…" was
refused with "I could not find Elijah Allen anywhere in this database. The
closest values I do have are: recmbOVHSyzXQZpQr, recoVo6dFSzG1ypX7, …".
Elijah Allen is a member; `first_name` and `last_name` are separate columns.

All offline: the DB boundary is faked.

Run from repo root:
    pytest backend/tests/test_opaque_ids_and_name_spans.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.retrieval.value_index import (
    ValueIndex, _column_is_opaque, _looks_opaque,
)

MEMBER = {
    "member_id":    ["recmbOVHSyzXQZpQr", "recoVo6dFSzG1ypX7", "reckYL4xtQWpkTJ8k"],
    "first_name":   ["Elijah", "Sacha", "Grant"],
    "last_name":    ["Allen", "Harrison", "Gilmour"],
    "position":     ["Member", "President"],
}


class _Cur:
    def __init__(self, data):
        self._data, self._rows = data, []

    def execute(self, sql, params=None):
        if "information_schema" in sql:
            self._rows = [(c,) for c in self._data.get(params[0], {})]
        else:
            for table, cols in self._data.items():
                for col, values in cols.items():
                    if f'"{col}"' in sql and f'"{table}"' in sql:
                        self._rows = [(v,) for v in values]

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _index(data=None):
    conn = MagicMock()
    conn.cursor.return_value = _Cur(data or {"member": MEMBER})
    return ValueIndex(lambda: conn)


# ── opaque identifiers ────────────────────────────────────────────────────────

class TestOpaqueDetection:

    @pytest.mark.parametrize("v", [
        "recmbOVHSyzXQZpQr", "recoVo6dFSzG1ypX7", "reckYL4xtQWpkTJ8k",
        "550e8400-e29b-41d4-a716-446655440000",
        "a3f5c9e1b7d2408fa6c4",
    ])
    def test_identifiers_are_opaque(self, v):
        assert _looks_opaque(v)

    @pytest.mark.parametrize("v", [
        "LAM", "EUR", "Medium", "Slovnaft",
        "Emery Secondary", "Julian Charter",       # spaces: a human wrote it
        "debit_card_specializing",                 # single case
        "Elijah", "Allen",
    ])
    def test_real_values_are_not_opaque(self, v):
        assert not _looks_opaque(v)

    def test_a_column_needs_a_clear_majority_to_be_dropped(self):
        """One odd entry must not disqualify a column of real values."""
        assert _column_is_opaque(["recmbOVHSyzXQZpQr", "recoVo6dFSzG1ypX7"])
        assert not _column_is_opaque(["Elijah", "Sacha", "recmbOVHSyzXQZpQr"])
        assert not _column_is_opaque([])

    def test_an_identifier_column_is_never_indexed(self):
        cols = _index()._load_table("member")
        assert "member_id" not in cols
        assert set(cols) == {"first_name", "last_name", "position"}


# ── a name spread across two columns ──────────────────────────────────────────

class TestMultiWordSpans:

    def test_the_reported_name_is_no_longer_reported_absent(self):
        idx = _index()
        resolved = idx.resolve_spans(
            "Calculate the total cost that Elijah Allen spent", ["member"])
        assert resolved["Elijah Allen"] == "Elijah Allen"

    def test_a_name_whose_parts_are_not_all_present_stays_absent(self):
        """Strictness matters: one matching word out of two would resolve
        'Elijah Baggins', and the refusal exists to catch exactly that."""
        idx = _index()
        resolved = idx.resolve_spans("What did Elijah Baggins spend?", ["member"])
        assert resolved.get("Elijah Baggins") is None

    def test_a_single_word_miss_is_still_a_miss(self):
        idx = _index()
        assert idx.resolve_spans("customers from Wakanda", ["member"]).get(
            "Wakanda") is None

    def test_parts_must_match_within_one_table(self):
        idx = _index({"member": {"first_name": ["Elijah"]},
                      "event":  {"name": ["Allen Cup"]}})
        # 'Allen' lives in another table — a cross-table coincidence is not
        # evidence that a member called 'Elijah Allen' exists.
        assert idx.resolve_spans("Elijah Allen", ["member"]).get(
            "Elijah Allen") is None


# ── suggestions in the refusal ────────────────────────────────────────────────

class TestNearestValues:

    def test_suggestions_are_ranked_by_similarity_to_what_was_typed(self):
        out = _index().nearest_values("Elijah Bagginz", ["member"], limit=3)
        assert out[0] == "Elijah"

    def test_identifiers_can_never_be_suggested(self):
        out = _index().nearest_values("Elijah Allen", ["member"], limit=6)
        assert not any(v.startswith("rec") for v in out)

    def test_no_tables_or_span_is_safe(self):
        assert _index().nearest_values("", ["member"]) == []
        assert _index().nearest_values("x", []) == []

    def test_a_load_failure_degrades_to_no_suggestion(self):
        idx = ValueIndex(MagicMock(side_effect=OSError("no pg")))
        assert idx.nearest_values("x", ["member"]) == []


class TestRefusalUsesIt:

    def test_the_refusal_ranks_values_against_the_missing_span(self):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a.value_index = MagicMock()
        a.value_index.nearest_values.return_value = ["Elijah", "Elias"]
        out = a._nearest_values(["Elijah Baggins"],
                                {"retrieved_tables": ["member", "event"]})
        a.value_index.nearest_values.assert_called_once_with(
            "Elijah Baggins", ["member", "event"], limit=6)
        assert out == "`Elijah`, `Elias`"

    def test_a_failure_just_omits_the_suggestion(self):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a.value_index = MagicMock()
        a.value_index.nearest_values.side_effect = RuntimeError("x")
        assert a._nearest_values(["q"], {"retrieved_tables": ["t"]}) == ""


# ── high-cardinality columns make absence unprovable ───────────────────

class TestUnindexedValuesBlockRefusal:
    """
    A column above the 2 000-value cap is not indexed, so a real value in it
    matches nothing — for exactly the same reason a fictional value matches
    nothing. Refusing on that evidence tells the user their own data does not
    exist, which is the most expensive mistake this system can make.
    """

    def _big_table(self, n=2001):
        return {"customers": {"full_name": [f"Person {i}" for i in range(n)],
                              "segment": ["LAM", "SME"]}}

    def test_an_overflow_column_is_recorded_not_silently_dropped(self):
        idx = _index(self._big_table())
        cols = idx._load_table("customers")
        assert "full_name" not in cols          # not indexed
        assert "segment" in cols                # small column still is
        assert idx._overflow["customers"] == ["full_name"]

    def test_has_unindexed_values_reports_it(self):
        assert _index(self._big_table()).has_unindexed_values(["customers"])

    def test_a_fully_indexed_table_reports_nothing_unindexed(self):
        assert not _index().has_unindexed_values(["member"])

    def test_no_refusal_when_values_could_not_be_checked(self):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a.value_index = _index(self._big_table())
        assert a._unsatisfiable_spans("spend by Elijah Allen", ["customers"]) == []

    def test_a_refusal_still_fires_when_everything_was_checkable(self):
        from backend.agents.db_agent import OrchestrationAgent
        a = OrchestrationAgent.__new__(OrchestrationAgent)
        a.value_index = _index()
        assert a._unsatisfiable_spans(
            "members from Wakanda", ["member"]) == ["Wakanda"]

    def test_a_column_exactly_at_the_cap_is_still_indexed(self):
        idx = _index(self._big_table(n=2000))
        assert "full_name" in idx._load_table("customers")
        assert not idx.has_unindexed_values(["customers"])
