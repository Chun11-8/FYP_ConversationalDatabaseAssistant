"""
Tests for column-description keying in backend/retrieval/data_processing.py.

Covers the table.column keying fix (previously descriptions were keyed by bare
column name, so same-named columns across tables clobbered each other).

Run from repo root:
    pytest backend/tests/test_column_meanings.py -v
"""

from __future__ import annotations

from backend.retrieval.data_processing import resolve_column_meaning


_MEANINGS = {
    "customers.segment":      ("business segment", "LAM, SME, KAM"),
    "customers.customerid":   ("customer id in customers", ""),
    "yearmonth.customerid":   ("customer id in yearmonth", ""),
    "yearmonth.consumption":  ("monthly consumption", ""),
}


class TestResolveColumnMeaning:

    def test_qualified_match_no_collision(self):
        # The two customerid columns must resolve to their OWN descriptions.
        assert resolve_column_meaning(_MEANINGS, "customers", "customerid")[0] == "customer id in customers"
        assert resolve_column_meaning(_MEANINGS, "yearmonth", "customerid")[0] == "customer id in yearmonth"

    def test_value_description_returned(self):
        desc, val = resolve_column_meaning(_MEANINGS, "customers", "segment")
        assert desc == "business segment"
        assert val == "LAM, SME, KAM"

    def test_case_insensitive(self):
        assert resolve_column_meaning(_MEANINGS, "Customers", "Segment")[0] == "business segment"

    def test_unambiguous_bare_fallback(self):
        # 'segment' exists in only one table → safe to resolve even if table differs.
        assert resolve_column_meaning(_MEANINGS, "mystery_table", "segment")[0] == "business segment"

    def test_ambiguous_bare_returns_empty(self):
        # 'customerid' exists in two tables → must NOT guess when table is unknown.
        assert resolve_column_meaning(_MEANINGS, "unknown_table", "customerid") == ("", "")

    def test_missing_column_returns_empty(self):
        assert resolve_column_meaning(_MEANINGS, "customers", "does_not_exist") == ("", "")

    def test_empty_meanings_returns_empty(self):
        assert resolve_column_meaning({}, "customers", "segment") == ("", "")
