"""
Tests: an executed query's results are never withheld by the trust
verdict.

The verdict is an OPINION about an answer, not a gate on it. If the SQL ran,
the user gets the rows and judges them against the checks; hiding them would
make a heuristic the arbiter of whether real data is visible, and would teach
the user that a red banner means "your results were taken away".

The frontend assertions are source-level: the render condition is JSX, and a
future edit adding `trust.level === 'passed' &&` to it is exactly the
regression worth catching.

Run from repo root:
    pytest backend/tests/test_results_always_shown.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"
_CHAT_MESSAGE = _FRONTEND / "components" / "ChatMessage.tsx"


class TestBackendSendsRowsRegardlessOfVerdict:

    def _result(self, rows):
        """The row/row_count contract run() puts on the response."""
        row_count = len(rows) if isinstance(rows, list) else None
        serialised = ([dict(r) for r in rows[:500]]
                      if isinstance(rows, list) and rows else None)
        return {"rows": serialised, "row_count": row_count}

    def test_rows_are_sent_when_the_query_returned_data(self):
        out = self._result([{"a": 1}, {"a": 2}])
        assert out["row_count"] == 2 and len(out["rows"]) == 2

    def test_zero_rows_is_distinguishable_from_never_executed(self):
        """The whole fix turns on this: 0 means it RAN and matched nothing;
        None means no query ran at all."""
        assert self._result([])["row_count"] == 0
        assert self._result(None)["row_count"] is None

    def test_the_row_payload_carries_no_verdict_dependency(self):
        """run() builds rows from execution_result alone — a grep guard,
        because adding a risk check to that branch would be easy and wrong."""
        import inspect
        from backend.agents.db_agent import OrchestrationAgent
        src = inspect.getsource(OrchestrationAgent.run)
        idx = src.index('"rows":')
        window = src[max(0, idx - 800):idx]
        for gate in ("risk_score", "trust", "untrusted"):
            assert gate not in window, (
                f"rows payload appears gated on {gate!r} — an executed query's "
                f"results must not depend on the verdict")


def _without_comments(text: str) -> str:
    """JSX and block comments stripped.

    Necessary, not cosmetic: the comment above <RecordsPanel> explains that it
    is NOT gated on trust, and a naive scan of the surrounding source matches
    the word in that very explanation.
    """
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


@pytest.fixture(scope="module")
def src() -> str:
    return _without_comments(_CHAT_MESSAGE.read_text(encoding="utf-8"))


class TestFrontendRendersResultsRegardlessOfVerdict:

    def test_the_records_panel_is_not_gated_on_trust(self, src):
        idx = src.index("<RecordsPanel")
        # the JSX condition sits just above the element
        condition = src[max(0, idx - 300):idx]
        for gate in ("trust", "risk_score", "untrusted", "verdict"):
            assert gate not in condition, (
                f"RecordsPanel appears gated on {gate!r}; an executed query's "
                f"rows must render whatever the verdict says")

    def test_a_zero_row_result_renders_its_own_panel(self, src):
        assert "EmptyResultPanel" in src
        assert "message.row_count === 0" in src, (
            "the empty state must key on row_count === 0 — `!rows` would also "
            "match a turn that never executed and claim it returned nothing")

    def test_the_sql_panel_is_not_gated_on_trust_either(self, src):
        idx = src.index("<SqlPanel")
        condition = src[max(0, idx - 200):idx]
        assert "trust" not in condition and "risk" not in condition

    def test_the_banner_says_results_are_still_shown(self, src):
        """A red verdict above a table invites 'were my results withheld?' —
        answered in the banner rather than left to inference."""
        assert "trust-still-shown" in src
        assert "executed" in src
