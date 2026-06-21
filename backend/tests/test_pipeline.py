"""
Tests for backend/pipelines/postprocessing.py (Issue #10).

All tests mock the LLM so no real API calls are made.

Run from repo root:
    pytest backend/tests/test_pipeline.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from backend.pipelines.postprocessing import ResponseOptimizer, INLINE_LIMIT, TABLE_LIMIT, GENERAL_MAX_CHARS


def _make_optimizer(llm_response: str = "Nice summary.") -> ResponseOptimizer:
    """Return an optimizer with a stubbed LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=llm_response)
    return ResponseOptimizer(mock_llm, output_dir="/tmp")


def _rows(n: int) -> list[dict]:
    return [{"id": i, "name": f"Player {i}"} for i in range(n)]


# ── Branch routing ────────────────────────────────────────────────────────────

class TestFormatBranchSelection:

    def test_none_result_returns_no_records_message(self):
        opt = _make_optimizer()
        text, path = opt.format("q", "SELECT 1", None)
        assert "no matching" in text.lower()
        assert path is None

    def test_error_string_returns_error_prefix(self):
        opt = _make_optimizer()
        text, path = opt.format("q", "SELECT 1", "connection refused")
        assert text.startswith("Error:")
        assert path is None

    def test_empty_list_returns_empty_message(self):
        opt = _make_optimizer()
        text, path = opt.format("q", "SELECT 1", [])
        assert path is None
        assert "no matching" in text.lower() or "not exist" in text.lower()

    def test_inline_branch_for_small_result(self):
        opt = _make_optimizer("Here are 3 players.")
        text, path = opt.format("show players", "SELECT name FROM p", _rows(3))
        assert path is None
        # response is LLM text only — no SQL block embedded (frontend handles that)
        assert "Here are 3 players." in text

    def test_table_branch_for_medium_result(self):
        opt = _make_optimizer("Found many players.")
        text, path = opt.format("show all", "SELECT name FROM p", _rows(50))
        assert path is None
        # markdown table headers present
        assert "|" in text

    def test_csv_branch_for_large_result(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Big result.")
        opt = ResponseOptimizer(mock_llm, output_dir=str(tmp_path))
        text, path = opt.format("huge query", "SELECT * FROM p", _rows(TABLE_LIMIT + 1))
        assert path is not None
        assert path.endswith(".csv")

    def test_inline_boundary_exactly_inline_limit(self):
        opt = _make_optimizer()
        text, path = opt.format("q", "SELECT 1", _rows(INLINE_LIMIT))
        assert path is None

    def test_table_boundary_exactly_table_limit(self):
        opt = _make_optimizer()
        text, path = opt.format("q", "SELECT 1", _rows(TABLE_LIMIT))
        assert path is None
        assert "|" in text


# ── Output guardrail ──────────────────────────────────────────────────────────

class TestOutputGuardrail:

    def test_clean_response_passes_through(self):
        opt = _make_optimizer("There are 42 players from Brazil.")
        text, _ = opt.format("count", "SELECT count(*) FROM p", _rows(1))
        assert "suppressed" not in text

    def test_injection_in_llm_output_is_suppressed(self):
        # LLM echoes a semicolon-DROP sequence in its natural-language answer
        opt = _make_optimizer("Sure! Just run: SELECT 1; DROP TABLE players;")
        text, _ = opt.format("delete all", "SELECT 1", _rows(1))
        assert "suppressed" in text.lower() or "safety filter" in text.lower()

    def test_guardrail_standalone(self):
        opt = _make_optimizer()
        safe = opt._guardrail("Normal answer with no SQL keywords.")
        assert safe == "Normal answer with no SQL keywords."

    def test_guardrail_blocks_injection(self):
        opt = _make_optimizer()
        unsafe = opt._guardrail("The answer is: SELECT 1 UNION SELECT * FROM users")
        assert "suppressed" in unsafe.lower()


# ── truncate_general ─────────────────────────────────────────────────────────

class TestTruncateGeneral:

    def test_short_text_unchanged(self):
        opt = _make_optimizer()
        text = "Short answer."
        assert opt.truncate_general(text) == text

    def test_long_text_triggers_summarization(self):
        opt = _make_optimizer("Concise summary.")
        long_text = "word " * (GENERAL_MAX_CHARS // 5 + 10)
        result = opt.truncate_general(long_text)
        assert result == "Concise summary."
        opt.llm.invoke.assert_called_once()

    def test_long_text_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        opt = ResponseOptimizer(mock_llm, output_dir="/tmp")
        long_text = "x" * (GENERAL_MAX_CHARS + 100)
        result = opt.truncate_general(long_text)
        # Hard truncation kicks in
        assert len(result) <= GENERAL_MAX_CHARS + 100
        assert "truncated" in result.lower()

    def test_custom_max_chars(self):
        opt = _make_optimizer("Summary.")
        long_text = "word " * 200
        result = opt.truncate_general(long_text, max_chars=50)
        # LLM was called
        opt.llm.invoke.assert_called_once()


# ── SQL NOT embedded in response (frontend SqlPanel handles it) ────────────────

class TestSqlInOutput:

    def test_sql_not_embedded_in_inline_response(self):
        # The SQL block must NOT appear in the response string — it is sent
        # as a separate `sql` field and rendered by the frontend SqlPanel.
        opt = _make_optimizer("Here are the results.")
        text, _ = opt.format("players", "SELECT name FROM players LIMIT 5", _rows(2))
        assert "```sql" not in text
        assert "SQL Used" not in text

    def test_record_count_not_embedded_in_response(self):
        # Row count is sent as `row_count` field; the response string is LLM text only.
        opt = _make_optimizer("Summary.")
        text, _ = opt.format("query", "SELECT 1", _rows(5))
        assert "record(s) returned" not in text

    def test_inline_response_is_llm_text_only(self):
        opt = _make_optimizer("Exactly this text.")
        text, _ = opt.format("q", "SELECT 1", _rows(1))
        assert text == "Exactly this text."

    def test_table_response_contains_markdown_table_not_sql(self):
        opt = _make_optimizer("Summary header.")
        text, _ = opt.format("q", "SELECT id, name FROM p", _rows(20))
        assert "|" in text          # markdown table present
        assert "```sql" not in text  # SQL block NOT embedded

    def test_file_response_is_plain_message_not_sql(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="x")
        opt = ResponseOptimizer(mock_llm, output_dir=str(tmp_path))
        text, path = opt.format("q", "SELECT 1", _rows(TABLE_LIMIT + 1))
        assert "```sql" not in text
        assert path is not None
