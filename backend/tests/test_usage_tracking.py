"""
Tests for GH#57 per-request LLM token/cost accounting (backend/agents/usage.py).

All offline — no OpenAI calls. The "LLM" is a stub object exposing the same
surface invoke_tracked() reads: .invoke(), .model_name, and a response carrying
usage_metadata.

Run from repo root:
    pytest backend/tests/test_usage_tracking.py -v
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.agents import usage


class _Response:
    def __init__(self, content="ok", usage_metadata=None, response_metadata=None):
        self.content = content
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        if response_metadata is not None:
            self.response_metadata = response_metadata


class _LLM:
    def __init__(self, model="gpt-4o-mini", response=None):
        self.model_name = model
        self._response = response or _Response(
            usage_metadata={"input_tokens": 1000, "output_tokens": 500})

    def invoke(self, messages):
        return self._response


@pytest.fixture(autouse=True)
def _clean_tls():
    yield
    usage.clear()


# ── cost math ────────────────────────────────────────────────────────────────

class TestCostUsd:

    def test_known_model_prices_per_million(self):
        # gpt-4o-mini defaults: $0.15/1M in, $0.60/1M out
        assert usage.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        assert usage.cost_usd("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)

    def test_unknown_model_costs_zero_but_does_not_raise(self):
        assert usage.cost_usd("some-future-model", 1_000_000, 1_000_000) == 0.0

    def test_none_model_costs_zero(self):
        assert usage.cost_usd(None, 5000, 5000) == 0.0

    def test_set_pricing_overrides_default(self):
        try:
            usage.set_pricing({"gpt-4o-mini": {"input": 1.0, "output": 2.0}})
            assert usage.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(1.0)
        finally:
            usage.set_pricing({"gpt-4o-mini": {"input": 0.15, "output": 0.60}})

    def test_malformed_pricing_entry_is_ignored(self):
        usage.set_pricing({"weird": "not-a-dict"})
        assert usage.cost_usd("weird", 1_000_000, 0) == 0.0

    def test_set_pricing_tolerates_non_dict_table(self):
        usage.set_pricing(None)      # must not raise
        usage.set_pricing("nope")


# ── tracker accumulation ─────────────────────────────────────────────────────

class TestUsageTracker:

    def test_totals_accumulate_across_calls(self):
        t = usage.UsageTracker()
        t.record("generate_sql", "gpt-4o-mini", 100, 50)
        t.record("generate_sql", "gpt-4o-mini", 200, 25)
        d = t.as_dict()
        assert d["calls"] == 2
        assert d["input_tokens"] == 300
        assert d["output_tokens"] == 75

    def test_per_stage_attribution(self):
        t = usage.UsageTracker()
        t.record("classify_intent", "gpt-4o-mini", 10, 1)
        t.record("generate_sql", "gpt-4o-mini", 4000, 300)
        d = t.as_dict()
        assert d["by_stage"]["classify_intent"]["calls"] == 1
        assert d["by_stage"]["generate_sql"]["input_tokens"] == 4000
        assert set(d["by_stage"]) == {"classify_intent", "generate_sql"}

    def test_cost_sums_across_stages(self):
        t = usage.UsageTracker()
        t.record("a", "gpt-4o-mini", 1_000_000, 0)   # $0.15
        t.record("b", "gpt-4o-mini", 1_000_000, 0)   # $0.15
        assert t.as_dict()["est_cost_usd"] == pytest.approx(0.30)

    def test_fresh_tracker_is_empty(self):
        assert usage.UsageTracker().as_dict() == {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "est_cost_usd": 0.0, "by_stage": {},
            # GH#114: no call has been billed yet, so nothing is unpriced and
            # the (zero) cost is trivially known.
            "models": [], "cost_known": True, "unpriced_models": [],
        }


# ── invoke_tracked ───────────────────────────────────────────────────────────

class TestInvokeTracked:

    def test_records_against_current_request(self):
        tracker = usage.start_request()
        usage.invoke_tracked(_LLM(), [], "generate_sql")
        d = tracker.as_dict()
        assert d["calls"] == 1 and d["input_tokens"] == 1000

    def test_returns_response_unchanged(self):
        usage.start_request()
        out = usage.invoke_tracked(_LLM(), [], "s")
        assert out.content == "ok"

    def test_no_tracker_is_a_no_op_not_an_error(self):
        usage.clear()
        assert usage.invoke_tracked(_LLM(), [], "s").content == "ok"

    def test_missing_usage_metadata_counts_zero_tokens(self):
        tracker = usage.start_request()
        usage.invoke_tracked(_LLM(response=_Response()), [], "s")
        d = tracker.as_dict()
        assert d["calls"] == 1
        assert d["input_tokens"] == 0 and d["output_tokens"] == 0

    def test_falls_back_to_response_metadata_token_usage(self):
        tracker = usage.start_request()
        resp = _Response(response_metadata={
            "token_usage": {"prompt_tokens": 77, "completion_tokens": 11}})
        usage.invoke_tracked(_LLM(response=resp), [], "s")
        d = tracker.as_dict()
        assert d["input_tokens"] == 77 and d["output_tokens"] == 11

    def test_explicit_tracker_beats_thread_local(self):
        other = usage.UsageTracker()
        usage.start_request()          # thread-local tracker, must stay empty
        usage.invoke_tracked(_LLM(), [], "s", tracker=other)
        assert other.as_dict()["calls"] == 1
        assert usage.current().as_dict()["calls"] == 0

    def test_pool_workers_attribute_to_the_captured_tracker(self):
        """The generator fans out path-diversity calls across a pool; those
        threads have no thread-local, so the tracker must ride along."""
        tracker = usage.start_request()
        llm = _LLM()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(usage.invoke_tracked, llm, [], "generate_sql", tracker)
                       for _ in range(3)]
            for f in futures:
                f.result()
        assert tracker.as_dict()["calls"] == 3
        assert tracker.as_dict()["input_tokens"] == 3000

    def test_start_request_replaces_previous_tracker(self):
        first = usage.start_request()
        usage.invoke_tracked(_LLM(), [], "s")
        second = usage.start_request()
        usage.invoke_tracked(_LLM(), [], "s")
        assert first.as_dict()["calls"] == 1
        assert second.as_dict()["calls"] == 1
