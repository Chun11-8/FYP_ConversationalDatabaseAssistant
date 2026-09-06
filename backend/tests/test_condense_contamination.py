"""
Tests for GH#87 — follow-up condensation must not merge unrelated questions.

Reported: turn 1 asked about SME consumption in 2013; turn 2, a self-contained
question about virtual schools in a DIFFERENT database, came back refusing
because it "could not find SME". The bare word `that` in a relative clause was
tripping the anaphora gate, and the condenser had no instruction permitting it
to decline.

All offline: no LLM, no database.

Run from repo root:
    pytest backend/tests/test_condense_contamination.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.agents.db_agent as dbm
from backend.agents.db_agent import OrchestrationAgent as A

HISTORY = [{"role": "user", "content": "consumption in SME for 2013"},
           {"role": "assistant", "content": "The average was 1234."}]

REPORTED = ("List the names of virtual schools that are among the top 5 in "
            "their respective counties based on average reading scores.")
CONTAMINATED = ("List the names of virtual schools in SME that are among the "
                "top 5 in their counties by average reading scores for 2013.")


# ── 1. the gate ───────────────────────────────────────────────────────────────

class TestAnaphoraGate:

    @pytest.mark.parametrize("q", [
        REPORTED,
        "Show me schools that have more than 500 students",
        "List all cards that are legal in commander",
        "Which customers pay in EUR?",
        "How many players are from Brazil?",
        "What types of events does the club hold?",
    ])
    def test_self_contained_questions_are_not_condensed(self, q):
        """`that` in a question is a relative pronoun, not a reference — and a
        short question is not a dependent one."""
        assert not A._looks_anaphoric(q, HISTORY)

    @pytest.mark.parametrize("q", [
        "What about 2014?",
        "And CZK?",
        "How many of them wear Medium?",
        "Only those with a Large t-shirt.",
        "Same for 2015",
        "And the previous year?",
    ])
    def test_real_follow_ups_are_still_condensed(self, q):
        """GH#63 must not regress: a genuine anaphor left un-condensed embeds
        to garbage and the cache replays the previous answer."""
        assert A._looks_anaphoric(q, HISTORY)

    def test_a_first_turn_is_never_condensed(self):
        assert not A._looks_anaphoric("What about 2014?", [])

    def test_the_relative_pronoun_is_gone_from_the_pattern(self):
        assert not dbm._ANAPHORIC_RE.search("schools that are large")
        # …while the demonstratives that genuinely refer back are kept.
        assert dbm._ANAPHORIC_RE.search("only those")
        assert dbm._ANAPHORIC_RE.search("how many of them")


# ── 2. the condenser may decline ──────────────────────────────────────────────

class TestCondenserMayDecline:

    def test_the_prompt_permits_returning_the_question_verbatim(self):
        from backend.agents.utils import PromptHarness
        text = PromptHarness.get_file("condense_query.md").lower()
        assert "verbatim" in text
        assert "different subject" in text

    def test_the_inline_fallback_says_the_same_thing(self):
        """The file may be missing; the fallback template must not then
        reinstate the merge-everything behaviour."""
        import inspect
        src = inspect.getsource(A.condense_query)
        assert "VERBATIM" in src


# ── 3. the merge veto ─────────────────────────────────────────────────────────

class TestMergeVeto:

    def test_the_reported_contamination_is_vetoed(self):
        reason = A._rewrite_merges_topics(REPORTED, CONTAMINATED)
        assert reason and "sme" in reason.lower()

    @pytest.mark.parametrize("raw,rewrite", [
        ("what about 2014?",
         "What was the total consumption of customer 6 in 2014?"),
        ("And CZK?", "How many customers pay in CZK?"),
        ("How many of them wear Medium?",
         "How many members of the club wear a Medium t-shirt?"),
        ("Only those with a Large t-shirt.",
         "How many members wear a Large t-shirt?"),
    ])
    def test_a_real_anaphor_is_never_vetoed(self, raw, rewrite):
        """The rewrite SHOULD add literals here — that is the whole job. A
        veto would reinstate GH#63, which returns a confident wrong answer
        rather than a refusal, so it is the worse failure of the two."""
        assert A._rewrite_merges_topics(raw, rewrite) is None

    def test_an_unchanged_rewrite_is_not_a_merge(self):
        assert A._rewrite_merges_topics(REPORTED, REPORTED) is None
        assert A._rewrite_merges_topics(REPORTED, "  " + REPORTED.upper()) is None

    def test_rewording_without_new_material_is_allowed(self):
        raw = "Show me the schools in Alameda County with the highest scores"
        assert A._rewrite_merges_topics(
            raw, "Show me schools in ALAMEDA COUNTY with the highest scores") is None

    def test_empty_input_is_safe(self):
        assert A._rewrite_merges_topics("", "x") is None
        assert A._rewrite_merges_topics("x", "") is None

    def test_a_span_extraction_failure_does_not_block_the_turn(self):
        with patch("backend.retrieval.value_index.extract_value_spans",
                   side_effect=RuntimeError("boom")):
            assert A._rewrite_merges_topics(REPORTED, CONTAMINATED) is None


# ── the node wires the veto in ────────────────────────────────────────────────

class TestCondenseQueryNode:

    def _agent(self):
        a = A.__new__(A)
        a._condense = True
        a.intent_llm = None
        a.llm = MagicMock()
        return a

    def _run(self, agent, question, llm_reply):
        reply = MagicMock()
        reply.content = llm_reply
        with patch.object(dbm, "invoke_tracked", return_value=reply):
            return agent.condense_query({
                "user_query": question,
                "conversation_history": HISTORY,
                "previous_sql": "SELECT 1",
            })

    def test_a_merging_rewrite_is_discarded(self):
        """Even when the gate fires and the LLM merges anyway, the turn runs
        on the question the user actually asked."""
        out = self._run(self._agent(), REPORTED, CONTAMINATED)
        assert out.get("standalone_query") is None

    def test_a_good_rewrite_is_kept(self):
        out = self._run(self._agent(), "what about 2014?",
                        "What was the consumption of customer 6 in 2014?")
        assert out["standalone_query"] == \
            "What was the consumption of customer 6 in 2014?"

    def test_the_gate_still_short_circuits_before_any_llm_call(self):
        agent = self._agent()
        with patch.object(dbm, "invoke_tracked") as llm:
            out = agent.condense_query({
                "user_query": REPORTED,
                "conversation_history": HISTORY,
                "previous_sql": None,
            })
        llm.assert_not_called()
        assert out.get("standalone_query") is None
