"""
Tests for backend/agents/sql_code_agent.py — Tier 1 (Issue #14/#18).

Covers SQLite dialect normalization and the reranker's JOIN-quality criterion.
All LLM calls are mocked, so these run offline.

Run from repo root:
    pytest backend/tests/test_sql_code_agent.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.agents.sql_code_agent import SQLCodeAgent, _to_sqlite
from backend.agents.utils import PromptHarness


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _agent(dialect: str) -> SQLCodeAgent:
    """Build an agent with a dummy LLM and the given target dialect."""
    return SQLCodeAgent({"db": {"dialect": dialect}}, MagicMock())


# ── _to_sqlite regex pass ─────────────────────────────────────────────────────

class TestToSqliteRegex:

    def test_ilike_becomes_like(self):
        assert "ILIKE" not in _to_sqlite("SELECT id FROM t WHERE name ILIKE '%a%'").upper().replace("LIKE", "")
        assert "LIKE" in _to_sqlite("SELECT id FROM t WHERE name ILIKE '%a%'")

    def test_string_agg_becomes_group_concat(self):
        out = _to_sqlite("SELECT STRING_AGG(name, ',') FROM t")
        assert "GROUP_CONCAT(" in out
        assert "STRING_AGG" not in out.upper()

    def test_double_colon_numeric_cast(self):
        out = _to_sqlite("SELECT price::NUMERIC FROM t")
        assert "::" not in out
        assert "CAST(price AS REAL)" in out

    def test_double_colon_numeric_with_precision(self):
        out = _to_sqlite("SELECT price::numeric(10,2) FROM t")
        assert "::" not in out
        assert "CAST(price AS REAL)" in out

    def test_cast_numeric_becomes_real(self):
        out = _to_sqlite("SELECT CAST(price AS NUMERIC(10,2)) FROM t")
        assert "NUMERIC" not in out.upper()
        assert "CAST(price AS REAL)" in out

    def test_boolean_literals_become_integers(self):
        out = _to_sqlite("SELECT id FROM t WHERE active = TRUE AND deleted = FALSE")
        assert "TRUE" not in out.upper()
        assert "FALSE" not in out.upper()
        assert "= 1" in out and "= 0" in out

    def test_idempotent(self):
        once = _to_sqlite("SELECT STRING_AGG(name, ',') FROM t WHERE x ILIKE 'a'")
        twice = _to_sqlite(once)
        assert once == twice


# ── normalize_dialect ─────────────────────────────────────────────────────────

class TestNormalizeDialect:

    def test_noop_for_postgres(self):
        agent = _agent("postgres")
        sql = "SELECT price::NUMERIC FROM t WHERE name ILIKE '%a%'"
        assert agent.normalize_dialect(sql) == sql

    def test_noop_for_empty(self):
        agent = _agent("sqlite")
        assert agent.normalize_dialect("") == ""

    def test_preserves_valid_select(self):
        agent = _agent("sqlite")
        sql = "SELECT id, name FROM players WHERE rating > 80 LIMIT 100"
        out = agent.normalize_dialect(sql)
        # Logic preserved — same tables/columns/filter survive
        assert "players" in out
        assert "rating" in out
        assert "LIMIT 100" in out.upper().replace("  ", " ")

    def test_cast_ilike_stringagg_rewritten(self):
        agent = _agent("sqlite")
        out = agent.normalize_dialect(
            "SELECT STRING_AGG(name, ',') FROM t WHERE flag = TRUE AND x ILIKE '%a%'"
        )
        assert "STRING_AGG" not in out.upper()
        assert "ILIKE" not in out.upper()
        assert "GROUP_CONCAT" in out.upper()

    def test_malformed_sql_does_not_raise(self):
        agent = _agent("sqlite")
        # sqlglot may fail to parse; regex fallback must still run without raising
        out = agent.normalize_dialect("SELECT STRING_AGG(((")
        assert isinstance(out, str)


# ── generate() applies normalization ──────────────────────────────────────────

class TestGenerateNormalizes:

    def test_generated_candidates_are_normalized(self):
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT STRING_AGG(name, ',') FROM players LIMIT 10\n```"
        )
        sqls = agent.generate(query="q", schema_context="players(name)", few_shots=[], history=[])
        assert len(sqls) == 1
        assert "STRING_AGG" not in sqls[0].upper()
        assert "GROUP_CONCAT" in sqls[0].upper()


# ── Default few-shots (decomposition + grain) injected when cache empty ────────

class TestDefaultFewShots:

    def test_default_fewshots_injected_when_cache_empty(self):
        from backend.agents.sql_code_agent import DEFAULT_FEWSHOTS

        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="q", schema_context="s", few_shots=[], history=[])

        # The prompt sent to the LLM should embed the default examples.
        sent_prompt = agent.llm.invoke.call_args[0][0][1].content
        assert "AVG(t2.consumption) / 12" in sent_prompt   # grain/formula example
        assert "ORDER BY SUM(t2.consumption) ASC" in sent_prompt

    def test_cache_fewshots_take_precedence(self):
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="q", schema_context="s",
                       few_shots=["CACHED_EXAMPLE_X"], history=[])
        sent_prompt = agent.llm.invoke.call_args[0][0][1].content
        assert "CACHED_EXAMPLE_X" in sent_prompt
        assert "AVG(t2.consumption) / 12" not in sent_prompt


# ── ReAct regenerate (refine on execution observation) ────────────────────────

class TestRegenerate:

    def test_regenerate_fills_template_and_extracts_sql(self):
        agent = _agent("postgres")
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT AVG(c) / 12 FROM t\n```"
        )
        sqls = agent.regenerate(
            query="avg monthly consumption",
            schema_context="t(c)",
            original_sql="SELECT AVG(c) FROM t",
            error="The query returned 0 rows.",
            history=[],
        )
        assert sqls == ["SELECT AVG(c) / 12 FROM t"]
        # The observation + original SQL + question must reach the prompt.
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "0 rows" in sent
        assert "SELECT AVG(c) FROM t" in sent
        assert "avg monthly consumption" in sent

    def test_regenerate_normalizes_dialect(self):
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT STRING_AGG(name, ',') FROM t\n```"
        )
        sqls = agent.regenerate(query="q", schema_context="t", original_sql="bad",
                                error="boom", history=[])
        assert "GROUP_CONCAT" in sqls[0].upper()
        assert "STRING_AGG" not in sqls[0].upper()

    def test_regenerate_empty_output_returns_empty(self):
        agent = _agent("postgres")
        agent.llm.invoke.return_value = MagicMock(content="sorry, no idea")
        assert agent.regenerate(query="q", schema_context="t", original_sql="x",
                                error="e", history=[]) == []


# ── Evidence-first handling + conditional date rule (Issue #18) ────────────────

class TestEvidenceHandling:

    def test_prompt_states_evidence_verbatim_for_filters(self):
        prompt = PromptHarness.get("SQL_CODE_AGENT")
        low = prompt.lower()
        # Evidence must be applied verbatim, and to filters (WHERE/HAVING), not only SELECT.
        assert "verbatim" in low
        assert "where" in low and "having" in low
        assert "not only" in low and "select" in low

    def test_date_rule_is_conditional_not_always_substr(self):
        prompt = PromptHarness.get("SQL_CODE_AGENT")
        low = prompt.lower()
        # SUBSTR/LIKE is now tied to a partial / year-only match, and exact values
        # must be matched exactly (no month truncation).
        assert "partial match" in low or "year-only" in low
        assert "exactly" in low

    def test_inline_build_prompt_has_evidence_and_conditional_date(self):
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="q", schema_context="s", few_shots=["x"], history=[])
        sent = agent.llm.invoke.call_args[0][0][1].content.lower()
        assert "verbatim" in sent
        assert "201208" in sent              # exact-date example present
        assert "partial match" in sent

    def test_default_fewshot_has_exact_date_match_and_nullif(self):
        from backend.agents.sql_code_agent import DEFAULT_FEWSHOTS
        joined = "\n".join(DEFAULT_FEWSHOTS).lower()
        assert "date = '201208'" in joined          # exact-value date match
        assert "nullif(" in joined                  # computed filter guarded
        assert "price / nullif" in joined

    def test_evidence_formula_as_target_guidance_present(self):
        # Issue #50 step 2: "how many times more" must not be reinterpreted as
        # COUNT(*)/HAVING — the formula IS the SELECT expression.
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        agent.generate(query="q", schema_context="s", few_shots=["x"], history=[])
        sent = agent.llm.invoke.call_args[0][0][1].content
        assert "how many times" in sent.lower()
        assert "HAVING SUM(X) > SUM(Y)" in sent

    def test_regeneration_prompt_includes_evidence_from_question(self):
        agent = _agent("sqlite")
        agent.llm.invoke.return_value = MagicMock(content="```sql\nSELECT 1\n```")
        q = "<domain_context>Date = '201208'; per unit = Price / Amount</domain_context>\n\nGive consumption"
        agent.regenerate(query=q, schema_context="s", original_sql="bad",
                         error="0 rows", history=[])
        sent = agent.llm.invoke.call_args[0][0][1].content
        # The evidence embedded in the question reaches the regeneration prompt,
        # and the template instructs applying it verbatim.
        assert "Date = '201208'" in sent
        assert "VERBATIM" in sent


# ── Compositional coverage + value grounding note (Issues #20 / #19) ───────────

class TestCompositionalCoverage:

    def test_prompt_covers_having_vs_where(self):
        low = PromptHarness.get("SQL_CODE_AGENT").lower()
        assert "having" in low and "group by" in low
        # per-group aggregate threshold → HAVING, not WHERE
        assert "per-group" in low or "per group" in low

    def test_prompt_covers_exclusion_setops(self):
        prompt = PromptHarness.get("SQL_CODE_AGENT")
        low = prompt.lower()
        assert "except" in low
        assert "not in" in low or "<>" in prompt
        assert "exclusion" in low or "without" in low or "other than" in low

    def test_prompt_covers_bridge_join(self):
        low = PromptHarness.get("SQL_CODE_AGENT").lower()
        assert "junction" in low or "bridge" in low
        assert "link_to_event" in low  # the concrete example

    def test_prompt_covers_computed_filter_and_nullif(self):
        prompt = PromptHarness.get("SQL_CODE_AGENT")
        assert "NULLIF" in prompt
        assert "/ NULLIF" in prompt or "NULLIF(Amount" in prompt

    def test_prompt_has_value_list_note(self):
        prompt = PromptHarness.get("SQL_CODE_AGENT")
        assert "values: {" in prompt           # references the schema values list
        assert "Women''s Soccer" in prompt      # apostrophe-escaping example

    def test_default_fewshots_include_having_except_and_bridge(self):
        from backend.agents.sql_code_agent import DEFAULT_FEWSHOTS
        joined = "\n".join(DEFAULT_FEWSHOTS)
        assert "HAVING COUNT(" in joined
        assert "EXCEPT" in joined
        assert "t2.link_to_member = t3.member_id" in joined   # bridge join through junction
        assert "'Women''s Soccer'" in joined                  # escaped value

    def test_reranker_prompt_mentions_having_and_bridge(self):
        prompt = PromptHarness.get_file("sql_reranker.md")
        low = prompt.lower()
        assert "having" in low
        assert "junction" in low or "bridge" in low
        assert "except" in low


# ── Reranker prompt + JOIN criterion ──────────────────────────────────────────

class TestReranker:

    def test_reranker_prompt_loads_and_mentions_join(self):
        prompt = PromptHarness.get_file("sql_reranker.md")
        assert "JOIN" in prompt.upper()
        assert "foreign-key" in prompt.lower() or "foreign key" in prompt.lower()

    def test_rerank_returns_single_candidate_untouched(self):
        agent = _agent("postgres")
        assert agent.rerank(["SELECT 1"], "q", "schema") == "SELECT 1"

    def test_rerank_empty_returns_empty(self):
        agent = _agent("postgres")
        assert agent.rerank([], "q", "schema") == ""

    def test_rerank_picks_llm_choice(self):
        agent = _agent("postgres")
        # Candidate 2 joins on the correct FK column; the (mocked) judge picks it.
        agent.llm.invoke.return_value = MagicMock(content="2")
        candidates = [
            "SELECT p.name FROM players p JOIN teams t ON p.team_api_id = t.id LIMIT 10",
            "SELECT p.name FROM players p JOIN teams t ON p.team_id = t.team_id LIMIT 10",
        ]
        best = agent.rerank(candidates, "players and their teams", "schema with FK players.team_id -> teams.team_id")
        assert best == candidates[1]

    def test_rerank_falls_back_to_first_on_llm_error(self):
        agent = _agent("postgres")
        agent.llm.invoke.side_effect = RuntimeError("LLM down")
        candidates = ["SELECT 1", "SELECT 2"]
        assert agent.rerank(candidates, "q", "schema") == "SELECT 1"
