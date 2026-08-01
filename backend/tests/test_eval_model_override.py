"""
Tests for the eval-CLI --model override (companion to GH#52's frontend/API
model picker): backend/eval/bird.py::resolve_model, its reuse from
bird_predict.py, run_eval.py's model= passthrough into agent.run(), and that
each CLI's argparse actually accepts --model.

All offline — no network, no OpenAI key, no BIRD dataset required.

Run from repo root:
    pytest backend/tests/test_eval_model_override.py -v
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

import backend.eval.bird as bird
import backend.eval.bird_predict as bird_predict
import backend.eval.run_eval as run_eval_mod
from backend.eval.bird import resolve_model


CONFIG = {"llm": {"model": "gpt-4o-mini",
                   "selectable_models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]}}


# ── resolve_model ──────────────────────────────────────────────────────────────

class TestResolveModel:

    def test_none_falls_back_to_config_default(self):
        assert resolve_model(None, CONFIG) == "gpt-4o-mini"

    def test_cli_value_wins(self):
        assert resolve_model("gpt-4o", CONFIG) == "gpt-4o"

    def test_blank_string_treated_as_unset(self):
        assert resolve_model("   ", CONFIG) == "gpt-4o-mini"

    def test_empty_string_treated_as_unset(self):
        assert resolve_model("", CONFIG) == "gpt-4o-mini"

    def test_model_outside_allowlist_still_returned(self, caplog):
        # Free-form input is intentional for the eval CLI — never blocked,
        # only warned (unlike the public chat API's strict allow-list).
        with caplog.at_level(logging.WARNING):
            result = resolve_model("gpt-5-research-preview", CONFIG)
        assert result == "gpt-5-research-preview"
        assert any("not in config.yaml" in r.message for r in caplog.records)

    def test_model_in_allowlist_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_model("gpt-4o", CONFIG)
        assert not any("not in config.yaml" in r.message for r in caplog.records)

    def test_no_selectable_models_configured_skips_check(self, caplog):
        config = {"llm": {"model": "gpt-4o-mini"}}
        with caplog.at_level(logging.WARNING):
            result = resolve_model("anything-goes", config)
        assert result == "anything-goes"
        assert not any("not in config.yaml" in r.message for r in caplog.records)

    def test_bird_predict_reexports_same_function(self):
        # bird_predict.py imports resolve_model from bird.py — must be the
        # identical function, not a divergent copy.
        assert bird_predict.resolve_model is bird.resolve_model


# ── run_eval.py: model= passthrough ─────────────────────────────────────────────

class TestRunEvalModelPassthrough:

    def _agent(self):
        agent = MagicMock()
        agent.run.return_value = {
            "response": "", "sql": "SELECT 1", "intent": "READ",
            "validation": {"valid": True}, "row_count": 1, "csv_path": None,
        }
        return agent

    def test_model_forwarded_for_answerable_questions(self):
        agent = self._agent()
        with patch.object(run_eval_mod, "execute_gold", return_value=([{"x": 1}], None)):
            run_eval_mod.run_eval(
                agent,
                ground_truth=[{"question_id": "q1", "question": "q",
                               "gold_sql": "SELECT 1", "difficulty": "simple"}],
                unanswerable=[],
                db_name="FYP",
                session_prefix="t",
                model="gpt-4o",
            )
        assert agent.run.call_args.kwargs["model"] == "gpt-4o"

    def test_model_forwarded_for_unanswerable_questions(self):
        agent = self._agent()
        run_eval_mod.run_eval(
            agent,
            ground_truth=[],
            unanswerable=[{"question_id": "u1", "question": "q"}],
            db_name="FYP",
            session_prefix="t",
            model="gpt-4o",
        )
        assert agent.run.call_args.kwargs["model"] == "gpt-4o"

    def test_omitted_model_forwards_none(self):
        # Unchanged default behavior: no CLI override → agent uses its own default.
        agent = self._agent()
        with patch.object(run_eval_mod, "execute_gold", return_value=([{"x": 1}], None)):
            run_eval_mod.run_eval(
                agent,
                ground_truth=[{"question_id": "q1", "question": "q",
                               "gold_sql": "SELECT 1", "difficulty": "simple"}],
                unanswerable=[],
                db_name="FYP",
                session_prefix="t",
            )
        assert agent.run.call_args.kwargs["model"] is None


# ── CLI argparse smoke tests ─────────────────────────────────────────────────────
# Each main() lazily imports + calls load_config() right after parsing. Patching
# it to raise a sentinel proves argparse accepted every flag (including --model)
# without a SystemExit — execution reached exactly one line past parsing.

class _StopAfterParse(Exception):
    pass


class TestCliAcceptsModelFlag:

    def test_bird_main_accepts_model_flag(self):
        with patch("backend.config.config_loader.load_config",
                    side_effect=_StopAfterParse):
            with pytest.raises(_StopAfterParse):
                bird.main([
                    "--dev-json", "dev.json", "--db-dir", "dbs",
                    "--model", "gpt-4o",
                ])

    def test_bird_predict_main_accepts_model_flag(self):
        with patch("backend.config.config_loader.load_config",
                    side_effect=_StopAfterParse):
            with pytest.raises(_StopAfterParse):
                bird_predict.main([
                    "--questions", "q.json", "--db-dir", "dbs",
                    "--output", "out.json", "--model", "gpt-4o",
                ])

    def test_run_eval_main_accepts_model_flag(self):
        with patch("backend.config.config_loader.load_config",
                    side_effect=_StopAfterParse):
            with pytest.raises(_StopAfterParse):
                run_eval_mod.main([
                    "--ground-truth", "gt.json", "--model", "gpt-4o",
                ])
