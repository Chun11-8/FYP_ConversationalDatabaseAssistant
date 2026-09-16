"""
Tests for backend/eval/compare_models.py (GH#121).

Offline: no LLM, no BIRD data, no subprocess actually launched. The point of
the sweep is that N models are run under IDENTICAL settings, so most of these
guard that property rather than the plumbing.

Run from repo root:
    pytest backend/tests/test_compare_models.py -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.eval.compare_models import (
    DEFAULT_MODELS,
    build_argv,
    check_same_questions,
    main,
    markdown_table,
    output_path,
    slugify,
    summarise,
)


def _args(**over) -> argparse.Namespace:
    base = dict(dev_json="dev.json", db_dir="dbs", subset=20, sample="head",
                seed=42, difficulty="moderate", db_id="student_club",
                question_timeout=None)
    base.update(over)
    return argparse.Namespace(**base)


def _row(model: str, ex: float, qids=(1, 2, 3), **rc) -> dict:
    run_config = dict(sample="head", seed=42, subset=20,
                      difficulty="moderate", db_id="student_club")
    run_config.update(rc)
    return {
        "model": model,
        "metrics": {"EX": ex, "F1": ex, "VES": ex, "total": 20,
                    "llm_calls_per_question": 2.1, "tokens_in": 1000,
                    "tokens_out": 500, "cost_usd": 0.01, "timeouts": 0},
        "run_config": run_config,
        "_qids": tuple(qids),
        "seconds": 60.0,
        "output": f"outputs/moderate/moderate_{model}.json",
    }


class TestSlugAndPaths:

    def test_the_vendor_prefix_is_kept(self):
        """Two vendors can ship the same bare name. Dropping the prefix would
        let one model's results overwrite another's in the same sweep."""
        assert slugify("qwen/qwen3.8-flash") == "qwen_qwen3.8-flash"
        assert slugify("minimax/minimax-m3") == "minimax_minimax-m3"

    def test_distinct_models_never_collide(self):
        slugs = [slugify(m) for m in DEFAULT_MODELS]
        assert len(slugs) == len(set(slugs))

    def test_a_slug_is_filesystem_safe(self):
        for model in DEFAULT_MODELS:
            assert "/" not in slugify(model)
            assert slugify(model) == slugify(model).strip("_")

    def test_output_path_groups_by_difficulty(self):
        p = output_path(Path("outputs"), "moderate", "qwen/qwen3.8-flash")
        assert p == Path("outputs/moderate_qwen_qwen3.8-flash.json")

    def test_no_difficulty_is_labelled_all_not_none(self):
        p = output_path(Path("outputs"), None, "gpt-4o-mini")
        assert p == Path("outputs/all_gpt-4o-mini.json")


class TestBuildArgv:
    """Every model must get the SAME flags — only --model and --output differ."""

    def test_only_model_and_output_differ_between_models(self):
        args = _args()
        a = build_argv("gpt-4o-mini", Path("a.json"), args, [])
        b = build_argv("qwen/qwen3.8-flash", Path("b.json"), args, [])

        def strip(argv):
            out, skip = [], False
            for tok in argv:
                if skip:
                    skip = False
                    continue
                if tok in ("--model", "--output"):
                    skip = True
                    continue
                out.append(tok)
            return out

        assert strip(a) == strip(b)

    def test_sampling_is_always_passed_explicitly(self):
        """Relying on bird.py's default would let the two CLIs drift apart and
        silently change which questions a sweep covers."""
        argv = build_argv("gpt-4o-mini", Path("o.json"), _args(sample="stratified"), [])
        assert "--sample" in argv and argv[argv.index("--sample") + 1] == "stratified"
        assert "--seed" in argv

    def test_optional_filters_are_omitted_when_unset(self):
        argv = build_argv("gpt-4o-mini", Path("o.json"),
                          _args(difficulty=None, db_id=None, subset=None), [])
        assert "--difficulty" not in argv
        assert "--db-id" not in argv
        assert "--subset" not in argv

    def test_question_timeout_is_omitted_rather_than_guessed(self):
        """Unset must mean the harness default, not a number invented here."""
        assert "--question-timeout" not in build_argv(
            "gpt-4o-mini", Path("o.json"), _args(), [])
        argv = build_argv("gpt-4o-mini", Path("o.json"),
                          _args(question_timeout=180.0), [])
        assert argv[argv.index("--question-timeout") + 1] == "180.0"

    def test_passthrough_reaches_every_run(self):
        argv = build_argv("gpt-4o-mini", Path("o.json"), _args(),
                          ["--no-refine", "--trust-eval"])
        assert argv[-2:] == ["--no-refine", "--trust-eval"]


class TestComparabilityGuard:
    """
    An EX column only means something when every row answered one question set.
    """

    def test_identical_runs_produce_no_warning(self):
        assert check_same_questions([_row("a", 70.0), _row("b", 80.0)]) == []

    def test_different_question_sets_are_flagged(self):
        rows = [_row("a", 70.0, qids=(1, 2, 3)), _row("b", 90.0, qids=(4, 5, 6))]
        warnings = check_same_questions(rows)
        assert any("did NOT answer the same questions" in w for w in warnings)
        assert any("b" in w for w in warnings)

    def test_a_mismatched_seed_is_flagged_even_with_equal_ids(self):
        """The seed can differ while a tiny sample coincides; the table would
        still be a comparison of two different sampling decisions."""
        rows = [_row("a", 70.0), _row("b", 80.0, seed=7)]
        assert any("run_config.seed differs" in w
                   for w in check_same_questions(rows))

    def test_a_mismatched_subset_or_difficulty_is_flagged(self):
        assert any("run_config.subset differs" in w for w in
                   check_same_questions([_row("a", 1.0), _row("b", 2.0, subset=50)]))
        assert any("run_config.difficulty differs" in w for w in
                   check_same_questions([_row("a", 1.0),
                                         _row("b", 2.0, difficulty="simple")]))


class TestRendering:

    def test_the_table_is_ranked_by_ex(self):
        text = summarise([_row("low", 40.0), _row("high", 90.0), _row("mid", 70.0)])
        body = text.splitlines()[2:]
        assert [ln.split()[0] for ln in body] == ["high", "mid", "low"]

    def test_markdown_ranks_and_escapes_model_ids(self):
        md = markdown_table([_row("qwen/qwen3.8-flash", 80.0), _row("gpt-4o-mini", 90.0)])
        assert md.index("gpt-4o-mini") < md.index("qwen/qwen3.8-flash")
        assert "`qwen/qwen3.8-flash`" in md

    def test_an_empty_sweep_does_not_crash_the_renderer(self):
        assert summarise([]).splitlines()[0].startswith("Model")


class TestSweepBehaviour:

    def _payload(self, ex: float) -> dict:
        return {"run_config": {"sample": "head", "seed": 42, "subset": 2,
                               "difficulty": "moderate", "db_id": "student_club"},
                "metrics": {"EX": ex, "F1": ex, "VES": ex, "total": 2,
                            "cost_usd": 0.01},
                "questions": [{"question_id": 1}, {"question_id": 2}]}

    def test_a_failing_model_does_not_abort_the_sweep(self, tmp_path: Path, capsys):
        """Six good runs are worth more than an aborted sweep — and by the time
        one model fails, the earlier ones have already been paid for."""
        calls = []

        def fake_run(cmd, *a, **k):
            model = cmd[cmd.index("--model") + 1]
            out = Path(cmd[cmd.index("--output") + 1])
            calls.append(model)
            if model == "bad":
                return type("R", (), {"returncode": 1})()
            out.write_text(json.dumps(self._payload(75.0)), encoding="utf-8")
            return type("R", (), {"returncode": 0})()

        with patch("backend.eval.compare_models.subprocess.run", side_effect=fake_run):
            rc = main(["--models", "good1", "bad", "good2",
                       "--out-dir", str(tmp_path), "--difficulty", "moderate"])

        assert calls == ["good1", "bad", "good2"], "the sweep stopped early"
        assert rc == 1, "a failed model must not report success"
        out = capsys.readouterr().out
        assert "FAILED : bad" in out
        assert "good1" in out and "good2" in out

    def test_exit_zero_only_when_every_model_succeeded(self, tmp_path: Path):
        def fake_run(cmd, *a, **k):
            Path(cmd[cmd.index("--output") + 1]).write_text(
                json.dumps(self._payload(80.0)), encoding="utf-8")
            return type("R", (), {"returncode": 0})()

        with patch("backend.eval.compare_models.subprocess.run", side_effect=fake_run):
            assert main(["--models", "a", "b", "--out-dir", str(tmp_path)]) == 0

    def test_a_missing_output_file_counts_as_failure(self, tmp_path: Path):
        """Exit code 0 is not proof of a result — a run can succeed and still
        write nothing if the path was wrong."""
        with patch("backend.eval.compare_models.subprocess.run",
                   return_value=type("R", (), {"returncode": 0})()):
            assert main(["--models", "a", "--out-dir", str(tmp_path)]) == 1

    def test_skip_existing_resumes_without_re_running(self, tmp_path: Path):
        done = tmp_path / "moderate_a.json"
        done.write_text(json.dumps(self._payload(88.0)), encoding="utf-8")
        ran = []

        def fake_run(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            Path(cmd[cmd.index("--output") + 1]).write_text(
                json.dumps(self._payload(60.0)), encoding="utf-8")
            return type("R", (), {"returncode": 0})()

        with patch("backend.eval.compare_models.subprocess.run", side_effect=fake_run):
            main(["--models", "a", "b", "--difficulty", "moderate",
                  "--out-dir", str(tmp_path), "--skip-existing"])

        assert ran == ["b"], "a finished model was paid for twice"
        summary = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
        assert {m["model"] for m in summary["models"]} == {"a", "b"}

    def test_dry_run_spends_nothing(self, tmp_path: Path, capsys):
        with patch("backend.eval.compare_models.subprocess.run") as sp:
            rc = main(["--models", "a", "b", "--out-dir", str(tmp_path), "--dry-run"])
        assert rc == 0
        sp.assert_not_called()
        assert "backend.eval.bird" in capsys.readouterr().out

    def test_duplicate_models_are_rejected(self, tmp_path: Path):
        """The second run would overwrite the first's output and the table
        would show one model twice with one set of numbers."""
        with pytest.raises(SystemExit):
            main(["--models", "a", "a", "--out-dir", str(tmp_path)])

    def test_the_sweep_settings_are_written_to_the_summary(self, tmp_path: Path):
        def fake_run(cmd, *a, **k):
            Path(cmd[cmd.index("--output") + 1]).write_text(
                json.dumps(self._payload(80.0)), encoding="utf-8")
            return type("R", (), {"returncode": 0})()

        with patch("backend.eval.compare_models.subprocess.run", side_effect=fake_run):
            main(["--models", "a", "--out-dir", str(tmp_path),
                  "--subset", "33", "--sample", "stratified", "--seed", "9"])

        summary = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
        assert summary["sweep"]["subset"] == 33
        assert summary["sweep"]["sample"] == "stratified"
        assert summary["sweep"]["seed"] == 9
        assert (tmp_path / "comparison.md").exists()

    def test_the_default_model_list_is_routable(self):
        """Every default id must be an EXACT entry in llm.selectable_models —
        anything else carries no provider tag, reaches OpenAI and 400s
        (GH#117), which would fail all seven runs of a default sweep."""
        import yaml
        cfg = yaml.safe_load(
            open("backend/config/config.yaml", encoding="utf-8"))
        known = {m if isinstance(m, str) else m["id"]
                 for m in cfg["llm"]["selectable_models"]}
        assert set(DEFAULT_MODELS) <= known, set(DEFAULT_MODELS) - known
