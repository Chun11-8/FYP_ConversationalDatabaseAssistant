"""
Tests for backend/eval/run_all_models.py (GH#124).

Offline: no subprocess is really launched, no LLM, no BIRD data.

The sweep's whole value is that only --model and --output change between runs.
Most of these guard that, plus the confirmation gate -- an unattended sweep is
seven paid benchmark runs.

Run from repo root:
    pytest backend/tests/test_run_all_models.py -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.eval.run_all_models import (
    MODELS, build_command, choose_mode, is_complete, main, resume_command,
)


def _args(**over) -> argparse.Namespace:
    base = dict(difficulty="moderate", subset=40, sample="stratified", seed=42,
                db_id=None, question_timeout=None,
                dev_json="dev.json", db_dir="dbs")
    base.update(over)
    return argparse.Namespace(**base)


class TestModelTable:

    def test_short_names_match_the_files_already_in_outputs(self):
        """The short names are how a v2 run sits beside its predecessor in
        outputs/simple/. Renaming one orphans the comparison."""
        assert set(MODELS) == {
            "gpt4oMini", "gpt4.1Mini", "gemini3.7Flash", "llama4",
            "minimaxM3", "qwen3.8Flash", "deepseekV4Flash"}

    def test_every_id_is_in_selectable_models(self):
        """An id that is not an EXACT entry carries no provider tag, falls back
        to OpenAI and 400s -- which would fail all seven runs."""
        import yaml
        cfg = yaml.safe_load(open("backend/config/config.yaml", encoding="utf-8"))
        known = {m if isinstance(m, str) else m["id"]
                 for m in cfg["llm"]["selectable_models"]}
        missing = set(MODELS.values()) - known
        assert not missing, missing

    def test_ids_are_distinct(self):
        assert len(set(MODELS.values())) == len(MODELS)


class TestBuildCommand:

    def test_output_path_follows_the_repo_convention(self):
        cmd = build_command("qwen3.8Flash", _args(difficulty="simple"), [])
        out = cmd[cmd.index("--output") + 1]
        assert Path(out) == Path("outputs/simple/simplev2_qwen3.8Flash.json")

    def test_the_difficulty_drives_folder_prefix_and_filter(self):
        cmd = build_command("llama4", _args(difficulty="challenging"), [])
        out = cmd[cmd.index("--output") + 1]
        assert "challenging/challengingv2_llama4.json" in out.replace("\\", "/")
        assert cmd[cmd.index("--difficulty") + 1] == "challenging"

    def test_only_model_and_output_differ_between_models(self):
        """If anything else drifts, the scores are not comparable and nothing
        in the results files would reveal it."""
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

        args = _args()
        base = strip(build_command("gpt4oMini", args, []))
        for short in MODELS:
            assert strip(build_command(short, args, [])) == base

    def test_sampling_and_seed_are_always_explicit(self):
        """Relying on bird.py's defaults would let the two CLIs drift and
        silently change which questions the sweep covers."""
        cmd = build_command("gpt4oMini", _args(), [])
        assert cmd[cmd.index("--sample") + 1] == "stratified"
        assert cmd[cmd.index("--seed") + 1] == "42"

    def test_optional_flags_are_omitted_rather_than_guessed(self):
        cmd = build_command("gpt4oMini", _args(), [])
        assert "--db-id" not in cmd
        assert "--question-timeout" not in cmd

    def test_question_timeout_is_passed_when_given(self):
        cmd = build_command("gpt4oMini", _args(question_timeout=300.0), [])
        assert cmd[cmd.index("--question-timeout") + 1] == "300.0"

    def test_passthrough_reaches_every_run(self):
        cmd = build_command("gpt4oMini", _args(), ["--no-refine", "--trust-eval"])
        assert cmd[-2:] == ["--no-refine", "--trust-eval"]


class TestSweepBehaviour:

    @staticmethod
    def _ok(cmd, *a, **k):
        Path(cmd[cmd.index("--output") + 1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[cmd.index("--output") + 1]).write_text("{}", encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    def test_pause_mode_gates_BETWEEN_models_not_before_the_first(
            self, tmp_path, monkeypatch):
        """The gap that needs approval is the one between runs. Prompting
        before the first asks permission for nothing, and prompting after the
        last asks about a run that will never happen."""
        monkeypatch.chdir(tmp_path)
        prompts, ran = [], []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._ok(cmd)

        def fake_input(prompt):
            prompts.append(prompt)
            return ""

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", side_effect=fake_input):
            main(["--difficulty", "simple", "--mode", "pause"])
        assert len(ran) == len(MODELS)
        assert len(prompts) == len(MODELS) - 1
        assert all("Press Enter" in p for p in prompts)

    def test_auto_mode_never_prompts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", side_effect=AssertionError("must not prompt")):
            main(["--difficulty", "simple", "--mode", "auto"])
        assert len(ran) == len(MODELS)

    def test_declining_between_models_stops_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", side_effect=["", "q"]):
            main(["--difficulty", "simple", "--mode", "pause"])
        assert len(ran) == 2, "first ran, Enter allowed the second, q stopped the rest"

    def test_eof_between_models_stops_rather_than_running_unattended(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", side_effect=EOFError):
            main(["--difficulty", "simple", "--mode", "pause"])
        assert len(ran) == 1


    def test_a_failing_model_does_not_abort_the_rest(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        ran = []

        def flaky(cmd, *a, **k):
            model = cmd[cmd.index("--model") + 1]
            ran.append(model)
            if model == "gpt-4.1-mini":
                return type("R", (), {"returncode": 1})()
            return self._ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=flaky), \
             patch("builtins.input", return_value=""):
            rc = main(["--difficulty", "simple", "--mode", "auto"])
        assert len(ran) == len(MODELS), "the sweep stopped at the failure"
        assert rc == 1
        assert "Failed: gpt4.1Mini" in capsys.readouterr().out

    def test_only_restricts_the_run_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto",
              "--only", "qwen3.8Flash", "gpt4oMini"])
        assert ran == ["qwen/qwen3.8-flash", "gpt-4o-mini"]

    def test_dry_run_spends_nothing_and_asks_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        with patch("backend.eval.run_all_models.subprocess.run") as sp, \
             patch("builtins.input", side_effect=AssertionError("must not prompt")):
            rc = main(["--difficulty", "moderate", "--dry-run"])
        assert rc == 0
        sp.assert_not_called()
        assert "backend.eval.bird" in capsys.readouterr().out

    def test_an_existing_output_file_is_flagged_before_overwriting(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        done = tmp_path / "outputs" / "simple"
        done.mkdir(parents=True)
        (done / "simplev2_gpt4oMini.json").write_text("{}", encoding="utf-8")
        with patch("backend.eval.run_all_models.subprocess.run", side_effect=self._ok), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto", "--only", "gpt4oMini"])
        assert "OVERWRITTEN" in capsys.readouterr().out

    def test_console_output_is_ascii_only(self, tmp_path, monkeypatch, capsys):
        """A non-ASCII character in a console line crashes outright on a
        Windows cp1252 terminal."""
        monkeypatch.chdir(tmp_path)
        with patch("backend.eval.run_all_models.subprocess.run", side_effect=self._ok), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto", "--only", "gpt4oMini"])
        out = capsys.readouterr().out
        out.encode("cp1252")          # raises if anything is un-encodable
        assert out.isascii()

    def test_an_unknown_short_name_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--difficulty", "simple", "--mode", "auto", "--only", "gpt5mini"])

    def test_difficulty_is_required(self):
        with pytest.raises(SystemExit):
            main(["--subset", "10"])



class TestModeSelection:
    """Auto and pause are not interchangeable -- auto spends money unattended,
    so the choice is made explicitly, never inferred."""

    def test_an_explicit_mode_is_not_second_guessed(self):
        assert choose_mode("auto") == "auto"
        assert choose_mode("pause") == "pause"

    def test_it_asks_when_the_mode_was_not_given(self):
        with patch("backend.eval.run_all_models.sys.stdin") as stdin, \
             patch("builtins.input", return_value="a"):
            stdin.isatty.return_value = True
            assert choose_mode(None) == "auto"

    def test_bare_enter_chooses_the_safer_option(self):
        with patch("backend.eval.run_all_models.sys.stdin") as stdin, \
             patch("builtins.input", return_value=""):
            stdin.isatty.return_value = True
            assert choose_mode(None) == "pause"

    def test_an_unrecognised_answer_is_re_asked(self):
        with patch("backend.eval.run_all_models.sys.stdin") as stdin, \
             patch("builtins.input", side_effect=["yes", "maybe", "p"]):
            stdin.isatty.return_value = True
            assert choose_mode(None) == "pause"

    def test_no_terminal_means_no_guess(self, tmp_path, monkeypatch):
        """With a piped stdin there is nothing to ask and no safe default;
        silently picking auto is the mistake this prevents."""
        with patch("backend.eval.run_all_models.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            assert choose_mode(None) is None

        monkeypatch.chdir(tmp_path)
        with patch("backend.eval.run_all_models.subprocess.run") as sp, \
             patch("backend.eval.run_all_models.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            assert main(["--difficulty", "simple"]) == 1
        sp.assert_not_called()


class TestResume:
    """
    A sweep is seven long, paid runs. When one dies partway -- a crash, a
    dropped connection, Ctrl-C -- the next thing needed is to continue without
    re-paying for the models that finished.
    """

    @staticmethod
    def _complete(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metrics": {"EX": 80.0}, "questions": []}),
                        encoding="utf-8")

    @staticmethod
    def _run_ok(cmd, *a, **k):
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metrics": {"EX": 70.0}}), encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    def test_a_finished_run_is_recognised(self, tmp_path):
        f = tmp_path / "done.json"
        self._complete(f)
        assert is_complete(f) is True

    def test_a_truncated_file_is_NOT_treated_as_done(self, tmp_path):
        """The whole point. A run killed partway leaves a half-written file;
        counting it as complete would silently drop that model from the
        comparison with nothing to show it happened."""
        f = tmp_path / "crash.json"
        f.write_text('{"metrics": {"EX"', encoding="utf-8")
        assert is_complete(f) is False

    def test_an_empty_or_missing_file_is_not_done(self, tmp_path):
        missing = tmp_path / "nope.json"
        assert is_complete(missing) is False
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        assert is_complete(empty) is False

    def test_valid_json_without_metrics_is_not_done(self, tmp_path):
        f = tmp_path / "partial.json"
        f.write_text(json.dumps({"run_config": {"model": "x"}}), encoding="utf-8")
        assert is_complete(f) is False

    def test_resume_skips_complete_and_reruns_the_crashed_one(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "outputs" / "simple"
        for m in ("gpt4oMini", "gpt4.1Mini"):
            self._complete(d / f"simplev2_{m}.json")
        (d / "simplev2_gemini3.7Flash.json").write_text('{"metri', encoding="utf-8")

        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._run_ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto", "--resume"])

        assert MODELS["gpt4oMini"] not in ran
        assert MODELS["gpt4.1Mini"] not in ran
        assert MODELS["gemini3.7Flash"] in ran, "a truncated run was not redone"
        assert len(ran) == len(MODELS) - 2

    def test_without_resume_everything_runs_again(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._complete(tmp_path / "outputs" / "simple" / "simplev2_gpt4oMini.json")
        ran = []

        def record(cmd, *a, **k):
            ran.append(cmd[cmd.index("--model") + 1])
            return self._run_ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=record), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto"])
        assert len(ran) == len(MODELS)

    def test_resume_with_nothing_left_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        for m in MODELS:
            self._complete(tmp_path / "outputs" / "simple" / f"simplev2_{m}.json")
        with patch("backend.eval.run_all_models.subprocess.run") as sp, \
             patch("builtins.input", side_effect=AssertionError("must not prompt")):
            rc = main(["--difficulty", "simple", "--mode", "auto", "--resume"])
        assert rc == 0
        sp.assert_not_called()
        assert "Nothing left to run" in capsys.readouterr().out

    def test_the_summary_prints_a_resume_command_when_work_remains(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        with patch("backend.eval.run_all_models.subprocess.run", side_effect=self._run_ok), \
             patch("builtins.input", side_effect=["", "q"]):
            main(["--difficulty", "moderate", "--mode", "pause"])
        out = capsys.readouterr().out
        assert "Still to run" in out
        assert "--resume" in out
        assert "run_all_models" in out

    def test_a_finished_sweep_prints_no_resume_command(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        with patch("backend.eval.run_all_models.subprocess.run", side_effect=self._run_ok), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto"])
        assert "Still to run" not in capsys.readouterr().out

    def test_a_failed_model_stays_in_the_outstanding_list(
            self, tmp_path, monkeypatch, capsys):
        """Exit code 0 is not proof of a result, and a model that errored must
        not be skipped by the next --resume."""
        monkeypatch.chdir(tmp_path)

        def flaky(cmd, *a, **k):
            if "llama" in cmd[cmd.index("--model") + 1]:
                return type("R", (), {"returncode": 1})()
            return self._run_ok(cmd)

        with patch("backend.eval.run_all_models.subprocess.run", side_effect=flaky), \
             patch("builtins.input", return_value=""):
            main(["--difficulty", "simple", "--mode", "auto"])
        out = capsys.readouterr().out
        assert "Failed: llama4" in out
        assert "llama4" in out.split("Still to run")[1]


class TestResumeCommand:

    def test_it_carries_the_run_settings_forward(self):
        args = _args(difficulty="simple", subset=40, sample="stratified",
                     seed=7, question_timeout=300.0)
        args.mode, args.only, args.resume = "pause", None, False
        cmd = resume_command(args, [])
        assert "--resume" in cmd
        for flag, value in (("--difficulty", "simple"), ("--subset", "40"),
                            ("--sample", "stratified"), ("--seed", "7"),
                            ("--question-timeout", "300.0"),
                            ("--mode", "pause")):
            assert cmd[cmd.index(flag) + 1] == value

    def test_it_preserves_only_and_passthrough(self):
        args = _args()
        args.mode, args.only, args.resume = None, ["qwen3.8Flash"], False
        cmd = resume_command(args, ["--no-refine"])
        assert cmd[cmd.index("--only") + 1] == "qwen3.8Flash"
        assert cmd[-2:] == ["--", "--no-refine"]

    def test_it_omits_unchanged_dataset_paths(self):
        """A resume line cluttered with defaults is one people stop reading."""
        from backend.eval.run_all_models import DB_DIR, DEV_JSON
        args = _args(dev_json=DEV_JSON, db_dir=DB_DIR)
        args.mode, args.only, args.resume = None, None, False
        cmd = resume_command(args, [])
        assert "--dev-json" not in cmd and "--db-dir" not in cmd

    def test_it_KEEPS_custom_dataset_paths(self):
        """Dropping a non-default path would make the resume line point at a
        dataset the original run never used."""
        args = _args(dev_json="my/dev.json", db_dir="my/dbs")
        args.mode, args.only, args.resume = None, None, False
        cmd = resume_command(args, [])
        assert cmd[cmd.index("--dev-json") + 1] == "my/dev.json"
        assert cmd[cmd.index("--db-dir") + 1] == "my/dbs"
