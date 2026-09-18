"""
run_all_models.py -- run the BIRD benchmark over the seven models, one at a
time, either straight through or pausing between them (GH#124).

Replaces typing the same `python -m backend.eval.bird` line seven times with
only --model and --output changed -- the edit that is easy to get wrong, and
whose mistakes (a different --subset or --sample on one run) are invisible in
the results afterwards.

Every run is launched as its own subprocess. A model that aborts -- a bad id
exits before the loop -- does not take the rest of the sweep with it, no state
carries between models, and the exact command for any single model is printed
so it can be re-run on its own.

Output lands at outputs/<difficulty>/<difficulty>v2_<ShortName>.json, matching
the naming already in the repo.

Run from the repo root:

    python -m backend.eval.run_all_models --difficulty moderate --subset 40
    #   -> asks once whether to run in auto or pause mode, then goes

    # decide up front instead of being asked
    python -m backend.eval.run_all_models --difficulty moderate --mode auto
    python -m backend.eval.run_all_models --difficulty moderate --mode pause

    # a slow reasoning model needs more head-room per question
    python -m backend.eval.run_all_models --difficulty simple --subset 40 \
        --question-timeout 300

    # just two of them, and pass extra flags straight to bird.py
    python -m backend.eval.run_all_models --difficulty simple \
        --only qwen3.8Flash gpt4oMini -- --no-refine

Output is deliberately plain ASCII: a non-ASCII character in a console line
crashes outright on a Windows cp1252 terminal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Short name -> model id. The short names are the ones already used by the
# files in outputs/simple/, so a v2 run sits beside its predecessor; the ids
# must match llm.selectable_models EXACTLY or the run aborts before the loop
# (an untagged id falls back to OpenAI and 400s). Verify a list with:
#     python -m backend.config.check_models
MODELS: Dict[str, str] = {
    "gpt4oMini":       "gpt-4o-mini",
    "gpt4.1Mini":      "gpt-4.1-mini",
    "gemini3.7Flash":  "google/gemini-3.7-flash",
    "llama4":          "meta-llama/llama-4-maverick",
    "minimaxM3":       "minimax/minimax-m3",
    "qwen3.8Flash":    "qwen/qwen3.8-flash",
    "deepseekV4Flash": "deepseek/deepseek-v4-flash",
}

DEV_JSON = "dataset/minidev/MINIDEV/mini_dev_postgresql.json"
DB_DIR = "dataset/minidev/MINIDEV/dev_databases"


def is_complete(out_file: Path) -> bool:
    """
    Did this model already finish, or is the file the wreckage of a crash?

    Existence alone is the wrong test. A run killed partway -- Ctrl-C, a lost
    connection, the machine sleeping -- can leave a truncated or empty file,
    and treating that as done would silently drop a model from the comparison
    with nothing to show it happened. A finished run always carries a metrics
    block with an EX figure, so that is what is checked.
    """
    try:
        with open(out_file, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "EX" in (payload.get("metrics") or {})


def build_command(short: str, args: argparse.Namespace,
                  passthrough: List[str]) -> List[str]:
    """The bird.py command for one model. Only --model and --output differ
    between models -- everything else is shared, which is the whole point."""
    out = Path("outputs") / args.difficulty / f"{args.difficulty}v2_{short}.json"
    cmd = [
        sys.executable, "-m", "backend.eval.bird",
        "--dev-json", args.dev_json,
        "--engine", args.engine,
        "--model", MODELS[short],
        "--output", str(out),
        "--difficulty", args.difficulty,
        "--subset", str(args.subset),
        "--sample", args.sample,
        "--seed", str(args.seed),
    ]
    # --db-dir is the SQLite layout; PostgreSQL reads one database instead.
    if args.engine == "sqlite":
        cmd += ["--db-dir", args.db_dir]
    else:
        if args.pg_dbname:
            cmd += ["--pg-dbname", args.pg_dbname]
        if args.pg_schema != "public":
            cmd += ["--pg-schema", args.pg_schema]
    if args.db_id:
        cmd += ["--db-id", args.db_id]
    if args.question_timeout is not None:
        cmd += ["--question-timeout", str(args.question_timeout)]
    return cmd + passthrough


def resume_command(args: argparse.Namespace,
                   passthrough: List[str]) -> List[str]:
    """This same sweep, with --resume added, so it skips what is finished."""
    cmd = [sys.executable, "-m", "backend.eval.run_all_models",
           "--difficulty", args.difficulty,
           "--subset", str(args.subset),
           "--sample", args.sample,
           "--seed", str(args.seed),
           "--resume"]
    if args.mode:
        cmd += ["--mode", args.mode]
    if args.db_id:
        cmd += ["--db-id", args.db_id]
    if args.question_timeout is not None:
        cmd += ["--question-timeout", str(args.question_timeout)]
    if args.dev_json != DEV_JSON:
        cmd += ["--dev-json", args.dev_json]
    if args.db_dir != DB_DIR:
        cmd += ["--db-dir", args.db_dir]
    if args.only:
        cmd += ["--only"] + list(args.only)
    if passthrough:
        cmd += ["--"] + passthrough
    return cmd


def confirm(prompt: str) -> bool:
    """Wait for Enter between models. Ctrl-C or EOF (a piped stdin) stops the
    sweep rather than silently running the remaining paid benchmarks."""
    try:
        reply = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")
        return False
    if reply in ("q", "quit", "n", "no"):
        print("Stopped.")
        return False
    return True


def choose_mode(explicit: Optional[str]) -> Optional[str]:
    """
    "auto" or "pause", asked once at the start when --mode was not given.

    Asked rather than defaulted because the two are not interchangeable: auto
    runs every remaining benchmark unattended, which costs real money. With no
    terminal to ask (a piped or redirected stdin) there is no safe default, so
    the sweep stops and says to pass --mode -- silently picking auto there is
    exactly the mistake worth preventing.
    """
    if explicit:
        return explicit
    if not sys.stdin.isatty():
        print("\n  stdin is not a terminal, so the mode cannot be asked for.")
        print("  Re-run with --mode auto or --mode pause.")
        return None
    print("\n  How should the models be run?")
    print("    [a] auto  -- run all of them straight through, no prompts")
    print("    [p] pause -- stop for Enter after each model completes")
    while True:
        try:
            reply = input("  Choose a/p (q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return None
        if reply in ("a", "auto"):
            return "auto"
        if reply in ("p", "pause", ""):
            return "pause"
        if reply in ("q", "quit"):
            print("Stopped.")
            return None
        print("  Please answer a, p or q.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the BIRD benchmark over the seven models, pausing "
                    "for Enter before each.",
        epilog="Anything after `--` is passed to every bird.py run unchanged.")
    parser.add_argument("--difficulty", required=True,
                        choices=("simple", "moderate", "challenging"),
                        help="Tier to evaluate; also names the output folder "
                             "and file prefix.")
    parser.add_argument("--subset", type=int, default=40)
    parser.add_argument("--sample", default="stratified",
                        choices=("head", "random", "stratified"),
                        help="How --subset picks questions. Default stratified "
                             "-- dev.json is grouped by database, so 'head' "
                             "returns one database's questions.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Keep this fixed across models, or they answer "
                             "different questions and the scores are not "
                             "comparable (default: 42)")
    parser.add_argument("--db-id", default=None,
                        help="Restrict to one database (optional)")
    parser.add_argument("--question-timeout", type=float, default=None,
                        help="Seconds per question before it is abandoned and "
                             "scored 0. Raise it for a slow reasoning model; "
                             "omit for bird.py's default.")
    parser.add_argument("--engine", choices=("postgres", "sqlite"),
                        default="postgres",
                        help="Where the SQL runs. Default postgres, matching "
                             "--dev-json's mini_dev_postgresql.json: the gold "
                             "dialect and the engine must agree or every gold "
                             "query errors.")
    parser.add_argument("--pg-dbname", default=None,
                        help="PostgreSQL database holding the BIRD tables")
    parser.add_argument("--pg-schema", default="public")
    parser.add_argument("--dev-json", default=DEV_JSON)
    parser.add_argument("--db-dir", default=DB_DIR)
    parser.add_argument("--only", nargs="+", metavar="SHORTNAME",
                        choices=list(MODELS),
                        help=f"Run only these: {', '.join(MODELS)}")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models whose output file is already a "
                             "COMPLETE run, so a sweep interrupted partway "
                             "picks up where it stopped instead of paying for "
                             "the finished models again. A truncated file from "
                             "a crash is not treated as done.")
    parser.add_argument("--mode", choices=("auto", "pause"), default=None,
                        help="auto: run every model straight through. "
                             "pause: wait for Enter after each model completes. "
                             "Omit and you are asked once at the start.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands and exit. Costs nothing.")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    requested = args.only or list(MODELS)
    out_dir = Path("outputs") / args.difficulty
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(short: str) -> Path:
        return out_dir / f"{args.difficulty}v2_{short}.json"

    skipped: List[str] = []
    if args.resume:
        skipped = [m for m in requested if is_complete(out_path(m))]
        shorts = [m for m in requested if m not in skipped]
    else:
        shorts = list(requested)

    print()
    print("=" * 66)
    print(f"  BIRD sweep -- {len(shorts)} model(s), tier '{args.difficulty}'")
    print(f"  subset={args.subset}  sample={args.sample}  seed={args.seed}"
          + (f"  db_id={args.db_id}" if args.db_id else ""))
    print(f"  output -> {out_dir}/{args.difficulty}v2_<model>.json")
    if passthrough:
        print(f"  extra  -> {' '.join(passthrough)}")
    if skipped:
        print(f"  resume -> skipping {len(skipped)} already complete: "
              f"{', '.join(skipped)}")
    print("=" * 66)

    if not shorts:
        print("\n  Nothing left to run -- every requested model is complete.")
        print(f"  Results in {out_dir}/\n")
        return 0

    if args.dry_run:
        for short in shorts:
            print("\n  " + " ".join(build_command(short, args, passthrough)))
        return 0

    mode = choose_mode(args.mode)
    if mode is None:
        return 1
    print(f"\n  Mode: {mode}"
          + ("  (running straight through)" if mode == "auto"
             else "  (stops for Enter after each model)"))

    failed: List[str] = []
    completed_count = 0
    interrupted = False
    for idx, short in enumerate(shorts, start=1):
        cmd = build_command(short, args, passthrough)
        out_file = cmd[cmd.index("--output") + 1]
        print()
        print("-" * 66)
        print(f"[{idx}/{len(shorts)}] {short}  ({MODELS[short]})")
        print(f"        -> {out_file}")
        if Path(out_file).exists():
            print("        NOTE: this file already exists and will be OVERWRITTEN.")
        print(f"        {' '.join(cmd)}")
        print()

        # Output is NOT captured: a run takes many minutes, and watching the
        # per-question lines is how a model failing every question is noticed
        # now rather than at the end.
        try:
            completed = subprocess.run(cmd)
        except KeyboardInterrupt:
            # Ctrl-C reaches the child first; it dies and we land here. Report
            # it as a stop rather than a traceback, and fall through to the
            # summary so the resume line is still printed.
            print(f"\n\n  INTERRUPTED during {short}.")
            interrupted = True
            break
        completed_count += 1
        if completed.returncode == 0:
            print(f"\n  DONE {short} -> {out_file}")
        else:
            print(f"\n  FAILED {short} (exit code {completed.returncode})")
            failed.append(short)

        # The gate sits BETWEEN models: after one finishes, before the next
        # starts. No prompt before the first (nothing has happened yet) and
        # none after the last (there is nothing left to approve).
        is_last = idx == len(shorts)
        if mode == "pause" and not is_last:
            remaining = len(shorts) - idx
            if not confirm(f"\n  Press Enter for {shorts[idx]} "
                           f"({remaining} left, q to stop): "):
                print(f"\n  Stopped after {short}. "
                      f"{completed_count} of {len(shorts)} model(s) run.")
                break

    print()
    print("=" * 66)
    print(f"  Ran {completed_count} of {len(shorts)} model(s).")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  No failures.")
    print(f"  Results in {out_dir}/")

    # Whatever stopped the sweep -- a crash, Ctrl-C, q, a failing model -- the
    # next thing anyone needs is the command that picks up the rest without
    # paying for the finished ones again. Print it here rather than leaving it
    # to be reconstructed from which files happen to exist.
    outstanding = [m for m in requested if not is_complete(out_path(m))]
    if outstanding:
        print()
        print(f"  Still to run ({len(outstanding)}): {', '.join(outstanding)}")
        print("  Resume with:")
        print(f"    {' '.join(resume_command(args, passthrough))}")
    print("=" * 66)
    print()
    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
