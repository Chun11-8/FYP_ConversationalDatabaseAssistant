"""
compare_models.py — run the BIRD benchmark over several models in one command
(GH#121).

Comparing N models means running the harness N times with identical settings
and only the model changing. Doing that by hand invites the failure this exists
to prevent: one run at a different --subset, --sample or --difficulty, and the
comparison table silently compares different question sets.

Each model runs as a SEPARATE `python -m backend.eval.bird` subprocess. In-
process looping would be faster to start, but a model that aborts (a bad id
exits before the loop, GH#117) or dies mid-run would take the whole sweep with
it — and after six paid runs that is the expensive failure. Isolation also
means no module-level state carries between models, and the exact command for
any single model is printed and re-runnable on its own.

Run from repo root:

    python -m backend.eval.compare_models \\
        --dev-json dataset/mini_dev_sqlite.json \\
        --db-dir   dataset/dev_databases \\
        --difficulty moderate --db-id student_club \\
        --subset 20 --sample stratified

    # every model, cross-domain subset, resume what is already done
    python -m backend.eval.compare_models --subset 50 --sample stratified \\
        --skip-existing

Anything after `--` is passed through to each bird run unchanged:

    python -m backend.eval.compare_models --subset 20 -- --no-refine --trust-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The seven models benchmarked in outputs/{simple,moderate,challenging}/.
# Ids must match `llm.selectable_models` in config.yaml EXACTLY — an id that is
# not an exact entry carries no provider tag, falls back to OpenAI, and fails
# with "invalid model ID" (GH#117). `check_models` verifies the whole list.
DEFAULT_MODELS: Tuple[str, ...] = (
    "gpt-4o-mini",                    # OpenAI direct — the reasoning baseline
    "gpt-4.1-mini",                   # OpenAI direct
    "google/gemini-3.7-flash",
    "meta-llama/llama-4-maverick",
    "minimax/minimax-m3",
    "qwen/qwen3.8-flash",
    "deepseek/deepseek-v4-flash",
)


def slugify(model: str) -> str:
    """A filesystem-safe, collision-free stem for a model id.

    The vendor prefix is KEPT (`qwen_qwen3.8-flash`, not `qwen3.8-flash`):
    two vendors can ship the same bare name, and a comparison table that
    silently overwrites one model's results with another's is worse than an
    ugly filename.
    """
    return re.sub(r"[^a-z0-9.\-]+", "_", model.lower()).strip("_")


def output_path(out_dir: Path, difficulty: Optional[str], model: str) -> Path:
    prefix = (difficulty or "all").lower()
    return out_dir / f"{prefix}_{slugify(model)}.json"


def build_argv(model: str, out_file: Path, args: argparse.Namespace,
               passthrough: List[str]) -> List[str]:
    """The exact `python -m backend.eval.bird` command for one model."""
    argv = [
        sys.executable, "-m", "backend.eval.bird",
        "--dev-json", args.dev_json,
        "--db-dir", args.db_dir,
        "--model", model,
        "--output", str(out_file),
        "--sample", args.sample,
        "--seed", str(args.seed),
    ]
    if args.subset:
        argv += ["--subset", str(args.subset)]
    if args.difficulty:
        argv += ["--difficulty", args.difficulty]
    if args.db_id:
        argv += ["--db-id", args.db_id]
    if args.question_timeout is not None:
        argv += ["--question-timeout", str(args.question_timeout)]
    return argv + passthrough


def read_result(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def question_ids(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(q.get("question_id") for q in payload.get("questions", []))


def check_same_questions(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Warnings if the models were not asked the SAME questions.

    An EX column only means something when every row answered one question set.
    A stray --seed or --sample between runs produces a table that looks
    comparable and is not, so this is checked rather than assumed.
    """
    warnings: List[str] = []
    sets = {r["model"]: r["_qids"] for r in rows if r.get("_qids")}
    if len(set(sets.values())) > 1:
        reference = next(iter(sets.values()))
        odd = [m for m, ids in sets.items() if ids != reference]
        warnings.append(
            "models did NOT answer the same questions — EX is not comparable "
            f"across this table. Differing: {', '.join(sorted(odd))}")
    for key in ("sample", "seed", "subset", "difficulty", "db_id"):
        values = {r["model"]: r["run_config"].get(key) for r in rows
                  if r.get("run_config")}
        if len(set(map(repr, values.values()))) > 1:
            warnings.append(
                f"run_config.{key} differs between models: "
                + ", ".join(f"{m}={v!r}" for m, v in sorted(values.items())))
    return warnings


def summarise(rows: List[Dict[str, Any]]) -> str:
    """The comparison table, ranked by EX."""
    head = ("Model", "EX", "F1", "VES", "n", "calls/q", "tok in", "tok out",
            "cost $", "t/o", "mins")
    body = []
    for r in sorted(rows, key=lambda x: -(x["metrics"].get("EX") or 0)):
        m = r["metrics"]
        body.append((
            r["model"],
            f"{m.get('EX', 0):.1f}",
            f"{m.get('F1', 0):.1f}",
            f"{m.get('VES', 0):.1f}",
            str(m.get("total", 0)),
            f"{m.get('llm_calls_per_question', 0):.1f}",
            f"{m.get('tokens_in', 0):,}",
            f"{m.get('tokens_out', 0):,}",
            f"{m.get('cost_usd', 0):.4f}",
            str(m.get("timeouts", 0)),
            f"{r['seconds'] / 60:.1f}",
        ))
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(head)]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    rule = "  ".join("-" * w for w in widths)
    out = [line, rule]
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(out)


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    out = ["| Model | EX | F1 | VES | n | LLM calls/q | Tokens in | Tokens out "
           "| Cost (USD) | Timeouts | Minutes |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: -(x["metrics"].get("EX") or 0)):
        m = r["metrics"]
        out.append(
            f"| `{r['model']}` | {m.get('EX', 0):.1f} | {m.get('F1', 0):.1f} "
            f"| {m.get('VES', 0):.1f} | {m.get('total', 0)} "
            f"| {m.get('llm_calls_per_question', 0):.1f} "
            f"| {m.get('tokens_in', 0):,} | {m.get('tokens_out', 0):,} "
            f"| {m.get('cost_usd', 0):.4f} | {m.get('timeouts', 0)} "
            f"| {r['seconds'] / 60:.1f} |")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the BIRD benchmark over several models and compare them.",
        epilog="Anything after `--` is passed to every bird run unchanged.")
    parser.add_argument("--dev-json", default="dataset/mini_dev_sqlite.json")
    parser.add_argument("--db-dir", default="dataset/dev_databases")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="Model ids to compare. Must match "
                             "llm.selectable_models exactly (GH#117). "
                             f"Default: the {len(DEFAULT_MODELS)} benchmarked models.")
    parser.add_argument("--subset", type=int, default=20)
    parser.add_argument("--sample", choices=("head", "random", "stratified"),
                        default="head",
                        help="How --subset picks questions (GH#120). dev.json "
                             "is grouped by database, so 'head' returns ONE "
                             "database's questions; use 'stratified' for a "
                             "cross-domain subset. Default: head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=("simple", "moderate", "challenging"),
                        default=None)
    parser.add_argument("--db-id", default=None)
    parser.add_argument("--question-timeout", type=float, default=None,
                        help="Passed through to each run (GH#119). Omit for "
                             "the harness default.")
    parser.add_argument("--out-dir", default=None,
                        help="Where per-model results land. Default: "
                             "outputs/<difficulty> when --difficulty is set, "
                             "else outputs/compare")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a model whose output file already exists — "
                             "resume a sweep interrupted partway without "
                             "paying for the finished models again")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would run and exit. "
                             "Costs nothing; use it to check the sweep before "
                             "spending on it.")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir or (
        f"outputs/{args.difficulty}" if args.difficulty else "outputs/compare"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dupes = [m for m in set(args.models) if args.models.count(m) > 1]
    if dupes:
        parser.error(f"duplicate model id(s): {', '.join(sorted(dupes))}")

    plan = [(m, output_path(out_dir, args.difficulty, m)) for m in args.models]

    print("\n── Model comparison sweep ─────────────────────────────────────")
    print(f"  Models   : {len(plan)}")
    print(f"  Subset   : {args.subset}  Sample: {args.sample}  Seed: {args.seed}")
    print(f"  Filters  : difficulty={args.difficulty or '-'}  db_id={args.db_id or '-'}")
    print(f"  Output   : {out_dir}/")
    if passthrough:
        print(f"  Extra    : {' '.join(passthrough)}")
    print()

    if args.dry_run:
        for model, out_file in plan:
            print("  " + " ".join(build_argv(model, out_file, args, passthrough)))
        return 0

    rows: List[Dict[str, Any]] = []
    failed: List[Tuple[str, str]] = []
    swept_t0 = time.perf_counter()

    for idx, (model, out_file) in enumerate(plan, start=1):
        if args.skip_existing and out_file.exists():
            print(f"[{idx}/{len(plan)}] SKIP {model} — {out_file} exists")
            payload = read_result(out_file)
            if payload:
                rows.append({"model": model, "metrics": payload.get("metrics", {}),
                             "run_config": payload.get("run_config", {}),
                             "_qids": question_ids(payload), "seconds": 0.0,
                             "output": str(out_file), "skipped": True})
            continue

        cmd = build_argv(model, out_file, args, passthrough)
        print(f"[{idx}/{len(plan)}] {model}")
        print(f"          {' '.join(cmd)}")
        t0 = time.perf_counter()
        # Output is NOT captured: a sweep can run for an hour and watching the
        # per-question lines is how you notice a model failing every question
        # rather than finding out at the end.
        completed = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

        if completed.returncode != 0:
            # Keep going. Six good runs are worth more than an aborted sweep,
            # and the failures are listed again at the end.
            reason = f"exit code {completed.returncode}"
            print(f"          FAILED ({reason}) — continuing with the rest\n")
            failed.append((model, reason))
            continue

        payload = read_result(out_file)
        if payload is None:
            reason = f"exited 0 but {out_file} is missing or unreadable"
            print(f"          FAILED ({reason})\n")
            failed.append((model, reason))
            continue

        rows.append({"model": model, "metrics": payload.get("metrics", {}),
                     "run_config": payload.get("run_config", {}),
                     "_qids": question_ids(payload), "seconds": elapsed,
                     "output": str(out_file), "skipped": False})
        print(f"          EX={payload.get('metrics', {}).get('EX', 0):.1f} "
              f"in {elapsed / 60:.1f} min → {out_file}\n")

    if not rows:
        print("No model produced results.")
        return 1

    warnings = check_same_questions(rows)

    print("\n── Comparison ──────────────────────────────────────────────────")
    print(summarise(rows))
    print()
    for w in warnings:
        print(f"  WARNING: {w}")
    if failed:
        for model, reason in failed:
            print(f"  FAILED : {model} — {reason}")
    if all((r["metrics"].get("cost_usd") or 0) == 0 for r in rows):
        print("  NOTE   : every cost reads $0.00 — add rates under `llm.pricing` "
              "in config.yaml (python -m backend.config.check_models --sync-pricing).")

    summary_path = out_dir / "comparison.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "sweep": {
                "subset": args.subset, "sample": args.sample, "seed": args.seed,
                "difficulty": args.difficulty, "db_id": args.db_id,
                "question_timeout": args.question_timeout,
                "passthrough": passthrough,
                "minutes": round((time.perf_counter() - swept_t0) / 60, 2),
            },
            "warnings": warnings,
            "failed": [{"model": m, "reason": r} for m, r in failed],
            "models": [{k: v for k, v in r.items() if k != "_qids"} for r in rows],
        }, fh, indent=2)

    md_path = out_dir / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Model comparison — {args.difficulty or 'all tiers'}"
                 f"{f' / {args.db_id}' if args.db_id else ''}\n\n"
                 f"`--subset {args.subset} --sample {args.sample} "
                 f"--seed {args.seed}`\n\n")
        fh.write(markdown_table(rows) + "\n")
        if warnings:
            fh.write("\n**Warnings**\n\n"
                     + "".join(f"- {w}\n" for w in warnings))

    print(f"\n  Summary → {summary_path}")
    print(f"  Table   → {md_path}")
    print("────────────────────────────────────────────────────────────────\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
