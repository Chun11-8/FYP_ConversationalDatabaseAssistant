"""
backend/eval/bird_predict.py — Generate predictions in the OFFICIAL BIRD format.

This script runs *your* text-to-SQL system architecture (SQLCodeAgent +
SQLValidatorAgent, with schema extracted from each SQLite database) over the
BIRD mini-dev questions and writes a predictions file that the **official**
BIRD evaluation scripts can score.

Why this exists
---------------
The official BIRD evaluator (mini_dev/evaluation/evaluation_ex.py etc.) does
NOT run your model. It only scores a predictions JSON of the form:

    {
      "0": "SELECT ... \\t----- bird -----\\t california_schools",
      "1": "SELECT ... \\t----- bird -----\\t financial",
      ...
    }

where each key is the question index (0-based, matching gold-file line order)
and each value is "<single-line SQL>\\t----- bird -----\\t<db_id>".

This generator produces exactly that file. You then point the official
evaluation_ex.py / evaluation_ves.py / evaluation_f1.py at it.

For a quick *self-contained* score (generation + scoring in one go, with
--subset / --difficulty support) use backend/eval/bird.py instead. Use THIS
script when you want the credible, official benchmark numbers.

Usage
-----
    python -m backend.eval.bird_predict \\
        --questions  path/to/mini_dev_data/mini_dev_sqlite.json \\
        --db-dir     path/to/mini_dev_data/dev_databases \\
        --output     path/to/mini_dev/sql_result/predict_mini_dev_MYSYS_SQLite.json

The output filename matters to the official run_evaluation.sh: it derives the
log name from `basename <predicted_sql_path> .json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.eval.bird import extract_sqlite_schema, load_dev_json, resolve_model

logger = logging.getLogger(__name__)

# Official BIRD separator between the SQL and the db_id inside a prediction value.
BIRD_SEP = "\t----- bird -----\t"


def _to_single_line(sql: str) -> str:
    """Collapse newlines/tabs/runs of whitespace so the prediction stays on one line.

    The official format stores each prediction as a single JSON string; embedded
    tabs would also collide with the BIRD separator. We normalise to single
    spaces and strip a trailing semicolon.
    """
    if not sql:
        return ""
    one_line = " ".join(sql.split())
    return one_line.rstrip(";").strip()


def generate_predictions(
    questions_path: str,
    db_dir: str,
    sql_agent: Any,
    subset: Optional[int] = None,
    difficulty_filter: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run the system over the BIRD questions and return the official predictions dict.

    The dict is keyed by the *positional index* of the question within the loaded
    file (as a string), which is what the official evaluator expects so that
    predictions line up with the gold SQL file by order.

    NOTE: for OFFICIAL scoring do NOT pass --subset/--difficulty — the prediction
    indices must align 1:1 with the full gold file. Those flags are for quick
    local smoke checks only.
    """
    questions = load_dev_json(questions_path)

    predictions: Dict[str, str] = {}
    total = len(questions)
    logger.info(f"[BIRD-PREDICT] Generating predictions for {total} question(s)")

    for idx, item in enumerate(questions):
        difficulty = item.get("difficulty", "unknown")
        if difficulty_filter and difficulty != difficulty_filter:
            continue
        if subset is not None and idx >= subset:
            break

        question: str = item["question"]
        evidence: str = item.get("evidence", "") or ""
        db_id: str = item["db_id"]
        db_path = str(Path(db_dir) / db_id / f"{db_id}.sqlite")

        # ── Schema context from the SQLite database ──────────────────────────
        try:
            schema_context = extract_sqlite_schema(db_path)
        except Exception as exc:
            logger.error(f"[BIRD-PREDICT] schema fail idx={idx} db={db_id}: {exc}")
            predictions[str(idx)] = "SELECT 1" + BIRD_SEP + db_id
            continue

        # ── Promote evidence into a structured block (Tier 1) ────────────────
        full_query = (
            f"<domain_context>{evidence.strip()}</domain_context>\n\n{question}"
            if evidence.strip()
            else question
        )

        # ── Generate + rerank with the real system ───────────────────────────
        pred_sql = ""
        t0 = time.perf_counter()
        try:
            sqls = sql_agent.generate(
                query=full_query,
                schema_context=schema_context,
                few_shots=[],
                history=[],
            )
            if sqls:
                pred_sql = sql_agent.rerank(sqls, full_query, schema_context)
        except Exception as exc:
            logger.warning(f"[BIRD-PREDICT] generation failed idx={idx}: {exc}")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        pred_sql = _to_single_line(pred_sql) or "SELECT 1"
        predictions[str(idx)] = pred_sql + BIRD_SEP + db_id

        logger.info(
            f"[{idx + 1}/{total}] db={db_id} diff={difficulty} {elapsed_ms}ms "
            f"| {question[:55]}"
        )

    return predictions


def write_predictions(predictions: Dict[str, str], output_path: str) -> None:
    """Write the predictions dict to a JSON file in the official format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[BIRD-PREDICT] Wrote {len(predictions)} predictions → {output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate BIRD-format predictions from the FYP system architecture"
    )
    parser.add_argument("--questions", required=True,
                        help="Path to mini_dev question JSON (e.g. mini_dev_sqlite.json)")
    parser.add_argument("--db-dir", required=True,
                        help="Path to dev_databases/ (contains {db_id}/{db_id}.sqlite)")
    parser.add_argument("--output", required=True,
                        help="Output predictions JSON (e.g. predict_mini_dev_MYSYS_SQLite.json)")
    parser.add_argument("--subset", type=int, default=None,
                        help="LOCAL SMOKE ONLY: first N questions (breaks official index alignment)")
    parser.add_argument("--difficulty",
                        choices=["simple", "moderate", "challenging"], default=None,
                        help="LOCAL SMOKE ONLY: filter by difficulty (breaks official alignment)")
    parser.add_argument("--model", default=None,
                        help="Override config.yaml's llm.model for this run (any model id)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-COMPATIBLE endpoint to predict against, e.g. "
                             "https://openrouter.ai/api/v1. Overrides $LLM_BASE_URL "
                             "and config llm.base_url; omit for OpenAI (GH#65)")
    args = parser.parse_args(argv)

    if args.subset is not None or args.difficulty is not None:
        logger.warning(
            "[BIRD-PREDICT] --subset/--difficulty set: predictions will NOT align "
            "with the full gold file. Use only for local smoke checks, not official scoring."
        )

    from backend.config.config_loader import load_config
    from backend.agents.sql_code_agent import SQLCodeAgent
    from backend.agents.llm_factory import (
        build_chat_llm, describe_target, resolve_base_url,
    )

    config = load_config()
    # Target SQLite syntax so dialect normalization (Tier 1) kicks in.
    bird_config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}

    model_name = resolve_model(args.model, config)
    base_url = resolve_base_url(config, args.base_url)
    logger.info("[BIRD-PREDICT] target → %s", describe_target(model_name, base_url))
    llm = build_chat_llm(model_name, config["llm"]["api_key"], base_url)
    sql_agent = SQLCodeAgent(bird_config, llm)

    predictions = generate_predictions(
        questions_path=args.questions,
        db_dir=args.db_dir,
        sql_agent=sql_agent,
        subset=args.subset,
        difficulty_filter=args.difficulty,
    )
    write_predictions(predictions, args.output)

    print("\n── BIRD Prediction Generation Complete ───────────────────────────────────")
    print(f"  Predictions written : {len(predictions)}")
    print(f"  Output file         : {args.output}")
    print("\n  Next: score with the OFFICIAL evaluator, e.g.")
    print("    python3 evaluation_ex.py \\")
    print("        --db_root_path <dev_databases>/ \\")
    print(f"        --predicted_sql_path {args.output} \\")
    print("        --ground_truth_path <mini_dev_sqlite_gold.sql> \\")
    print("        --diff_json_path <mini_dev_sqlite.json> \\")
    print("        --num_cpus 4 --meta_time_out 30.0 --sql_dialect SQLite \\")
    print("        --output_log_path <eval_result>/mysys.txt")
    print("──────────────────────────────────────────────────────────────────────────\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
