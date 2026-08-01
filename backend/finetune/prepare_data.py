"""
backend/finetune/prepare_data.py — build an OpenAI fine-tuning JSONL file from
this system's own data (BIRD gold pairs and/or validated semantic-cache pairs).

Each training example reconstructs the EXACT production prompt
(SQLCodeAgent._system_prompt() + SQLCodeAgent._build_prompt()) so the
fine-tune transfers to real inference-time behavior, not a generic
question→SQL mapping. The assistant turn is the gold SQL wrapped in the same
```sql fence the system already expects to parse out of a real completion.

Usage
-----
    python -m backend.finetune.prepare_data \\
        --bird-results outputs/bird_results_simple_0108.json \\
                        outputs/bird_results_moderate_0108.json \\
        --output backend/finetune/train.jsonl \\
        --split-eval 0.1

    # Add real usage examples from your own database's validated pairs:
    python -m backend.finetune.prepare_data \\
        --bird-results outputs/bird_results_simple_0108.json \\
        --from-semantic-cache \\
        --output backend/finetune/train.jsonl

See backend/finetune/README.md for the full walkthrough (what fine-tuning
does and doesn't help with, cost, how to run the job, how to evaluate it).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── one training example ────────────────────────────────────────────────────

def build_example(
    sql_agent,
    question: str,
    evidence: str,
    schema_context: str,
    gold_sql: str,
) -> Dict[str, Any]:
    """
    One {"messages": [...]} example matching the real production prompt shape.

    ``sql_agent`` is an SQLCodeAgent instance (its llm is never called here —
    only its pure prompt-building methods are used, so `llm=None` is fine).
    """
    query = f"<domain_context>{evidence}</domain_context>\n\n{question}" if evidence else question
    user_prompt = sql_agent._build_prompt(
        query=query,
        schema_context=schema_context,
        few_shots=[],
        history=[],
        n_candidates=1,
    )
    return {
        "messages": [
            {"role": "system", "content": sql_agent._system_prompt()},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"```sql\n{gold_sql.strip()}\n```"},
        ]
    }


# ── BIRD source ──────────────────────────────────────────────────────────────

def examples_from_bird_results(
    sql_agent,
    results_paths: List[str],
    db_dir: Optional[str],
    min_difficulty: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build examples from one or more backend/eval/bird.py output JSON files
    (each has {"questions": [{question, evidence, db_id, difficulty,
    gold_sql, gold_error, ...}, ...]}). Questions whose gold SQL itself
    errored on SQLite (``gold_error`` set) are skipped — never train on a
    known-bad target.
    """
    from backend.eval.bird import extract_sqlite_schema

    examples: List[Dict[str, Any]] = []
    schema_cache: Dict[str, str] = {}
    skipped_gold_error = 0
    skipped_no_schema = 0

    for path in results_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            if q.get("gold_error"):
                skipped_gold_error += 1
                continue
            if min_difficulty and q.get("difficulty") not in min_difficulty:
                continue

            db_id = q["db_id"]
            if db_id not in schema_cache:
                if not db_dir:
                    skipped_no_schema += 1
                    continue
                db_path = str(Path(db_dir) / db_id / f"{db_id}.sqlite")
                try:
                    schema_cache[db_id] = extract_sqlite_schema(db_path)
                except Exception as exc:
                    logger.warning("Schema unavailable for %s: %s", db_id, exc)
                    schema_cache[db_id] = ""
            schema_context = schema_cache[db_id]
            if not schema_context:
                skipped_no_schema += 1
                continue

            examples.append(build_example(
                sql_agent,
                question=q["question"],
                evidence=q.get("evidence", ""),
                schema_context=schema_context,
                gold_sql=q["gold_sql"],
            ))

    if skipped_gold_error:
        logger.info("Skipped %d question(s) with a gold_error (unwinnable gold SQL)",
                     skipped_gold_error)
    if skipped_no_schema:
        logger.info("Skipped %d question(s) with no available schema (pass --db-dir)",
                     skipped_no_schema)
    return examples


# ── live Postgres schema (for semantic-cache examples) ─────────────────────

def render_postgres_schema(db_name: Optional[str] = None) -> str:
    """
    Plain-text schema context for the live Postgres database, in the same
    "Table: x / Columns: - col: TYPE [PK]" shape backend.eval.bird's
    extract_sqlite_schema() produces — so SQLCodeAgent's prompt-building sees
    a consistent format regardless of source. Reuses the same introspection
    the /schema API endpoint is built on (backend.retrieval.data_processing).
    """
    from backend.retrieval.data_processing import get_schema_info

    info = get_schema_info(db_name)
    sections = []
    for t in info["tables"]:
        lines = [f"Table: {t['table_name']}", "Columns:"]
        for c in t["columns"]:
            markers = []
            if c.get("primary_key"):
                markers.append("PK")
            if c.get("not_null"):
                markers.append("NOT NULL")
            suffix = f" [{', '.join(markers)}]" if markers else ""
            lines.append(f"  - {c['name']}: {c['type']}{suffix}")
        sections.append("\n".join(lines))

    fk_lines = [
        f"  {t['table_name']}.{c['name']} → {c['foreign_key']}"
        for t in info["tables"] for c in t["columns"] if c.get("foreign_key")
    ]
    if fk_lines:
        sections.append("Foreign Keys:\n" + "\n".join(fk_lines))
    return "\n\n".join(sections)


# ── semantic-cache source ───────────────────────────────────────────────────

def examples_from_semantic_cache(
    sql_agent,
    db_config: dict,
    schema_context: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Pull validated (question → SQL) pairs your own live system has produced
    (backend/memory/semantic_cache.py's query_sql_pairs table) as additional
    real-usage training examples. ``schema_context`` is your live database's
    schema (see render_postgres_schema()) — these pairs have no evidence text
    and no per-question schema, so one schema context is reused for all of
    them.
    """
    from backend.memory.semantic_cache import SemanticCache

    cache = SemanticCache(db_config=db_config)
    if not cache.is_enabled:
        logger.warning("Semantic cache is disabled (no reachable Postgres) — skipping.")
        return []

    conn = cache._connect()
    try:
        with conn.cursor() as cur:
            sql = "SELECT user_query, sql FROM query_sql_pairs WHERE status = 'active'"
            if limit:
                sql += " LIMIT %s"
                cur.execute(sql, (limit,))
            else:
                cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        build_example(sql_agent, question=q, evidence="", schema_context=schema_context, gold_sql=sql)
        for q, sql in rows
    ]


# ── JSONL writing + split ───────────────────────────────────────────────────

def write_jsonl(examples: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def split_train_eval(
    examples: List[Dict[str, Any]], eval_fraction: float, seed: int = 0,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shuffle deterministically (fixed seed) then split off eval_fraction."""
    if not (0.0 <= eval_fraction < 1.0):
        raise ValueError("--split-eval must be in [0, 1)")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_fraction)
    return shuffled[n_eval:], shuffled[:n_eval]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Build an OpenAI fine-tuning JSONL file from this system's own data"
    )
    parser.add_argument("--bird-results", nargs="*", default=[],
                        help="One or more backend/eval/bird.py output JSON files")
    parser.add_argument("--db-dir", default=None,
                        help="BIRD dev_databases/ root (needed to rebuild schema context "
                             "for --bird-results examples)")
    parser.add_argument("--difficulty", nargs="*", default=None,
                        choices=["simple", "moderate", "challenging"],
                        help="Restrict BIRD examples to these difficulty tiers")
    parser.add_argument("--from-semantic-cache", action="store_true",
                        help="Also pull validated pairs from your live semantic cache")
    parser.add_argument("--cache-limit", type=int, default=None,
                        help="Cap on semantic-cache pairs pulled (default: all)")
    parser.add_argument("--db-name", default=None,
                        help="Postgres database name for --from-semantic-cache's schema "
                             "context (default: config.yaml's db.name)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--split-eval", type=float, default=0.1,
                        help="Fraction held out as a validation file (default: 0.1); "
                             "0 disables the split")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap total examples (cost control) — applied after collecting "
                             "all sources, before the train/eval split")
    args = parser.parse_args(argv)

    if not args.bird_results and not args.from_semantic_cache:
        parser.error("Provide --bird-results and/or --from-semantic-cache")

    from backend.config.config_loader import load_config
    from backend.agents.sql_code_agent import SQLCodeAgent

    config = load_config()
    bird_config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}
    sql_agent = SQLCodeAgent(bird_config, llm=None)

    examples: List[Dict[str, Any]] = []
    if args.bird_results:
        examples += examples_from_bird_results(
            sql_agent, args.bird_results, args.db_dir, min_difficulty=args.difficulty)
        logger.info("Built %d example(s) from BIRD results", len(examples))

    if args.from_semantic_cache:
        schema_context = render_postgres_schema(args.db_name)
        cache_examples = examples_from_semantic_cache(
            sql_agent, config.get("db", {}), schema_context, limit=args.cache_limit)
        logger.info("Built %d example(s) from the semantic cache", len(cache_examples))
        examples += cache_examples

    if not examples:
        logger.error("No training examples produced — nothing to write.")
        return 1

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    if args.split_eval > 0:
        train, eval_ = split_train_eval(examples, args.split_eval)
        write_jsonl(train, args.output)
        eval_path = str(Path(args.output).with_suffix(".eval.jsonl"))
        write_jsonl(eval_, eval_path)
        logger.info("Wrote %d training example(s) -> %s", len(train), args.output)
        logger.info("Wrote %d validation example(s) -> %s", len(eval_), eval_path)
    else:
        write_jsonl(examples, args.output)
        logger.info("Wrote %d training example(s) -> %s", len(examples), args.output)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
