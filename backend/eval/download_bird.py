"""
backend/eval/download_bird.py — Download the BIRD mini-dev dataset via HuggingFace.

Saves data in the exact structure expected by bird.py:

    <output-dir>/
      mini_dev_sqlite.json          # questions array
      mini_dev_sqlite_gold.sql      # one "SQL\\tdb_id" per line
      dev_databases/                # SQLite database files (see note below)

NOTE ON DATABASE FILES
----------------------
The HuggingFace dataset `birdsql/bird_mini_dev` includes question metadata but
NOT the SQLite database files (they are large binary blobs, ~1-2 GB total).
You must obtain the `.sqlite` files separately:

  Option A — Download the official mini-dev archive:
    https://bird-bench.github.io/ → "Download Mini-Dev"
    Unzip and copy the `dev_databases/` folder to <output-dir>/dev_databases/

  Option B — Clone the mini-dev GitHub repo (which bundles the DBs):
    git clone --depth 1 https://github.com/bird-bench/mini_dev.git
    cp -r mini_dev/mini_dev_sqlite/dev_databases <output-dir>/dev_databases/

This script will create the <output-dir>/dev_databases/ directory and warn you
if the databases are missing when bird.py would need them.

Usage
-----
    python -m backend.eval.download_bird                      # saves to ./mini_dev_data
    python -m backend.eval.download_bird --output-dir /data/bird

After running this script, follow the NOTE above to populate dev_databases/,
then run:
    python -m backend.eval.bird --dev-json <dev.json> --engine postgres
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Field mapping ─────────────────────────────────────────────────────────────
# The HuggingFace dataset schema may differ slightly from the official JSON.
# We normalise to the fields bird.py expects.

_HF_FIELD_MAP = {
    # HF field name → our canonical name
    "question_id": "question_id",
    "db_id":       "db_id",
    "question":    "question",
    "evidence":    "evidence",
    "SQL":         "SQL",
    "difficulty":  "difficulty",
    # alternate field names seen in some dataset versions
    "query":       "SQL",
    "hard_level":  "difficulty",
}


def _normalise_row(row: dict, idx: int) -> dict:
    """Map one HuggingFace row to the canonical question dict."""
    out: dict = {}
    for hf_key, canon_key in _HF_FIELD_MAP.items():
        if hf_key in row and canon_key not in out:
            out[canon_key] = row[hf_key]

    # Guarantee required fields exist
    out.setdefault("question_id", idx)
    out.setdefault("db_id", "")
    out.setdefault("question", "")
    out.setdefault("evidence", "")
    out.setdefault("SQL", "")
    out.setdefault("difficulty", "simple")
    return out


def _gold_line(sql: str, db_id: str) -> str:
    """Format one line for mini_dev_sqlite_gold.sql."""
    # Official format: "SELECT ...\t<db_id>"  (no newlines inside SQL)
    single = " ".join(sql.split())
    return f"{single}\t{db_id}"


# ── Main download logic ───────────────────────────────────────────────────────

def download(output_dir: Path) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.error(
            "The 'datasets' package is required. Install it with:\n"
            "    pip install datasets"
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir = output_dir / "dev_databases"
    db_dir.mkdir(exist_ok=True)

    logger.info("Loading birdsql/bird_mini_dev from HuggingFace …")
    try:
        ds = load_dataset("birdsql/bird_mini_dev")
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        sys.exit(1)

    # The dataset may expose a single split or named splits.
    # Try common split names; fall back to the first available split.
    split_names = list(ds.keys()) if hasattr(ds, "keys") else []
    for candidate in ("validation", "dev", "mini_dev", "test", "train"):
        if candidate in split_names:
            split = ds[candidate]
            logger.info(f"Using split '{candidate}' ({len(split)} questions)")
            break
    else:
        if split_names:
            split = ds[split_names[0]]
            logger.info(f"Using split '{split_names[0]}' ({len(split)} questions)")
        else:
            # Dataset itself might be the split (DatasetDict with no named splits)
            split = ds
            logger.info(f"Using dataset directly ({len(split)} questions)")

    questions: list[dict] = []
    gold_lines: list[str] = []

    for idx, row in enumerate(split):
        norm = _normalise_row(dict(row), idx)
        questions.append(norm)
        gold_lines.append(_gold_line(norm["SQL"], norm["db_id"]))

    # Write mini_dev_sqlite.json
    questions_path = output_dir / "mini_dev_sqlite.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(questions)} questions → {questions_path}")

    # Write mini_dev_sqlite_gold.sql
    gold_path = output_dir / "mini_dev_sqlite_gold.sql"
    with open(gold_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gold_lines) + "\n")
    logger.info(f"Wrote gold file → {gold_path}")

    # Warn about missing database files
    db_ids = sorted({q["db_id"] for q in questions if q["db_id"]})
    missing_dbs = [d for d in db_ids if not (db_dir / d / f"{d}.sqlite").exists()]

    if missing_dbs:
        logger.warning(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  DATABASE FILES NOT FOUND\n"
            "  The HuggingFace dataset does not include SQLite .sqlite files.\n"
            "  bird.py needs them under:\n"
            f"    {db_dir}/<db_id>/<db_id>.sqlite\n\n"
            "  To populate them choose one option:\n\n"
            "  Option A — Official mini-dev archive:\n"
            "    1. Go to https://bird-bench.github.io/ and download Mini-Dev\n"
            "    2. Unzip and run:\n"
            f"         cp -r <unzipped>/dev_databases {db_dir}\n\n"
            "  Option B — GitHub clone (includes DBs):\n"
            "    git clone --depth 1 https://github.com/bird-bench/mini_dev.git\n"
            f"    cp -r mini_dev/mini_dev_sqlite/dev_databases {db_dir}\n\n"
            f"  Missing ({len(missing_dbs)}): {', '.join(missing_dbs[:10])}"
            + (" …" if len(missing_dbs) > 10 else "") + "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
    else:
        logger.info(f"All {len(db_ids)} database directories found under {db_dir}")

    logger.info(
        "\nDone. Next steps:\n"
        "  1. (If warned above) populate dev_databases/ with .sqlite files\n"
        "  2. Run:  python -m backend.eval.bird --dev-json <dev.json>\n"
        "     or:  python -m backend.eval.bird \\\n"
        f"              --questions {questions_path} \\\n"
        f"              --db-dir    {db_dir} \\\n"
        "              --output    predictions.json"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download BIRD mini-dev dataset via HuggingFace and save it "
                    "in the structure expected by bird.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir",
        default="./mini_dev_data",
        help="Directory to save downloaded data (default: ./mini_dev_data)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    download(Path(args.output_dir))


if __name__ == "__main__":
    main()
