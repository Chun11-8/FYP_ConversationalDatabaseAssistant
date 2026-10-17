"""
backend/eval/download_bird.py — Download the BIRD mini-dev QUESTIONS via HuggingFace.

BIRD publishes mini-dev once per SQL dialect. The questions differ — the gold
SQL is written for that engine — so the file must match `bird.py --engine` or
the gold cannot execute and the run scores whatever survives:

    <output-dir>/
      mini_dev_postgresql.json        # questions array   (--engine postgres)
      mini_dev_postgresql_gold.sql    # one "SQL\\tdb_id" per line
      mini_dev_sqlite.json            # questions array   (--engine sqlite)
      mini_dev_sqlite_gold.sql
      dev_databases/                  # SQLite database files (see below)

WHAT THIS SCRIPT DOES NOT FETCH
-------------------------------
`load_dataset()` returns the question ROWS — not the database contents. (The
dataset repo's Files tab does host the full archive; this script reads the
rows, so it cannot give you the data.) Download the archive by hand from
https://huggingface.co/datasets/birdsql/bird_mini_dev or
https://bird-bench.github.io/, and take from it:

  * PostgreSQL — one dump holding every database's tables in ONE namespace,
    `MINIDEV_postgresql/BIRD_dev.sql` in the official archive. Load it with:
        psql -h 127.0.0.1 -U postgres -d FYP -f .../MINIDEV_postgresql/BIRD_dev.sql

  * SQLite — one file per database, used directly via `--db-dir`:
        git clone --depth 1 https://github.com/bird-bench/mini_dev.git
        cp -r mini_dev/mini_dev_sqlite/dev_databases <output-dir>/dev_databases/

Both ship in the official archive at https://bird-bench.github.io/ →
"Download Mini-Dev". The root README's "Loading Data into PostgreSQL" section
walks through it end to end.

Usage
-----
    python -m backend.eval.download_bird                       # both dialects
    python -m backend.eval.download_bird --dialect postgresql
    python -m backend.eval.download_bird --output-dir dataset/minidev
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Dialects ──────────────────────────────────────────────────────────────────
# The HuggingFace split names are not the file names, and upstream has used
# more than one spelling. Each dialect lists the candidates in preference
# order; whichever the dataset actually exposes wins, and an unresolvable
# dialect reports the splits that DO exist rather than a bare KeyError.

DIALECTS: dict[str, tuple[str, ...]] = {
    "postgresql": ("mini_dev_pg", "mini_dev_postgresql", "postgresql", "pg"),
    "sqlite":     ("mini_dev_sqlite", "sqlite"),
    "mysql":      ("mini_dev_mysql", "mysql"),
}

# Used only when the dataset exposes no recognisable per-dialect split (an
# older snapshot with one unnamed split). The questions are then whatever that
# split holds, so we say so rather than labelling them with a dialect.
_GENERIC_SPLITS = ("validation", "dev", "mini_dev", "test", "train")


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
    """Format one line for a *_gold.sql file."""
    # Official format: "SELECT ...\t<db_id>"  (no newlines inside SQL)
    single = " ".join(sql.split())
    return f"{single}\t{db_id}"


def resolve_split(available: list[str], dialect: str) -> str | None:
    """
    The split name this dataset uses for *dialect*, or None.

    Matching is case-insensitive and ignores '-'/'_' so a renamed split does
    not silently fall through to the wrong dialect's questions.
    """
    def norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    index = {norm(s): s for s in available}
    for candidate in DIALECTS[dialect]:
        if norm(candidate) in index:
            return index[norm(candidate)]
    return None


def write_dialect(split, output_dir: Path, dialect: str) -> Path:
    """Write <dialect>'s questions + gold file. Returns the questions path."""
    questions: list[dict] = []
    gold_lines: list[str] = []
    for idx, row in enumerate(split):
        norm = _normalise_row(dict(row), idx)
        questions.append(norm)
        gold_lines.append(_gold_line(norm["SQL"], norm["db_id"]))

    questions_path = output_dir / f"mini_dev_{dialect}.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(questions)} {dialect} questions → {questions_path}")

    gold_path = output_dir / f"mini_dev_{dialect}_gold.sql"
    with open(gold_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gold_lines) + "\n")
    logger.info(f"Wrote gold file → {gold_path}")
    return questions_path


# ── Main download logic ───────────────────────────────────────────────────────

def download(output_dir: Path, dialects: list[str]) -> None:
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

    available = list(ds.keys()) if hasattr(ds, "keys") else []
    logger.info(f"Splits available: {', '.join(available) or '(none named)'}")

    written: list[Path] = []
    unresolved: list[str] = []
    for dialect in dialects:
        name = resolve_split(available, dialect)
        if name is None:
            unresolved.append(dialect)
            continue
        logger.info(f"{dialect}: using split '{name}' ({len(ds[name])} questions)")
        written.append(write_dialect(ds[name], output_dir, dialect))

    if unresolved and not written:
        # An older snapshot with one unnamed split: write it once, unlabelled,
        # rather than claiming it is a dialect we could not find.
        fallback = next((s for s in _GENERIC_SPLITS if s in available), None)
        if fallback is None and available:
            fallback = available[0]
        if fallback is not None:
            logger.warning(
                f"No per-dialect split found for {', '.join(unresolved)}. "
                f"Falling back to '{fallback}' and writing it as sqlite — "
                f"CHECK the gold SQL dialect before scoring against it.")
            written.append(write_dialect(ds[fallback], output_dir, "sqlite"))
        else:
            logger.error("Dataset exposes no usable split.")
            sys.exit(1)
    elif unresolved:
        logger.warning(
            f"No split found for: {', '.join(unresolved)}. "
            f"Available: {', '.join(available)}")

    _warn_missing_databases(output_dir, db_dir, written)


def _warn_missing_databases(output_dir: Path, db_dir: Path,
                            written: list[Path]) -> None:
    """Tell the user what still has to be fetched by hand, per engine."""
    db_ids: set[str] = set()
    for path in written:
        with open(path, encoding="utf-8") as f:
            db_ids |= {q["db_id"] for q in json.load(f) if q.get("db_id")}

    missing = sorted(d for d in db_ids
                     if not (db_dir / d / f"{d}.sqlite").exists()
                     and not (db_dir / d / f"{d}.db").exists())

    if missing:
        logger.warning(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  DATABASE CONTENTS NOT FOUND — questions only were downloaded.\n"
            "  load_dataset() returns rows, not the database files. Download\n"
            "  the mini-dev archive by hand and take the data from it.\n\n"
            "  For --engine postgres (the default): load the dump into the\n"
            "  SAME database the assistant uses, then no --db-dir is needed:\n"
            "    psql -h 127.0.0.1 -U postgres -d FYP \\\n"
            "         -f <unzipped>/MINIDEV_postgresql/BIRD_dev.sql\n\n"
            "  For --engine sqlite: bird.py reads\n"
            f"    {db_dir}/<db_id>/<db_id>.sqlite\n"
            "    git clone --depth 1 https://github.com/bird-bench/mini_dev.git\n"
            f"    cp -r mini_dev/mini_dev_sqlite/dev_databases {db_dir}\n\n"
            "  Both ship in the official archive:\n"
            "    https://bird-bench.github.io/ → Download Mini-Dev\n\n"
            f"  Databases referenced ({len(missing)}): "
            f"{', '.join(missing[:10])}" + (" …" if len(missing) > 10 else "") + "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
    else:
        logger.info(f"All {len(db_ids)} SQLite database directories found under {db_dir}")

    if written:
        logger.info(
            "\nDone. Next:\n"
            f"  python -m backend.eval.bird --dev-json {written[0]} --subset 2"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download BIRD mini-dev questions via HuggingFace and save "
                    "them in the structure expected by bird.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir",
        default="./mini_dev_data",
        help="Directory to save downloaded data (default: ./mini_dev_data)",
    )
    p.add_argument(
        "--dialect",
        default="all",
        choices=("all", *DIALECTS),
        help="Which dialect's questions to fetch. The gold SQL differs per "
             "dialect and must match --engine (default: all)",
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
    dialects = (["postgresql", "sqlite"] if args.dialect == "all"
                else [args.dialect])
    download(Path(args.output_dir), dialects)


if __name__ == "__main__":
    main()
