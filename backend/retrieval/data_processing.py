"""
data_processing.py
------------------
Loads and cleans metadata from multiple .sqlite files.
Extracts: table schema, column meanings, sample rows, knowledge base (kb.jsonl)

Expected folder structure:
    data/
        alien/
            alien_tempalte.sqlite
            alien_column_meaning_base.json
            alien_kb.jsonl
        archeology/
            archeology_tempalte.sqlite
            ...
"""

import os
import json
import psycopg2
import re
import csv
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / "secrets.env", override=False)


# ── helpers ────────────────────────────────────────────────────────────────────
def connect_postgres(
    db_name=None,
    user=None,
    password=None,
    host=None,
    port=None,
):
    try:
        conn = psycopg2.connect(
            dbname=db_name   if db_name   is not None else os.getenv("PG_DB",   "FYP"),
            user=user         if user      is not None else os.getenv("PG_USER", "postgres"),
            password=password if password  is not None else os.getenv("PG_PASSWORD"),
            host=host         if host      is not None else os.getenv("PG_HOST", "127.0.0.1"),
            port=port         if port      is not None else int(os.getenv("PG_PORT", 5432)),
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Cannot connect to PostgreSQL: {e}")
        return None

def clean_text(text: str) -> str:
    """Remove excess whitespace and non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n]", "", text)   # keep printable ASCII + newline
    text = re.sub(r" {2,}", " ", text)             # collapse multiple spaces
    return text.strip()



# ── schema extraction ──────────────────────────────────────────────────────────

def extract_schema_pg(conn, db_name: str) -> list[dict]:
    cursor = conn.cursor()

    # Get tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    table_names = [row[0] for row in cursor.fetchall()]

    tables = []

    for table_name in table_names:
        # Columns
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
        """, (table_name,))
        cols = cursor.fetchall()

        columns = []
        for col in cols:
            columns.append({
                "name": col[0],
                "type": col[1],
                "not_null": col[2] == "NO",
                "default": col[3],
                "primary_key": False,  # fill later
            })

        # Primary keys
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = %s
              AND tc.constraint_type = 'PRIMARY KEY'
        """, (table_name,))
        pk_cols = {row[0] for row in cursor.fetchall()}

        for col in columns:
            if col["name"] in pk_cols:
                col["primary_key"] = True

        # Foreign keys
        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name = %s
        """, (table_name,))

        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append({
                "from_col": row[0],
                "to_table": row[1],
                "to_col": row[2],
            })

        tables.append({
            "db_name": db_name,
            "table_name": table_name,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "col_description": "",  # placeholder for future use
            "val_description": "",  # placeholder for future use
        })

    return tables


# ── sample rows ────────────────────────────────────────────────────────────────

def extract_sample_rows_pg(conn, table_name: str, n: int = 3):
    cursor = conn.cursor()

    try:
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {n}')
        colnames = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(colnames, row)) for row in rows]
    except Exception as e:
        print(f"[WARN] Could not sample {table_name}: {e}")
        return []


# ── column meanings ────────────────────────────────────────────────────────────

def load_column_meanings(dataset_root: Path = None) -> dict:
    """
    Load column descriptions from CSV files in
    dataset/dev_databases/*/database_description/

    Each CSV is named after its table (file stem == table name) and has the
    columns:
        original_column_name, column_name, column_description,
        data_format, value_description

    IMPORTANT: ``original_column_name`` is the *real* SQL/Postgres column name
    (e.g. ``account_id``). ``column_name`` is only a human-friendly label
    (e.g. ``account id``) and is frequently blank. We therefore key meanings by
    ``original_column_name`` so they actually match the columns that live in
    Postgres — keying by ``column_name`` was why descriptions came back empty.

    Returns a dict keyed by the FULLY-QUALIFIED, lowercased "table.column":
        { "account.account_id": (col_description, value_description), ... }

    Keying by table.column prevents columns that share a name across tables
    from clobbering each other. Use :func:`resolve_column_meaning` to look up a
    description (it handles the qualified key plus a safe unambiguous bare-name
    fallback).
    """
    if dataset_root is None:
        dataset_root = Path("dataset/dev_databases")

    if not dataset_root.exists():
        print(f"  [WARN] Dataset root not found: {dataset_root}")
        return {}

    meanings = {}
    loaded = 0

    # Iterate through each database folder
    for db_folder in dataset_root.iterdir():
        if not db_folder.is_dir():
            continue

        db_description_dir = db_folder / "database_description"
        if not db_description_dir.exists():
            continue

        # Find all CSV files in database_description folder
        for csv_file in db_description_dir.glob("*.csv"):
            table_name = csv_file.stem  # filename without .csv extension == table name

            try:
                # utf-8-sig strips the BOM that prefixes these CSVs, otherwise
                # the first header becomes "﻿original_column_name".
                with open(csv_file, "r", encoding="utf-8-sig", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        # Prefer the real SQL column name; fall back to the
                        # friendly label only when original is missing.
                        raw_col = (row.get("original_column_name")
                                   or row.get("column_name") or "")
                        # Match the real Postgres column verbatim — do NOT
                        # collapse spaces to underscores: columns like
                        # frpm."Academic Year" keep their spaces. Only trim
                        # whitespace; resolve_column_meaning lowercases the key.
                        col_name = raw_col.strip()
                        col_desc = (row.get("column_description") or "").strip()
                        val_desc = (row.get("value_description") or "").strip()

                        if col_name and col_desc:
                            key = f"{table_name}.{col_name}".lower()
                            meanings[key] = (clean_text(col_desc), clean_text(val_desc))
                            loaded += 1
            except Exception as e:
                print(f"  [WARN] Error reading {csv_file}: {e}")

    print(f"  [INFO] load_column_meanings: {loaded} column descriptions "
          f"across {len(meanings)} unique table.column keys")
    return meanings


def resolve_column_meaning(meanings: dict, table_name: str, col_name: str) -> tuple[str, str]:
    """
    Look up a column's (description, value_description) from a meanings dict that
    is keyed by lowercased "table.column".

    Resolution order:
      1. Exact "table.column" match (correct, unambiguous).
      2. Fallback: a bare column-name match, but ONLY when that column name is
         unique across all tables — so we never attach a wrong description when
         the same name exists in multiple tables.

    Returns ("", "") when nothing matches.
    """
    if not meanings:
        return ("", "")

    qualified = f"{table_name}.{col_name}".lower()
    if qualified in meanings:
        return meanings[qualified]

    target = col_name.lower()
    hits = [v for k, v in meanings.items() if k.split(".", 1)[-1] == target]
    if len(hits) == 1:
        return hits[0]
    return ("", "")



# ── main processor ─────────────────────────────────────────────────────────────

def process_database(db_name: str) -> Optional[dict]:
    """
    Process a single database directory.
    Returns a structured dict with all metadata, or None on failure.
    """
    
    print(f"\n[INFO] Processing: {db_name}")

    try:
        conn = connect_postgres(db_name)
    except Exception as e:
        print(f"  [ERROR] Cannot open {db_name}: {e}")
        return None

    # 1. Schema
    tables = extract_schema_pg(conn, db_name)
    print(f"  → {len(tables)} tables found")
    # 3. Column meanings

    column_meanings = load_column_meanings()
    print(f"  → {len(column_meanings)} column meanings loaded")

    # 2. Sample rows per table
    for table in tables:
        table["sample_rows"] = extract_sample_rows_pg(conn, table["table_name"])
        for col in table["columns"]:
            # Resolve by table.column (unambiguous), falling back to a unique
            # bare column name only when it exists in exactly one table.
            col["description"], col["value_description"] = resolve_column_meaning(
                column_meanings, table["table_name"], col["name"]
            )
    
    
    conn.close()
    return {
        "db_name": db_name,
        "tables": tables,
    }





# ── schema info for API ────────────────────────────────────────────────────────

def get_schema_info(db_name: str | None = None) -> dict:
    """
    Return a clean table + column dict for the /schema API endpoint.

    Only includes name, type, nullability, PK, and FK info — no sample rows.
    """
    resolved_db = db_name or os.getenv("PG_DB", "FYP")
    conn = connect_postgres(db_name=resolved_db)
    if conn is None:
        raise RuntimeError(f"Cannot connect to database '{resolved_db}'")
    try:
        raw_tables = extract_schema_pg(conn, resolved_db)
    finally:
        conn.close()

    # Column descriptions (best-effort) from the dataset CSVs. Keyed by column
    # name → (description, value_description). Never fatal if the files are absent.
    try:
        column_meanings = load_column_meanings()
    except Exception as e:
        print(f"  [WARN] Could not load column meanings: {e}")
        column_meanings = {}

    clean: list[dict] = []
    for t in raw_tables:
        fk_map = {
            fk["from_col"]: f"{fk['to_table']}.{fk['to_col']}"
            for fk in t.get("foreign_keys", [])
        }
        columns = []
        for c in t["columns"]:
            desc, val_desc = resolve_column_meaning(
                column_meanings, t["table_name"], c["name"]
            )
            columns.append({
                "name": c["name"],
                "type": c["type"],
                "not_null": c["not_null"],
                "primary_key": c["primary_key"],
                "foreign_key": fk_map.get(c["name"]),
                "description": desc or None,
                "value_description": val_desc or None,
            })
        clean.append({
            "table_name": t["table_name"],
            "column_count": len(t["columns"]),
            "columns": columns,
        })

    return {
        "db_name": resolved_db,
        "table_count": len(clean),
        "tables": clean,
    }


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db_name = "FYP"  # your PostgreSQL DB
    db = process_database(db_name)

"""
    if db:
        print(f"\n── Preview: {db['db_name']} ──")
        for t in db["tables"]:
            print(f"  Table: {t['table_name']}")
            for c in t["columns"]:
                print(f"    {c['name']} ({c['type']})")
                print(f"      Description: {c['description']}")
                print(f"      Value Description: {c['value_description']}")
            if t["sample_rows"]:
                print(f"    Sample: {t['sample_rows'][0]}")"""