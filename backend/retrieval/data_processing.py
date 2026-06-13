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


# ── helpers ────────────────────────────────────────────────────────────────────
def connect_postgres(db_name="FYP", user="postgres", password="Toyo20001108!", host="localhost", port=5432):
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
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
    Load column descriptions from CSV files in dataset/minidev/MINIDEV/dev_databases/*/database_description/
    Returns a flat dict: { "table.column" -> "meaning" }
    """
    if dataset_root is None:
        dataset_root = Path("dataset/minidev/MINIDEV/dev_databases")
    
    if not dataset_root.exists():
        print(f"  [WARN] Dataset root not found: {dataset_root}")
        return {}
    
    meanings = {}
    
    # Iterate through each database folder
    for db_folder in dataset_root.iterdir():
        if not db_folder.is_dir():
            continue
        
        db_description_dir = db_folder / "database_description"
        if not db_description_dir.exists():
            continue
        
        # Find all CSV files in database_description folder
        for csv_file in db_description_dir.glob("*.csv"):
            table_name = csv_file.stem  # filename without .csv extension
            
            try:
                with open(csv_file, "r", encoding="utf-8",errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row and "column_name" in row and "column_description" in row:
                            col_name = row["column_name"].replace(" ", "_").strip()  # normalize column name
                            col_desc = row["column_description"].strip()
                            val_desc = row.get("value_description", "").strip()  # optional value description
                            #print(f"  [INFO] Loaded meaning for {table_name}.{col_name}: {col_desc}")
                            if col_name and col_desc:
                                key = col_name
                                meanings[key] = (clean_text(col_desc), clean_text(val_desc))    
            except Exception as e:
                print(f"  [WARN] Error reading {csv_file}: {e}")
    #print(meanings)
    return meanings



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
            key = col["name"]
            #print(col)
            if key in column_meanings:
                col["description"], col["value_description"] = column_meanings[key]
            else:
                col["description"] = ""
                col["value_description"] = ""
            #print(f"  → Sampled {len(table['sample_rows'])} rows from {table['table_name']}")
    
    
    conn.close()
    return {
        "db_name": db_name,
        "tables": tables,
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