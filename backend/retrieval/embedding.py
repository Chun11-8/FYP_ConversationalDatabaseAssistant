"""
embedding.py
------------
Converts processed database metadata into embeddable text chunks,
then generates embeddings using OpenAI text-embedding-3-small.

Designed to work with data_processing.py which:
  - Connects to PostgreSQL
  - Loads column meanings from CSV (database_description/*.csv)
  - Has NO knowledge base yet (KB support commented out for future use)

Each chunk represents ONE of:
  - A full table (schema + column descriptions + value descriptions + sample rows)
  - [FUTURE] A knowledge base entry

Usage:
    from embedding import build_chunks, embed_chunks
"""

import os
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / "secrets.env", override=False)

# ── OpenAI client ──────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. "
            "Run: export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


EMBED_MODEL      = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100

# Schema chunk format (GH#49): "verbose" (default, human-readable) or "mschema"
# (XiYan-SQL's compact semi-structured layout). Switching requires a re-index.
SCHEMA_FORMAT = os.getenv("SCHEMA_FORMAT", "verbose").strip().lower()


# ── table chunk builder ────────────────────────────────────────────────────────

def build_mschema_chunk(table: dict) -> dict:
    """
    Render one table in M-Schema format (XiYan-SQL / XGenerationLab/M-Schema):

        【DB_ID】 db_name
        # Table: table_name
        [
        (col:TYPE, Primary Key, NOT NULL, description, Examples: [v1, v2]),
        (col:TYPE, description),
        ]
        【Foreign keys】
        table.from_col=to_table.to_col

    Compact and explicit — the ensemble leader treats the format as
    load-bearing. Same metadata/chunk_id as :func:`build_table_chunk`; only the
    ``text`` differs. PK markers ("Primary Key") and the `(col:TYPE` column
    syntax are recognized by the downstream grain/identifier passes.
    """
    db_name    = table["db_name"]
    table_name = table["table_name"]

    lines = [f"【DB_ID】 {db_name}", f"# Table: {table_name}", "["]
    col_reprs: list[str] = []
    for col in table["columns"]:
        parts = [f"{col['name']}:{col['type']}"]
        if col.get("primary_key"):
            parts.append("Primary Key")
        if col.get("not_null") and not col.get("primary_key"):
            parts.append("NOT NULL")
        desc = (col.get("description") or "").strip()
        if desc:
            parts.append(desc)
        distinct = col.get("distinct_values")
        val_desc = (col.get("value_description") or "").strip()
        if distinct:
            parts.append("Examples: [" + ", ".join(str(v) for v in distinct) + "]")
        elif val_desc:
            parts.append(f"Values: {val_desc}")
        col_reprs.append(f"({', '.join(parts)})")
    # Trailing comma on every column line but the last (M-Schema convention).
    for i, rep in enumerate(col_reprs):
        lines.append(rep + ("," if i < len(col_reprs) - 1 else ""))
    lines.append("]")

    if table.get("foreign_keys"):
        lines.append("【Foreign keys】")
        for fk in table["foreign_keys"]:
            lines.append(
                f"{table_name}.{fk['from_col']}={fk['to_table']}.{fk['to_col']}"
            )

    sample_rows = table.get("sample_rows", [])
    if sample_rows:
        lines.append("【Sample rows】")
        for row in sample_rows[:3]:
            short_row = {
                k: (str(v)[:80] if v is not None else "NULL")
                for k, v in row.items()
            }
            lines.append(f"{short_row}")

    return {
        "chunk_id":   f"{db_name}__{table_name}__schema",
        "chunk_type": "table_schema",
        "db_name":    db_name,
        "table_name": table_name,
        "text":       "\n".join(lines),
    }

def build_table_chunk(table: dict) -> dict:
    """
    Build a single text chunk for one table containing:
      - Database name
      - Table name
      - All columns with type, PK flag, description, value_description
      - Foreign keys
      - Sample rows (up to 3)

    Matches the column structure from data_processing.py:
        col["name"], col["type"], col["not_null"], col["primary_key"]
        col["description"], col["value_description"]

    Returns a chunk dict with 'text' and metadata fields.
    """
    db_name    = table["db_name"]
    table_name = table["table_name"]

    lines = [
        f"Database: {db_name}",
        f"Table: {table_name}",
        "",
        "Columns:",
    ]

    for col in table["columns"]:
        pk_tag  = " [PK]"     if col.get("primary_key") else ""
        nn_tag  = " NOT NULL" if col.get("not_null")    else ""

        # Column description + value description loaded from CSV
        desc     = col.get("description", "").strip()
        val_desc = col.get("value_description", "").strip()
        # Observed distinct values for low-cardinality text columns (Issue #19)
        distinct = col.get("distinct_values")

        col_line = f"  {col['name']} ({col['type']}{pk_tag}{nn_tag})"
        if desc:
            col_line += f" — {desc}"
        if distinct:
            col_line += " | values: {" + ", ".join(str(v) for v in distinct) + "}"
        elif val_desc:
            col_line += f" | values: {val_desc}"

        lines.append(col_line)

    # Foreign keys
    if table.get("foreign_keys"):
        lines.append("")
        lines.append("Foreign Keys:")
        for fk in table["foreign_keys"]:
            lines.append(
                f"  {fk['from_col']} → {fk['to_table']}.{fk['to_col']}"
            )

    # Sample rows
    
    sample_rows = table.get("sample_rows", [])
    #print(sample_rows)
    if sample_rows:
        lines.append("")
        lines.append("Sample Rows:")
        for row in sample_rows[:3]:
            # Truncate long values for readability
            short_row = {
                k: (str(v)[:80] if v is not None else "NULL")
                for k, v in row.items()
            }
            lines.append(f"  {short_row}")

    text = "\n".join(lines)

    return {
        "chunk_id":   f"{db_name}__{table_name}__schema",
        "chunk_type": "table_schema",
        "db_name":    db_name,
        "table_name": table_name,
        "text":       text,
    }


# ── [FUTURE] knowledge base chunk builder ──────────────────────────────────────
# Uncomment when KB support is added to data_processing.py
# KB files live at: dataset/minidev/MINIDEV/dev_databases/<db>/*_kb.jsonl
#
# def build_kb_chunk(entry: dict, db_name: str) -> dict:
#     """
#     Build a text chunk for one knowledge base entry.
#     Expected entry fields (from *_kb.jsonl):
#       id, knowledge, description, definition, type, children_knowledge
#     """
#     kb_id    = entry.get("id", "unknown")
#     name     = entry.get("knowledge", "")
#     desc     = entry.get("description", "")
#     defn     = entry.get("definition", "")
#     ktype    = entry.get("type", "")
#     children = entry.get("children_knowledge", [])
#
#     lines = [
#         f"Database: {db_name}",
#         f"Knowledge: {name}",
#     ]
#     if ktype:
#         lines.append(f"Type: {ktype}")
#     if desc:
#         lines.append(f"Description: {desc}")
#     if defn:
#         lines.append(f"Definition: {defn}")
#     if children and children != [-1]:
#         lines.append(f"Depends on knowledge IDs: {children}")
#
#     text = "\n".join(lines)
#
#     return {
#         "chunk_id":   f"{db_name}__kb__{kb_id}",
#         "chunk_type": "knowledge_base",
#         "db_name":    db_name,
#         "kb_id":      str(kb_id),
#         "kb_name":    name,
#         "text":       text,
#     }


# ── chunk orchestrator ─────────────────────────────────────────────────────────

def build_chunks(all_dbs: list[dict]) -> list[dict]:
    """
    Convert all processed databases into a flat list of text chunks.

    Args:
        all_dbs: Output from data_processing.process_database() —
                 list of dicts with keys: db_name, tables

    Returns:
        List of chunk dicts, each with:
            chunk_id, chunk_type, db_name, table_name, text
    """
    chunks = []

    # Format is env-selected once per build (a full re-index applies it).
    builder = build_mschema_chunk if SCHEMA_FORMAT == "mschema" else build_table_chunk

    for db in all_dbs:
        # Table schema chunks (schema + col descriptions + sample rows)
        for table in db["tables"]:
            chunks.append(builder(table))

        # ── [FUTURE] KB chunks ─────────────────────────────────────────────
        # Uncomment when knowledge_base is returned by data_processing
        #
        # for entry in db.get("knowledge_base", []):
        #     chunks.append(build_kb_chunk(entry, db["db_name"]))

    n_table = sum(1 for c in chunks if c["chunk_type"] == "table_schema")
    # n_kb  = sum(1 for c in chunks if c["chunk_type"] == "knowledge_base")
    print(f"[INFO] Built {len(chunks)} chunks total ({n_table} table schema)")
    #print(chunks)
    return chunks


# ── embedding ──────────────────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for one batch of texts."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def embed_chunks(
    chunks: list[dict],
    client: Optional[OpenAI] = None,
    retry_delay: float = 5.0,
) -> list[dict]:
    """
    Add an 'embedding' field (list[float]) to each chunk dict in-place.
    Processes in batches with basic retry on rate-limit/transient errors.

    Args:
        chunks:      List of chunk dicts from build_chunks().
        client:      Reuse an existing OpenAI client, or None to create one.
        retry_delay: Seconds to wait before retrying a failed batch.

    Returns:
        Same list of chunks, each now containing an 'embedding' key.
    """
    if client is None:
        client = get_openai_client()

    total   = len(chunks)
    batches = [
        chunks[i : i + EMBED_BATCH_SIZE]
        for i in range(0, total, EMBED_BATCH_SIZE)
    ]

    print(f"[INFO] Embedding {total} chunks in {len(batches)} batch(es) "
          f"using '{EMBED_MODEL}' ...")

    embedded = 0
    for batch_idx, batch in enumerate(batches):
        texts = [c["text"] for c in batch]

        for attempt in range(3):
            try:
                embeddings = embed_batch(client, texts)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [WARN] Batch {batch_idx} attempt {attempt + 1} failed: {e}. "
                      f"Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)

        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb

        embedded += len(batch)
        print(f"  → {embedded}/{total} chunks embedded")

    dim = len(chunks[0]["embedding"]) if chunks else "N/A"
    print(f"[INFO] Embedding complete. Vector dimension: {dim}")
    return chunks


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_processing import process_database

    # Process your PostgreSQL DB
    db_name = "FYP"
    db_data = process_database(db_name)
    
    
    if db_data:
        chunks = build_chunks([db_data])

        print(f"\n── Sample chunk text ──\n")
        print(chunks[0]["text"])

        # Only embed if API key is available
        if api_key:
            chunks = embed_chunks(chunks)
            print(f"\nFirst embedding (first 5 dims): {chunks[0]['embedding'][:5]}")
        else:
            print("\n[SKIP] Set OPENAI_API_KEY to test embedding.")