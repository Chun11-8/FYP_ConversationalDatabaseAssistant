"""
store_pinecone.py
-----------------
Persists embedded chunks into Pinecone (managed cloud vector DB).

Setup:
    1. Create a Pinecone account at https://www.pinecone.io
    2. Create an index with:
          dimension = 1536   (text-embedding-3-small)
          metric    = cosine
    3. Set environment variables:
          export PINECONE_API_KEY=your-api-key
          export PINECONE_INDEX=your-index-name

Two namespaces inside one index:
    "table_schemas"   → table schema + column meanings + sample rows
    "knowledge_base"  → [FUTURE] KB entries

Usage:
    from store_pinecone import get_store, save_chunks, retrieve
"""

import os
from typing import Optional
from pinecone import Pinecone, ServerlessSpec


# ── config ─────────────────────────────────────────────────────────────────────

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX",   "text-to-sql-metadata")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD",   "aws")
PINECONE_REGION    = os.getenv("PINECONE_REGION",  "us-east-1")

VECTOR_DIM         = 1536   # text-embedding-3-small
NAMESPACE_SCHEMA   = "table_schemas"
NAMESPACE_KB       = "knowledge_base"   # [FUTURE]
UPSERT_BATCH_SIZE  = 100    # Pinecone recommends ≤ 100 vectors per upsert


# ── connection ─────────────────────────────────────────────────────────────────

def get_store(
    api_key: str = None,
    index_name: str = None,
) -> "pinecone.Index":
    """
    Connect to Pinecone and return the index object.
    Creates the index if it does not exist yet.

    Args:
        api_key:    Pinecone API key. Falls back to PINECONE_API_KEY env var.
        index_name: Index name. Falls back to PINECONE_INDEX env var.

    Returns:
        A Pinecone Index object ready for upsert/query.
    """
    api_key    = api_key    or PINECONE_API_KEY
    index_name = index_name or PINECONE_INDEX

    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY not set. "
            "Run: export PINECONE_API_KEY=your-api-key"
        )

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"[INFO] Creating Pinecone index '{index_name}' ...")
        pc.create_index(
            name      = index_name,
            dimension = VECTOR_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(
                cloud  = PINECONE_CLOUD,
                region = PINECONE_REGION,
            ),
        )
        print(f"[INFO] Index '{index_name}' created.")
    else:
        print(f"[INFO] Pinecone index '{index_name}' already exists.")

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"[INFO] Connected. Index stats: {stats}")
    return index


# ── save chunks ────────────────────────────────────────────────────────────────

def save_chunks(
    chunks: list[dict],
    index = None,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Upsert all embedded chunks into Pinecone.
    Each chunk goes into a namespace based on its chunk_type.

    Args:
        chunks:     List of chunk dicts with 'embedding' field.
        index:      Existing Pinecone Index object, or None to create one.
        batch_size: Vectors per upsert call (max 100 recommended).
    """
    if index is None:
        index = get_store()

    table_chunks = [c for c in chunks if c["chunk_type"] == "table_schema"]

    # ── [FUTURE] KB chunks ─────────────────────────────────────────────────
    # kb_chunks = [c for c in chunks if c["chunk_type"] == "knowledge_base"]

    _upsert_batches(index, table_chunks, NAMESPACE_SCHEMA, batch_size)

    # ── [FUTURE] uncomment to save KB chunks ──────────────────────────────
    # _upsert_batches(index, kb_chunks, NAMESPACE_KB, batch_size)

    print("[INFO] All chunks upserted to Pinecone.")


def _build_pinecone_record(chunk: dict) -> dict:
    """
    Convert a chunk dict into a Pinecone upsert record.
    Metadata values must be str, int, float, bool, or list[str].
    """
    metadata = {
        "chunk_type": chunk.get("chunk_type", ""),
        "db_name":    chunk.get("db_name", ""),
        "text":       chunk.get("text", "")[:3000],   # Pinecone metadata limit ~40KB; truncate to be safe
    }

    if chunk["chunk_type"] == "table_schema":
        metadata["table_name"] = chunk.get("table_name", "")

    # ── [FUTURE] KB metadata ───────────────────────────────────────────────
    # elif chunk["chunk_type"] == "knowledge_base":
    #     metadata["kb_id"]   = chunk.get("kb_id", "")
    #     metadata["kb_name"] = chunk.get("kb_name", "")

    return {
        "id":       chunk["chunk_id"],
        "values":   chunk["embedding"],
        "metadata": metadata,
    }


def _upsert_batches(index, chunks: list[dict], namespace: str, batch_size: int) -> None:
    """Batch upsert chunks into a specific Pinecone namespace."""
    if not chunks:
        return

    total = len(chunks)
    saved = 0

    for i in range(0, total, batch_size):
        batch   = chunks[i : i + batch_size]
        records = [_build_pinecone_record(c) for c in batch]
        index.upsert(vectors=records, namespace=namespace)
        saved += len(batch)
        print(f"  → [{namespace}] {saved}/{total} upserted")


# ── retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query_embedding: list[float],
    db_name: Optional[str] = None,
    top_k_tables: int = 3,
    index = None,
) -> dict:
    """
    Find the most relevant table chunks for a query embedding.

    Args:
        query_embedding: The embedded user query (list of 1536 floats).
        db_name:         Filter results to this database only (optional).
        top_k_tables:    Number of table chunks to return.
        index:           Existing Pinecone Index, or None to create one.

    Returns:
        {
            "tables":         [ { chunk_id, db_name, table_name, text, score }, ... ],
            "knowledge_base": [],   # empty until KB is implemented
        }
    """
    if index is None:
        index = get_store()

    results = {"tables": [], "knowledge_base": []}

    # Build optional metadata filter
    # Pinecone filter syntax: { "field": { "$eq": value } }
    filter_query = {"db_name": {"$eq": db_name}} if db_name else None

    raw = index.query(
        vector          = query_embedding,
        top_k           = top_k_tables,
        namespace       = NAMESPACE_SCHEMA,
        filter          = filter_query,
        include_metadata= True,
    )

    for match in raw.get("matches", []):
        results["tables"].append({
            "chunk_id":   match["id"],
            "db_name":    match["metadata"].get("db_name", ""),
            "table_name": match["metadata"].get("table_name", ""),
            "text":       match["metadata"].get("text", ""),
            "score":      match["score"],   # cosine similarity (higher = more similar)
        })

    # ── [FUTURE] KB retrieval ──────────────────────────────────────────────
    # raw_kb = index.query(
    #     vector           = query_embedding,
    #     top_k            = top_k_kb,
    #     namespace        = NAMESPACE_KB,
    #     filter           = filter_query,
    #     include_metadata = True,
    # )
    # for match in raw_kb.get("matches", []):
    #     results["knowledge_base"].append({
    #         "chunk_id": match["id"],
    #         "db_name":  match["metadata"].get("db_name", ""),
    #         "kb_id":    match["metadata"].get("kb_id", ""),
    #         "kb_name":  match["metadata"].get("kb_name", ""),
    #         "text":     match["metadata"].get("text", ""),
    #         "score":    match["score"],
    #     })

    return results


# ── inspection helpers ─────────────────────────────────────────────────────────

def get_stats(index) -> dict:
    """Return vector counts per namespace."""
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    return {
        ns: info.get("vector_count", 0)
        for ns, info in namespaces.items()
    }


def delete_database(index, db_name: str) -> None:
    """
    Delete all vectors for a specific database.
    Uses Pinecone's delete-by-metadata-filter (requires paid plan).
    On free plan: you must delete by IDs instead.
    """
    try:
        index.delete(
            filter    = {"db_name": {"$eq": db_name}},
            namespace = NAMESPACE_SCHEMA,
        )
        # index.delete(filter={"db_name": {"$eq": db_name}}, namespace=NAMESPACE_KB)  # [FUTURE]
        print(f"[INFO] Deleted all chunks for database '{db_name}'")
    except Exception as e:
        print(f"[WARN] Delete by filter may require a paid Pinecone plan: {e}")
        print("       Alternatively, delete by chunk IDs manually.")


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    index = get_store()
    stats = get_stats(index)
    print(f"\nPinecone index stats: {stats}")

    if not any(stats.values()):
        print("Index is empty. Run indexing.py first.")