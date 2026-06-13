"""
indexing.py
-----------
Orchestrates the full Text-to-SQL pipeline:
    1. data_processing  → connect to PostgreSQL, load & clean metadata
    2. embedding        → build text chunks + OpenAI embeddings
    3. store            → persist to pgvector OR Pinecone

Switch between backends by setting the VECTOR_STORE env var:
    export VECTOR_STORE=pgvector    # default (recommended)
    export VECTOR_STORE=pinecone

Also provides retrieve() and build_sql_prompt() for runtime use.

Usage (build index):
    python indexing.py build --db FYP
    python indexing.py build --db FYP --store pinecone --force

Usage (query):
    python indexing.py query "show all customers with high credit score" --db FYP
    python indexing.py query "show all customers" --db FYP --store pinecone --show-prompt

Usage (import):
    from indexing import build_index, retrieve, build_sql_prompt
"""

import os
import argparse
from typing import Optional
from langchain.messages import HumanMessage, SystemMessage
from langchain_core import messages
from langchain_openai import ChatOpenAI
from openai import OpenAI

from backend.retrieval.data_processing import process_database
from backend.retrieval.embedding import build_chunks, embed_chunks, get_openai_client, EMBED_MODEL

#from data_processing import process_database
#from embedding import build_chunks, embed_chunks, get_openai_client, EMBED_MODEL
# ── backend selector ───────────────────────────────────────────────────────────

SUPPORTED_STORES = ("pgvector", "pinecone")
VECTOR_STORE="pgvector"

def _get_backend(store: str):
    """
    Dynamically import the correct store module.
    Returns the store module so callers can use store.get_store(),
    store.save_chunks(), store.retrieve(), store.delete_database().
    """
    store = store.lower().strip()
    if store not in SUPPORTED_STORES:
        raise ValueError(
            f"Unknown store '{store}'. Choose from: {SUPPORTED_STORES}"
        )
    if store == "pgvector":
        import backend.retrieval.pgvector_store as backend
        #import pgvector_store as backend
    elif store == "pinecone":
        import backend.retrieval.pinecone_store as backend
        #import pinecone_store as backend

    return backend


def _default_store() -> str:
    return VECTOR_STORE.lower()



# ── embed query helper ─────────────────────────────────────────────────────────

def _embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single natural-language query string."""
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


# ── build index pipeline ───────────────────────────────────────────────────────

def build_index(
    db_name: str,
    store: str = None,
    force_reindex: bool = False,
    pg_config: dict = None,
) -> None:
    """
    Run the complete pipeline for ONE PostgreSQL database:
        process → embed → store

    Args:
        db_name:       PostgreSQL database name (e.g. "FYP").
        store:         "pgvector" or "pinecone". Defaults to VECTOR_STORE env var.
        force_reindex: If True, delete existing vectors before re-indexing.
        pg_config:     Optional custom PostgreSQL connection config dict.
                       If None, uses defaults from data_processing.py.
    """
    store = store or _default_store()
    backend = _get_backend(store)
    
    print("=" * 60)
    print(f"STEP 1/3 — Data Processing  [{db_name}]")
    print("=" * 60)

    # Pass optional pg_config if provided
    if pg_config:
        db_data = process_database(db_name, **pg_config)
    else:
        db_data = process_database(db_name)

    if not db_data:
        print(f"[ERROR] Could not process database '{db_name}'. Aborting.")
        return

    print("\n" + "=" * 60)
    print("STEP 2/3 — Building & Embedding Chunks")
    print("=" * 60)
    oai_client = get_openai_client()
    chunks     = build_chunks([db_data])
    chunks     = embed_chunks(chunks, client=oai_client)

    print("\n" + "=" * 60)
    print(f"STEP 3/3 — Storing in {store.upper()}")
    print("=" * 60)

    # ── pgvector ───────────────────────────────────────────────────────────
    if store == "pgvector":
        conn = backend.get_store(config=pg_config)

        if force_reindex:
            print(f"[INFO] force_reindex=True → deleting existing data for '{db_name}' ...")
            backend.delete_database(conn, db_name)

        backend.save_chunks(chunks, conn=conn)
        stats = backend.get_stats(conn)
        conn.close()

        print("\n" + "=" * 60)
        print("✅ Index build complete!")
        print(f"   Database  : {db_name}")
        print(f"   Chunks    : {len(chunks)}")
        print(f"   Store     : pgvector → {stats}")
        print("=" * 60)

    # ── pinecone ───────────────────────────────────────────────────────────
    elif store == "pinecone":
        index = backend.get_store()

        if force_reindex:
            print(f"[INFO] force_reindex=True → deleting existing data for '{db_name}' ...")
            backend.delete_database(index, db_name)

        backend.save_chunks(chunks, index=index)
        stats = backend.get_stats(index)

        print("\n" + "=" * 60)
        print("✅ Index build complete!")
        print(f"   Database  : {db_name}")
        print(f"   Chunks    : {len(chunks)}")
        print(f"   Store     : Pinecone → {stats}")
        print("=" * 60)


# ── retrieve ───────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    db_name: Optional[str] = None,
    top_k_tables: int = 3,
    store: str = None,
    oai_client: Optional[OpenAI] = None,
    store_conn=None,
) -> dict:
    """
    Given a natural-language query, retrieve the most relevant table schemas
    (and KB entries in the future) from the chosen vector store.

    Args:
        query:         Natural language question from the user.
        db_name:       Filter results to this database only (optional).
        top_k_tables:  Number of table chunks to return.
        store:         "pgvector" or "pinecone". Defaults to VECTOR_STORE env var.
        oai_client:    Reuse an existing OpenAI client (avoids re-init cost).
        store_conn:    Reuse an existing store connection (pgvector conn or
                       Pinecone index). If None, a new one is created.

    Returns:
        {
            "tables":         [ { chunk_id, db_name, table_name, text, distance/score }, ... ],
            "knowledge_base": [],    # empty until KB is implemented
            "prompt_context": str,   # ready-to-use context block for LLM prompt
        }
    """
    store   = store or _default_store()
    backend = _get_backend(store)

    # Embed the query
    if oai_client is None:
        oai_client = get_openai_client()
    query_embedding = _embed_query(query, oai_client)

    # ── pgvector retrieve ──────────────────────────────────────────────────
    if store == "pgvector":
        conn = store_conn or backend.get_store()
        raw  = backend.retrieve(
            query_embedding = query_embedding,
            db_name         = db_name,
            top_k_tables    = top_k_tables,
            conn            = conn,
        )
        # Normalise field names → always use "text" and "distance"
        tables = []
        for r in raw["tables"]:
            tables.append({
                "chunk_id":   r["chunk_id"],
                "db_name":    r["db_name"],
                "table_name": r["table_name"],
                "text":       r["content"],          # pgvector uses "content"
                "distance":   r["distance"],         # lower = more similar
            })

    # ── pinecone retrieve ──────────────────────────────────────────────────
    elif store == "pinecone":
        index = store_conn or backend.get_store()
        raw   = backend.retrieve(
            query_embedding = query_embedding,
            db_name         = db_name,
            top_k_tables    = top_k_tables,
            index           = index,
        )
        # Normalise field names → always use "text" and "distance"
        tables = []
        for r in raw["tables"]:
            tables.append({
                "chunk_id":   r["chunk_id"],
                "db_name":    r["db_name"],
                "table_name": r["table_name"],
                "text":       r["text"],
                "distance":   1 - r["score"],        # convert cosine similarity → distance
            })

    # ── [FUTURE] KB results ────────────────────────────────────────────────
    # kb_results = raw.get("knowledge_base", [])

    # Build prompt context string
    context_parts = ["### Relevant Table Schemas\n"]
    for item in tables:
        context_parts.append(item["text"])
        context_parts.append("")

    # ── [FUTURE] KB context ────────────────────────────────────────────────
    # if kb_results:
    #     context_parts.append("### Relevant Knowledge Base\n")
    #     for item in kb_results:
    #         context_parts.append(item["text"])
    #         context_parts.append("")

    prompt_context = "\n".join(context_parts).strip()

    return {
        "tables":         tables,
        "knowledge_base": [],       # populated when KB is implemented
        "prompt_context": prompt_context,
    }

def execute_query(sql: str) -> list[dict]:
    """
    Execute a SQL query against the PostgreSQL database and return results.

    Args:
        sql:       The SQL query string to execute.
        pg_config: PostgreSQL connection config dict (host, port, user, password).
    Returns:
        List of dicts representing query results (column names as keys).
    """ 
    backend = _get_backend("pgvector")
    conn  = backend.get_store()
    try:
        return backend.run_query(conn, sql)
    except Exception as e:
        print(f"Error executing query: {e}")
        raise

# ── prompt builder ─────────────────────────────────────────────────────────────

def build_sql_prompt(user_query: str, context: dict) -> str:
    """
    Assemble a complete Text-to-SQL prompt from retrieved context.

    Args:
        user_query: The natural language question from the user.
        context:    Output dict from retrieve().

    Returns:
        A formatted prompt string ready to send to your LLM.
    """
    prompt = f"""You are an expert SQL assistant. Use ONLY the schema provided below to write a correct PostgreSQL SQL query.

    {context['prompt_context']}

    ### User Question
    {user_query}

    ### Instructions
    - Use only the tables and columns shown in the schema above.
    - Match column names exactly as shown.
    - Do not invent tables or columns that are not listed.
    - Return ONLY the SQL query with no explanation or markdown.

    ### SQL Query
    """
    return prompt



# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Indexing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--store",
        default=_default_store(),
        choices=SUPPORTED_STORES,
        help="Vector store backend (default: pgvector, or set VECTOR_STORE env var)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── build subcommand ───────────────────────────────────────────────────
    build_parser = subparsers.add_parser("build", help="Build the vector index")
    build_parser.add_argument(
        "--db", required=True,
        help="PostgreSQL database name to process (e.g. FYP)"
    )
    build_parser.add_argument(
        "--force", action="store_true",
        help="Delete existing vectors for this DB before rebuilding"
    )

    # ── query subcommand ───────────────────────────────────────────────────
    query_parser = subparsers.add_parser("query", help="Query the vector index")
    query_parser.add_argument("query", help="Natural language question")
    query_parser.add_argument(
        "--db", default=None,
        help="Filter results to this database name (optional)"
    )
    query_parser.add_argument("--top-k-tables", type=int, default=3)
    query_parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the full assembled LLM prompt"
    )

    # ── stats subcommand ───────────────────────────────────────────────────
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")

    args = parser.parse_args()
    store = args.store
    print(f"[INFO] Using vector store: {args}")
    # ── build ──────────────────────────────────────────────────────────────
    if args.command == "build":
        print(f"\n[INFO] Building index for database '{args.db}' using store '{store}' ...")
        build_index(
            db_name       = args.db,
            store         = store,
            force_reindex = args.force,
        )

    # ── query ──────────────────────────────────────────────────────────────
    elif args.command == "query":
        results = retrieve(
            query         = args.query,
            db_name       = args.db,
            top_k_tables  = args.top_k_tables,
            store         = store,
        )
        
        

        print(f"\n── Retrieved Tables [{store}] ──")
        for r in results["tables"]:
            dist_label = "dist" if store == "pgvector" else "sim"
            print(f"  [{r['distance']:.4f} {dist_label}] "
                  f"{r['table_name']} ({r['db_name']})")
        
        if args.show_prompt:
            prompt = build_sql_prompt(args.query, results)
            print("\n── LLM Prompt ──\n")
            print(prompt)

    # ── stats ──────────────────────────────────────────────────────────────
    elif args.command == "stats":
        backend = _get_backend(store)
        if store == "pgvector":
            conn  = backend.get_store()
            stats = backend.get_stats(conn)
            dbs   = backend.list_databases(conn)
            tables = backend.list_tables(conn, "FYP")
            conn.close()
            print(f"\n── pgvector stats ──")
            print(f"  {stats}")
            print(f"  Databases indexed: {dbs}")
            print(f"  Tables indexed: {tables}")
        elif store == "pinecone":
            index = backend.get_store()
            stats = backend.get_stats(index)
            print(f"\n── Pinecone stats ──")
            print(f"  {stats}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()