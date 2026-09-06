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
from backend.retrieval.db_catalog import load_catalog
from backend.retrieval.embedding import build_chunks, embed_chunks, get_openai_client, EMBED_MODEL
import logging as _logging
_logger = _logging.getLogger(__name__)

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


def _print_scope_summary(chunks: list[dict]) -> None:
    """
    GH#81: show which logical database each table was tagged with, so a build
    makes the retrieval scopes visible instead of silently labelling
    everything with the --db name.
    """
    by_db: dict[str, list[str]] = {}
    for c in chunks:
        if c.get("table_name"):
            by_db.setdefault(c.get("db_name", "?"), []).append(c["table_name"])
    if len(by_db) <= 1:
        return
    print(f"   Scopes    : {len(by_db)} logical databases "
          "(set retrieval.active_db to search just one)")
    for db in sorted(by_db):
        tables = sorted(by_db[db])
        shown = ", ".join(tables[:6]) + (" …" if len(tables) > 6 else "")
        print(f"     - {db:<28} {len(tables):>3} tables  [{shown}]")


# ── build index pipeline ───────────────────────────────────────────────────────

def build_index(
    db_name: str,
    store: str = None,
    force_reindex: bool = False,
    pg_config: dict = None,
    tables_json: str = None,
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
        tables_json:   GH#81 — BIRD tables.json used to tag each table with the
                       logical database (db_id) that owns it. None searches the
                       default locations; nothing found leaves every table
                       tagged with *db_name*.
    """
    catalog = load_catalog(tables_json)
    if catalog:
        print(f"[INFO] Table catalog: {len(catalog)} tables across "
              f"{len(catalog.databases())} logical databases "
              f"({catalog.source})")
    store = store or _default_store()
    backend = _get_backend(store)
    
    print("=" * 60)
    print(f"STEP 1/3 — Data Processing  [{db_name}]")
    print("=" * 60)

    # Pass optional pg_config if provided
    if pg_config:
        db_data = process_database(db_name, catalog=catalog, **pg_config)
    else:
        db_data = process_database(db_name, catalog=catalog)

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
        # GH#79: drop chunks left behind by an index built before the
        # assistant's own tables were excluded (save_chunks upserts, so a
        # plain rebuild would never remove them).
        backend.purge_internal_chunks(conn, db_name)
        # GH#81: same for tables that have just moved to their real db_id.
        backend.prune_relabelled_chunks(conn, chunks)
        stats = backend.get_stats(conn)
        conn.close()

        print("\n" + "=" * 60)
        print("✅ Index build complete!")
        print(f"   Database  : {db_name}")
        print(f"   Chunks    : {len(chunks)}")
        print(f"   Store     : pgvector → {stats}")
        _print_scope_summary(chunks)
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
    embedding: Optional[list[float]] = None,
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
        embedding:     Reuse a vector already computed for this query (GH#57) —
                       the caller must have produced it with the same
                       EMBED_MODEL. None → embed here.

    Returns:
        {
            "tables":         [ { chunk_id, db_name, table_name, text, distance/score }, ... ],
            "knowledge_base": [],    # empty until KB is implemented
            "prompt_context": str,   # ready-to-use context block for LLM prompt
        }
    """
    store   = store or _default_store()
    backend = _get_backend(store)

    # Embed the query — unless the caller already has the vector.
    if embedding is not None:
        query_embedding = embedding
    else:
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

    _logger.info(f"[RETRIEVE] {len(tables)} table(s) returned for query: {query[:60]}")
    for t in tables:
        _logger.info(f"[RETRIEVE]   dist={t['distance']:.4f}  table={t['table_name']}")

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

def execute_query(sql: str, timeout_ms: int | None = 15000) -> list[dict]:
    """
    Execute a USER SQL query against PostgreSQL and return results.

    Issue #25/GH#41 hardening: runs on the read-only app connection when
    PG_APP_USER is configured, and under a statement timeout (default 15 s)
    so a pathological query cannot hang the request thread.
    """
    _logger.info(f"[EXECUTE_QUERY] Running SQL (timeout={timeout_ms}ms):\n{sql}")
    backend = _get_backend("pgvector")
    conn = backend.get_query_conn()
    try:
        rows = backend.run_query(conn, sql, timeout_ms=timeout_ms)
        _logger.info(f"[EXECUTE_QUERY] Returned {len(rows)} row(s)")
        return rows
    except Exception as e:
        _logger.error(f"[EXECUTE_QUERY] Error: {e}")
        print(f"Error executing query: {e}")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ── Full-schema retrieval (GH#85) ─────────────────────────────────────────────

def fetch_full_schema(
    db_name: str, store: str = "pgvector", store_conn=None,
) -> Optional[dict]:
    """
    Every indexed table of ONE logical database, in :func:`retrieve`'s shape.

    This is what the BIRD harness does: `extract_sqlite_schema(db_path)` hands
    the generator the COMPLETE schema of the question's `db_id`, so a needed
    table can never be missing. Cosine top-k cannot make that promise — the
    table holding the filter ("who is in segment LAM?") often has low textual
    similarity to the question, and once it is absent the model quietly drops
    the constraint instead of failing.

    Returns None when there is no scope, the store has nothing for it, or the
    lookup fails — the caller falls back to cosine retrieval.

    ``distance`` is 0.0 on every chunk: these were not ranked, and a fake
    ranking would be read as one downstream.
    """
    if not db_name:
        return None
    backend = _get_backend(store)
    if not hasattr(backend, "list_tables"):
        return None                      # pinecone: no name lookup, use cosine
    conn = None
    try:
        conn = store_conn or backend.get_store()
        names = backend.list_tables(conn, db_name)
        if not names:
            return None
        chunks = backend.get_chunks_by_tables(conn, names, db_name)
        if not chunks:
            return None
        tables = [{
            "chunk_id":   c["chunk_id"],
            "db_name":    c["db_name"],
            "table_name": c["table_name"],
            "text":       c["content"],
            "distance":   0.0,
        } for c in chunks]
    except Exception as exc:
        _logger.warning(f"[FULL_SCHEMA] lookup failed for '{db_name}': {exc}")
        return None
    finally:
        if store_conn is None and conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    parts = ["### Relevant Table Schemas\n"]
    for t in tables:
        parts.append(t["text"])
        parts.append("")
    _logger.info(f"[FULL_SCHEMA] {len(tables)} table(s) for db '{db_name}'")
    return {
        "tables":         tables,
        "knowledge_base": [],
        "prompt_context": "\n".join(parts).strip(),
        "full_schema":    True,
    }


# ── Backward schema linking (GH#49 P4) ────────────────────────────────────────

def fetch_table_chunks(
    table_names: list[str], db_name: Optional[str] = None,
    store: str = "pgvector"
) -> list[dict]:
    """
    Direct name lookup of schema chunks (no embedding search) — used to recover
    tables the linker plan names but forward retrieval missed (RSL-SQL-style
    backward linking). Table names are matched case-insensitively against the
    indexed names. *db_name* None (GH#81) spans every retrieval scope.
    Returns [] on any failure — never raises.
    """
    if not table_names:
        return []
    backend = _get_backend(store)
    conn = None
    try:
        conn = backend.get_store()
        indexed = {t.casefold(): t for t in backend.list_tables(conn, db_name)}
        resolved = [indexed[n.casefold()] for n in table_names
                    if n and n.casefold() in indexed]
        if not resolved:
            return []
        return backend.get_chunks_by_tables(conn, resolved, db_name)
    except Exception as exc:
        _logger.warning(f"[BACKWARD_LINK] chunk fetch failed: {exc}")
        return []
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


# ── FK-aware retrieval expansion ──────────────────────────────────────────────

def expand_fk_tables(
    retrieve_result: dict,
    db_name: Optional[str] = None,
    max_hops: int = 2,
    store_conn=None,
    max_expansion: int = 5,
    max_schema_tokens: int = 4000,
) -> dict:
    """
    Expand a retrieve() result with FK-linked tables.

    After cosine retrieval picks the top-K tables by similarity, this function:
      1. Builds a bidirectional FK graph from information_schema (one cheap query).
      2. BFS-expands up to *max_hops* from the retrieved table frontier.
      3. Fetches schema chunks for the newly discovered tables.
      4. Returns an enriched dict with an updated prompt_context that has three
         sections:
           ### Relevant Table Schemas (Retrieved by Query)
           ### FK-Related Tables (Included via join relationships)
           ### Foreign Key Relationships

    Args:
        retrieve_result:   Output from retrieve() — must contain "tables" list.
        db_name:           Retrieval scope for schema_embeddings lookups;
                           None (GH#81) spans every logical database.
        max_hops:          FK traversal depth (default 2 so a middle "bridge"
                           table pulls in its far side, e.g. players→stats→teams).
        store_conn:        Reuse an existing pgvector connection (None = create one).
        max_expansion:     Cap on FK-added tables (closest hops win) to prevent
                           context explosion on star schemas.
        max_schema_tokens: Approximate token budget for the final prompt_context.
                           When exceeded, FK-expansion tables are dropped from the
                           END (farthest hop first); primary tables are never dropped.

    Returns:
        A copy of retrieve_result with keys added/updated:
            "fk_tables"      — list of expansion schema chunk dicts
            "fk_edges"       — list of FK edge dicts relevant to the expanded set
            "prompt_context" — enriched, ready-to-inject context string
    """
    import backend.retrieval.pgvector_store as pgstore
    import logging
    logger = logging.getLogger(__name__)

    conn = store_conn or pgstore.get_store()

    primary_table_names = {t["table_name"] for t in retrieve_result.get("tables", [])}

    if not primary_table_names:
        return {**retrieve_result, "fk_tables": [], "fk_edges": []}

    try:
        fk_graph, fk_edges = pgstore.get_fk_graph(conn)
    except Exception as exc:
        logger.warning(f"[FK_EXPAND] FK graph query failed: {exc}; skipping expansion.")
        return {**retrieve_result, "fk_tables": [], "fk_edges": []}

    # BFS from primary tables — hop-ordered so, under the expansion cap, the
    # closest (most join-relevant) tables survive.
    visited: set[str] = set(primary_table_names)
    frontier: set[str] = set(primary_table_names)
    expansion_ordered: list[str] = []
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for tbl in sorted(frontier):            # deterministic order
            for neighbour in sorted(fk_graph.get(tbl, set())):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
                    if len(expansion_ordered) < max_expansion:
                        expansion_ordered.append(neighbour)
        if len(expansion_ordered) >= max_expansion:
            logger.info(f"[FK_EXPAND] expansion cap ({max_expansion}) reached")
            break
        frontier = next_frontier

    expansion_names = set(expansion_ordered)

    if expansion_names:
        logger.info(f"[FK_EXPAND] Adding {len(expansion_names)} FK-linked table(s): {expansion_ordered}")

    # Fetch schema chunks for expansion tables, keeping hop order
    fk_chunks: list[dict] = []
    if expansion_names:
        try:
            fk_chunks = pgstore.get_chunks_by_tables(conn, expansion_ordered, db_name)
            hop_rank = {t: i for i, t in enumerate(expansion_ordered)}
            fk_chunks.sort(key=lambda c: hop_rank.get(c.get("table_name"), len(hop_rank)))
        except Exception as exc:
            logger.warning(f"[FK_EXPAND] chunk fetch failed: {exc}")

    from backend.agents.utils import estimate_tokens

    def _assemble(kept_chunks: list[dict]) -> tuple[str, list[dict]]:
        """Build the prompt_context for primary tables + the given FK chunks."""
        kept_names = primary_table_names | {c["table_name"] for c in kept_chunks}
        edges = [
            e for e in fk_edges
            if e["from_table"] in kept_names and e["to_table"] in kept_names
        ]
        parts: list[str] = []
        parts.append("### Relevant Table Schemas (Retrieved by Query)\n")
        for item in retrieve_result.get("tables", []):
            parts.append(item["text"])
            parts.append("")
        if kept_chunks:
            parts.append("### FK-Related Tables (Included via join relationships)\n")
            for chunk in kept_chunks:
                parts.append(chunk["content"])
                parts.append("")
        if edges:
            parts.append("### Foreign Key Relationships\n")
            seen_edges: set[tuple] = set()
            for e in edges:
                key = (e["from_table"], e["from_col"], e["to_table"], e["to_col"])
                if key not in seen_edges:
                    parts.append(
                        f"  {e['from_table']}.{e['from_col']} → {e['to_table']}.{e['to_col']}"
                    )
                    seen_edges.add(key)
        return "\n".join(parts).strip(), edges

    # Token budget: drop FK-expansion chunks from the END (farthest hop first)
    # until the context fits. Primary tables are never dropped.
    kept = list(fk_chunks)
    prompt_context, relevant_edges = _assemble(kept)
    while kept and estimate_tokens(prompt_context) > max_schema_tokens:
        dropped = kept.pop()
        logger.info(
            f"[FK_EXPAND] token budget ({max_schema_tokens}) exceeded; "
            f"dropping FK table '{dropped.get('table_name')}'"
        )
        prompt_context, relevant_edges = _assemble(kept)

    return {
        **retrieve_result,
        "fk_tables":      kept,
        "fk_edges":       relevant_edges,
        "prompt_context": prompt_context,
    }


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
    build_parser.add_argument(
        "--tables-json", default=None,
        help="BIRD tables.json mapping each table to its owning db_id (GH#81).\n"
             "Default: $BIRD_TABLES_JSON, then outputs/dev_tables.json."
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
            tables_json   = args.tables_json,
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