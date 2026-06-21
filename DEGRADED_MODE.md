# Degraded Mode & Minimal Executable Components (Issue #13)

This document describes exactly which components must be running for end-to-end
operation, and how the system degrades when an optional component is missing.
The goal is **no silent failures**: a missing component produces a clear log
warning and a usable (if reduced) experience rather than a crash.

---

## Mandatory components

| Component | How to start | What breaks without it |
|---|---|---|
| **OpenAI API key** | `OPENAI_API_KEY` in `backend/config/secrets.env` | All LLM nodes degrade. Intent classification falls back to `AMBIGUOUS`, SQL generation routes to the graceful fallback message, GENERAL answers return a "temporarily unable" message. The server still responds (no 500). |
| **FastAPI server** | `uvicorn backend.api.main:app --reload --port 8000` | Frontend has no backend; it shows the error banner. |
| **PostgreSQL 17 + pgvector** | `docker compose -f backend/config/compose.yml up -d` | `/health` reports `db_connected: false`. SQL execution, persistent memory, semantic cache, and schema retrieval are unavailable, but `/health` and `/chat` still respond. |
| **Schema vector index** | `python -m backend.retrieval.indexing build --db FYP` | `/health` reports `vector_index: false`. `retrieve_schema` returns empty context → `fallback_handler` runs a broader scan; if still empty the user is asked to clarify. |

---

## Optional components (degrade gracefully)

| Component | Impact when missing | Fallback behaviour |
|---|---|---|
| **Session persistence** (`sessions`/`turns` tables) | Memory lost on restart | `SessionMemory` falls back to in-RAM `_RamBackend` (`is_persistent == False`). |
| **Semantic cache** (`query_sql_pairs`) | No few-shot examples injected | `SemanticCache(db_config=None)` → `is_enabled == False`; every method is a no-op (`retrieve_few_shots` → `[]`, `save_pair`/`log_interaction`/`add_feedback` → no-op). |
| **pgvector extension** | Vector search fails | `SemanticCache` constructor catches the error and disables itself. |
| **`outputs/` directory** | Large CSV results | Created on startup and re-created on demand in `ResponseOptimizer._format_file` (belt-and-suspenders). |
| **Custom per-request API key** | Uses the system default key | `agent.run(api_key=None)` silently uses the system key. |

---

## Health endpoint

`GET /health` returns HTTP 200 even when PostgreSQL is down, so liveness checks
never depend on the database:

```json
{
  "status": "ok",
  "db_connected": true,
  "vector_index": true,
  "cache_enabled": true,
  "memory_persistent": true
}
```

Each boolean reports the live availability of an optional component. A value of
`false` means that capability is degraded but the service is still up.

---

## Lite mode (local development without Docker)

You can run the backend with **no PostgreSQL and no vector index** for quick UI
or routing work:

1. Set only `OPENAI_API_KEY` and a dummy `PG_PASSWORD` in `secrets.env`.
2. Start the API: `uvicorn backend.api.main:app --reload --port 8000`.
3. `SessionMemory` and `SemanticCache` automatically fall back to RAM / disabled.
4. `/health` will show `db_connected: false`, `vector_index: false`,
   `memory_persistent: false`, `cache_enabled: false`.

In lite mode, READ queries can't execute SQL (no DB), but intent
classification, the GENERAL/web-search path, and the full frontend remain
usable. Conversations live only in process memory and are lost on restart.

---

## Failure-mode coverage

| Scenario | Behaviour |
|---|---|
| DB unreachable at startup | ✅ `_check_db()` catches; `/health` → `db_connected:false` |
| Vector index empty/absent | ✅ `_check_vector_index()` → `vector_index:false`; `fallback_handler` broadens scan |
| Memory DB unreachable | ✅ Falls back to `_RamBackend` |
| Cache DB unreachable | ✅ `SemanticCache.is_enabled=False`, all ops no-op |
| Schema retrieval fails | ✅ `retrieve_schema` catches, sets `fallback_level=1` |
| SQL validation fails | ✅ `fallback_handler` retries with broader context (max 2 rounds) |
| **LLM call fails (classify/generate/general)** | ✅ Each LLM node is wrapped in try/except → safe response, no 500 |
| `outputs/` dir missing at write time | ✅ Re-created in `_format_file` |
