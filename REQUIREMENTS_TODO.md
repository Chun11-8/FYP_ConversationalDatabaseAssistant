# Conversational Database Assistant — Remaining Requirements

> Gap analysis of the current repo against `FYPWorkPlan(System Design).pdf` and the
> system workflow diagram. Generated 2026-06-13.

## Legend
- ✅ Done · ⚠️ Partial / buggy · ❌ Not started / empty stub

---

## 1. Current State Summary

| Layer (from spec) | Status | Where |
|---|---|---|
| Orchestration (LangGraph) | ⚠️ | `backend/agents/db_agent.py` — graph works; history injection bug |
| Intent classification | ⚠️ | passes literal `"history_str"` (db_agent.py:92) |
| Agentic RAG — schema vectors | ✅ | `retrieval/indexing.py`, `retrieval/pgvector_store.py` |
| Agentic RAG — Knowledge Graph | ❌ | spec wants LlamaIndex `PropertyGraphIndex`; not present |
| SQL Code Agent (top-K + rerank) | ✅ | `agents/sql_code_agent.py` |
| Validation (syntax/safety/semantic) | ✅ | `agents/sql_validator_agent.py` |
| Response Optimizer | ✅ | `pipelines/postprocessing.py` |
| Session memory | ⚠️ | in-RAM dict only — `memory/session_memory.py` |
| Semantic cache (few-shot pairs) | ❌ | `few_shot_examples` hardcoded `[]` |
| Feedback collection | ❌ | not built |
| Business Layer (HTTP API) | ❌ | no server exists |
| Frontend ↔ backend | ❌ | `frontend/src/App.tsx` fully mocked |
| Schema Versioning Layer | ❌ | `agents/schema_versioning_agent.py` calls non-existent methods |
| Tools / Connectivity Layer | ❌ | all `tools/*.py` empty |
| User isolation / roles | ❌ | `memory/user_profiles.json` empty |
| Config loader | ❌ | `config/config.yaml` empty |
| Tests | ❌ | all `tests/*.py` empty |

---

## 2. Remaining Requirements

### R1 — Business Layer: HTTP API Server  *(priority 1, blocker)*
- No FastAPI/Flask server exists; orchestration only runs from a `__main__` block.
- Build a FastAPI app exposing:
  - `POST /chat` — `{session_id, message}` → `{response, sql, intent}`
  - `POST /session` — create a new session, return `session_id`
  - `GET /history/{session_id}` — return turn history
  - `GET /download/{filename}` — serve the >500-row CSV from `outputs/`
- CORS enabled for the Vite dev origin.
- Instantiate `OrchestrationAgent` once at startup from the config loader (R10).

### R2 — Frontend ↔ Backend Integration  *(priority 2)*
- Replace mocked `handleSend` / `addBotMessage` / `SIMULATED_DELAY` in `App.tsx` with real `fetch` to the API.
- Render assistant markdown (tables, ```sql``` blocks) — add a markdown renderer.
- Wire **New Chat** → `POST /session`; sidebar history → `GET /history`.
- Implement the MCQ clarification flow against the backend `AMBIGUOUS` intent (currently MCQ is hardcoded).
- Add a download link/button for CSV responses.

### R3 — Memory: fix history + persistence  *(priority 3)*
- **Bug:** intent classifier injects the literal string `"history_str"` (db_agent.py:92). Format and inject real history.
- Move `SessionMemory` from in-RAM dict → PostgreSQL tables (`sessions`, `turns`) so returning users reload past sessions (addresses the spec's stated "Session & Memory Persistence" limitation).
- Implement **short-term memory summarization** to cap context size (spec requirement).

### R4 — Memory: Semantic Cache / few-shot  *(priority 4, core RAG feature)*
- `few_shot_examples` is always `[]`. Implement the spec's "Semantic Retrieval":
  - On each successful, validated query, embed the `(user_query, SQL)` pair and store in pgvector (the commented-out KB table in `pgvector_store.py`).
  - At generation time, retrieve top-K similar past pairs and inject as few-shot examples.
- Add a **Query Interactions** physical-text log table (diagram's "POSTGres DB write").

### R5 — Feedback Collection  *(priority 5)*
- UI: thumbs up/down on each assistant response.
- API: `POST /feedback` — `{turn_id, rating}`.
- Store ratings; gate whether a query→SQL pair is written to the semantic cache on approval (diagram's "Approve? → store pair").

### R6 — Schema Versioning Layer  *(priority 6)*
- The agent calls `self.retriever.db.get_schema_diff()`, `embed_table()`, `update_table_embedding()`, `remove_table_embedding()`, `persist_hash()` — **none exist.** Implement them.
- **Schema Changes:** hash `information_schema`, compare to stored hash, extract structured diff.
- **Change Classifier:** Additive (append embeddings) / Modified (lower confidence on pairs) / Breaking (archive pairs, drop vector chunks + KG nodes).
- **Knowledge Update:** surgical re-embed of affected tables only.
- **Schema Version Store:** persist hash + timestamp + diff log each cycle.
- **Admin Hook:** `POST /admin/schema-sync` (admin-only) to trigger it.

### R7 — Tools / Connectivity Layer  *(priority 8)*
- `tools/*.py` empty. Build:
  - A **web search** tool so `GENERAL` intent answers DB how-to questions via search instead of plain LLM (`handle_general`).
  - Tool registry / dispatcher the orchestrator can route to (diagram's "API Registry / Data Transformation / Tool").
  - The flight/hotel/weather stubs appear to be scaffold from a template — confirm whether they belong; remove if not.

### R8 — User Isolation & Roles  *(priority 6)*
- Implement 2 roles (admin / user) per spec.
- Group-based memory isolation (not per-user).
- Admin-only access to the schema-versioning trigger (R6).
- Populate `user_profiles.json` or a `users` table + minimal auth.

### R9 — Response Optimizer guardrails  *(priority 8)*
- Wire an explicit SQL-injection / output guardrail at the response layer.
- Summarize over-long `GENERAL` answers (spec mentions response-layer summarization).

### R10 — Config, Secrets, Tests  *(priority 1 for config; ongoing)*
- `config/config.yaml` is empty but code reads `config["llm"]`/`config["db"]`. Build a loader (yaml + `secrets.env`) and stop hardcoding config in `__main__` blocks.
- **Security:** a live OpenAI API key is hardcoded in `retrieval/indexing.py` (~line 394). Revoke it and move to env.
- `requirements.txt`: `panda` → `pandas`; add `llama-index` (if R-KG chosen), `fastapi`, `uvicorn`, `pyyaml`.
- Fill empty `tests/*.py` with at least smoke tests per agent.

### R-KG (DEFERRED) — Knowledge Graph via LlamaIndex
- Spec mandates `PropertyGraphIndex` / `SQLTableRetriever` / `NLSQLTableQueryEngine` for multi-table FK join-path resolution. Current code is 100% LangChain with no `llama-index`.
- **Decision pending:** keep pgvector-only RAG, or add a LlamaIndex KG module on top. Listed separately so it can be scheduled independently.

---

## 3. Suggested Build Order
1. R10 (config + secret cleanup) → R1 (API) → R2 (frontend wiring)
2. R3 (history fix + persistence) → R4 (semantic cache) → R5 (feedback)
3. R8 (roles) → R6 (schema versioning + admin hook)
4. R7 (tools/web search) → R9 (guardrails) → R10 (tests)
5. R-KG (when decided)
