# GitHub Issues — Detailed Requirement Breakdown

Each `##` section below is one GitHub issue. Copy the title into the issue title, the
body into the description. Checkboxes are sub-tasks you can split into smaller issues if
needed. "Improvements" are optional enhancements beyond the minimum.

Suggested labels are listed per issue. Recommended milestones:
- **M1 – Make it run end-to-end** (#0, #1, #2, #3)
- **M2 – Core RAG & learning** (#4, #5)
- **M3 – Admin & safety** (#6, #7, #8)
- **M4 – Tools, guardrails, polish** (#9, #10, #11)
- **M5 – Optional KG** (#12)

---

## Issue #0 — Security & config hardening (do first)
**Labels:** `security`, `chore`, `priority:critical`

### Why
Live secrets are committed and config is hardcoded in `__main__` blocks. This must be
fixed before anything else touches the repo history further.

### Sub-tasks
- [ ] Revoke the OpenAI key currently hardcoded in `backend/retrieval/indexing.py` (~line 394) and the Postgres password in `backend/retrieval/pgvector_store.py` (line 31, `Toyo20001108!`).
- [ ] Delete both hardcoded values; read them only from `backend/config/secrets.env` via `python-dotenv`.
- [ ] Add `backend/config/secrets.env` to `.gitignore` (verify it isn't already tracked; if it is, `git rm --cached`).
- [ ] Scrub secrets from git history (`git filter-repo` or BFG) **or**, if acceptable for an FYP, document that old keys are dead.
- [ ] Build `load_config()` in `backend/config/config_loader.py` that merges `config.yaml` + env vars into the dict shape the code expects: `config["llm"]["model"]`, `config["llm"]["api_key"]`, `config["db"]["dialect"]`, `config["db"]` connection fields.
- [ ] Populate `backend/config/config.yaml` with non-secret defaults (model name, dialect, db host/port/name, top_k values, table/inline limits).
- [ ] Replace hardcoded config dicts in `db_agent.py` and `indexing.py` `__main__` blocks with `load_config()`.
- [ ] Fix `requirements.txt`: `panda` → `pandas`; add `fastapi`, `uvicorn[standard]`, `pyyaml`, `python-multipart`. Pin versions.

### Improvements
- Use `pydantic-settings` (`BaseSettings`) instead of a hand-rolled loader — gives typed config + validation + clear error when a var is missing.
- Centralize the Postgres `DB_CONFIG` (currently duplicated in `pgvector_store.py`) so there's one source of truth.

### Acceptance criteria
- Repo contains no secrets in tracked files. `grep -ri "sk-" backend/` and the password string return nothing.
- `from backend.config.config_loader import load_config; load_config()` returns a valid dict and the agent runs.

---

## Issue #1 — Business Layer: FastAPI server
**Labels:** `backend`, `feature`, `priority:critical`

### Why
No HTTP server exists — `OrchestrationAgent` only runs from a `__main__` block. This is
the blocker for turning the project into a website.

### Sub-tasks
- [ ] Create `backend/api/main.py` with a FastAPI app; instantiate `OrchestrationAgent` **once** at startup (FastAPI `lifespan`) using `load_config()`.
- [ ] `POST /session` → generate `session_id` (uuid4), persist an empty session, return `{session_id}`.
- [ ] `POST /chat` → body `{session_id, message}`; call `agent.run(session_id, message)`; return `{response, sql, intent, validation}`. Add `try/except` → HTTP 500 with a clean error body.
- [ ] `GET /history/{session_id}` → return turn list from memory.
- [ ] `GET /download/{filename}` → `FileResponse` from `outputs/`, with a path-traversal guard (reject `..`, only serve files inside `outputs/`).
- [ ] Add Pydantic request/response models (`ChatRequest`, `ChatResponse`) so the schema is typed and self-documented at `/docs`.
- [ ] Enable CORS for `http://localhost:5173`.
- [ ] Return structured output: have `agent.run` return a dict (not just a string) so the API can expose `sql`, `intent`, row count, and any CSV path separately. (Requires a small refactor of `run()`.)
- [ ] Add `GET /health` returning `{status, db_connected}`.
- [ ] Document `uvicorn backend.api.main:app --reload --port 8000` in the README.

### Improvements
- **Streaming:** add `POST /chat/stream` using Server-Sent Events so the frontend can show tokens / pipeline-stage progress ("classifying → retrieving → generating → validating").
- Make `OrchestrationAgent.run` async or run the graph in a threadpool (`run_in_executor`) so the event loop isn't blocked by LLM calls.
- Add request-id logging middleware to trace a query through all layers.
- Rate-limit `/chat` (e.g. `slowapi`) to protect the OpenAI bill.

### Acceptance criteria
- `curl -X POST /chat` with a real session returns a real answer + the SQL used.
- `/docs` renders all endpoints.

### Depends on
#0

---

## Issue #2 — Frontend ↔ Backend integration
**Labels:** `frontend`, `feature`, `priority:high`

### Why
`frontend/src/App.tsx` is 100% mocked (`SIMULATED_DELAY`, hardcoded replies, fake MCQ).
It never calls the backend.

### Sub-tasks
- [ ] Add a typed API client `frontend/src/api/client.ts` (`createSession`, `sendChat`, `getHistory`) with a configurable `VITE_API_BASE_URL`.
- [ ] On first load, call `POST /session`, store `session_id` in state/localStorage.
- [ ] Replace `handleSend` mock with a real `POST /chat`; append the actual response.
- [ ] Replace the `setTimeout`/`SIMULATED_DELAY` typing indicator with one tied to the real in-flight request.
- [ ] Install `react-markdown` + `remark-gfm`; render assistant messages as markdown so ```sql``` blocks and tables display. Add syntax highlighting (`rehype-highlight`).
- [ ] Add a `sql` field to the assistant message type and render it in a collapsible "SQL used" panel.
- [ ] Detect CSV-download responses and render a real download button hitting `/download/{filename}`.
- [ ] Wire **New Chat** → `POST /session` + clear messages.
- [ ] Replace the hardcoded `chatHistory` sidebar with real sessions (needs a `GET /sessions` endpoint — add to #1 or #3).
- [ ] Error UI: show a toast/inline error when the API returns 500.

### Improvements
- Persist the current `session_id` + message list in `localStorage` so a refresh doesn't lose the chat (bridges to #3's server persistence).
- Add an "Export conversation" button.
- Show the detected `intent` as a small badge (READ / GENERAL / WRITE_BLOCKED) for transparency.
- Extend `types.ts` `Message` union with a `table` and `sql` variant rather than overloading `text`.

### Acceptance criteria
- Typing a real DB question returns a real, markdown-rendered answer from Postgres.
- New Chat starts a fresh server session.

### Depends on
#1

---

## Issue #3 — Memory: fix history injection + Postgres persistence
**Labels:** `backend`, `bug`, `feature`, `priority:high`

### Why
Two problems: (1) a real bug breaks multi-turn context, (2) memory is in-RAM only so
sessions vanish on restart — which is exactly the "Session & Memory Persistence"
limitation the spec calls out.

### Sub-tasks — Bug fix
- [ ] In `db_agent.py::classify_intent`, the prompt contains the literal string `"history_str"` (line ~92) and the real `_format_history(...)` call is commented out. Restore it and inject formatted history.
- [ ] Verify `generate_sql` actually passes `history` through to `SQLCodeAgent` (it does receive `conversation_history[-4:]` — confirm format matches `_build_prompt`).
- [ ] Add a test: ask "show 5 players", then "now only the top 3" — the second query must use prior context.

### Sub-tasks — Persistence
- [ ] Create tables: `sessions(session_id, user_id/group_id, created_at, title)` and `turns(id, session_id, role, content, sql, valid, created_at)`.
- [ ] Rewrite `SessionMemory` to read/write Postgres while keeping the **same method signatures** (`add`, `get_history`, `get_last_sql`, `clear_session`, `active_sessions`) so `db_agent.py` is unchanged.
- [ ] Add `list_sessions(user_or_group)` for the sidebar.
- [ ] Auto-generate a session `title` from the first user message (truncate or LLM one-liner).
- [ ] Add `GET /sessions` and `GET /history/{session_id}` wiring in the API.

### Sub-tasks — Short-term summarization (spec requirement)
- [ ] When a session exceeds N turns, summarize the oldest turns into one rolling "context summary" instead of dropping them (current code just trims).
- [ ] Store the summary on the session row; prepend it to the history passed into the graph.

### Improvements
- Use SQLAlchemy + Alembic for schema/migrations instead of raw `CREATE TABLE` strings scattered across modules.
- Add a `last_sql` resolver so pronoun references ("filter *that* by year") can re-use the previous query as a CTE.
- Cache hot sessions in memory (write-through) to avoid a DB round-trip every turn.

### Acceptance criteria
- Restarting the server preserves prior conversations.
- The two-turn context test passes.

### Depends on
#1

---

## Issue #4 — Memory: Semantic cache (few-shot query→SQL pairs)
**Labels:** `backend`, `feature`, `rag`, `priority:high`

### Why
`few_shot_examples` is hardcoded to `[]`. The spec's headline "Semantic Retrieval /
semantic cache" feature (embed past query→SQL pairs, inject top-K as few-shot) does not
exist. The pgvector KB table is commented out in `pgvector_store.py`.

### Sub-tasks — Storage
- [ ] Add a `query_sql_pairs` table: `id, group_id, user_query, sql, embedding vector(1536), confidence REAL, status TEXT ('active'|'archived'|'low_conf'), created_at`. (Model it on the commented `kb_embeddings` block.)
- [ ] Add an ivfflat cosine index on `embedding`.
- [ ] Add `save_pair(user_query, sql, group_id)` and `retrieve_pairs(query_embedding, top_k, group_id)` in a new `backend/memory/semantic_cache.py` (or extend `pgvector_store.py`).

### Sub-tasks — Write path
- [ ] After a query is validated **and** executed successfully in the graph, embed `(user_query, best_sql)` and upsert a pair.
- [ ] Deduplicate near-identical pairs (cosine distance threshold) so the cache doesn't bloat.

### Sub-tasks — Read path
- [ ] Add `retrieve_few_shots(query, top_k=3)`; call it in `retrieve_schema` (or a new node) and populate `state["few_shot_examples"]`.
- [ ] Format pairs into the `<few_shot_examples>` block already present in `SQLCodeAgent._build_prompt`.

### Sub-tasks — Logging
- [ ] Add a `query_interactions` log table (diagram's "POSTGres DB write / physical text"): every turn's query, sql, response, valid, latency.

### Improvements
- **Semantic cache hit shortcut:** if an incoming query is a near-exact match (distance < ε) to a high-confidence pair, optionally skip generation and reuse the cached SQL (huge latency/cost win) — gate behind a config flag.
- Weight few-shot retrieval by feedback score (from #5) so only "good" examples are injected.
- Add a cross-encoder rerank of retrieved pairs for quality before injection.
- Track and expose cache hit-rate as a metric.

### Acceptance criteria
- Asking a question similar to a past one visibly injects a relevant few-shot example (log it).
- Pairs accumulate in `query_sql_pairs` after successful queries.

### Depends on
#3

---

## Issue #5 — Feedback collection
**Labels:** `fullstack`, `feature`, `priority:medium`

### Why
The diagram's "Feedback Collection → Approve? → store pair" loop is entirely absent.
Needed to make the semantic cache self-improving.

### Sub-tasks
- [ ] `feedback` table: `id, turn_id, rating ('up'|'down'), comment NULL, created_at`.
- [ ] `POST /feedback` endpoint `{turn_id, rating, comment?}`.
- [ ] Frontend: thumbs up/down on each assistant message; optimistic UI + call.
- [ ] Gate the semantic-cache write (#4): only persist/keep a pair when not down-voted (or only when up-voted, configurable).
- [ ] Down-voted pairs → set `status='low_conf'` or archive so they're never retrieved as few-shot.

### Improvements
- Optional free-text "what was wrong?" on down-vote → store for error analysis.
- Auto-eval fallback: when no human rating, use an LLM judge to score the answer (the diagram mentions "user-eval / auto-eval").
- Admin dashboard showing thumbs-down queries to find systematic failures.

### Acceptance criteria
- Rating a message stores a row and influences whether its pair is reused.

### Depends on
#4

---

## Issue #6 — User isolation & roles (admin/user)
**Labels:** `backend`, `feature`, `auth`, `priority:medium`

### Why
Spec requires exactly 2 roles with **group-based** memory isolation (not per-user), and
admin-only access to schema versioning. `user_profiles.json` is empty.

### Sub-tasks
- [ ] `users` table: `id, username, password_hash, role ('admin'|'user'), group_id`.
- [ ] Minimal auth: `POST /login` returning a token (JWT or signed session). For an FYP a simple bearer token is acceptable.
- [ ] FastAPI dependency `get_current_user` + `require_admin` guard.
- [ ] Scope `sessions`, `query_sql_pairs`, `feedback` by `group_id`; every memory read/write filters on it.
- [ ] Frontend: login screen; store token; attach `Authorization` header to all API calls; hide admin UI for non-admins.
- [ ] Seed one admin + one user account (migration/script).

### Improvements
- Use `passlib[bcrypt]` for hashing; never store plaintext.
- Add per-group config (e.g. which database a group can query).
- Audit log of admin actions (ties into #7's audit requirement).

### Acceptance criteria
- A user in group A cannot see group B's sessions or cache.
- Only admins can call the schema-sync endpoint.

### Depends on
#3

---

## Issue #7 — Schema Versioning Layer + Admin Hook
**Labels:** `backend`, `feature`, `rag`, `priority:medium`

### Why
`backend/agents/schema_versioning_agent.py` calls methods that don't exist
(`retriever.db.get_schema_diff`, `embed_table`, `update_table_embedding`,
`remove_table_embedding`, `persist_hash`). The whole layer is non-functional.

### Sub-tasks — Schema Changes (detection)
- [ ] `get_current_schema()` — query `information_schema.columns`/`tables` into a normalized dict.
- [ ] `hash_schema(schema)` — deterministic hash (sorted JSON → sha256).
- [ ] `schema_version_store` table: `id, schema_hash, snapshot JSONB, diff_log JSONB, created_at`.
- [ ] `get_schema_diff(old_snapshot, new_snapshot)` → list of `{action: add|modify|drop, table, column?}`.

### Sub-tasks — Change Classifier
- [ ] Classify each diff entry:
  - **Additive** (new table/column) → append new embeddings only.
  - **Modified** (rename/type change) → lower `confidence` on affected `query_sql_pairs`, don't delete.
  - **Breaking** (drop) → archive pairs referencing the object; remove the table's vector chunk(s).

### Sub-tasks — Knowledge Update (surgical)
- [ ] `embed_table(table)` / `update_table_embedding(table)` / `remove_table_embedding(table)` operating on `schema_embeddings` for **only** affected tables (reuse `data_processing` + `embedding` for one table, not the whole DB).
- [ ] `persist_hash(hash, snapshot, diff_log)` → write a `schema_version_store` row.

### Sub-tasks — Admin Hook
- [ ] `POST /admin/schema-sync` (admin-only, #6): run the agent **after** confirming the DB change committed; run async (BackgroundTask) so a RAG-update failure can't roll back the DB.
- [ ] Capture audit metadata (who triggered, when, diff summary).
- [ ] Frontend admin button to trigger + show last sync result.

### Improvements
- Wrap each store's update so a partial failure (vector ok, KG fails) is logged and retryable rather than leaving stores inconsistent.
- Provide a dry-run mode returning the diff + classification without applying.
- Schedule an optional nightly drift check (compare current hash vs stored) that only *alerts* (keeps the event-driven design but catches out-of-band DDL).

### Acceptance criteria
- Adding a column then hitting `/admin/schema-sync` re-embeds only that table and writes a new version row; dropping a table archives its pairs.

### Depends on
#4, #6

---

## Issue #8 — (folded into #6/#7) — see those issues
*(User isolation and the admin trigger are covered by #6 and #7. Kept as a pointer so
numbering matches the requirements doc.)*

---

## Issue #9 — Tools / Connectivity Layer (web search for GENERAL intent)
**Labels:** `backend`, `feature`, `priority:medium`

### Why
All `tools/*.py` are empty. `handle_general` answers DB how-to questions with the bare
LLM; the spec/diagram want a web-search tool and a tool registry. The flight/hotel/
weather stubs look like leftover scaffold.

### Sub-tasks
- [ ] Define a `Tool` interface (name, description, `run(args)`), and a `ToolRegistry` the orchestrator can look up.
- [ ] Implement a `web_search` tool (Tavily / SerpAPI / Bing — pick one, key in env).
- [ ] Route `GENERAL` intent through `web_search` + synthesis instead of a bare LLM call in `handle_general`.
- [ ] Decide on the flight/hotel/weather stubs: confirm they belong to this project; if not, delete `tools/flight_api.py`, `hotel_api.py`, `weather_api.py` and their test.
- [ ] Add a "Data Transformation" helper (diagram) that normalizes tool output into the response shape.

### Improvements
- Let the LLM decide *whether* a web search is needed (function-calling / tool-choice) rather than always searching on GENERAL.
- Cache web-search results by query to cut latency/cost.
- Cite sources (URLs) in the GENERAL answer for trustworthiness.

### Acceptance criteria
- "How do I write a SELECT in Postgres?" returns a synthesized answer with sources, no SQL generated.

### Depends on
#1

---

## Issue #10 — Response Optimizer guardrails
**Labels:** `backend`, `feature`, `security`, `priority:low`

### Why
Spec's response-layer guardrails (injection re-check, summarize over-long answers) aren't
wired into orchestration beyond the validator.

### Sub-tasks
- [ ] Add an output guardrail step that re-scans the final response/SQL for injection patterns before returning.
- [ ] Summarize over-long `GENERAL`/web answers to a configured max length.
- [ ] Enforce the `LIMIT`/row-cap policy at execution time, not just in the prompt (inject/append a `LIMIT` if missing).
- [ ] Centralize the safety regexes (currently in `sql_validator_agent.py`) so the optimizer and validator share one source.

### Improvements
- Add PII/column-level redaction policy per group (e.g. hide `password_hash`, `email`).
- Token-budget the schema_context + few-shots so prompts don't blow the context window.

### Acceptance criteria
- A query without `LIMIT` is capped automatically; over-long answers are summarized.

### Depends on
#1

---

## Issue #11 — Config validation, tests, CI
**Labels:** `chore`, `tests`, `priority:medium`

### Why
All `tests/*.py` are empty; `llm_pipeline.py` and `retriever.py` are stubs.

### Sub-tasks
- [ ] `tests/test_tools.py` (or rename): validator safety — INSERT/UPDATE/DELETE/DROP must be blocked; injection patterns rejected.
- [ ] `tests/test_agent.py`: mock the LLM, assert the graph routes GENERAL vs READ vs WRITE_BLOCKED correctly.
- [ ] `tests/test_pipeline.py`: `ResponseOptimizer` inline/table/CSV branch selection by row count.
- [ ] API test: `/chat` happy path with a mocked agent.
- [ ] Decide the fate of empty `pipelines/llm_pipeline.py` and the dataset-snippet `retrieval/retriever.py` — implement or delete.
- [ ] Add `pytest` to requirements; a `make test` / GitHub Actions workflow running lint + tests.

### Improvements
- Add a tiny seeded test Postgres (docker-compose `compose.yml` already exists) for integration tests.
- Add an eval harness over a handful of NL→SQL pairs to track accuracy as you change prompts.

### Acceptance criteria
- `pytest` runs green; CI passes on push.

---

## Issue #12 — (Optional, deferred) Knowledge Graph via LlamaIndex
**Labels:** `backend`, `feature`, `rag`, `research`, `priority:low`

### Why
Spec mandates LlamaIndex (`PropertyGraphIndex`, `SQLTableRetriever`,
`NLSQLTableQueryEngine`) for multi-table FK join-path resolution. Code is currently all
LangChain/pgvector with no `llama-index` dependency. **Decision pending** — keep as an
additive module.

### Sub-tasks
- [ ] Add `llama-index` deps; decide coexistence boundary with the existing pgvector RAG.
- [ ] Build a `PropertyGraphIndex` of tables (nodes) + FK relationships (edges) from `information_schema`.
- [ ] On multi-table queries, traverse the KG to resolve join paths and enrich `schema_context`.
- [ ] Wire into `db_agent.retrieve_schema` as an *additive* enrichment (keep pgvector retrieval).
- [ ] Keep the KG in sync via the Schema Versioning Layer (#7) — add KG node/edge updates there.

### Improvements
- A/B measure SQL accuracy on multi-join queries with vs without the KG to justify the added complexity.
- If LlamaIndex proves heavy, a lightweight alternative is to store FK edges in a `schema_fk_edges` table and do BFS join-path resolution yourself — same benefit, fewer dependencies.

### Acceptance criteria
- A 3-table join question produces correct JOINs using KG-resolved paths.

### Depends on
#7
