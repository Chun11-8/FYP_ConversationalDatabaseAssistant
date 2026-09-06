# CLAUDE.md — Agent guide for the Conversational Database Assistant

This file orients any AI agent working in this repo. Read it fully before editing.
Companion docs: `REQUIREMENTS_TODO.md` (gap analysis), `GITHUB_ISSUES.md` (work items),
`BUILD_PROMPTS.md` (sequenced prompts), `FYPWorkPlan(System Design).pdf` (the spec).

---

## 1. What this project is

A **conversational, read-only Text-to-SQL database assistant** (final-year project). A
user asks a question in natural language; the system classifies intent, retrieves the
relevant DB schema via RAG, generates + validates + executes a **SELECT-only** SQL query,
and returns a natural-language answer. It also answers general DB how-to questions.

**Framework intent (per spec):** LangGraph + LlamaIndex. **Reality:** the code is
currently LangGraph/LangChain + pgvector only. LlamaIndex / Knowledge Graph is NOT yet
present and is deferred (see Issue #12). Do not assume it exists.

---

## 2. Architecture (layered, multi-agent)

Pipeline implemented as a **LangGraph `StateGraph`** in `backend/agents/db_agent.py`:

```
classify_intent ─┬─ WRITE_BLOCKED → handle_write_blocked → END
                 ├─ GENERAL       → handle_general       → END
                 └─ READ/AMBIGUOUS → retrieve_schema → generate_sql
                                      → validate_sql ─┬─ valid    → execute_sql → format_response → END
                                                      └─ invalid  → fallback_handler ↺ (max 2 rounds)
```

Spec layers and where they live:
| Layer | Module |
|---|---|
| Orchestration | `backend/agents/db_agent.py` (`OrchestrationAgent`) |
| SQL generation + rerank | `backend/agents/sql_code_agent.py` (`SQLCodeAgent`) |
| Validation (syntax/safety/semantic) | `backend/agents/sql_validator_agent.py` (`SQLValidatorAgent`) |
| Agentic RAG (schema vectors) | `backend/retrieval/` (`indexing.py`, `pgvector_store.py`, `pinecone_store.py`, `embedding.py`, `data_processing.py`) |
| Memory | `backend/memory/session_memory.py` (in-RAM only today) |
| Response Optimizer | `backend/pipelines/postprocessing.py` (`ResponseOptimizer`) |
| Schema Versioning | `backend/agents/schema_versioning_agent.py` (**stub — calls non-existent methods**) |
| Tools / Connectivity | `backend/tools/*` (**empty stubs**) |
| Frontend (UI layer) | `frontend/` (React + Vite + TS — **currently fully mocked**) |
| Business Layer (HTTP API) | **does not exist yet** — top priority |

---

## 3. Repo map

```
backend/
  agents/        db_agent.py (orchestrator), sql_code_agent.py, sql_validator_agent.py,
                 schema_versioning_agent.py (stub), prompt.md (intent prompt)
  retrieval/     indexing.py (CLI + retrieve/execute), pgvector_store.py (canonical store),
                 pinecone_store.py (alt), embedding.py, data_processing.py
  memory/        session_memory.py, user_profiles.json (empty)
  pipelines/     postprocessing.py (ResponseOptimizer), prompt_response.md,
                 llm_pipeline.py (EMPTY)
  tools/         flight/hotel/weather_api.py (EMPTY scaffold — likely leftover)
  config/        config.yaml (EMPTY), secrets.env, compose.yml, postgres/schema.sql
  context/       CLAUDE.md/system_prompt.md/skills.md (all EMPTY)
  tests/         test_agent.py, test_pipeline.py, test_tools.py (all EMPTY)
frontend/        Vite + React 19 + TS. src/App.tsx (mocked), components/, types.ts
dataset/         minidev sample data
outputs/         CSV exports from large query results
```

---

## 4. Tech stack & conventions

- **Backend:** Python 3.13, LangGraph, LangChain, OpenAI (`gpt-4o-mini`,
  `text-embedding-3-small` → 1536-dim), `psycopg2`, `sqlglot`, pgvector.
- **Frontend:** React 19, TypeScript, Vite, `lucide-react`. No state lib — plain hooks.
- **DB:** PostgreSQL 17 with the `pgvector` extension (see `backend/config/compose.yml`).
- **Imports:** use **absolute imports rooted at `backend.`** (e.g.
  `from backend.retrieval.indexing import retrieve`). The codebase mixes these with
  commented-out relative imports — always prefer the `backend.` form. Run modules from the
  **repo root** so the package resolves.
- **LLM calls:** via `langchain_openai.ChatOpenAI` with `temperature=0` and
  `SystemMessage`/`HumanMessage`. Keep prompts in `.md`/`.tpl` files where possible
  (pattern already used: `agents/prompt.md`, `pipelines/prompt_response.md`).
- **Style:** type hints + `from __future__ import annotations`; module-level `logger =
  logging.getLogger(__name__)`. Match the existing section-comment style (`# ── … ──`).

---

## 5. How to run

```bash
# 1. Start Postgres + pgvector
docker compose -f backend/config/compose.yml up -d

# 2. Build the schema vector index (from repo root)
#    Tables are tagged with the logical database that owns them (GH#81, read
#    from outputs/dev_tables.json); the assistant's own tables are excluded
#    (GH#79, backend/retrieval/internal_tables.py). Scope retrieval to one
#    domain with retrieval.active_db in config.yaml; unset = search all.
python -m backend.retrieval.indexing build --db FYP

# 3. Run the orchestrator (until the API exists)
python -m backend.agents.db_agent

# Frontend
cd frontend && npm install && npm run dev    # http://localhost:5173
```
Env: copy real values into `backend/config/secrets.env` (`OPENAI_API_KEY`, `PG_*`).
Once the API exists: `uvicorn backend.api.main:app --reload --port 8000`.

---

## 6. ⚠️ Known issues & gotchas (read before coding)

1. **Secrets are committed.** A live OpenAI key is in `retrieval/indexing.py`, and the
   Postgres password `Toyo20001108!` is hardcoded in `pgvector_store.py` (line 31) AND
   `config/compose.yml`. Treat these as compromised. **Never** add new hardcoded secrets —
   read from `secrets.env`. Fixing this is Issue #0; do it first.
2. **History injection bug.** In `db_agent.py::classify_intent` the prompt embeds the
   literal string `"history_str"` (the real `_format_history` call is commented out).
   Multi-turn context is broken until Issue #3 fixes it.
3. **Memory is in-RAM.** `SessionMemory` is a dict — lost on restart. Don't build features
   assuming durable history until #3.
4. **Schema versioning is a stub.** `schema_versioning_agent.py` calls
   `retriever.db.get_schema_diff()`, `embed_table()`, etc. that **don't exist**. It will
   crash if invoked. See Issue #7.
5. **`few_shot_examples` is always `[]`.** The semantic cache is not wired (Issue #4). The
   pgvector store has a fully-written but commented-out `kb_embeddings` table/upsert/
   retrieve block — adapt that rather than writing from scratch.
6. **`agent.run()` returns a bare string.** For the API/feedback features you'll want it to
   return a dict (`response`, `sql`, `intent`, row count, csv path). Refactor early (#1).
7. **Windows/pgvector quirk:** `localhost` may resolve to IPv6 and hit the wrong Postgres
   instance. The store defaults to `127.0.0.1` on purpose — keep it.
8. **`pgvector_store.py` is canonical**, `pinecone_store.py` is the swappable alt. Keep
   both behind the `_get_backend(store)` selector in `indexing.py`.

---

## 7. Safety rules (domain-critical — never weaken)

- The system is **strictly read-only**. Never generate or permit
  INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE. The validator and the intent classifier
  both block these — keep both layers.
- Every generated query must be `SELECT` only, use explicit columns (no `SELECT *`), and
  carry a `LIMIT` (default 100, max 500).
- Preserve the SQL-injection checks in `sql_validator_agent.py`.
- Admin-only actions (schema sync) must stay behind an auth guard once #6 lands.

---

## 8. Workflow for picking up an issue

1. Read the matching section in `GITHUB_ISSUES.md` (Why / Sub-tasks / Acceptance criteria
   / Depends-on). Respect the dependency order.
2. Keep changes scoped to one issue. Don't refactor unrelated layers in the same PR.
3. Match existing patterns (prompt-in-file, `backend.` imports, logger usage).
4. Add/adjust tests in `backend/tests/` (see Issue #11). Run `pytest` from repo root.
5. If you touch the pipeline, do a manual smoke run (`python -m backend.agents.db_agent`)
   before declaring done.
6. Update this file / the docs if you change architecture, run steps, or conventions.
7. Never commit secrets. Verify with `grep -ri "sk-" backend/` and the password string.

---

## 9. Definition of done (per change)

- [ ] Acceptance criteria for the issue met.
- [ ] Read-only safety preserved; no new hardcoded secrets.
- [ ] Tests added/updated and passing from repo root.
- [ ] Imports use the `backend.` root; module runs from repo root.
- [ ] Docs updated if behavior/architecture changed.
