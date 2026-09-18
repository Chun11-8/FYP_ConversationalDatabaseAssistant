# CLAUDE.md — Agent guide for the Conversational Database Assistant

This file orients any AI agent working in this repo. Read it fully before editing.
Companion docs: `README.md` (full reference), `DEGRADED_MODE.md` (what still works when a
component is down), `FYPWorkPlan(System Design).pdf` (the spec). Work items live in GitHub
Issues, not in a file.

---

## 1. What this project is

A **conversational, read-only Text-to-SQL database assistant** (final-year project). A
user asks a question in natural language; the system classifies intent, retrieves the
relevant DB schema via RAG, generates + validates + executes a **SELECT-only** SQL query,
and returns a natural-language answer. It also answers general DB how-to questions.

**Framework intent (per spec):** LangGraph + LlamaIndex. **Reality:** LangGraph/LangChain
+ pgvector only — there is no `llama_index` import anywhere in the repo. LlamaIndex and the
Knowledge Graph are deferred; do not assume they exist.

The project has **two halves**: generating SQL, and deciding whether that SQL can be
trusted (§6.8). Both are evaluated — `eval/bird.py` for generation, `eval/trust_eval.py`
for the detector.

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
| Memory | `backend/memory/session_memory.py` (`_PgBackend` when Postgres is up, `_RamBackend` otherwise — check `is_persistent`) |
| Semantic cache / few-shots | `backend/memory/semantic_cache.py`, `skeleton.py` |
| Response Optimizer | `backend/pipelines/postprocessing.py` (`ResponseOptimizer`) |
| Schema Versioning | `backend/agents/schema_versioning_agent.py` (`hash_schema`, `diff_schemas`, `snapshot_from_tables`) |
| Trust / abstention | `backend/agents/risk.py`, `back_translation.py`, `trust.py` (see §6.8) |
| Per-model provider routing | `backend/agents/llm_factory.py` (`resolve_model_target`, `build_for_target`) |
| Tools / Connectivity | `backend/tools/` (`base.py`, `registry.py`, `web_search.py`) |
| Frontend (UI layer) | `frontend/` (React + Vite + TS; talks to the API via `src/api/client.ts`) |
| Business Layer (HTTP API) | `backend/api/main.py` (FastAPI) |
| Evaluation | `backend/eval/` (`bird.py` benchmark, `trust_eval.py` detector metrics) |

---

## 3. Repo map

```
backend/
  api/           main.py (FastAPI: /chat, /models, /databases, /health, /feedback)
  agents/        db_agent.py (orchestrator), sql_code_agent.py, sql_validator_agent.py,
                 schema_linking_agent.py, schema_versioning_agent.py, llm_factory.py
                 (per-model provider routing), risk.py / back_translation.py / trust.py
                 (the trust half — see §6.8), usage.py (tokens + cost), safety.py,
                 trace.py, prompt.md (intent prompt)
  retrieval/     indexing.py (CLI + retrieve/execute), pgvector_store.py (canonical store),
                 pinecone_store.py (alt), embedding.py, data_processing.py,
                 db_catalog.py (table → db_id), internal_tables.py (self-exclusion),
                 value_index.py (distinct-value hints)
  memory/        session_memory.py (_PgBackend / _RamBackend), semantic_cache.py, skeleton.py
  pipelines/     postprocessing.py (ResponseOptimizer), prompt_response.md
  eval/          bird.py (BIRD mini-dev harness), trust_eval.py (detector AUROC),
                 db_backends.py (postgres/sqlite engines), error_analysis.py,
                 run_all_models.py (sweep), download_bird.py
  tools/         base.py, registry.py, web_search.py
  config/        config.yaml (models, providers, pricing, retrieval, sql),
                 config_loader.py, check_models.py, secrets.env, compose.yml,
                 postgres/schema.sql
  context/       CLAUDE.md/system_prompt.md/skills.md/Gemini.md (all still EMPTY)
  tests/         55 files, ~13.5k lines — run `pytest backend/tests` from the repo root
frontend/        Vite + React 19 + TS. src/App.tsx, src/api/client.ts, components/, types.ts
archive/         multiturn-eval (retired harness; README has restore steps)
outputs/         BIRD results by tier, CSV exports from large query results
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

# 3. Start the API
uvicorn backend.api.main:app --reload --port 8000

# 4. Frontend
cd frontend && npm install && npm run dev    # http://localhost:5173

# Tests (from the repo root)
pytest backend/tests

# BIRD benchmark. dev.json is GROUPED BY DATABASE, so --sample head with no
# --db-id returns one database's questions; use stratified for a cross-domain
# subset. --question-timeout abandons a stalled question (scored 0, still counted).
python -m backend.eval.bird --dev-json <dev.json> --db-dir <dbs> \
    --subset 20 --sample stratified --difficulty moderate
```
Env: copy real values into `backend/config/secrets.env` (`OPENAI_API_KEY`, `PG_*`,
and `OPENROUTER_API_KEY` for the gateway models).

---

## 6. ⚠️ Known issues & gotchas (read before coding)

1. **Secrets come from the environment.** `pgvector_store.py` reads `PG_PASSWORD`,
   `compose.yml` interpolates `${PG_PASSWORD}`, and API keys come from
   `secrets.env` / the process env. The earlier committed key and the hardcoded
   `Toyo20001108!` are gone — treat both as compromised regardless, and **never**
   add a new hardcoded secret. Verify with `grep -ri "sk-" backend/`.
2. **`llm.pricing` must stay indented under `llm:`.** Every loader reads
   `config["llm"]["pricing"]` (`db_agent.py`, `eval/bird.py`). At the top level of
   `config.yaml` it parses fine, is silently ignored, and every reported cost reads
   `$0.00` — which is exactly what happened to a whole set of benchmark runs.
3. **A model id must match `llm.selectable_models` EXACTLY** or it carries no
   `provider:` tag, falls back to OpenAI, and 400s with "invalid model ID" —
   display names, wrong case and near-misses all fail this way. `bird.py` aborts
   before the loop rather than billing every question; check a list with
   `python -m backend.config.check_models`.
4. **Read config and prompt files with `encoding="utf-8"`.** Windows defaults to
   cp1252, where five bytes are undefined and crash outright while everything
   else mojibakes silently. Non-ASCII in a `.md`/`.yaml` the runtime reads has
   broken this before.
5. **`backend/context/*.md` are all still empty** (`CLAUDE.md`, `system_prompt.md`,
   `skills.md`, `Gemini.md`). Don't assume they carry context — this file does.
6. **`OrchestrationAgent.run()` returns a dict**, not a string: `response`, `sql`,
   `intent`, row count, csv path, usage and the trust block. The API and the UI
   both depend on those keys.
7. **Windows/pgvector quirk:** `localhost` may resolve to IPv6 and hit the wrong Postgres
   instance. The store defaults to `127.0.0.1` on purpose — keep it.
8. **The project has two halves.** Generating SQL, and deciding whether the
   SQL can be trusted. The second half is `agents/risk.py` (ten gold-free
   checks), `agents/back_translation.py` (SQL → English → compare) and
   `agents/trust.py` (evidence → verdict), measured by `eval/trust_eval.py`.
   Two rules there are load-bearing: every signal is computed from the query
   in front of the user, and a check that could not run is reported as *not
   checked*, never as *nothing wrong*. Don't weaken either.
9. **Two execution paths exist.** `backend/eval/bird.py` has its OWN loop — it
   never touches `OrchestrationAgent`. It shares only the generation agents, so
   a change to the graph does not change the benchmark and vice versa. GH#85
   aligned what they feed those agents (full schema per scope; no tier
   inference live); evidence (`<domain_context>`) is still eval-only. When you
   change retrieval or generation inputs, check both.
10. **`pgvector_store.py` is canonical**, `pinecone_store.py` is the swappable alt. Keep
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

1. Read the GitHub issue (Why / Sub-tasks / Acceptance criteria / Depends-on).
   Respect the dependency order it states.
2. Keep changes scoped to one issue. Don't refactor unrelated layers in the same PR.
3. Match existing patterns (prompt-in-file, `backend.` imports, logger usage).
4. Add/adjust tests in `backend/tests/`. Run `pytest backend/tests` from the repo root.
5. If you touch the pipeline, smoke-run it through the API (`uvicorn
   backend.api.main:app --reload --port 8000`) before declaring done. If you touch
   retrieval or generation inputs, check the BIRD path too (§6.9).
6. Update this file / the docs if you change architecture, run steps, or conventions.
7. Never commit secrets. Verify with `grep -ri "sk-" backend/` and the password string.

---

## 9. Definition of done (per change)

- [ ] Acceptance criteria for the issue met.
- [ ] Read-only safety preserved; no new hardcoded secrets.
- [ ] Tests added/updated and passing from repo root.
- [ ] Imports use the `backend.` root; module runs from repo root.
- [ ] Docs updated if behavior/architecture changed.
