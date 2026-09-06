# Conversational Database Assistant

A final-year project — a **read-only, conversational Text-to-SQL assistant** powered by LangGraph, LangChain, OpenAI, and pgvector. Users ask questions about their database in plain English; the system classifies intent, retrieves relevant schema via vector RAG, generates and validates a `SELECT` query, executes it, and returns a natural-language answer.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Frontend](#frontend)
11. [Safety & Security](#safety--security)
12. [Accuracy & Evaluation](#accuracy--evaluation)
13. [Development Status](#development-status)
14. [Roadmap](#roadmap)
15. [Contributing](#contributing)

---

## Overview

The assistant operates as a **multi-agent pipeline** built on a LangGraph `StateGraph`. A user types a plain-English question; the system:

0. **Checks whether it has answered this before** — a question near-identical to a past validated one skips straight to validation with the cached SQL, bypassing steps 1–7 entirely. The SQL is still re-validated and re-executed against the live database every time.
1. **Classifies intent** — `READ` (SQL query), `GENERAL` (how-to answer), `WRITE_BLOCKED` (mutation rejected), or `AMBIGUOUS` (needs clarification).
2. **Condenses follow-ups** — anaphoric questions ("what about 2013?") are rewritten into standalone queries before retrieval, using prior turns and the last SQL.
3. **Retrieves schema context** via agentic RAG — pgvector embeddings narrow down the relevant tables, then FK-aware expansion pulls in linked tables and **value grounding** injects the actual stored values/casing for `WHERE` literals. A semantic cache supplies few-shot examples (and can short-circuit near-identical past questions).
4. **Links the schema** — a schema-linking agent produces a structured **query plan** (target, tables, join path, filters, evidence formula, aggregation, grouping, set-ops) that the generator follows. Tables named in the plan but missed by retrieval are recovered by **backward linking**.
5. **Generates SQL** — an LLM SQL agent produces multiple candidates. On the harder tiers, candidates come from **diverse reasoning paths** (direct, divide-and-conquer, query-plan chain-of-thought) so they fail differently. Deterministic repairs fix evidence pseudo-functions, mangled identifiers, and dialect quirks.
6. **Validates SQL** — syntax (sqlglot), safety (injection/mutation guard), and schema-grounded semantic checks run before any execution.
7. **Selects the best candidate** — valid candidates are executed and **plurality-voted on their result sets**; genuine ties are resolved by a unit-test screen + pairwise LLM comparison over the actual rows (falling back to a text reranker).
8. **Refines (ReAct)** — on an execution error or an empty result, the query is fed back for one or two repair rounds.
9. **Formats the response** — results are rendered as inline answers, markdown tables, or exported to CSV for large result sets.

Throughout, the turn streams live stage events over SSE, records per-stage timings and per-stage token/cost usage, and (on success) caches the validated question→SQL pair together with its answer and a fingerprint of the rows that produced it.

The system is **strictly read-only** — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, and `CREATE` are blocked at the intent-classification layer, the SQL validator, **and** an optional read-only database role.

---

## Architecture

```
User Query
    │
    ▼
route_entry ──── repeat question already resolved from cache ──────────┐
    │            (skips classification, condensation, generation)      │
    │ new question                                                     │
    ▼                                                                  │
classify_intent ──┬── WRITE_BLOCKED ──► handle_write_blocked ──► END    │
                  ├── GENERAL       ──► handle_general (web search + LLM) ──► END
                  ├── AMBIGUOUS     ──► ask_clarification ──► END       │
                  ├── (LLM call failed) ──► handle_intent_error ──► END │
                  └── READ                                             │
                          │                                            │
                          ▼                                            │
                   condense_query    (rewrite anaphoric follow-ups → standalone)
                          │                                            │
                          ▼                                            │
                   retrieve_schema ──── cache hit ─────────────────────┤
                          │            (pgvector RAG + FK expansion +  │
                          │             value grounding; cache few-shots)
                          ▼                                            │
                   link_schema       (structured query plan + backward linking)
                          │                                            │
                          ▼                                            │
                   generate_sql      (diverse-path candidates + deterministic repairs)
                          │                                            │
                          ▼                                            │
                   validate_sql ◄───────────────────────────────────────┘
                          │
                          ├── valid       ──► execute_sql
                          │                        │  (execution voting +
                          │                        │   unit-test/pairwise select)
                          │                        ▼
                          │                  route_after_execution
                          │                   ├── error / 0 rows ──► refine_sql ↺
                          │                   └── ok ──► format_response ──► END
                          ├── invalid, cached ──► retrieve_schema ↺  (stale cache →
                          │                                           full retrieval)
                          └── invalid         ──► fallback_handler ──► (retry, max 2 rounds)
```

Three paths reach `validate_sql`. A brand-new question walks the full pipeline.
A repeat question is resolved against the semantic cache **before the graph
runs** and enters directly at validation. A near-match discovered during
retrieval short-circuits linking and generation. In every case the SQL is
re-validated against the current schema and re-executed, so a cached query that
no longer fits the schema unwinds to full retrieval rather than returning
wrong rows.

A cache hit needs more than a close vector. Embeddings are near-blind to exact
numeric and date tokens — *"…in 2013?"* and *"…in 2014?"* differ by one
character and sit well inside the 0.03 threshold — so a **literal guard**
compares the concrete values each question pins down (numbers, quoted strings,
month names, normalized so `Aug` == `August` and `06` == `6`). Differing
values reject the hit no matter how close the vectors are, on both the read
path and the dedup write path. This is what keeps a follow-up like *"what
about 2014?"* from being answered with the previous turn's SQL; a genuine
repeat, with identical literals, still hits the cache.

`classify_intent` distinguishes a genuinely unclear question from a broken
model call: a call failure (an invalid custom model id, an auth error, a
timeout) routes to `handle_intent_error` with a real error message instead of
the misleading "your question is ambiguous" clarification prompt. A short,
on-topic reply to the assistant's own last clarifying question (e.g. naming a
table) is recognized and forced straight to `READ`, so `condense_query` — the
existing follow-up-merging step — resolves it against the original question
instead of re-asking indefinitely.

### Agent & Module Responsibilities

| Component | Module | Description |
|---|---|---|
| Orchestrator | `backend/agents/db_agent.py` | LangGraph `StateGraph`; wires all nodes, voting, selection, ReAct refine |
| Schema Linker | `backend/agents/schema_linking_agent.py` | Produces the structured query plan (tables, joins, filters, evidence formula) |
| SQL Generator | `backend/agents/sql_code_agent.py` | Diverse-path candidate generation, reranking, deterministic repairs |
| SQL Validator | `backend/agents/sql_validator_agent.py` | Syntax, safety, schema-grounded semantic validation |
| Safety helpers | `backend/agents/safety.py` | Shared DML/injection regexes + `LIMIT` enforcement |
| Cost accounting | `backend/agents/usage.py` | Per-request LLM token/cost tracking; `invoke_tracked()` wraps every LLM call |
| Prompt harness | `backend/agents/utils.py` | Loads prompt sections from `prompt.md` / `prompts/` |
| Pipeline tracing | `backend/agents/trace.py` | Config-gated console banners for interactive tuning |
| Schema Versioning | `backend/agents/schema_versioning_agent.py` | Snapshot/diff/hash; surgical vs full re-index |
| Schema RAG | `backend/retrieval/` | pgvector store, FK expansion, value index, indexing CLI |
| Memory | `backend/memory/session_memory.py` | Session turn history (in-RAM, optional Postgres) with rolling summaries |
| Semantic cache | `backend/memory/semantic_cache.py` | Validated question→SQL pairs; few-shots, hit shortcut, cached answers |
| LLM factory | `backend/agents/llm_factory.py` | Single construction point for query-stage clients; OpenAI-compatible `base_url` |
| Query skeletons | `backend/memory/skeleton.py` | Structural-shape signatures used to blend few-shot selection |
| Tools | `backend/tools/` | Web-search tool + registry, used by the `GENERAL` intent path |
| Response Optimizer | `backend/pipelines/postprocessing.py` | Formats results (inline / table / CSV) + output guardrail |
| BIRD evaluation | `backend/eval/` | BIRD mini-dev harness (EX / Soft-F1 / VES), error analysis |
| HTTP API | `backend/api/main.py` | FastAPI server: chat, SSE streaming, schema sync, feedback, metrics |
| Config Loader | `backend/config/config_loader.py` | Single config entry point; validates secrets |

---

## Tech Stack

### Backend

| Technology | Role |
|---|---|
| Python 3.13 | Runtime |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent orchestration state machine |
| [LangChain](https://www.langchain.com/) | LLM abstraction, prompt helpers |
| [OpenAI `gpt-4o-mini`](https://platform.openai.com/) | Intent classification, SQL generation, response formatting |
| [OpenAI `text-embedding-3-small`](https://platform.openai.com/) | Schema + KB embeddings (1536-dim) |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity search inside PostgreSQL |
| PostgreSQL 17 | Primary database + vector store |
| [sqlglot](https://github.com/tobymao/sqlglot) | SQL parsing and syntax validation |
| psycopg2-binary | PostgreSQL driver |
| FastAPI + Uvicorn | HTTP API layer |
| python-dotenv / PyYAML | Configuration management |

### Frontend

| Technology | Role |
|---|---|
| React 19 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool / dev server |
| react-markdown + remark-gfm | Markdown + table rendering |
| rehype-highlight + highlight.js | SQL and code syntax highlighting |
| lucide-react | Icon set |
| uuid | Client-side message IDs |

---

## Repository Structure

```
FYP_ConversationalDatabaseAssistant/
├── backend/
│   ├── agents/
│   │   ├── db_agent.py                 # Orchestrator (LangGraph StateGraph)
│   │   ├── schema_linking_agent.py     # Structured query-plan generation
│   │   ├── sql_code_agent.py           # SQL generation, reranking, deterministic repairs
│   │   ├── sql_validator_agent.py      # Syntax / safety / semantic validation
│   │   ├── schema_versioning_agent.py  # Schema snapshot / diff / re-index
│   │   ├── safety.py                   # Shared DML + injection guards, LIMIT enforcement
│   │   ├── usage.py                    # Per-request LLM token + cost accounting
│   │   ├── trace.py                    # Config-gated pipeline console tracing
│   │   ├── utils.py                    # PromptHarness — loads prompt sections from disk
│   │   ├── prompt.md                   # Intent + schema-linking + SQL system prompts
│   │   └── prompts/                    # Condense / regenerate / rerank templates
│   ├── api/
│   │   ├── main.py                     # FastAPI app (lifespan, routes, CORS)
│   │   └── models.py                   # Pydantic request/response models
│   ├── config/
│   │   ├── config.yaml                 # Non-secret configuration
│   │   ├── config_loader.py            # load_config() — single entry point
│   │   ├── secrets.env                 # Real secrets — git-ignored
│   │   ├── secrets.env.example         # Template — safe to commit
│   │   ├── compose.yml                 # Docker Compose for Postgres + pgvector
│   │   └── postgres/
│   │       └── schema.sql              # DB bootstrap (enables pgvector extension)
│   ├── memory/
│   │   ├── session_memory.py           # Session turn history + rolling summaries
│   │   ├── semantic_cache.py           # Validated query→SQL pairs (few-shots, shortcut, answers)
│   │   └── skeleton.py                 # Structural-shape signatures for few-shot blending
│   ├── pipelines/
│   │   ├── postprocessing.py           # ResponseOptimizer (inline/table/CSV) + guardrail
│   │   └── prompt_response.md          # Response formatting prompt template
│   ├── retrieval/
│   │   ├── pgvector_store.py           # Canonical vector store (pgvector)
│   │   ├── pinecone_store.py           # Alternate vector store (Pinecone)
│   │   ├── indexing.py                 # CLI: build/query index, FK expansion, backward link
│   │   ├── embedding.py                # Embedding client + verbose / M-Schema chunk builders
│   │   ├── value_index.py              # Distinct-value index for WHERE-literal grounding
│   │   ├── data_processing.py          # Schema extraction from Postgres
│   │   └── retriever.py                # High-level retrieve() interface
│   ├── eval/                           # BIRD mini-dev harness (EX/F1/VES) + error analysis
│   ├── tests/                          # 776 unit/integration tests (offline)
│   └── tools/                          # Web-search tool + registry
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Root component — session + chat loop
│   │   ├── main.tsx                    # React entry point
│   │   ├── index.css                   # Design system (dark-minimal theme)
│   │   ├── types.ts                    # TypeScript interfaces (Message, McqMessage…)
│   │   ├── api/
│   │   │   └── client.ts               # API client (createSession, sendChat…)
│   │   └── components/
│   │       ├── ChatLayout.tsx          # App shell (sidebar + top bar + tabs)
│   │       ├── ChatMessage.tsx         # Message renderer (markdown, SQL panel, records, badges)
│   │       ├── ChatInput.tsx           # Auto-resize textarea + send button
│   │       └── DatabaseTab.tsx         # Schema browser + Index Sync panel
│   ├── package.json
│   └── vite.config.ts
├── dataset/                            # BIRD mini-dev data — git-ignored, fetched by
│                                       #   python -m backend.eval.download_bird
├── outputs/                            # CSV exports + BIRD result JSON — git-ignored
├── requirements.txt
├── CLAUDE.md                           # Agent/developer guide
├── GITHUB_ISSUES.md                    # Detailed issue backlog
├── REQUIREMENTS_TODO.md                # Gap analysis vs. system design spec
└── BUILD_PROMPTS.md                    # Sequenced implementation prompts
```

---

## Prerequisites

- **Python 3.13+**
- **Node.js 18+** and npm
- **Docker** (for the PostgreSQL + pgvector container)
- **OpenAI API key** (`gpt-4o-mini` + `text-embedding-3-small`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Chun11-8/FYP_ConversationalDatabaseAssistant.git
cd FYP_ConversationalDatabaseAssistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Configuration

### 1. Create `backend/config/secrets.env`

Copy the example file and fill in real values:

```bash
cp backend/config/secrets.env.example backend/config/secrets.env
```

Edit `backend/config/secrets.env`:

```env
OPENAI_API_KEY=sk-your-openai-key-here

PG_DB=FYP
PG_USER=postgres
PG_PASSWORD=your_postgres_password_here
PG_HOST=127.0.0.1
PG_PORT=5432

# Optional — a read-only role for executing USER queries (defense-in-depth).
# When both are set, generated queries run on this role instead of PG_USER.
# PG_APP_USER=fyp_readonly
# PG_APP_PASSWORD=your_readonly_password_here
```

> **Never commit `secrets.env`** — it is git-ignored. Only `secrets.env.example` is tracked.

### 2. Review `backend/config/config.yaml`

Non-secret settings live here; edit as needed:

```yaml
llm:
  model: gpt-4o-mini          # OpenAI model used for all LLM calls
  selectable_models:          # curated list shown by the frontend's model picker —
    - gpt-4o-mini              # a "Custom..." entry there also accepts any other
    - gpt-4o                   # well-formed OpenAI model id
    - gpt-4.1-mini
  # pricing:                  # USD per 1M tokens, for the cost figures in
  #   gpt-4o-mini:            # /metrics-lite and each turn's `usage` block.
  #     input: 0.15           # Built-in defaults cover the models above.
  #     output: 0.60

db:
  dialect: postgres
  host: "127.0.0.1"
  port: 5432
  name: FYP
  user: postgres
  statement_timeout_ms: 15000  # per-statement cap on user queries

retrieval:
  vector_store: pgvector       # or "pinecone"
  top_k_tables: 5              # schema chunks to retrieve per query
  top_k_kb: 3                  # few-shot examples to retrieve
  fk_max_hops: 2               # FK-aware context expansion depth
  fk_max_expansion: 5          # cap on FK-added tables
  max_schema_tokens: 4000      # approximate token budget for schema context
  value_grounding: true        # correct WHERE-literal casing against real values
  backward_linking: true       # recover plan-named tables retrieval missed

sql:
  schema_linking: true         # produce a structured query plan before generation
  n_candidates: 5              # SQL candidates per question
  adaptive_prompting: true     # per-tier strategy blocks + candidate counts
  path_diversity: true         # divide-and-conquer + query-plan CoT candidates
  execution_voting: true       # plurality-vote on candidate result sets
  selection: pairwise          # tie-break: "pairwise" (unit-test + pairwise) | "reranker"
  unit_tests: true             # LLM-derived result checks before pairwise
  max_refine_rounds: 1         # ReAct refine rounds (challenging tier: 2)
  strict_semantic: true        # block SQL naming tables/columns absent from the schema
  enforce_limit: true          # append a safety LIMIT in the live pipeline
  fewshot_skeleton_blend: true # blend SQL-shape overlap into few-shot selection
  fewshot_skeleton_weight: 0.3 # 0 = pure embedding distance (the old ranking)
  cache_shortcut: true         # skip generation on a near-identical cached question
  cache_hit_threshold: 0.03    # cosine distance below which a pair counts as a hit
  fast_path: true              # resolve a repeat question BEFORE the graph runs
  answer_reuse: true           # reuse the stored answer when re-executed rows match

response:
  inline_row_limit: 10         # rows returned inline in markdown
  table_row_limit: 100         # rows rendered as a markdown table
  csv_row_limit: 500           # rows written to CSV (larger results)

logging:
  pipeline_trace: true         # rich console stage banners (false = structured logs only)
```

> Abbreviated — see `backend/config/config.yaml` for the full set (per-tier
> candidate counts, refine depth per tier, DISTINCT post-pass, and more).

> Every accuracy feature is config-gated so it can be A/B'd independently. The
> schema-chunk format is selected at **build time** via the `SCHEMA_FORMAT` env
> var (`verbose` default, or `mschema` for XiYan-SQL's compact layout —
> switching requires a re-index).

### 3. (Optional) Frontend API base URL

By default the frontend calls `http://localhost:8000`. To override, create `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://your-backend-host:8000
```

---

## Running the Application

### Step 1 — Start PostgreSQL with pgvector

```bash
docker compose -f backend/config/compose.yml up -d
```

This starts a PostgreSQL 17 container with the `pgvector` extension enabled. Data is persisted in the `pgdata` Docker volume.

### Step 2 — Build the schema vector index

Run once (and re-run whenever your database schema changes):

```bash
python -m backend.retrieval.indexing build --db FYP
```

This introspects your PostgreSQL schema, embeds table/column descriptions using `text-embedding-3-small`, and stores the vectors in pgvector for retrieval.

The assistant's own tables (`sessions`, `turns`, `query_sql_pairs`, `query_interactions`, `feedback`, `schema_embeddings`, `kb_embeddings`, `schema_version_store`) are **excluded** from the index — they are plumbing, not user data, and indexing them let them win a top-K retrieval slot over the tables that hold the answer. The list lives in `backend/retrieval/internal_tables.py`; add any new app-owned table to it. A rebuild also purges chunks for those tables left behind by an older index, so re-run the command above once after upgrading.

### Step 3 — Start the backend API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

### Step 4 — Start the frontend dev server

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

### Quick health check

Once both servers are running, confirm everything is connected:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok", "db_connected": true}
```

If `db_connected` is `false`, the backend is running but cannot reach PostgreSQL — check that the Docker container is up and that the password in `secrets.env` matches `compose.yml`.

---

### Optional — Run tests

```bash
pytest backend/tests/ -v
```

### Optional — Smoke-test the agent directly (no API)

```bash
python -m backend.agents.db_agent
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Missing required environment variables: ['OPENAI_API_KEY']` | `secrets.env` not found or key is missing | Complete Step 1 of Configuration |
| `"Could not connect to the server"` banner in the UI | Backend not running or wrong port | Start `uvicorn` on port 8000 (Step 3) |
| `db_connected: false` on `/health` | Docker container not running | `docker compose -f backend/config/compose.yml up -d` |
| `connection refused` on port 5432 | pgvector container crashed or wrong host | Check `PG_HOST=127.0.0.1` (not `localhost`) in `secrets.env` |
| Frontend shows blank page | Vite dev server not running | `cd frontend && npm run dev` (Step 4) |
| `ModuleNotFoundError: No module named 'backend'` | Running from wrong directory | Always run Python commands from the **repo root** |
| Schema index build fails | DB not reachable or schema not loaded | Ensure Docker is up and your DB has tables before indexing |

---

## API Reference

All endpoints are served at `http://localhost:8000`. Interactive Swagger UI is available at `/docs`.

---

### `GET /health`

Checks service and database liveness.

**Response**
```json
{
  "status": "ok",
  "db_connected": true
}
```

---

### `POST /session`

Creates a new conversation session.

**Response**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### `POST /chat`

Sends a user message and returns a structured response.

**Request body**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Show the top 10 players by overall rating"
}
```

**Response**
```json
{
  "response": "Here are the top 10 players by overall rating:\n\n| Name | Overall |\n|---|---|\n| Lionel Messi | 94 |\n...",
  "sql": "SELECT name, overall FROM players ORDER BY overall DESC LIMIT 10",
  "intent": "READ",
  "validation": { "valid": true, "syntax_ok": true, "safety_ok": true, "semantic_ok": true },
  "row_count": 10,
  "rows": [{ "name": "Lionel Messi", "overall": 94 }],
  "csv_path": null,
  "turn_id": 42,
  "cache_hit": false,
  "latency_ms": 1830,
  "timings": { "retrieve_schema": 210, "generate_sql": 940, "execute_sql": 60 },
  "usage": {
    "calls": 6, "input_tokens": 9840, "output_tokens": 512,
    "est_cost_usd": 0.001783,
    "by_stage": { "classify_intent": { "calls": 1, "input_tokens": 180, "output_tokens": 1, "est_cost_usd": 0.000027 } }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `response` | string | Natural-language answer in markdown |
| `sql` | string \| null | The SQL query used (null for GENERAL intent) |
| `intent` | string | `READ`, `GENERAL`, `WRITE_BLOCKED`, or `AMBIGUOUS` |
| `validation` | object | `valid` + per-check flags (`syntax_ok`, `safety_ok`, `semantic_ok`) |
| `row_count` | integer \| null | Number of rows returned |
| `rows` | array \| null | Result rows (capped at 500) for the frontend records panel |
| `csv_path` | string \| null | CSV filename for large result sets (>100 rows) |
| `turn_id` | integer \| null | Turn id, used to submit feedback |
| `cache_hit` | boolean | Answered via the semantic-cache shortcut |
| `verification` | object | How this answer was chosen — `candidates`, `executed`, `agreement`, `distinct_result_sets`, `decided_by`, `refine_rounds`, `validated`, `validation_checks` (GH#67) |
| `standalone_query` | string \| null | A follow-up rewritten as a self-contained question; `null` when the turn needed no context |
| `latency_ms` | integer | End-to-end turn latency |
| `timings` | object | Per-stage wall-clock ms |
| `usage` | object | LLM calls, token counts, and estimated USD cost for the turn, in total and per stage |

An optional `api_key` may be included in the request body to use a per-request
OpenAI key instead of the server default. An optional `model` field similarly
overrides the LLM model for that request only. It need not be one of the
models returned by `GET /models` — any well-formed OpenAI model id is accepted
(`^[a-zA-Z0-9][a-zA-Z0-9._:-]{1,63}$`, e.g. `gpt-4.1` or a fine-tuned id like
`ft:gpt-4o-mini-2024-07-18:org::abc123`); malformed values return `400`. This
is a shape check, not an allow-list — `GET /models` still curates the set the
frontend's picker shows by default.

---

### `POST /chat/stream`

Same request shape as `/chat`, but streams **Server-Sent Events** so the UI can
show live progress through the pipeline. Each event is a `data: <json>` line:

```
data: {"type": "stage", "stage": "retrieve_schema"}
data: {"type": "sql", "sql": "SELECT ..."}
data: {"type": "done", "response": "...", "sql": "...", ...}   ← same fields as /chat
```

A terminal `done` (or `error`) event always closes the stream.

---

### `GET /models`

Returns the models the per-request picker (sidebar) may choose from.

**Response**
```json
{
  "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
  "default": "gpt-4o-mini"
}
```

---

### `GET /history/{session_id}`

Returns the full conversation turn history for a session.

**Response**
```json
[
  { "role": "user",      "content": "How many players are from Brazil?" },
  { "role": "assistant", "content": "There are 1,237 players from Brazil.", "sql": "SELECT COUNT(*)..." }
]
```

---

### `GET /download/{filename}`

Serves a CSV file from the `outputs/` directory.

- Only files directly inside `outputs/` are served (path-traversal protected).
- Use the `csv_path` value from a `/chat` response as the filename.

---

### `GET /sessions`

Lists known sessions with a short summary (id, turn count, last activity).

---

### `POST /feedback`

Records thumbs up/down on an assistant turn. Body: `turn_id`, `session_id`,
`rating` (`up`/`down`), optional `comment`.

---

### `GET /schema`

Returns the connected database's tables and columns (name, type, PK/FK flags,
descriptions) for the frontend Database tab.

---

### `GET /schema/diff`

Compares the live database schema against the last indexed snapshot and reports
added/dropped tables, added/dropped/modified columns, and any unindexed tables.

---

### `POST /schema/sync`

Re-embeds the RAG index to match the live schema. Body: `full` (re-embed every
table — use when table **data** changed) and optional `tables` filter. The
default is a **surgical** sync that only touches changed/unindexed tables and
removes chunks for dropped ones. A dropped table also **archives** (not
deletes) its semantic-cache few-shot pairs; a modified/dropped column
**demotes** their confidence — both reported in the response as
`archived_pairs`/`demoted_pairs` (`{table: count}`), so stale examples stop
surfacing without losing history. No admin-auth guard yet (Issue #6/GH#7).

---

### `GET /metrics-lite`

Lightweight in-process counters — turn count, error count, cache-hit rate,
rolling latency, and cumulative LLM cost (`llm_calls`, `llm_calls_per_turn`,
`tokens_in`, `tokens_out`, `cost_usd`, `cost_usd_per_turn`) — for quick
operational visibility. Reset on restart.

---

## Frontend

The React frontend provides a dark-minimal chat interface:

| Feature | Description |
|---|---|
| **Welcome screen** | Three example query chips for a quick start |
| **Markdown rendering** | Full GFM support — tables, lists, code blocks |
| **Intent badges** | Colour-coded `READ` / `GENERAL` / `WRITE BLOCKED` / `AMBIGUOUS` |
| **Row count chip** | Shows how many rows the query returned |
| **Collapsible SQL panel** | First 3 lines as preview with line-count badge; expands to full syntax-highlighted SQL |
| **CSV download button** | Appears when the response includes a downloadable file |
| **Copy button** | Copies the assistant's response to the clipboard with a 2-second confirmation flash |
| **Timestamps** | Shown on every message |
| **Typing indicator** | Animated dots while the backend is processing |
| **Error banner** | Dismissible error display for network or server errors |
| **New chat** | Starts a fresh session from the sidebar button |
| **Responsive layout** | Collapsible sidebar on mobile; the chat column widens with the screen (48rem → 60rem ≥1600px → 72rem ≥2000px) and the input bar stays aligned with it |
| **Live streaming stages** | Pipeline stage labels update in real time over SSE (falls back to `/chat`) |
| **Records panel** | First 10 result rows inline, the rest behind a toggle |
| **Cache chip** | Marks a turn answered from the semantic cache rather than regenerated |
| **Cost chip** | Per-turn LLM calls, tokens and estimated USD; hover for the per-stage breakdown. A reused answer reads `0 LLM calls` |
| **Database tab** | Browse tables/columns with descriptions and value hints |
| **Index Sync panel** | Detect schema/data changes and re-index (surgical or full) from the UI |
| **Feedback** | Thumbs up/down per assistant turn |
| **Model picker** | Sidebar row (above the OpenAI API Key section) to switch the per-request LLM model, plus a "Custom…" entry to type any OpenAI model id not in the curated list |

---

## Safety & Security

The system enforces read-only semantics at multiple independent layers:

| Layer | Mechanism |
|---|---|
| Intent classifier | Rejects any query classified as `WRITE_BLOCKED` before SQL is generated |
| SQL validator | `sqlglot` parse-tree check blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE` |
| SQL validator | Injection guard — rejects string-concatenation patterns, stacked statements, and dangerous functions |
| SQL validator | `LIMIT` enforcement — every generated query must carry a `LIMIT` (max 500) |
| Read-only DB role | When `PG_APP_USER`/`PG_APP_PASSWORD` are set, user queries run on a read-only role — a write cannot succeed even if every prior layer were bypassed |
| Statement timeout | Every user query runs under a per-statement timeout so a pathological query cannot hang the request |
| Response optimizer | Row limits + an output guardrail re-scan applied before display or CSV export |
| Download endpoint | Path-traversal guard — only files inside `outputs/` are served |

**Secret management:**
- All secrets are loaded at startup by `load_config()` from environment variables or `secrets.env`.
- No secrets appear in any tracked file — `secrets.env` is git-ignored.
- `config.yaml` holds only non-sensitive defaults.

---

## Accuracy & Evaluation

Accuracy is measured on the **BIRD mini-dev** benchmark (`backend/eval/`), which
reports Execution Accuracy (EX), Soft-F1, and VES per difficulty tier. The
harness transpiles MySQL-dialect gold queries to SQLite, flags unwinnable
questions (gold that errors on SQLite), and supports execution voting, ReAct
refinement, and schema linking as toggles for A/B runs.

```bash
# End-to-end BIRD run (predictions + official EX/F1/VES scoring)
bash backend/eval/run_bird.sh

# Override the model for this run (any OpenAI model id, not just the
# frontend's allow-list — see backend/eval/bird.py::resolve_model)
MODEL=gpt-4o bash backend/eval/run_bird.sh

# Or the lightweight ground-truth harness on your own labelled set
python -m backend.eval.run_eval --ground-truth backend/eval/ground_truth.json --db FYP --model gpt-4o
```

### Benchmarking across providers (GH#65)

`--base-url` points a run at any **OpenAI-compatible** endpoint, so the same
BIRD harness can score open-weight models from different vendors against the
identical prompt stack, schema linking and voting configuration:

```bash
export OPENAI_API_KEY=sk-...            # still required — embeddings/RAG use OpenAI
export PROVIDER_KEY=sk-or-...           # your OpenRouter / Groq / DeepSeek key

# Baseline
python -m backend.eval.bird --output out/gpt4omini.json --model gpt-4o-mini

# Meta Llama via Groq
python -m backend.eval.bird --output out/llama33.json \
  --model llama-3.3-70b-versatile --base-url https://api.groq.com/openai/v1

# DeepSeek direct
python -m backend.eval.bird --output out/deepseek.json \
  --model deepseek-chat --base-url https://api.deepseek.com

# Alibaba Qwen via OpenRouter (one key reaches many vendors)
python -m backend.eval.bird --output out/qwen.json \
  --model qwen/qwen-2.5-coder-32b-instruct --base-url https://openrouter.ai/api/v1

# Fully local, no API cost
python -m backend.eval.bird --output out/local.json \
  --model qwen2.5-coder --base-url http://localhost:11434/v1
```

`--base-url` also works on `bird_predict.py` and `smoke_examples.py`, and can
be set once via `$LLM_BASE_URL` or `llm.base_url` in `config.yaml`. Each run
records both `model` and `base_url` in the results file's `run_config` — the
same model id served by two providers can differ in weights, quantisation and
sampling defaults, so a number without its endpoint isn't reproducible.

Two things to know before reading the numbers:

- **Cost shows `$0.00` for non-OpenAI models** until you add them to the
  `llm.pricing` block in `config.yaml`. Token counts are still accurate.
- **Reasoning models** (DeepSeek-R1, QwQ, Qwen-thinking, GPT-OSS) emit a
  chain of thought before the answer. `strip_reasoning()`
  (`backend/agents/utils.py`) removes it ahead of all four free-text parsers —
  intent, query plan, SQL extraction and follow-up condensation — so a `<think>`
  block doesn't get mistaken for the answer.

Research-backed techniques implemented to raise accuracy **without a stronger
model** (all config-gated) — see GitHub issues #48/#49/#50:

| Technique | Targets | Source |
|---|---|---|
| Diverse-path candidates (divide-and-conquer, query-plan CoT) | grain, grouped superlatives, multi-joins | CHASE-SQL |
| Unit-test screen + pairwise selection | wrong-candidate selection | CHESS / CHASE-SQL |
| Value index for `WHERE`-literal grounding | value casing/format mismatches | CHESS |
| Backward schema linking | broken ≥3-table joins | RSL-SQL |
| Verbatim evidence formula in the plan | evidence paraphrase | TA-SQL |
| M-Schema compact schema format | schema legibility | XiYan-SQL |
| Join-minimality demotion | spurious joins silently dropping rows | GH#50 |
| Evidence-target guidance | evidence formulas misread as filters instead of the SELECT target | GH#50 |
| Lexical-collision guidance | column names colliding with unrelated question vocabulary | GH#50 |
| Skeleton-similarity few-shot blending | few-shots that match topic but not SQL shape | DAIL-SQL |

### Showing the evidence behind an answer (GH#67)

A text-to-SQL system gets no natural signal telling it whether its own answer
is right. Syntax errors are self-reporting and safety violations are
detectable, but a query that joins the wrong table parses cleanly, executes,
and returns plausible rows. The 150-question runs in `outputs/` bear this out:

| Run | Failed validation | Failed execution | Wrong but executed |
|---|---|---|---|
| gpt-4o-mini (EX 82/62/42) | 1 | 2 | 54 |
| gpt-5-mini (EX 76/58/46) | 0 | 0 | **57 (100 %)** |

Every failure a user actually meets runs cleanly. The pipeline already
manufactures evidence about which answer to trust — execution voting over
candidate result sets, a unit-test screen, pairwise comparison — so each turn
now reports it in `verification` and the UI shows it: an agreement chip
(*"4/5 agreed"*) beside the answer and a collapsed **Why this answer** panel
with the full breakdown.

These are **facts, not a verdict**. There is deliberately no High/Medium/Low
confidence badge: a miscalibrated badge would teach users to trust exactly the
answers they should question.

Measured against the labelled `error_category` data, agreement is a real
signal — and has a hard ceiling:

| | unanimous | candidates disagreed |
|---|---|---|
| gpt-5-mini | 67.5 % EX (n=126) | 23.8 % EX (n=21) |
| gpt-4o-mini | 69.6 % EX (n=115) | 40.6 % EX (n=32) |

**41 of gpt-5-mini's 57 errors happened on turns where every candidate
agreed** — wrong together. Agreement can never catch those.

### Risk checks (GH#69)

An `assess_risk` node runs seven checks that need no ground truth: zero rows,
a WHERE literal absent from the data (via the GH#49 value index), an
unconstrained cross join, an aggregate with no GROUP BY on a per-group
question, `LIMIT 1` on a plural question, column count against the linker
plan, and a table outside the retrieved schema. Flags are attached to
`verification.risks` and shown above the result rows — each stating one
specific, checkable objection (*"'Brazil' does not appear in the data — use
'BRA'"*) rather than a generic warning.

They **annotate only**: nothing is blocked, rewritten or retried. Config gate
`sql.risk_checks`.

Replayed over the 150 labelled questions (agreement OR a risk flag):

| | errors caught | turns flagged | precision |
|---|---|---|---|
| agreement alone | 28.1 % | 14.3 % | 76.2 % |
| **+ risk checks** | **40.4 %** | 19.0 % | **82.1 %** |

Zero false alarms on the 85 unanimous-and-correct answers. This is a **floor**:
the two checks aimed at the largest error class (`disjoint`, 25 of the 41)
need a live database and stored schema context, so they could not be replayed
offline and contribute nothing to the figure above.

### Cost & minimum execution (GH#57)

A full first-time READ turn makes ~6 LLM calls (intent → linking → generation ×N
→ formatting). Repeat questions no longer pay that. Every LLM call is now
metered — `run()` returns a `usage` block and `GET /metrics-lite` accumulates
tokens and estimated USD — so the saving is measured, not assumed.

| Situation | LLM calls | What runs |
|---|---|---|
| First time asked | ~6 | Full pipeline |
| Asked again, data changed | 1 | Cached SQL re-validated + re-executed, answer re-formatted |
| Asked again, data unchanged | **0** | Cached SQL re-validated + re-executed, stored answer reused |

The cached SQL is **always** re-validated against the current schema and
re-executed against the live database — only the wording is reused, and only
when the fresh rows hash identically to the stored ones, so stale data can never
be served. A cached query that no longer validates (schema drift) unwinds to
full retrieval and normal generation. Anaphoric follow-ups ("what about last
year?") always take the full path, since they mean nothing until condensed
against the conversation history.

Set `sql.fast_path: false` and `sql.answer_reuse: false` to reproduce the
previous behavior exactly.

---

## Development Status

| Issue | Area | Status | Notes |
|---|---|---|---|
| #0 | Security & config hardening | ✅ Done | Secrets in env, `load_config()` with validation |
| #1 | FastAPI HTTP API | ✅ Done | Chat, SSE streaming, schema sync, feedback, metrics |
| #2 | Frontend ↔ Backend integration | ✅ Done | Real API wiring, streaming, Database tab |
| #3 | Memory — history + condensation | ✅ Done | History fixed; rolling summaries; multi-turn condensation |
| #4 | Semantic cache / few-shot | ✅ Done | Validated pairs as few-shots + hit shortcut |
| #5 | Feedback collection (thumbs) | ✅ Done | `/feedback` endpoint + UI |
| #6 | User isolation & roles | ⏳ Planned | Read-only DB role landed; auth still open |
| #7 | Schema versioning | ✅ Done | Snapshot/diff/hash + surgical/full re-index + UI; drop/modify cascades the semantic cache (archive/demote) |
| #9 | Tools / external APIs | ✅ Done | Web-search tool wired into GENERAL intent |
| #10 | Response optimizer guardrails | ✅ Done | Output guardrail + entity-first formatting |
| #11 | Full test suite | ✅ Done | ~776 tests; CI still to wire |
| #38–42 | Latency, streaming, multi-turn, hardening, observability | ✅ Done | Model routing, SSE, timeouts, request IDs, metrics |
| #48–50 | Accuracy techniques (see above) | ✅ Done | BIRD A/B rerun pending live env |
| #52 | Per-request model selector | ✅ Done | `/models` endpoint, sidebar picker, `--model` on eval CLIs |
| #56 | Skeleton-similarity few-shot selection | ✅ Done | DAIL-SQL-style structural-shape blending |
| #57 | Cost telemetry + minimum-execution routing | ✅ Done | Token/cost accounting, repeat-query fast path, answer reuse |
| #12 | Knowledge Graph (LlamaIndex) | 🔮 Deferred | Not started |

---

## Roadmap

**Remaining work**
- [ ] Issue #6 — User isolation, roles, JWT auth (read-only DB role already landed)
- [ ] GitHub Actions CI wired to the test suite
- [ ] BIRD 3-tier A/B rerun to quantify the #48–50 accuracy techniques
- [ ] BIRD rerun with `sql.fast_path` / `sql.answer_reuse` on vs. off — EX must be
      unchanged (GH#57 is accuracy-neutral by design); the new `cost_usd` metric
      shows the saving
- [ ] Issue #12 — Knowledge Graph via LlamaIndex (deferred)

---

## Contributing

1. Pick an issue from [GITHUB_ISSUES.md](./GITHUB_ISSUES.md). Respect the dependency order listed there.
2. Work on the designated feature branch (see `CLAUDE.md` for the current branch name).
3. Follow the conventions in [CLAUDE.md](./CLAUDE.md):
   - Absolute imports rooted at `backend.` (e.g. `from backend.retrieval.indexing import retrieve`)
   - Type hints + `from __future__ import annotations`
   - Module-level `logger = logging.getLogger(__name__)`
   - Always run from the **repo root** so the `backend` package resolves
4. Add/update tests in `backend/tests/`. Run `pytest` from the repo root before pushing.
5. Do a manual smoke run (`python -m backend.agents.db_agent`) if you touch the pipeline.
6. Never commit secrets. Verify with `grep -ri "sk-" backend/` before pushing.
7. Update this README and `CLAUDE.md` if you change architecture, run steps, or conventions.

### Definition of Done

- [ ] Acceptance criteria for the issue met
- [ ] Read-only safety preserved; no new hardcoded secrets
- [ ] Tests added/updated and passing from repo root
- [ ] Imports use the `backend.` root; module runs from repo root
- [ ] Docs updated if behaviour or architecture changed
