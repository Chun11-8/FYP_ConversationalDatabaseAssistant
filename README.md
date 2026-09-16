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
12. [Accuracy & Evaluation](#accuracy--evaluation) — including
    [Can this answer be trusted?](#can-this-answer-be-trusted-gh91) and
    [Measuring the detector](#measuring-the-detector-gh101)
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
9. **Grades the answer** — ten gold-free checks run against the executed result, the SQL is read back into plain English and compared with the question, and the evidence is condensed into a verdict: *checks passed*, *needs your review*, or *do not rely on this*.
10. **Formats the response** — results are rendered as inline answers, markdown tables, or exported to CSV for large result sets.

Throughout, the turn streams live stage events over SSE, records per-stage timings and per-stage token/cost usage, and (on success) caches the validated question→SQL pair together with its answer and a fingerprint of the rows that produced it.

The system is **strictly read-only** — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, and `CREATE` are blocked at the intent-classification layer, the SQL validator, **and** an optional read-only database role.

### Why a text-to-SQL system needs a detector

A wrong SQL query is not a broken one. It parses, it executes, and it returns
plausible rows for a question nobody asked. Across the 750 labelled benchmark
records in `outputs/`, **1 answer failed validation and 2 failed execution —
while 54 executed cleanly and were wrong**. Syntax and safety checks cannot see
that class of failure, because nothing about it is malformed.

So the second half of this project is **selective prediction**: estimating,
without knowing the right answer, whether this one can be trusted — and saying
so on screen. Five gates feed a verdict, and the verdict is measured against
ground truth rather than asserted (see
[Measuring the detector](#measuring-the-detector-gh101)). Every signal is
computed from the query in front of the user; no benchmark statistic is ever
presented as a property of an individual answer.

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
                          │            (routes the question to its own │
                          │             database → full schema; else   │
                          │             pgvector top-k + FK expansion; │
                          │             value grounding; cache few-shots)
                          │
                          ├── question names a value that does not exist
                          │            ──► handle_unanswerable ──► END  │
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
                          │                   └── ok ──► assess_risk
                          │                                │  (ten gold-free checks
                          │                                │   + SQL read back into
                          │                                │   English and compared
                          │                                │   with the question)
                          │                                ▼
                          │                          format_response ──► END
                          │                           (+ trust verdict)
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
| Trust verdict | `backend/agents/trust.py` | Grades the evidence into *checks passed* / *needs review* / *do not rely on this* — pure function, no LLM, no I/O |
| Gold-free checks | `backend/agents/risk.py` | Ten checks on the executed result: unknown value, dropped constraint, aggregate grain/intent, plan conformance, cross join, arity, … |
| Back-translation | `backend/agents/back_translation.py` | Describes the SQL in plain English *without* showing the model the question, then compares |
| Schema RAG | `backend/retrieval/` | pgvector store, FK expansion, value index, per-database scoping, indexing CLI |
| Database catalog | `backend/retrieval/db_catalog.py` | Maps each table to the logical database that owns it, so retrieval can be scoped to one domain |
| Memory | `backend/memory/session_memory.py` | Session turn history (in-RAM, optional Postgres) with rolling summaries |
| Semantic cache | `backend/memory/semantic_cache.py` | Validated question→SQL pairs; few-shots, hit shortcut, cached answers |
| LLM factory | `backend/agents/llm_factory.py` | Single construction point for query-stage clients; OpenAI-compatible `base_url` |
| Query skeletons | `backend/memory/skeleton.py` | Structural-shape signatures used to blend few-shot selection |
| Tools | `backend/tools/` | Web-search tool + registry, used by the `GENERAL` intent path |
| Response Optimizer | `backend/pipelines/postprocessing.py` | Formats results (inline / table / CSV) + output guardrail |
| BIRD evaluation | `backend/eval/bird.py` | BIRD mini-dev harness (EX / Soft-F1 / VES), error analysis |
| Detector evaluation | `backend/eval/trust_eval.py` | AUROC per gate, per-check precision/false-alarm, risk–coverage curve |
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
│   │   ├── risk.py                     # Ten gold-free checks on the executed result
│   │   ├── trust.py                    # Evidence → verdict (pure; no LLM, no I/O)
│   │   ├── back_translation.py         # SQL → plain English → compare with the question
│   │   ├── llm_factory.py              # Query-stage LLM clients; OpenAI-compatible base_url
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
│   │   ├── db_catalog.py               # table → owning database, for per-domain scoping
│   │   ├── internal_tables.py          # The assistant's own tables — never indexed as data
│   │   ├── data_processing.py          # Schema extraction from Postgres
│   ├── eval/
│   │   ├── bird.py                     # BIRD mini-dev harness (EX / Soft-F1 / VES)
│   │   ├── trust_eval.py               # Does the detector detect? AUROC, risk–coverage
│   │   ├── error_analysis.py           # Failure taxonomy for BIRD runs
│   │   └── smoke_examples.py           # Self-contained SQLite fixtures
│   ├── tests/                          # 1287 unit/integration tests (offline)
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

### Step 2b — (Optional) Scope retrieval to one logical database

Several BIRD mini-dev databases were loaded into the same PostgreSQL schema, so
`public` holds 75 tables from 11 unrelated domains. An un-scoped search picks
its top-5 tables out of all 75, while the BIRD harness only ever sees the 4
tables of the question's own `db_id` — the same question, a strictly harder
retrieval problem, and the reason the chat UI and the benchmark could disagree.

The build now tags each schema chunk with the logical database that owns its
table, read from a BIRD `tables.json`:

```bash
# auto-detected: $BIRD_TABLES_JSON, then outputs/dev_tables.json,
# then dataset/minidev/MINIDEV/dev_tables.json
python -m backend.retrieval.indexing build --db FYP
python -m backend.retrieval.indexing build --db FYP --tables-json path/to/tables.json
```

The build prints the scopes it produced, and `GET /databases` lists what is
indexed plus which scope is active. To answer questions against one domain —
the benchmark's conditions — set it in `backend/config/config.yaml` and restart
the API:

```yaml
retrieval:
  active_db: debit_card_specializing
```

In the UI, the sidebar's **Database** picker overrides this per request (no
restart needed), the welcome screen names the domain a question will be
answered from, and the Database tab groups the schema browser by domain with
the active one marked. Leaving `active_db` unset searches every scope, which is
exactly what a single-domain install has always done. Nothing here creates, moves or drops a
table: `db_name` is a **label** on a vector row, and queries always execute
against the one database configured under `db:`.

### How the chat UI and the BIRD harness stay aligned

`backend/eval/bird.py` runs its own loop over `SQLCodeAgent` /
`SchemaLinkingAgent` / `SQLValidatorAgent`; the chat UI runs the LangGraph in
`db_agent.py`. They share the generation components, so any difference in what
those components are *fed* shows up as a UI/eval gap. Two such differences have
been closed (GH#85):

| | BIRD harness | Chat UI |
|---|---|---|
| Schema | complete schema of the question's `db_id` | **complete schema of the question's database**, cosine top-k only as fallback |
| Effort tier | the dataset's real difficulty label | **one fixed tier** (`sql.live_tier`, default `moderate`) — no inference |
| Evidence | `<domain_context>` from the dataset | none — a live user types no evidence hint |

The schema change matters because the table carrying a filter is often the one
*least* similar to the question's wording, and a missing table does not fail:
the model quietly drops the constraint and answers a different question. 80% of
BIRD's gold SQL needs a join, so this is the common case.

The harness gets the question's `db_id` from the dataset. The chat UI has no
such label, so it recovers one (GH#95): every schema chunk carries the database
that owns its table, which makes the cosine ranking a **vote**. A database that
contributes two or more of the top-k tables, and outscores every other, wins the
question — and its complete schema is fetched. No extra embedding, no extra LLM
call, no second query. Explicit intent (`retrieval.active_db`, or the sidebar
picker) always overrides the vote.

Schema selection therefore runs in three steps:

1. **Configured scope** — the full schema of `active_db`.
2. **Inferred scope** — the full schema of the database the ranking chose.
3. **Cosine top-k** — when no database wins outright, nothing is indexed for
   it, or the full schema exceeds `retrieval.max_schema_tokens`.

`retrieval.auto_scope: false` disables step 2;
`retrieval.full_schema_when_scoped: false` disables both and restores pure
top-k retrieval for an A/B.

Evidence remains the one real asymmetry: the benchmark supplies a domain hint
for 148 of 150 questions and the deployed system has none.

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

### `GET /databases`

Lists the retrieval scopes present in the schema vector index, and which one is
in force (`retrieval.active_db`, or every scope when unset). Read-only —
switching scope is a config change plus a restart. `503` when the index cannot
be read.

**Response**
```json
{
  "databases": [
    { "name": "debit_card_specializing", "tables": 5,  "active": true  },
    { "name": "superhero",               "tables": 10, "active": false }
  ],
  "active": "debit_card_specializing"
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
Selecting what to run — model, scope and size:

```bash
# one model
python -m backend.eval.bird --dev-json … --db-dir … --model gpt-4.1-mini

# one database (GH#116) — applied BEFORE --subset, so this really is 20
# questions from that database
python -m backend.eval.bird --dev-json … --db-dir … \
    --db-id debit_card_specializing --subset 20

# one difficulty tier, one database
python -m backend.eval.bird --dev-json … --db-dir … \
    --difficulty moderate --db-id student_club
```

`--db-id`, `--difficulty` and `--subset` combine; the two filters narrow the
set and `--subset` then slices what is left.

#### `--sample` — a subset that is actually cross-domain (GH#120)

`mini_dev_sqlite.json` stores each database as one **contiguous block**:

```
  30 debit_card_specializing | 48 student_club | 50 thrombosis_prediction
  51 european_football_2     | 66 formula_1    | 52 superhero  …
```

So `--subset 20` with no `--db-id` slices the front of the file and returns
**20 questions from `debit_card_specializing`** — a single-domain score that
reads like an overall one. `--sample` decides how the subset is drawn:

| Strategy | 20 questions from mini-dev cover |
|---|---|
| `head` (default) | **1** database — the file's own order |
| `random` | ~9 of 11, seeded shuffle |
| `stratified` | **all 11**, proportional to each database's size |

```bash
# cross-domain subset — every database represented, reproducible
python -m backend.eval.bird --dev-json … --db-dir … \
    --subset 20 --sample stratified --seed 42
```

`head` stays the default so existing result files remain reproducible. Use
`stratified` for any number you intend to call an overall or cross-domain
score; `random` is the plain shuffle, and with 11 databases a small random
subset may miss some of them.

`sample`, `seed` and `databases_evaluated` are recorded in the results file's
`run_config`: the same `--subset 20` means a different 20 under a different
strategy, so without them a run cannot be re-derived or even interpreted. The
harness also warns when a `head` subset turns out to cover one database.

**Cross-domain means generalisation, not cross-database joins.** Every BIRD
question carries exactly one `db_id` and its gold SQL runs against that one
SQLite file — no question joins two databases, and none uses `ATTACH`. What a
cross-domain run demonstrates is that the model reads a schema it has not seen
before, across 11 unrelated subject areas.

#### `--question-timeout` — don't let one question hold the run (GH#119)

A wall-clock budget for **one question's LLM work**, default **60 s**. On
expiry the question is abandoned and the loop moves on:

```bash
python -m backend.eval.bird --dev-json … --db-dir … \
    --question-timeout 180        # a slow reasoning model
python -m backend.eval.bird --dev-json … --db-dir … \
    --question-timeout 0          # disabled — pre-GH#119 behaviour
```

Three separate guards exist and they are easy to confuse:

| Knob | Bounds | Where |
|---|---|---|
| `llm.request_timeout_s` | one HTTP request | `config.yaml` (GH#118) |
| `--question-timeout` | one question, all its calls | this flag (GH#119) |
| `--timeout` | one SQLite execution | this flag |

The middle one is not implied by the first: a moderate question makes roughly
ten LLM calls, so a run can crawl without any single request timing out.

**An abandoned question is scored 0 and stays in the denominator.** It is
*not* treated like a question whose gold SQL fails on SQLite — those are
genuinely unwinnable and are excluded from `EX_winnable`. Excluding timeouts
would mean a model that stalls on the hardest questions outscores one that
answers them slowly. The run summary prints the count, `metrics.timeouts`
records it, and `run_config.question_timeout` stores the budget, because two
runs at different budgets are not comparable if either abandoned work.

Caveat worth knowing: Python cannot interrupt a thread blocked on a socket
read, so the abandoned call is **abandoned, not cancelled** — it keeps running
until its own HTTP timeout fires, and it keeps billing. `--question-timeout`
buys you the loop moving on; `llm.request_timeout_s` is what reclaims the
socket.

**`--trust-eval` is not needed for accuracy.** EX / F1 / VES are computed from
the gold SQL on every run regardless. The flag adds the back-translation judge
— evidence for `trust_eval.py`, which measures the DETECTOR rather than the
generator — at a cost of two extra LLM calls per question. Leave it off for
model comparisons and ablations; switch it on for the one or two runs where
you want the judge gate scored.

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
export LLM_API_KEY=sk-or-v1-...         # key for the QUERY-STAGE provider
                                        # (OpenRouter / Groq / DeepSeek / …)
                                        # unset → falls back to OPENAI_API_KEY

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

**Keys travel with the endpoint (GH#106).** `base_url` says *where* to send a
request and `LLM_API_KEY` says *who you are there*; pointing `base_url` at
another provider while still sending your OpenAI key returns 401. The
resolution order is explicit value → `$LLM_API_KEY` → `OPENAI_API_KEY`, so a
plain OpenAI run needs no new variable at all.

| Variable | Used for | Unset falls back to |
|---|---|---|
| `OPENAI_API_KEY` | embeddings / pgvector RAG — **always OpenAI** | (required) |
| `LLM_API_KEY` | the query stage: every chat call | `OPENAI_API_KEY` |
| `JUDGE_API_KEY` | the `--trust-eval` back-translation judge | `LLM_API_KEY` |

### Mixing providers in one deployment (GH#108)

A model's provider is a property of the **model**, not of the process, so one
deployment serves OpenAI models directly and others through a gateway — and
the sidebar picker switches provider implicitly.

```yaml
llm:
  providers:
    openrouter:
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
  selectable_models:
    - gpt-4o-mini                       # untagged → DIRECT to OpenAI, unchanged
    - id: qwen/qwen3.5-122b-a10b
      provider: openrouter
    - id: some/reasoning-model
      provider: openrouter
      temperature: null                 # OMIT the parameter — some reasoning
                                        # models reject an explicit temperature
```

An **untagged** entry behaves exactly as before: the process-wide `base_url`
and the query-stage key. The client built for `gpt-4o-mini` adds nothing to the
pre-GH#108 construction except the call limits below, and a test asserts the
exact kwarg set.

#### When a call appears to hang (GH#118)

A non-streaming request to a gateway returns its `200 OK` as soon as the
**headers** are written, then holds the body open for the whole generation. So
this line in the log means the request was accepted, not that an answer came
back:

```
INFO HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

A reasoning model that emits a long chain of thought before the SQL can sit
there for minutes, and it is indistinguishable from a dead connection. Left
uncapped the OpenAI SDK default applies — 600 s per attempt with 2 retries, so
one stuck call occupies a benchmark run for half an hour. Both limits are
configurable, per process and per model:

```yaml
llm:
  request_timeout_s: 120        # null restores the SDK default (no cap)
  max_retries: 2
  selectable_models:
    - id: deepseek/deepseek-v4-flash-0731
      provider: openrouter
      request_timeout_s: 300    # a reasoner needs the head-room
```

The ceiling belongs on the model rather than the process: three minutes from a
reasoning model is it working, while three minutes from a small instruct model
means the connection is gone.

**Verify slugs before spending.** Providers retire ids, and a wrong one fails
at the first call — mid-run, after everything before it has been paid for:

```bash
python -m backend.config.check_models          # id + key + routing
python -m backend.config.check_models --call   # also one tiny real call each
```

It exits non-zero on failure, so it can gate a run:
`python -m backend.config.check_models && python -m backend.eval.bird …`

An unreachable catalogue is reported as *could not check*, never as *the model
does not exist*.

The eval CLI uses the same routing, so `--model qwen/qwen3.5-122b-a10b` reaches
OpenRouter without repeating the endpoint; an explicit `--base-url` still wins.

### Which model answered, and what it cost (GH#114)

Each assistant turn carries two chips: the **model** that answered (with the
provider in its tooltip) and the token/cost figure.

Cost reads **`cost n/a`**, not `$0.0000`, when no rate is configured for the
model. A missing price is not a zero bill, and showing one would quietly tell
you an OpenRouter run was free. Tokens are counted either way.

To make it a real number, pull the provider's own rates:

```bash
python -m backend.config.check_models --sync-pricing
```

That prints a ready-to-paste `llm.pricing` block. OpenRouter quotes per-TOKEN
prices and this project works in per-MILLION, so the conversion happens there
rather than being left to whoever pastes it. Rates change — re-run before
quoting a cost figure in a report.

`JUDGE_API_KEY` exists so the generator and the judge can sit on **different**
providers — the configuration the literature reports at 0.82 AUROC against
0.72–0.78 for a single judge. Running both through one gateway (OpenRouter)
needs only `LLM_API_KEY`.

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

### Can this answer be trusted? (GH#91)

Every answer carries a verdict, above the rows it judges:

| | When | What the user is told |
|---|---|---|
| ✅ **Checks passed** | every gate green | "Every check we can run without knowing the right answer passed… but it is not a guarantee." |
| ⚠️ **Needs your review** | candidates disagreed, plan mismatch, or a medium-severity flag | which check, in plain words |
| ⛔ **Do not rely on this** | unknown value, dropped constraint, out-of-schema table, nothing executed, or a back-translation mismatch | which check, plus what would resolve it |

Expanding it shows **how the grading was reached** — five gates, each with a
pass/warn/fail mark, the reason, and a suggested remedy:

| Gate | Question it answers |
|---|---|
| Safe and well-formed | read-only, parses, every table and column exists |
| Matches your question | the nine gold-free checks in `risk.py` |
| Follows its own plan | does the SQL implement the linker's plan? (aggregate, GROUP BY, grain) |
| Agrees with itself | how many independently written candidates returned this result |
| Describes what you asked | the SQL read back into plain English, compared with the question |

Two deliberate constraints:

- **No percentage.** Every signal is computed from *this* query, *this* schema
  and *this* result. A hit rate measured on the BIRD databases describes the
  **detector**, not an answer over your own data — that number belongs in the
  evaluation, never beside an individual answer.
- **"Not checked" never counts as "found wrong".** A gate that cannot run is
  skipped and does not lower the verdict, so turning `back_translation` off
  does not make every answer look untrustworthy.

The read-back is shown under the SQL whether or not the comparison reached a
verdict — *"This query returns the lowest single monthly reading for each
customer"* against a question that meant the lowest yearly total is something
a reader spots instantly, with no verdict needed. It is generated **without**
showing the model the question; shown it, the model paraphrases the question
back and the check becomes a mirror.

`sql.back_translation: false` disables the round trip (two LLM calls per
answered turn); everything else is deterministic and free.

### Keeping one turn out of another's way (GH#87 / GH#89)

Conversation history reaches four stages — intent classification, follow-up
condensation, planning and generation — so a question about a different
subject can be degraded by the previous turn in four places. Two layers guard
against that:

- **Condensation** only rewrites a follow-up that genuinely depends on the
  conversation, and a deterministic veto discards a rewrite that *injects*
  literals the question never mentioned (GH#87). A relative clause — "schools
  **that** are among the top 5" — is no longer mistaken for a reference.
- **Topic switch** (GH#89): when a turn's retrieved tables are disjoint from
  the tables the previous answer queried, the turn runs with **no history** at
  all. The comparison reuses the retrieved set and the previous SQL already in
  state — no extra retrieval, no extra LLM call. It is reported in the answer's
  evidence panel, and `sql.topic_switch_detection: false` turns it off.

Both are covered by `topic_switch` cases in the multi-turn harness, including
one that moves the conversation between databases mid-session — archived under
`archive/multiturn-eval/`.

### Measuring the detector (GH#101)

A verdict without a number is a claim. `backend/eval/trust_eval.py` replays the
labelled BIRD records in `outputs/` through the **same** `trust.grade()` the
live pipeline uses, and reports what each signal is worth:

```bash
python -m backend.eval.trust_eval
python -m backend.eval.trust_eval --output outputs/trust_eval.json
```

No API key, no database, no extra dependency — AUROC is computed rank-wise
(Mann-Whitney U with average ranks, so the coarse three-level gate scores are
not inflated by ties).

Reported: **AUROC** per gate and for the combined verdict; **precision,
recall and false-alarm rate** per individual risk code; the **risk–coverage
curve** (selective accuracy against the proportion of questions answered); and
an explicit list of what could *not* be measured.

That last part is load-bearing. A gate skipped on every record is reported as
`not_measured` rather than scored — its 0.5 AUROC would otherwise read as
"measured and worthless" instead of "never ran". A test asserts every code
`risk.py` can emit is classified, so a new check cannot go silently unreported.

**Capturing the judge signal.** The back-translation gate and the checks that
need the executed rows cannot be recovered from an old record — the rows are
gone. `bird --trust-eval` stores the real verification block at generation
time, so a fresh run makes them measurable:

```bash
# judge on the generator's own model — measures self-agreement
python -m backend.eval.bird --dev-json … --db-dir … --trust-eval

# judge on a DIFFERENT provider — the literature reports 0.82 AUROC for a
# two-provider ensemble, against 0.72–0.78 for a single judge
python -m backend.eval.bird --dev-json … --db-dir … --trust-eval \
    --judge-model deepseek-chat --judge-base-url https://api.deepseek.com
```

Costs two extra LLM calls per question, and the judge model and endpoint are
recorded in the run header so a cross-provider result is attributable.
`trust_eval.py` then prefers the stored block and reports the judge gate
automatically — no flag, it follows the data.

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
| #11 | Full test suite | ✅ Done | 1287 tests; CI still to wire |
| #38–42 | Latency, streaming, multi-turn, hardening, observability | ✅ Done | Model routing, SSE, timeouts, request IDs, metrics |
| #48–50 | Accuracy techniques (see above) | ✅ Done | BIRD A/B rerun pending live env |
| #52 | Per-request model selector | ✅ Done | `/models` endpoint, sidebar picker, `--model` on eval CLIs |
| #56 | Skeleton-similarity few-shot selection | ✅ Done | DAIL-SQL-style structural-shape blending |
| #57 | Cost telemetry + minimum-execution routing | ✅ Done | Token/cost accounting, repeat-query fast path, answer reuse |
| #65 | Open-weight / multi-provider models | ✅ Done | `llm_factory` + `base_url`; query stage only, embeddings stay on OpenAI |

**Trustworthiness — the second half of the project**

| Issue | Area | Status | Notes |
|---|---|---|---|
| #67 | Evidence behind an answer | ✅ Done | Candidate agreement, how the winner was chosen, refine rounds |
| #69 | Gold-free risk checks | ✅ Done | Seven checks on the executed result; annotate only |
| #71 | Multi-turn evaluation harness | 📦 Archived | Working and tested; moved to `archive/multiturn-eval/` — out of scope for the presented project |
| #73/#75 | Question coverage + unanswerable exit | ✅ Done | A question naming a value that does not exist is refused, not answered |
| #76 | Plan diversity + grain rationale | ✅ Done | Both now enabled in config — see the caveat under Roadmap |
| #87/#89 | Follow-up contamination + topic switch | ✅ Done | Condensation veto; history dropped when the subject changes |
| #91 | Trust verdict + plan conformance + back-translation | ✅ Done | Five gates, a verdict, and the grading shown on screen |
| #101 | Measuring the detector | ✅ Done | AUROC, per-check rates, risk–coverage — **4 of 5 gates measured** |
| #103 | Capturing the judge signal | ✅ Done | `bird --trust-eval`; **needs a run to produce numbers** |

**Retrieval correctness**

| Issue | Area | Status | Notes |
|---|---|---|---|
| #79 | Exclude the assistant's own tables from the index | ✅ Done | Requires a re-index |
| #81 | Tag schema chunks by owning database; scoping | ✅ Done | Requires a re-index |
| #83 | Surface the database scope in the UI | ✅ Done | Sidebar picker, per-request `db_scope` |
| #85 | Align live schema + effort budget with `bird.py` | ✅ Done | Full schema per scope; no tier inference live |
| #93/#95 | Grain false positive; automatic domain routing | ✅ Done | Routing from the retrieval ranking; budget 4k → 12k |
| #97/#99 | Record ids and unindexable columns | ✅ Done | No refusal on evidence the index never had |
| #6 | User isolation & roles | ⏳ Planned | Read-only DB role landed; auth still open |
| #12 | Knowledge Graph (LlamaIndex) | 🔮 Deferred | Not started |

---

## Roadmap

**Runs that have not been done.** Most of what is left is measurement, not
code. Every item below is a table or a figure the write-up currently cannot
show.

- [ ] **Judge gate** — `bird --trust-eval`, then re-run `trust_eval`. Turns
      *4 of 5 gates measured* into 5, and is the signal the literature puts
      well above self-consistency.
- [ ] **Cross-provider judge** — the same run with `--judge-model` /
      `--judge-base-url` pointed elsewhere. Comparing it against the self-judge
      is the single most citable result available here.
- [ ] **Ablations** — `--no-exec-voting`, `--no-schema-linking`, `--no-refine`,
      and the newer `retrieval.full_schema_when_scoped` / `retrieval.auto_scope`
      / `sql.live_tier` knobs. All shipped unmeasured.
- [ ] **`sql.fast_path` / `sql.answer_reuse` on vs off** — EX must be unchanged
      (GH#57 is accuracy-neutral by design); `cost_usd` shows the saving.

> ⚠️ `plan_diversity` and `plan_rationale` are now **enabled** in
> `config.yaml`. They were default-off pending an A/B, so their effect is
> currently unquantified — the 750 records already in `outputs/` were all
> produced with them off and serve as the baseline for a re-run.

**Code**
- [ ] Issue #6 — User isolation, roles, JWT auth (read-only DB role already landed)
- [ ] GitHub Actions CI wired to the test suite
- [ ] Selective prediction — the verdict annotates but never withholds; the
      risk–coverage curve exists, the abstention policy does not
- [ ] Fit the gate weights instead of the hand-written worst-gate rule, and
      report calibration error against it
- [ ] Dialect-aware `ValueIndex` — would let the three value-grounded checks be
      measured on the SQLite benchmark, and let the multi-turn harness ground
      values
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
