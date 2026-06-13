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
12. [Development Status](#development-status)
13. [Roadmap](#roadmap)
14. [Contributing](#contributing)

---

## Overview

The assistant operates as a **multi-agent pipeline** built on a LangGraph `StateGraph`. A user types a plain-English question; the system:

1. **Classifies intent** — `READ` (SQL query), `GENERAL` (how-to answer), `WRITE_BLOCKED` (mutation rejected), or `AMBIGUOUS` (needs clarification).
2. **Retrieves schema context** via agentic RAG — pgvector embeddings of table/column descriptions narrow down the relevant schema.
3. **Generates SQL** — an LLM-powered SQL agent produces candidate queries, which are ranked by a reranker.
4. **Validates SQL** — syntax (sqlglot), safety (injection/mutation guard), and semantic checks all run before execution.
5. **Executes the query** — read-only execution against PostgreSQL.
6. **Formats the response** — results are rendered as markdown tables, inline answers, or exported to CSV for large result sets.

The system is **strictly read-only** — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and `TRUNCATE` are blocked at both the intent-classification and validation layers.

---

## Architecture

```
User Query
    │
    ▼
classify_intent ──┬── WRITE_BLOCKED ──► handle_write_blocked ──► END
                  ├── GENERAL       ──► handle_general       ──► END
                  └── READ / AMBIGUOUS
                          │
                          ▼
                   retrieve_schema   (pgvector RAG — top-k tables + few-shot pairs)
                          │
                          ▼
                   generate_sql      (SQLCodeAgent — candidate SQL + reranking)
                          │
                          ▼
                   validate_sql ─────┬── valid   ──► execute_sql ──► format_response ──► END
                                     └── invalid ──► fallback_handler ──► (retry, max 2 rounds)
```

### Agent & Module Responsibilities

| Component | Module | Description |
|---|---|---|
| Orchestrator | `backend/agents/db_agent.py` | LangGraph `StateGraph`; wires all nodes |
| SQL Generator | `backend/agents/sql_code_agent.py` | Generates + reranks candidate SQL |
| SQL Validator | `backend/agents/sql_validator_agent.py` | Syntax, safety, semantic validation |
| Schema RAG | `backend/retrieval/` | pgvector-based schema embedding & retrieval |
| Memory | `backend/memory/session_memory.py` | In-RAM session turn history |
| Response Optimizer | `backend/pipelines/postprocessing.py` | Formats results (inline / table / CSV) |
| HTTP API | `backend/api/main.py` | FastAPI server exposing the agent over HTTP |
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
│   │   ├── sql_code_agent.py           # SQL generation + reranking
│   │   ├── sql_validator_agent.py      # Syntax / safety / semantic validation
│   │   ├── schema_versioning_agent.py  # Stub — not yet functional
│   │   └── prompt.md                   # Intent classification system prompt
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
│   │   └── session_memory.py           # In-RAM session turn history
│   ├── pipelines/
│   │   ├── postprocessing.py           # ResponseOptimizer (inline/table/CSV)
│   │   └── prompt_response.md          # Response formatting prompt template
│   ├── retrieval/
│   │   ├── pgvector_store.py           # Canonical vector store (pgvector)
│   │   ├── pinecone_store.py           # Alternate vector store (Pinecone)
│   │   ├── indexing.py                 # CLI: build/query the schema index
│   │   ├── embedding.py                # OpenAI embedding client
│   │   ├── data_processing.py          # Schema extraction from Postgres
│   │   └── retriever.py                # High-level retrieve() interface
│   ├── tests/
│   │   ├── test_config.py              # Config loader unit tests
│   │   ├── test_api.py                 # FastAPI integration tests (httpx)
│   │   ├── test_agent.py               # Agent tests (stub)
│   │   ├── test_pipeline.py            # Pipeline tests (stub)
│   │   └── test_tools.py               # Tool tests (stub)
│   └── tools/                          # External tool stubs (planned)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Root component — session + chat loop
│   │   ├── main.tsx                    # React entry point
│   │   ├── index.css                   # Design system (dark-minimal theme)
│   │   ├── types.ts                    # TypeScript interfaces (Message, McqMessage…)
│   │   ├── api/
│   │   │   └── client.ts               # API client (createSession, sendChat…)
│   │   └── components/
│   │       ├── ChatLayout.tsx          # App shell (sidebar + top bar)
│   │       ├── ChatMessage.tsx         # Message renderer (markdown, SQL panel, badges)
│   │       └── ChatInput.tsx           # Auto-resize textarea + send button
│   ├── package.json
│   └── vite.config.ts
├── dataset/                            # Sample data (minidev)
├── outputs/                            # CSV exports from large query results
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
```

> **Never commit `secrets.env`** — it is git-ignored. Only `secrets.env.example` is tracked.

### 2. Review `backend/config/config.yaml`

Non-secret settings live here; edit as needed:

```yaml
llm:
  model: gpt-4o-mini          # OpenAI model used for all LLM calls

db:
  dialect: postgres
  host: "127.0.0.1"
  port: 5432
  name: FYP
  user: postgres

retrieval:
  vector_store: pgvector       # or "pinecone"
  top_k_tables: 5              # schema chunks to retrieve per query
  top_k_kb: 3                  # few-shot examples to retrieve

response:
  inline_row_limit: 10         # rows returned inline in markdown
  table_row_limit: 100         # rows rendered as a markdown table
  csv_row_limit: 500           # rows written to CSV (larger results)
```

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

### Optional — Run tests

```bash
pytest backend/tests/ -v
```

### Optional — Smoke-test the agent directly (no API)

```bash
python -m backend.agents.db_agent
```

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
  "validation": { "syntax": true, "safety": true, "semantic": true },
  "row_count": 10,
  "csv_path": null
}
```

| Field | Type | Description |
|---|---|---|
| `response` | string | Natural-language answer in markdown |
| `sql` | string \| null | The SQL query used (null for GENERAL intent) |
| `intent` | string | `READ`, `GENERAL`, `WRITE_BLOCKED`, or `AMBIGUOUS` |
| `validation` | object | Validation check results |
| `row_count` | integer \| null | Number of rows returned |
| `csv_path` | string \| null | CSV filename for large result sets (>100 rows) |

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
| **Responsive layout** | Collapsible sidebar on mobile screens |

---

## Safety & Security

The system enforces read-only semantics at multiple independent layers:

| Layer | Mechanism |
|---|---|
| Intent classifier | Rejects any query classified as `WRITE_BLOCKED` before SQL is generated |
| SQL validator | `sqlglot` parse-tree check blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE` |
| SQL validator | Injection guard — rejects string-concatenation patterns, stacked statements, and dangerous functions |
| SQL validator | `LIMIT` enforcement — every generated query must carry a `LIMIT` (max 500) |
| Response optimizer | Row limits applied before display or CSV export |
| Download endpoint | Path-traversal guard — only files inside `outputs/` are served |

**Secret management:**
- All secrets are loaded at startup by `load_config()` from environment variables or `secrets.env`.
- No secrets appear in any tracked file — `secrets.env` is git-ignored.
- `config.yaml` holds only non-sensitive defaults.

---

## Development Status

| Issue | Area | Status | Notes |
|---|---|---|---|
| #0 | Security & config hardening | ✅ Done | Secrets in env, `load_config()` with validation |
| #1 | FastAPI HTTP API | ✅ Done | All endpoints live with Pydantic models |
| #2 | Frontend ↔ Backend integration | ✅ Done | Real API wiring, dark-minimal UI |
| #3 | Memory — history bug + persistence | ⏳ Planned | History injection bug; memory is in-RAM only |
| #4 | Semantic cache / few-shot | ⏳ Planned | `few_shot_examples` always `[]` currently |
| #5 | Feedback collection (thumbs) | ⏳ Planned | Not started |
| #6 | User isolation & roles | ⏳ Planned | No authentication today |
| #7 | Schema versioning | ⏳ Planned | Stub only — calls non-existent methods |
| #9 | Tools / external APIs | ⏳ Planned | Empty stubs |
| #10 | Response optimizer guardrails | ⏳ Planned | Basic implementation exists |
| #11 | Full test suite + CI | ⏳ Planned | Config + API tests exist; agent/pipeline stubs empty |
| #12 | Knowledge Graph (LlamaIndex) | 🔮 Deferred | Not started |

---

## Roadmap

**Phase 1 — Core pipeline (complete)**
- [x] Issue #0 — Security & config hardening
- [x] Issue #1 — FastAPI HTTP server
- [x] Issue #2 — Frontend integration + UI

**Phase 2 — Memory & intelligence**
- [ ] Issue #3 — Fix history injection bug; add Postgres-backed session persistence
- [ ] Issue #4 — Semantic cache: store query→SQL pairs; retrieve as few-shot examples
- [ ] Issue #5 — Feedback (thumbs up/down); gate cache writes on positive feedback

**Phase 3 — Access control & reliability**
- [ ] Issue #6 — User isolation, roles, JWT auth
- [ ] Issue #7 — Schema versioning: detect diffs, re-embed changed tables
- [ ] Issue #9 — Tools layer: web search for GENERAL intent
- [ ] Issue #10 — Response optimizer guardrails

**Phase 4 — Quality**
- [ ] Issue #11 — Complete test suite, GitHub Actions CI

**Phase 5 — Optional**
- [ ] Issue #12 — Knowledge Graph via LlamaIndex (PropertyGraphIndex + SQLTableRetriever)

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
