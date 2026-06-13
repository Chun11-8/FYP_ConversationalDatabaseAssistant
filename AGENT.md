# AGENT.md

Cross-tool agent guide for the **Conversational Database Assistant**. This is a concise
mirror of the critical rules — **`CLAUDE.md` is the canonical, full guide; read it first.**
Companion docs: `REQUIREMENTS_TODO.md`, `GITHUB_ISSUES.md`, `BUILD_PROMPTS.md`,
`FYPWorkPlan(System Design).pdf`.

## TL;DR
A read-only natural-language → SQL assistant. Pipeline is a LangGraph `StateGraph` in
`backend/agents/db_agent.py`: intent → schema RAG → SQL gen+rerank → validate → execute →
format. Frontend (React/Vite) is currently mocked; there is **no HTTP API yet** (top
priority). Framework intent is LangGraph + LlamaIndex, but LlamaIndex/KG is not built yet.

## Non-negotiable rules
1. **Read-only forever.** Never generate/allow INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/
   CREATE. Keep blocks in both the intent classifier and `sql_validator_agent.py`.
2. **SELECT-only, explicit columns, mandatory `LIMIT`** (default 100, max 500). No `SELECT *`.
3. **No hardcoded secrets.** Read from `backend/config/secrets.env`. Existing committed
   secrets (OpenAI key in `retrieval/indexing.py`; Postgres password `Toyo20001108!` in
   `pgvector_store.py` + `compose.yml`) are compromised — fix via Issue #0, never add more.
4. **Absolute imports rooted at `backend.`**; run modules from the repo root.
5. Keep prompts in `.md`/`.tpl` files (pattern: `agents/prompt.md`, `pipelines/prompt_response.md`).

## Known broken / stubbed (don't assume these work)
- No FastAPI server — build it (Issue #1).
- History bug: `classify_intent` injects the literal string `"history_str"` (Issue #3).
- `SessionMemory` is in-RAM only; lost on restart (Issue #3).
- `schema_versioning_agent.py` calls methods that don't exist — will crash (Issue #7).
- `few_shot_examples` is always `[]`; semantic cache not wired (Issue #4). Reuse the
  commented-out `kb_embeddings` block in `pgvector_store.py`.
- `tools/*.py`, `tests/*.py`, `config.yaml`, `llm_pipeline.py` are empty.
- `agent.run()` returns a bare string — refactor to a dict for the API/feedback work.

## Run
```bash
docker compose -f backend/config/compose.yml up -d
python -m backend.retrieval.indexing build --db FYP
python -m backend.agents.db_agent          # until the API exists
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## Workflow
Pick an issue from `GITHUB_ISSUES.md`, respect its Depends-on order, keep the change
scoped, match existing patterns, add tests in `backend/tests/`, smoke-run the pipeline,
and verify no secrets are committed (`grep -ri "sk-" backend/`). Full details, the layer
map, and the definition-of-done are in **`CLAUDE.md`**.
