# Build Prompts — paste these to Claude one at a time, in order

Each block is self-contained. Run them sequentially; verify the app still works
between blocks. Replace bracketed notes if your setup differs.

---

## Prompt 0 — Secret cleanup + config loader + deps  *(do first)*
```
There is a hardcoded live OpenAI API key in backend/retrieval/indexing.py (around
line 394, inside a commented block). Remove it entirely and make sure every module
reads the key from backend/config/secrets.env via python-dotenv instead.

Then build a real config loader: backend/config/config.yaml currently empty but the
code reads config["llm"]["model"], config["llm"]["api_key"], config["db"]["dialect"].
Create a load_config() function (backend/config/__init__.py or config_loader.py) that
merges config.yaml with secrets.env and returns that dict. Update db_agent.py and
indexing.py to use it instead of hardcoding config in __main__.

Finally fix requirements.txt: change "panda" to "pandas" and add fastapi, uvicorn,
pyyaml, python-multipart. Do not add llama-index yet.
```

## Prompt 1 — Business Layer: FastAPI server
```
Build a FastAPI server at backend/api/main.py that exposes the OrchestrationAgent
from backend/agents/db_agent.py over HTTP. Instantiate OrchestrationAgent ONCE at
startup using the config loader.

Endpoints:
- POST /session            -> create a new session_id (uuid), return {session_id}
- POST /chat               -> body {session_id, message}; call agent.run(...);
                              return {response, sql, intent}
- GET  /history/{session_id} -> return the turn history from SessionMemory
- GET  /download/{filename} -> serve a CSV from the outputs/ directory as a file

Enable CORS for http://localhost:5173 (Vite dev). Add a `uvicorn backend.api.main:app
--reload` run note to the README. Test that POST /chat returns a real answer.
```

## Prompt 2 — Connect the React frontend to the API
```
The frontend in frontend/src/App.tsx is fully mocked (SIMULATED_DELAY, hardcoded bot
replies, fake MCQ). Replace the mock with real calls to the FastAPI backend.

- On first load, call POST /session and keep the session_id in state.
- handleSend should POST /chat {session_id, message} and append the real response.
- Render assistant messages as markdown (install react-markdown + remark-gfm) so SQL
  blocks and tables display correctly.
- Wire "New Chat" to POST /session and clear messages.
- Add a small typing indicator tied to the real request (not a timer).
- If a response contains a CSV download path, render a download link to /download.
Keep the existing visual style/components.
```

## Prompt 3 — Fix history bug + persist memory to Postgres
```
Two memory fixes:

1) BUG: in backend/agents/db_agent.py classify_intent() and generate_sql(), the
conversation history is injected as the literal string "history_str". Format the real
history list into a string and inject it properly in both places.

2) backend/memory/session_memory.py is an in-RAM dict, so sessions are lost on restart.
Back it with PostgreSQL: create `sessions` and `turns` tables, and make add()/
get_history()/get_last_sql()/clear_session() read/write Postgres. Returning users with
the same session_id must reload prior turns. Keep the same method signatures so
db_agent.py needs no changes beyond construction.

Also add short-term-memory summarization: when a session exceeds N turns, summarize the
oldest turns into a single context string instead of dropping them.
```

## Prompt 4 — Semantic cache (few-shot query→SQL pairs)
```
Implement the Semantic Retrieval / semantic-cache feature. Right now few_shot_examples
is hardcoded to [] in db_agent.py.

- Use the commented-out knowledge-base/pairs table in backend/retrieval/pgvector_store.py
  (or add a `query_sql_pairs` table) to store embedded (user_query, sql) pairs.
- After a query is validated AND executed successfully, embed the (user_query, best_sql)
  pair and upsert it.
- Add retrieve_few_shots(query, top_k=3) that returns the most similar past pairs.
- In db_agent.retrieve_schema (or a new node), populate few_shot_examples from it and
  pass them into SQLCodeAgent.generate.
Also create a `query_interactions` log table that records every turn (query, sql,
response, valid) as physical text.
```

## Prompt 5 — Feedback collection
```
Add user feedback on responses.
- Frontend: thumbs up / thumbs down buttons on each assistant message.
- API: POST /feedback {turn_id, rating} storing into a `feedback` table.
- Gating: only write a (query, sql) pair to the semantic cache from Prompt 4 when the
  user rated it positively (or it passed validation and got no negative rating).
Surface a simple "was this helpful?" affordance without cluttering the UI.
```

## Prompt 6 — User roles & isolation
```
Implement 2 roles (admin, user) per the system design, with group-based memory
isolation (not per-user). Add a minimal users table + lightweight auth (a header token
or simple login is fine for an FYP). Scope sessions/memory by group. Restrict the
schema-versioning trigger (next prompt) to admins only. Populate
backend/memory/user_profiles.json or replace it with the users table.
```

## Prompt 7 — Schema Versioning Layer + Admin Hook
```
backend/agents/schema_versioning_agent.py calls methods that don't exist
(self.retriever.db.get_schema_diff, embed_table, update_table_embedding,
remove_table_embedding, persist_hash). Implement the whole layer:

- Schema Changes: hash the live schema by querying information_schema; compare to a
  stored hash; extract a structured diff of added/modified/dropped tables & columns.
- Change Classifier: Additive (append embeddings only), Modified (lower confidence on
  affected pairs), Breaking (archive pairs + remove vector chunks/KG nodes).
- Knowledge Update: surgically re-embed ONLY affected tables in pgvector.
- Schema Version Store: persist {hash, timestamp, diff_log} each successful cycle.
- Admin Hook: POST /admin/schema-sync (admin-only) that runs the above asynchronously
  AFTER confirming the DB change committed, so a RAG-update failure can't corrupt the DB.
```

## Prompt 8 — Tools / Connectivity Layer (web search for GENERAL)
```
backend/tools/*.py are empty. Build a small tool layer:
- A web_search tool, and route GENERAL-intent questions (e.g. "how do I write a SELECT
  in Postgres?") through it instead of db_agent.handle_general calling the LLM directly.
- A simple tool registry/dispatcher the orchestrator can call.
- Remove the flight/hotel/weather stub files unless they are intentionally part of this
  project (they look like leftover scaffold — confirm before deleting).
```

## Prompt 9 — Guardrails + tests
```
- Add an output-layer guardrail in pipelines/postprocessing.py / orchestration that
  re-checks for SQL-injection patterns and summarizes over-long GENERAL answers.
- Fill the empty tests in backend/tests/: smoke tests for SQLCodeAgent, SQLValidatorAgent,
  the validation safety rules (DML must be blocked), and an API /chat happy-path test.
```

## Prompt KG (optional, deferred) — Knowledge Graph via LlamaIndex
```
The system design mandates LlamaIndex (PropertyGraphIndex, SQLTableRetriever,
NLSQLTableQueryEngine) for resolving multi-table FK join paths, but the code is currently
all LangChain/pgvector. Add a LlamaIndex-based Knowledge Graph module that builds a
PropertyGraphIndex of tables + FK edges from the Postgres schema, and use it to enrich
schema_context with join paths for multi-table queries. Keep the existing pgvector RAG;
the KG is an additive enhancement, wired into db_agent.retrieve_schema.
```
