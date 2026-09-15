# Multi-turn evaluation — archived

Out of scope for the presented project. Kept here rather than deleted because
it works and is fully tested; nothing in `backend/` imports it.

## What it is

A harness that replays scripted conversations through the **full orchestration
graph** and scores every turn against gold SQL — the conversational counterpart
to the BIRD harness, which is single-turn only.

14 cases / 29 turns across seven categories: `literal_swap`, `ellipsis`,
`anaphora`, `refinement`, `topic_switch`, `clarification_reply`,
`legitimate_repeat`. It reports first-turn versus follow-up accuracy separately
— the gap between them is the actual measure of conversational competence — and
gives `stale_repeat` its own verdict, because "returned the previous answer" is
a specific diagnosable failure rather than generic wrongness.

Self-contained: it builds SQLite fixtures from `backend/eval/smoke_examples.py`,
so it needs no BIRD download and no PostgreSQL — only a key for SQL generation.

## Restoring it

```bash
git mv archive/multiturn-eval/multiturn.py          backend/eval/
git mv archive/multiturn-eval/multiturn_cases.json  backend/eval/
git mv archive/multiturn-eval/test_multiturn_eval.py backend/tests/
python -m pytest backend/tests/test_multiturn_eval.py -q
python -m backend.eval.multiturn --output outputs/multiturn.json
```

It depends only on `backend/eval/smoke_examples.py` and the orchestration
graph, both of which are still in the project.

## Note if you restore it later

`build_agent()` stubs four module-level seams in `db_agent` — `retrieve`,
`execute_query`, `expand_fk_tables` and `fetch_full_schema` — so the harness
runs against its own SQLite fixtures. If the graph grows another schema entry
point, that stub list needs the new one, or a `retrieval.active_db` in
`config.yaml` will silently send this harness to the live pgvector index
instead of the fixture it just built.
