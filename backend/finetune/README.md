# Fine-tuning `gpt-4o-mini` on this system's own data

This is a walkthrough, not just a reference — read it in order the first
time.

## 1. What fine-tuning actually is (and isn't)

**Prompt engineering** (everything in `backend/agents/prompt.md`,
`sql_code_agent.py`'s `TIER_STRATEGIES`/`PATH_STRATEGIES`, the schema-linking
plan, etc.) changes what you *ask* the model at inference time. The model's
weights never change — you're steering a fixed brain with better
instructions.

**Fine-tuning** changes the model's weights themselves, biasing it toward
the style of answer your training examples show, *before* you ask anything.
It's good at "remembering" narrow patterns — this project's evidence-formula
phrasing, this database's specific join shapes, your house style for
`CASE WHEN`/`STRFTIME`/quoting. It's not a substitute for reasoning through
a genuinely novel question it's never seen anything like.

**Practical upshot for this system**: fine-tuning is a *complement* to the
pipeline, not a replacement. Keep schema-linking, path-diversity,
execution-voting, everything — exactly as-is. A fine-tuned model just slots
in as a different `--model` value, same as the `gpt-5-mini` A/B run you
already did. Compare its BIRD EX against the base model on the *same*
pipeline to know if it actually helped.

## 2. Where the training data comes from

Two sources, both usable independently or combined:

- **BIRD gold pairs** — any `outputs/bird_results_*.json` file already has
  `question`, `evidence`, `gold_sql` for every question. Gold SQL is
  correct by definition, so every non-`gold_error` question is a valid
  training example.
- **Your live system's validated pairs** — every question→SQL pair the
  semantic cache accepted (`backend/memory/semantic_cache.py`'s
  `query_sql_pairs` table) is real usage on your actual database, not BIRD's
  toy databases.

`prepare_data.py` builds each training example using the **exact same
system + user prompt** `SQLCodeAgent` sends in production
(`_system_prompt()` + `_build_prompt()`), so the model learns to map that
specific prompt shape to correct SQL — not a generic Q→SQL mapping that
won't transfer to how you actually call it.

## 3. Step by step

### 3.1 Build the training JSONL

```bash
python -m backend.finetune.prepare_data \
    --bird-results outputs/bird_results_simple_0108.json \
                    outputs/bird_results_moderate_0108.json \
    --db-dir dataset/dev_databases \
    --output backend/finetune/train.jsonl \
    --split-eval 0.1
```

This writes `train.jsonl` (90% of examples) and `train.eval.jsonl` (10%,
held out — OpenAI scores the job against this during training so you can
see if it's overfitting). Open a couple of lines by hand:

```bash
head -1 backend/finetune/train.jsonl | python3 -m json.tool
```

Confirm the `assistant` message is the gold SQL you expect, fenced in
```` ```sql ```` — that's the exact shape `SQLCodeAgent._extract_sqls()`
parses at inference time, so the fine-tuned model's completions will be
parseable the same way.

**Useful flags:**
- `--difficulty simple moderate` — restrict to easier tiers first (cheaper,
  and isolates whether fine-tuning helps on the failure clusters that
  matter most before spending on the harder tier).
- `--max-examples 100` — cap cost while you're learning the workflow.
- `--from-semantic-cache` — also blend in real usage pairs (needs a live
  Postgres connection via `config.yaml`'s `db:` block). Schema context for
  these examples is introspected live from Postgres (the same code path the
  `/schema` API endpoint uses) — pass `--db-name` if it's not `config.yaml`'s
  default.

### 3.2 Run the fine-tuning job

```bash
python -m backend.finetune.run_job \
    --training-file backend/finetune/train.jsonl \
    --validation-file backend/finetune/train.eval.jsonl \
    --base-model gpt-4o-mini-2024-07-18
```

This uploads both files, creates the job, and polls until it finishes,
printing progress events as they arrive. A run over a few hundred examples
typically takes anywhere from several minutes to an hour or two — OpenAI
queues fine-tuning jobs, so wall-clock time varies.

On success you'll see something like:
```
Fine-tuning SUCCEEDED: ft:gpt-4o-mini-2024-07-18:your-org::abc123xyz
```
That string **is** your model id from now on.

Don't want to wait in the terminal? Pass `--no-wait` — it prints the job id
and exits immediately; check status later from a Python shell with
`client.fine_tuning.jobs.retrieve(job_id)`, or in the OpenAI dashboard under
Fine-tuning.

### 3.3 Cost

OpenAI bills fine-tuning per training token, separately from inference —
check [OpenAI's current pricing page](https://openai.com/api/pricing/) for
`gpt-4o-mini` fine-tuning rates rather than trusting a number here (it
changes). As an order-of-magnitude sense check: a few hundred short
SQL-generation examples is a small training file (tens of thousands of
tokens), so this is typically a few dollars, not a big spend — but always
check the live price before running a large job.

### 3.4 Evaluate it — the part that actually matters

A fine-tuned model existing proves nothing by itself. Run the same BIRD
comparison you already know how to do:

```bash
python -m backend.eval.bird \
    --dev-json dataset/mini_dev_sqlite.json \
    --db-dir dataset/dev_databases \
    --difficulty simple \
    --output outputs/bird_results_simple_finetuned.json \
    --model ft:gpt-4o-mini-2024-07-18:your-org::abc123xyz
```

Compare `metrics.EX` against your base-model run on the *same* questions.
If you trained on some of the simple/moderate questions, evaluate on a
**held-out slice you did NOT train on** (e.g. train on moderate+challenging,
evaluate on simple) — otherwise you're just measuring memorization, not
generalization.

`resolve_model()` (`backend/eval/bird.py`) never blocks a fine-tuned model
id — it's not in `config.yaml`'s `llm.selectable_models` allow-list, so
you'll see one warning line, and the run proceeds normally.

### 3.5 Promote it to live use (once validated)

Add the model id to `backend/config/config.yaml`:
```yaml
llm:
  selectable_models:
    - gpt-4o-mini
    - gpt-4o
    - gpt-4.1-mini
    - ft:gpt-4o-mini-2024-07-18:your-org::abc123xyz
```
It now shows up in the frontend's model picker (sidebar, above the OpenAI
API Key section — Issue #52) for anyone using the live chat UI to select.

## 4. What this does NOT change

No pipeline code changes — `db_agent.py`, `sql_code_agent.py`, the
schema-linking/voting/refine logic are all untouched. A fine-tuned model is
consumed purely through the existing `--model` (eval CLIs) or
`selectable_models` (live chat) mechanisms already built for GH#52's model
picker. If the fine-tuned model doesn't help, delete the config line and
you're back to exactly where you started.
