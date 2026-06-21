# Benchmarking the system against BIRD (mini-dev)

This guide shows how to score **your** conversational text-to-SQL system against
the **official** [BIRD mini-dev benchmark](https://github.com/bird-bench/mini_dev).

There are two ways to do it. Read the difference before you pick one.

| | Path A — **Official scoring** (use for the report) | Path B — Quick self-check |
|---|---|---|
| Scoring code | Official BIRD `evaluation_ex.py` / `evaluation_ves.py` / `evaluation_f1.py` | Our own `backend/eval/bird.py` |
| Credible for FYP write-up? | ✅ Yes — identical to the leaderboard | ⚠️ Re-implementation, good for iteration |
| Generates predictions | `backend/eval/bird_predict.py` | built in |
| Subset / difficulty flags | ❌ (must score full set) | ✅ `--subset`, `--difficulty` |
| Needs the official repo cloned | ✅ | ❌ |

**Recommendation:** use **Path A** for the numbers you put in your report, and
**Path B** while iterating on prompts/agents because it's faster and self-contained.

> **One-command shortcut:** once the dataset is downloaded and the official repo
> is cloned, `backend/eval/run_bird.sh` does all of Path A (generate predictions
> → score EX + F1 + VES) in a single run. See [TL;DR](#tldr--one-command-path-a) below.

---

## TL;DR — one command (Path A)

```bash
# from the repo root, after downloading the data and cloning the official repo
export OPENAI_API_KEY=sk-your-key
BIRD_DATA=path/to/mini_dev_data \
MINI_DEV_REPO=path/to/mini_dev \
bash backend/eval/run_bird.sh
```

That script generates predictions with your system and runs all three official
scorers, writing logs to `<mini_dev>/eval_result/predict_mini_dev_MYSYS_SQLite_{EX,F1,VES}.txt`.
Useful overrides (all optional):

| Env var | Default | Purpose |
|---|---|---|
| `BIRD_DATA` | `./mini_dev_data` | dataset root (questions, gold, dev_databases) |
| `MINI_DEV_REPO` | `./mini_dev` | cloned official evaluator |
| `SQL_DIALECT` | `SQLite` | `SQLite` \| `MySQL` \| `PostgreSQL` |
| `NUM_CPUS` | `4` | parallelism for the scorers |
| `SKIP_GENERATE` | `0` | set `1` to re-score existing predictions without regenerating |

The sections below explain each step that this wrapper automates.

---

## How the official BIRD evaluation actually works

This is the part that confused you. The official evaluator **does not run your
model.** It only scores a *predictions file* you give it. The flow is:

```
                 ┌─────────────────────────┐
  BIRD questions │ mini_dev_sqlite.json     │
  + SQLite DBs   │ dev_databases/<id>/*.sqlite
                 └───────────┬─────────────┘
                             │  (1) bird_predict.py  ← YOUR system runs here
                             ▼
                 ┌─────────────────────────┐
  predictions    │ predict_..._SQLite.json │   {"0":"SELECT ...\t----- bird -----\tdb_id", ...}
                 └───────────┬─────────────┘
                             │  (2) official evaluation_ex.py
                             ▼
            gold:  mini_dev_sqlite_gold.sql   (one "SQL<TAB>db_id" per line)
                             ▼
                      EX / VES / F1 scores
```

So the only thing your system has to produce is the **predictions JSON** in the
exact official format. `backend/eval/bird_predict.py` does that.

### The official predictions format

A JSON object keyed by question index (0-based, **in the same order as the gold
file**). Each value is the predicted SQL, the separator `\t----- bird -----\t`,
then the `db_id`:

```json
{
  "0": "SELECT COUNT(*) FROM schools\t----- bird -----\tcalifornia_schools",
  "1": "SELECT MAX(T1.rating) FROM ...\t----- bird -----\tfinancial"
}
```

---

# Path A — Official scoring (step by step)

### Step 1 — Get the dataset

Download the mini-dev data (questions + SQLite databases + gold SQL) from the
[HuggingFace dataset](https://huggingface.co/datasets/birdsql/bird_mini_dev) or
the Google-Drive link in the [mini_dev README](https://github.com/bird-bench/mini_dev).
You should end up with:

```
mini_dev_data/
  mini_dev_sqlite.json            ← 500 questions (db_id, question, evidence, SQL, difficulty)
  mini_dev_sqlite_gold.sql        ← gold SQL, one "SQL<TAB>db_id" per line
  dev_databases/
    california_schools/california_schools.sqlite
    financial/financial.sqlite
    ...                           ← 11 databases
```

### Step 2 — Clone the official evaluation suite

```bash
git clone https://github.com/bird-bench/mini_dev.git
# the scorers live in mini_dev/evaluation/
```

### Step 3 — Set your OpenAI key (your system needs it to generate SQL)

```bash
export OPENAI_API_KEY=sk-your-key
# or put it in backend/config/secrets.env
```

### Step 4 — Generate predictions with YOUR system

Run from the **repo root** of *this* project. Generate over the **full** set
(no `--subset` / `--difficulty`, or the indices won't line up with the gold file):

```bash
python -m backend.eval.bird_predict \
    --questions  path/to/mini_dev_data/mini_dev_sqlite.json \
    --db-dir     path/to/mini_dev_data/dev_databases \
    --output     path/to/mini_dev/sql_result/predict_mini_dev_MYSYS_SQLite.json
```

> The output filename matters: the official `run_evaluation.sh` derives the log
> name from `basename <file> .json`. Keep the `predict_..._SQLite.json` shape.

This takes ~20–40 min for 500 questions (one LLM round-trip per question).

### Step 5 — Score with the official scripts

From inside `mini_dev/evaluation/`:

```bash
# Execution Accuracy (EX)
python3 evaluation_ex.py \
    --db_root_path       path/to/mini_dev_data/dev_databases/ \
    --predicted_sql_path path/to/mini_dev/sql_result/predict_mini_dev_MYSYS_SQLite.json \
    --ground_truth_path  path/to/mini_dev_data/mini_dev_sqlite_gold.sql \
    --diff_json_path     path/to/mini_dev_data/mini_dev_sqlite.json \
    --num_cpus 4 --meta_time_out 30.0 --sql_dialect SQLite \
    --output_log_path    path/to/mini_dev/eval_result/mysys_ex.txt

# Soft F1
python3 evaluation_f1.py   <same args, --output_log_path ..._f1.txt>

# Valid Efficiency Score (VES)
python3 evaluation_ves.py  <same args, --output_log_path ..._ves.txt>
```

The official scripts print accuracy broken down by **simple / moderate /
challenging** plus the overall score, and write the same to the log file.

> Tip: `mini_dev/evaluation/run_evaluation.sh` does all three at once — edit the
> paths at the top of that script to point at your files and run `bash
> run_evaluation.sh`.

### Step 6 — (Optional) MySQL / PostgreSQL dialects

mini-dev also ships `mini_dev_mysql.json` / `mini_dev_postgresql.json`. To score
those, generate predictions against that dialect and pass `--sql_dialect MySQL`
or `PostgreSQL` to the scorers. Our generator currently targets **SQLite**
(`bird_config` sets `dialect: "sqlite"`); change that if you benchmark another
dialect.

---

# Path B — Quick self-contained score (fast iteration)

No official repo needed — `backend/eval/bird.py` generates **and** scores in one
pass, and supports `--subset` / `--difficulty` for fast checks.

```bash
# Smoke test: 10 questions, ~1 min
python -m backend.eval.bird \
    --dev-json  path/to/mini_dev_data/mini_dev_sqlite.json \
    --db-dir    path/to/mini_dev_data/dev_databases \
    --output    outputs/bird_smoke.json \
    --subset 10

# Only the hard ones
python -m backend.eval.bird \
    --dev-json ... --db-dir ... \
    --difficulty challenging --output outputs/bird_challenging.json
```

> ⚠️ `bird.py` expects each DB at `<db-dir>/<db_id>/<db_id>.db`. The official
> mini-dev SQLite files are named `<db_id>.sqlite`. Either symlink/rename them,
> or use Path A (the generator reads `.sqlite`). This is the main gotcha when
> switching between the two paths.

---

## Offline tests (no dataset, no API key)

Both harnesses are unit-tested. From the repo root:

```bash
pytest backend/tests/test_bird_eval.py backend/tests/test_bird_predict.py -v
```

---

## Interpreting the results & what to fix next

After an official run, drill into failures to see *why* queries fail:

```bash
python - <<'EOF'
import json
preds = json.load(open("path/to/predict_mini_dev_MYSYS_SQLite.json"))
# Inspect a few generated queries
for k in list(preds)[:5]:
    sql, db = preds[k].split("\t----- bird -----\t")
    print(f"[{k}] db={db}\n  {sql}\n")
EOF
```

Map symptoms to the improvement tiers (GitHub issues):

| Symptom in failures | Likely cause | Addressed by |
|---|---|---|
| `no such function: STRING_AGG`, `ILIKE`, `::` cast errors | PostgreSQL syntax run on SQLite | **Tier 1 (#18)** dialect normalization ✅ done |
| Valid SQL, 0 rows, wrong joins | join-column hallucination | **Tier 2 (#19)** column validation + zero-row retry |
| Times out / very slow | inefficient query | **Tier 3 (#21)** + VES tuning |

### Rough expectations (GPT-4o-mini class system)

| Difficulty | Expected EX |
|---|---|
| simple | 60–75% |
| moderate | 30–50% |
| challenging | 10–25% |
| **overall** | **35–55%** |

BIRD SOTA (full GPT-4 pipeline) is ~65–72% EX.
