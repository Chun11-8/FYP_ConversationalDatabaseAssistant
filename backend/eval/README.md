# Benchmarking the system against BIRD (mini-dev)

Scores this text-to-SQL system against the
[BIRD mini-dev benchmark](https://github.com/bird-bench/mini_dev) using
`backend/eval/bird.py`, which computes EX, Soft-F1 and VES itself.

> **Official scoring is no longer wired up.** `bird_predict.py` (which wrote
> predictions in the official file format) and `run_bird.sh` (which drove the
> official `evaluation_ex.py` / `_f1.py` / `_ves.py` scorers) were removed in
> GH#132. What remains is this harness's own re-implementation of the three
> metrics. It is the right tool for iterating, and for comparing models against
> each other on identical questions — but a number from it is **not** the same
> artefact as a leaderboard score, and a write-up should say so. Restoring the
> official path means re-adding a script that emits `{question_id: sql}` JSON
> and feeding it to the cloned mini_dev repo.

## Ablations — measuring one knob at a time (GH#136)

Three pipeline features are off by default and can be toggled per run. Each
arm needs its own `--tag` or the second overwrites the first:

```bash
# arm A — feature off (the configured default)
python -m backend.eval.run_all_models --difficulty challenging --subset 40 \
    --engine sqlite --question-timeout 300 --tag plandiv_off

# arm B — feature on
python -m backend.eval.run_all_models --difficulty challenging --subset 40 \
    --engine sqlite --question-timeout 300 --tag plandiv_on -- --plan-diversity
```

| Flag | Config key | What it changes |
|---|---|---|
| `--plan-diversity` | `sql.plan_diversity` | asks the linker for a SECOND plan that disagrees on grain or aggregation, so voting tests the plan instead of confirming it |
| `--path-diversity` / `--no-path-diversity` | `sql.path_diversity` | replaces two direct candidates with a divide-and-conquer and a query-plan-CoT one (moderate/challenging) |
| `--n-candidates N` | `sql.candidates_by_tier` | how many candidates execution voting chooses between |
| `--no-exec-voting` | `sql.execution_voting` | selects by text rerank instead of by executing candidates |
| `--no-refine` | — | disables the one ReAct refine round |

Unset flags keep the `config.yaml` value, so a run naming none of them
reproduces the configured pipeline exactly.

**Every arm is recorded in the results file's `run_config`** — `plan_diversity`,
`path_diversity`, `candidates_effective`, `exec_voting`, `refine`, plus
`engine`, `sample`, `seed` and `question_timeout`. An ablation whose arm cannot
be read back off the file is not an ablation.

Keep `--seed` and `--subset` identical across arms, or the two runs answered
different questions and the comparison is meaningless.

## Engines

> **Parser note (GH#135):** the last-resort SQL extractor accepts unfenced
> statements beginning `WITH` as well as `SELECT`. Matching only `SELECT` was
> worse than missing a CTE: it returned the CTE's trailing SELECT *without* the
> `WITH` clause, yielding SQL that referenced a CTE which no longer existed.

BIRD publishes the benchmark twice, and the gold dialect must match the engine:

| `--engine` | Question file | Data |
|---|---|---|
| `postgres` (default) | `mini_dev_postgresql.json` | all tables in ONE PostgreSQL database |
| `sqlite` | `mini_dev_sqlite.json` | `--db-dir` of `{db_id}/{db_id}.sqlite` files |

Pairing `--engine postgres` with the SQLite question file is refused before the
run starts (GH#131): gold that cannot execute is recorded as `gold_error`, and
those questions are dropped from the EX denominator, so the run would otherwise
report a plausible score over whatever survived.

---


# Running a scored run

`bird.py` generates **and** scores in one pass, and `--subset` / `--difficulty`
keep a check cheap. Loading the data is covered in the root README under
[Loading Data into PostgreSQL](../../README.md#loading-data-into-postgresql).

```bash
# Smoke test on PostgreSQL (the default engine): 10 questions, ~1 min
python -m backend.eval.bird \
    --dev-json dataset/minidev/MINIDEV/mini_dev_postgresql.json \
    --output   outputs/bird_smoke.json \
    --subset 10

# The same on SQLite — needs --db-dir and the SQLite question file
python -m backend.eval.bird \
    --dev-json dataset/minidev/MINIDEV/mini_dev_sqlite.json \
    --db-dir   dataset/minidev/MINIDEV/dev_databases \
    --engine   sqlite --output outputs/bird_smoke_sqlite.json --subset 10

# Only the hard ones
python -m backend.eval.bird --dev-json … \
    --difficulty challenging --output outputs/bird_challenging.json
```

Under `--engine sqlite`, each database is looked up at
`<db-dir>/<db_id>/<db_id>.sqlite`, falling back to `<db_id>.db` — the official
mini-dev layout works unchanged, and so do older self-built sets.

---

## Offline tests (no dataset, no API key)

Both harnesses are unit-tested. From the repo root:

```bash
pytest backend/tests/test_bird_eval.py backend/tests/test_db_backends.py -v
```

---

## Interpreting the results & what to fix next

Every run writes one JSON file with three top-level keys — `run_config`
(which arm produced it), `metrics` (EX / EX_winnable / F1 / VES, plus
`by_difficulty` and `error_categories`) and `questions` (one record per
question). `error_analysis.py` is the library that assigns each failure its
`error_category`; the counts are already summarised in `metrics`:

```bash
python -I - <<'PY'
import json, collections
r = json.load(open("outputs/<your_run>.json"))
print(r["run_config"])
print(r["metrics"]["error_categories"])

bad = [q for q in r["questions"] if not q.get("ex")]
for q in bad[:3]:
    print("\nQ   :", q["question"])
    print("PRED:", q["pred_sql"])
    print("GOLD:", q["gold_sql"])
    print("WHY :", q.get("error_category"), q.get("pred_error") or "")
PY
```

Each record also carries `gold_error` — gold SQL that fails on the engine. Those
questions are unwinnable and are excluded from `EX_winnable`, which is why that
figure and plain `EX` can differ.

| Symptom in failures | Likely cause | Where to look |
|---|---|---|
| Wrong dialect function (`strftime` on PostgreSQL, `STRING_AGG` on SQLite) | the generator wrote for the other engine | `DIALECT_NOTES` in `sql_code_agent.py` |
| Valid SQL, 0 rows, wrong joins | join-column hallucination | schema linking + `strict_semantic` |
| Right rows, wrong order on ties | NULL ordering differs between the engines | the superlative rules in `prompt.md` |
| Times out / very slow | inefficient query | `--timeout`, and VES |

### Rough expectations (GPT-4o-mini class system)

| Difficulty | Expected EX |
|---|---|
| simple | 60–75% |
| moderate | 30–50% |
| challenging | 10–25% |
| **overall** | **35–55%** |

BIRD SOTA (full GPT-4 pipeline) is ~65–72% EX.
