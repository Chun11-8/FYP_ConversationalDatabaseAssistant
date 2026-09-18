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

## Engines

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
pytest backend/tests/test_bird_eval.py backend/tests/test_db_backends.py -v
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
