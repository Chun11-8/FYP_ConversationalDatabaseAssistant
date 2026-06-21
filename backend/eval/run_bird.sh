#!/usr/bin/env bash
#
# run_bird.sh — End-to-end BIRD mini-dev benchmark for THIS system.
#
# Generates predictions with our system (backend.eval.bird_predict) and then
# scores them with the OFFICIAL BIRD evaluation scripts (EX, F1, VES).
#
# Prerequisites
# -------------
#   1. BIRD mini-dev dataset downloaded (questions json, gold sql, dev_databases).
#   2. Official evaluator cloned:  git clone https://github.com/bird-bench/mini_dev.git
#   3. OPENAI_API_KEY exported (or set in backend/config/secrets.env).
#   4. Run from the repo root:  bash backend/eval/run_bird.sh
#
# Configure paths below, or override any of them via environment variables, e.g.
#   BIRD_DATA=~/mini_dev_data MINI_DEV_REPO=~/mini_dev bash backend/eval/run_bird.sh
#
set -euo pipefail

# ── Paths (edit these, or override via env vars) ───────────────────────────────

# Root of the downloaded mini-dev dataset (contains mini_dev_sqlite.json,
# mini_dev_sqlite_gold.sql, and dev_databases/).
BIRD_DATA="${BIRD_DATA:-./mini_dev_data}"

# Root of the cloned official evaluator repo (contains evaluation/evaluation_ex.py).
MINI_DEV_REPO="${MINI_DEV_REPO:-./mini_dev}"

# Dialect to benchmark: SQLite | MySQL | PostgreSQL  (our generator targets SQLite).
SQL_DIALECT="${SQL_DIALECT:-SQLite}"

# Derived dataset files (SQLite dialect by default).
DIALECT_LOWER="$(echo "$SQL_DIALECT" | tr '[:upper:]' '[:lower:]')"
QUESTIONS="${QUESTIONS:-${BIRD_DATA}/mini_dev_${DIALECT_LOWER}.json}"
GOLD_SQL="${GOLD_SQL:-${BIRD_DATA}/mini_dev_${DIALECT_LOWER}_gold.sql}"
DB_ROOT="${DB_ROOT:-${BIRD_DATA}/dev_databases}"

# Where to write our predictions (the official scorer derives its log name from this).
PRED_DIR="${PRED_DIR:-${MINI_DEV_REPO}/sql_result}"
PRED_FILE="${PRED_FILE:-${PRED_DIR}/predict_mini_dev_MYSYS_${SQL_DIALECT}.json}"

# Where the official scorers write their logs.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${MINI_DEV_REPO}/eval_result}"

# Scorer tuning.
NUM_CPUS="${NUM_CPUS:-4}"
META_TIMEOUT="${META_TIMEOUT:-30.0}"

# Skip prediction generation and only re-score an existing PRED_FILE (1 = skip).
SKIP_GENERATE="${SKIP_GENERATE:-0}"

EVAL_DIR="${MINI_DEV_REPO}/evaluation"

# ── Sanity checks ──────────────────────────────────────────────────────────────

err() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$QUESTIONS" ]      || err "Questions file not found: $QUESTIONS"
[ -f "$GOLD_SQL" ]       || err "Gold SQL file not found: $GOLD_SQL"
[ -d "$DB_ROOT" ]        || err "dev_databases dir not found: $DB_ROOT"
[ -d "$EVAL_DIR" ]       || err "Official evaluator not found: $EVAL_DIR (clone bird-bench/mini_dev)"
[ -f "${EVAL_DIR}/evaluation_ex.py" ] || err "evaluation_ex.py missing in $EVAL_DIR"

mkdir -p "$PRED_DIR" "$EVAL_OUT_DIR"

base_name="$(basename "$PRED_FILE" .json)"

echo "──────────────────────────────────────────────────────────────────────────"
echo "  BIRD mini-dev benchmark"
echo "  dialect      : $SQL_DIALECT"
echo "  questions    : $QUESTIONS"
echo "  databases    : $DB_ROOT"
echo "  gold sql     : $GOLD_SQL"
echo "  predictions  : $PRED_FILE"
echo "  eval logs    : $EVAL_OUT_DIR"
echo "──────────────────────────────────────────────────────────────────────────"

# ── 1. Generate predictions with our system ────────────────────────────────────

if [ "$SKIP_GENERATE" = "1" ]; then
    echo "[1/2] SKIP_GENERATE=1 — reusing existing predictions: $PRED_FILE"
    [ -f "$PRED_FILE" ] || err "SKIP_GENERATE set but no predictions at $PRED_FILE"
else
    echo "[1/2] Generating predictions with the system architecture..."
    python -m backend.eval.bird_predict \
        --questions "$QUESTIONS" \
        --db-dir    "$DB_ROOT" \
        --output    "$PRED_FILE"
fi

# ── 2. Score with the official BIRD scripts (EX, F1, VES) ───────────────────────

echo "[2/2] Scoring with official BIRD evaluators..."

run_scorer() {
    local script="$1" metric="$2"
    echo ""
    echo "── ${metric} (${script}) ─────────────────────────────────────────────────"
    ( cd "$EVAL_DIR" && python3 -u "./${script}" \
        --db_root_path       "$DB_ROOT/" \
        --predicted_sql_path "$PRED_FILE" \
        --ground_truth_path  "$GOLD_SQL" \
        --diff_json_path     "$QUESTIONS" \
        --num_cpus           "$NUM_CPUS" \
        --meta_time_out      "$META_TIMEOUT" \
        --sql_dialect        "$SQL_DIALECT" \
        --output_log_path    "${EVAL_OUT_DIR}/${base_name}_${metric}.txt" )
}

run_scorer evaluation_ex.py  EX
run_scorer evaluation_f1.py  F1
run_scorer evaluation_ves.py VES

echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo "  Done. Logs written to: $EVAL_OUT_DIR/${base_name}_{EX,F1,VES}.txt"
echo "──────────────────────────────────────────────────────────────────────────"
