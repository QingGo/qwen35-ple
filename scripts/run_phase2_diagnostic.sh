#!/usr/bin/env bash
# Small post-Phase-2 diagnostic matrix for the pure-PLE grafting question.
#
# Goal: with a fixed instruction-style QA prompt and saved readers, decide
# whether real PLE content beats the shuffled control on task-level metrics,
# or whether the previous BoolQ improvement was only a reader-format effect.
#
# Defaults:
#   corpora : PURE_WIKI PURE_CODE FW_STEM
#   modes   : real control no-reader
#   seeds   : 0 1 2
#   steps   : 500
#   QA      : instruction-style prompt (set QA_PROMPT_TEMPLATE / QA_BOOLQ_PROMPT_TEMPLATE)
#   readers : saved per corpus/mode/seed
#
# Extra arguments are forwarded to run_phase1_matrix.sh, so paths can be
# overridden without editing this file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/phase2-diagnostic}"
ROWS_DIR="${ROWS_DIR:-/dev/shm/qwen38-rows}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen38_ple}"
MODEL="${MODEL:-$ROOT/models/Qwen3.5-0.8B}"
LAYER="${LAYER:-2}"
CORPORA="${CORPORA:-PURE_WIKI PURE_CODE FW_STEM}"
STEPS="${STEPS:-500}"
SEEDS="${SEEDS:-0 1 2}"
MAX_NEW="${MAX_NEW:-96}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-16}"
QA_BATCH_MAX_TOKENS="${QA_BATCH_MAX_TOKENS:-2048}"
DEFAULT_QA_PROMPT=$'Question: {question}\nAnswer:'
DEFAULT_BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'
export QA_PROMPT_TEMPLATE="${QA_PROMPT_TEMPLATE:-$DEFAULT_QA_PROMPT}"
export QA_BOOLQ_PROMPT_TEMPLATE="${QA_BOOLQ_PROMPT_TEMPLATE:-$DEFAULT_BOOLQ_PROMPT}"
export RESUME="${RESUME:-1}"

echo "=== [phase2-diagnostic] output=$OUTPUT_DIR ==="
echo "=== [phase2-diagnostic] corpora=$CORPORA ==="

exec bash "$SCRIPT_DIR/run_phase1_matrix.sh" \
  --pyenv "$PYTHON" \
  --rows-dir "$ROWS_DIR" \
  --model-dir "$MODEL_DIR" \
  --model "$MODEL" \
  --layer "$LAYER" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --seeds "$SEEDS" \
  --max-new "$MAX_NEW" \
  --qa-batch-size "$QA_BATCH_SIZE" \
  --qa-batch-max-tokens "$QA_BATCH_MAX_TOKENS" \
  --save-reader \
  --corpora $CORPORA \
  "$@"
