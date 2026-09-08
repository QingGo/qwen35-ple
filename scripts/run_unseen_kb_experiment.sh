#!/usr/bin/env bash
# Run an unseen-KB memory-reading experiment.
#
# Uses KB token streams produced by scripts/build_kb_token_streams.py:
#   train.tokens.npy  -> reader training (seen KB)
#   eval.tokens.npy   -> held-out/unseen KB evaluation
#
# Protocol:
#   1. Train a real reader on the seen KB.
#   2. Train a control reader on the seen KB with shuffled PLE rows.
#   3. Evaluate both loaded readers on the unseen KB, plus no-reader.
#
# This tests whether the reader learns to use arbitrary external PLE memory
# rather than merely memorizing a particular training corpus.
#
# Usage:
#   bash scripts/run_unseen_kb_experiment.sh \
#     --pyenv /path/to/python \
#     --train-tokens data/phase1/kb-wiki-tokens/train.tokens.npy \
#     --eval-tokens data/phase1/kb-wiki-tokens/eval.tokens.npy \
#     --rows-dir /path/to/qwen38-rows \
#     --model-dir /path/to/qwen38-ple \
#     --steps 20 --seeds 0 --output outputs/unseen-kb-wiki.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN_TOKENS=""
EVAL_TOKENS=""
ROWS_DIR="${ROWS_DIR:-/home/zeng/qwen38-rows}"
MODEL_DIR="${MODEL_DIR:-/home/zeng/qwen38-ple}"
MODEL="${MODEL:-data/models/Qwen3.5-0.8B}"
STEPS="${STEPS:-20}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-0}"
OFFICIAL_READER_PATH="${OFFICIAL_READER_PATH:-data/official_ple_reader.pt}"
OUTPUT="${OUTPUT:-outputs/unseen-kb.json}"
READER="${READER:-official}"
FORCE="${FORCE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyenv) PYTHON="$2"; shift 2 ;;
    --train-tokens) TRAIN_TOKENS="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --rows-dir) ROWS_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --official-reader-path) OFFICIAL_READER_PATH="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --reader) READER="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TRAIN_TOKENS" || -z "$EVAL_TOKENS" ]]; then
  echo "ERROR: --train-tokens and --eval-tokens are required" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
READER_DIR="$(dirname "$OUTPUT")/unseen-readers"
mkdir -p "$READER_DIR"

for SEED in $SEEDS; do
  for MODE in real control; do
    CHECKPOINT="${READER_DIR}/reader-${MODE}-seed${SEED}.pt"
    if [[ "$FORCE" == "0" && -f "$CHECKPOINT" ]]; then
      echo "[unseen-kb] skip training $CHECKPOINT"
      continue
    fi
    echo "[unseen-kb] train $MODE seed=$SEED on seen KB -> $CHECKPOINT"
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$TRAIN_TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --steps "$STEPS" \
      --seq-len "$SEQ_LEN" \
      --lr "$LR" \
      --seeds "$SEED" \
      --modes "$MODE" \
      --save-reader "$CHECKPOINT" \
      --output "${READER_DIR}/train-${MODE}-seed${SEED}.json"
  done
done

# Eval each loaded reader on the unseen KB. no-reader is computed once.
for SEED in $SEEDS; do
  for MODE in real control; do
    CHECKPOINT="${READER_DIR}/reader-${MODE}-seed${SEED}.pt"
    echo "[unseen-kb] eval $MODE seed=$SEED on unseen KB"
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$EVAL_TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --load-reader "$CHECKPOINT" \
      --seeds "$SEED" \
      --modes "$MODE" \
      --output "${READER_DIR}/eval-${MODE}-seed${SEED}.json"
  done
  echo "[unseen-kb] eval no-reader seed=$SEED on unseen KB"
  "$PYTHON" -u scripts/run_phase0.py \
    --live-store \
    --tokens-npy "$EVAL_TOKENS" \
    --rows-dir "$ROWS_DIR" \
    --model-dir "$MODEL_DIR" \
    --model "$MODEL" \
    --seeds "$SEED" \
    --modes no-reader \
    --output "${READER_DIR}/eval-no-reader-seed${SEED}.json"
done

echo "[unseen-kb] done. Readers in $READER_DIR; per-arm JSONs in $READER_DIR"
