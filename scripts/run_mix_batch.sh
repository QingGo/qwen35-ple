#!/usr/bin/env bash
# Run Phase 0 three-arm QA for the M1-M5 mixed 1M-token corpora.
#
# The mixed corpora are expected to already exist under data/mixes/M* (see
# scripts/build_mix.py with --exclude-qa).  On WSL:
#
#   bash scripts/run_mix_batch.sh \
#     --pyenv .venv/bin/python \
#     --rows-dir /home/zeng/qwen38-rows \
#     --model-dir data/models/Qwen3.5-0.8B \
#     --qa-file data/qa-expanded-150.json \
#     --mixes M1 M2 M3 M4 M5
#
# Each mix is run with real/control/no-reader, and with per-question progress
# logs enabled by the current run_phase0.py exact-match path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
ROWS_DIR="${ROWS_DIR:-/home/zeng/qwen38-rows}"
MODEL_DIR="${MODEL_DIR:-data/models/Qwen3.5-0.8B}"
MODEL="${MODEL:-data/models/Qwen3.5-0.8B}"
QA_FILE="${QA_FILE:-data/qa-expanded-150.json}"
STEPS="${STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
MIXES="${MIXES:-M1 M2 M3 M4 M5}"
FORCE="${FORCE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyenv) PYTHON="$2"; shift 2 ;;
    --rows-dir) ROWS_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --qa-file) QA_FILE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mixes)
      shift
      MIXES=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MIXES="${MIXES} $1"
        shift
      done
      MIXES="${MIXES# }"
      ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

for MIX in $MIXES; do
  TOKENS="data/mixes/${MIX}/tokens.npy"
  if [[ ! -f "$TOKENS" ]]; then
    echo "[mix-batch] ERROR: missing $TOKENS; run build_mix.py first" >&2
    exit 1
  fi

  OUT="${OUTPUT_DIR}/phase0-${MIX}-seed${SEEDS}.json"
  if [[ "$FORCE" == "0" && -f "$OUT" ]] && grep -q '"summary"' "$OUT"; then
    echo "[mix-batch] skip $MIX: $OUT already exists"
    continue
  fi
  echo "=== [mix-batch] $MIX -> $OUT ==="
  "$PYTHON" -u scripts/run_phase0.py \
    --live-store \
    --tokens-npy "$TOKENS" \
    --rows-dir "$ROWS_DIR" \
    --model-dir "$MODEL_DIR" \
    --model "$MODEL" \
    --reader official \
    --official-reader-path data/official_ple_reader.pt \
    --steps "$STEPS" \
    --seq-len "$SEQ_LEN" \
    --lr "$LR" \
    --seeds $SEEDS \
    --modes real control no-reader \
    --qa-exact-match \
    --qa-max-new-tokens 16 \
    --qa-file "$QA_FILE" \
    --output "$OUT"
done

echo "[mix-batch] all mixes done"
