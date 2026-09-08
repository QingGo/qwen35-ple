#!/usr/bin/env bash
# Run the Phase 1 1M three-arm matrix (real / control / no-reader).
#
# This is the next step after scripts/build_phase1_corpora.py has produced the
# six corpora under data/phase1.  Each corpus is evaluated with the improved
# protocol: longer generation (default 96 new tokens, can be changed) and the
# generated text can later be re-scored with evaluate_generated_answers.py.
#
# Usage:
#   bash scripts/run_phase1_matrix.sh \
#     --pyenv /home/zeng/qwen35-ple/.venv/bin/python \
#     --rows-dir /home/zeng/qwen38-rows \
#     --model-dir /home/zeng/qwen38-ple \
#     --model data/models/Qwen3.5-0.8B \
#     --output-dir outputs \
#     --device cuda \
#     --corpora PURE_WIKI PURE_FINEWEB PURE_STEM PURE_CODE FW_CODE FW_STEM
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
ROWS_DIR="${ROWS_DIR:-/home/zeng/qwen38-rows}"
MODEL_DIR="${MODEL_DIR:-/home/zeng/qwen38-ple}"
MODEL="${MODEL:-data/models/Qwen3.5-0.8B}"
QA_FILE="${QA_FILE:-data/qa-expanded-150.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
CORPORA="${CORPORA:-PURE_WIKI PURE_FINEWEB PURE_STEM PURE_CODE FW_CODE FW_STEM}"
STEPS="${STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-0 1 2}"
MAX_NEW="${MAX_NEW:-96}"
READER="${READER:-official}"
DEVICE="${DEVICE:-cuda}"
OFFICIAL_READER_PATH="${OFFICIAL_READER_PATH:-data/official_ple_reader.pt}"
FORCE="${FORCE:-0}"
SKIP_QA="${SKIP_QA:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyenv) PYTHON="$2"; shift 2 ;;
    --rows-dir) ROWS_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --qa-file) QA_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --reader) READER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --official-reader-path) OFFICIAL_READER_PATH="$2"; shift 2 ;;
    --skip-qa) SKIP_QA=1; shift ;;
    --force) FORCE=1; shift ;;
    --corpora)
      shift
      CORPORA=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CORPORA="${CORPORA} $1"
        shift
      done
      CORPORA="${CORPORA# }"
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

for C in $CORPORA; do
  TOKENS="data/phase1/${C}/tokens.npy"
  if [[ ! -f "$TOKENS" ]]; then
    echo "[phase1-matrix] ERROR: missing $TOKENS; run build_phase1_corpora.py first" >&2
    exit 1
  fi

  OUT="${OUTPUT_DIR}/phase1-${C}.json"
  if [[ "$FORCE" == "0" && -f "$OUT" ]] && grep -q '"summary"' "$OUT"; then
    echo "[phase1-matrix] skip $C: $OUT already exists"
    continue
  fi

  echo "=== [phase1-matrix] $C -> $OUT ==="
  if [[ "$SKIP_QA" == "1" ]]; then
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --device "$DEVICE" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --steps "$STEPS" \
      --seq-len "$SEQ_LEN" \
      --lr "$LR" \
      --seeds $SEEDS \
      --modes real control no-reader \
      --output "$OUT"
  else
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --device "$DEVICE" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --steps "$STEPS" \
      --seq-len "$SEQ_LEN" \
      --lr "$LR" \
      --seeds $SEEDS \
      --modes real control no-reader \
      --qa \
      --qa-exact-match \
      --qa-max-new-tokens "$MAX_NEW" \
      --qa-file "$QA_FILE" \
      --output "$OUT"
  fi
done

echo "[phase1-matrix] all corpora done"
