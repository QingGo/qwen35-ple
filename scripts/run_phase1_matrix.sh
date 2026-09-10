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
#     --save-reader \
#     --corpora PURE_WIKI PURE_FINEWEB PURE_STEM PURE_CODE FW_CODE FW_STEM
#
# QA-only rerun against saved readers (training is skipped):
#   bash scripts/run_phase1_matrix.sh ... --load-reader --corpora PURE_WIKI
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
LAYER="${LAYER:-8}"
DEVICE="${DEVICE:-cuda}"
SCALE="${SCALE:-}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-16}"
QA_BATCH_MAX_TOKENS="${QA_BATCH_MAX_TOKENS:-2048}"
QA_PROMPT_TEMPLATE="${QA_PROMPT_TEMPLATE:-}"
QA_BOOLQ_PROMPT_TEMPLATE="${QA_BOOLQ_PROMPT_TEMPLATE:-}"
QA_CHAT_TEMPLATE="${QA_CHAT_TEMPLATE:-0}"
QA_CHAT_ENABLE_THINKING="${QA_CHAT_ENABLE_THINKING:-0}"
QA_SFT_FILE="${QA_SFT_FILE:-}"
QA_SFT_WEIGHT="${QA_SFT_WEIGHT:-0}"
QA_SFT_MAX_LEN="${QA_SFT_MAX_LEN:-512}"
QA_SFT_FULL_LOSS="${QA_SFT_FULL_LOSS:-0}"
QA_SFT_LOG_EVERY="${QA_SFT_LOG_EVERY:-0}"
GATE_REG_WEIGHT="${GATE_REG_WEIGHT:-0}"
GATE_OVERRIDE="${GATE_OVERRIDE:-}"
QA_SFT_WARMUP_STEPS="${QA_SFT_WARMUP_STEPS:-0}"
QA_SFT_LAZY="${QA_SFT_LAZY:-0}"
QA_SFT_LAZY_CACHE="${QA_SFT_LAZY_CACHE:-128}"
QA_SFT_CHAT_TEMPLATE="${QA_SFT_CHAT_TEMPLATE:-0}"
QA_SFT_CHAT_ENABLE_THINKING="${QA_SFT_CHAT_ENABLE_THINKING:-0}"
UNFREEZE_OFFICIAL_SOURCE="${UNFREEZE_OFFICIAL_SOURCE:-0}"
MODES="${MODES:-real control no-reader}"
BRIDGE_MLP="${BRIDGE_MLP:-1}"
OUT_MLP="${OUT_MLP:-1}"
OFFICIAL_READER_PATH="${OFFICIAL_READER_PATH:-data/official_ple_reader.pt}"
FORCE="${FORCE:-0}"
SKIP_QA="${SKIP_QA:-0}"
SAVE_READER="${SAVE_READER:-0}"
LOAD_READER="${LOAD_READER:-0}"
RESUME="${RESUME:-0}"
PARTIAL_DIR="${PARTIAL_DIR:-}"
BACKUP_DIR="${BACKUP_DIR:-}"

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
    --modes)
      shift
      MODES=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODES="${MODES} $1"
        shift
      done
      MODES="${MODES# }"
      ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --reader) READER="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --qa-batch-size) QA_BATCH_SIZE="$2"; shift 2 ;;
    --qa-batch-max-tokens) QA_BATCH_MAX_TOKENS="$2"; shift 2 ;;
    --qa-prompt-template) QA_PROMPT_TEMPLATE="$2"; shift 2 ;;
    --qa-boolq-prompt-template) QA_BOOLQ_PROMPT_TEMPLATE="$2"; shift 2 ;;
    --qa-chat-template) QA_CHAT_TEMPLATE=1; shift ;;
    --qa-chat-enable-thinking) QA_CHAT_ENABLE_THINKING=1; shift ;;
    --qa-sft-file) QA_SFT_FILE="$2"; shift 2 ;;
    --qa-sft-weight) QA_SFT_WEIGHT="$2"; shift 2 ;;
    --qa-sft-max-len) QA_SFT_MAX_LEN="$2"; shift 2 ;;
    --qa-sft-full-loss) QA_SFT_FULL_LOSS=1; shift ;;
    --qa-sft-log-every) QA_SFT_LOG_EVERY="$2"; shift 2 ;;
    --gate-reg-weight) GATE_REG_WEIGHT="$2"; shift 2 ;;
    --gate-override) GATE_OVERRIDE="$2"; shift 2 ;;
    --qa-sft-warmup-steps) QA_SFT_WARMUP_STEPS="$2"; shift 2 ;;
    --qa-sft-lazy) QA_SFT_LAZY=1; shift ;;
    --qa-sft-lazy-cache) QA_SFT_LAZY_CACHE="$2"; shift 2 ;;
    --qa-sft-chat-template) QA_SFT_CHAT_TEMPLATE=1; shift ;;
    --qa-sft-chat-enable-thinking) QA_SFT_CHAT_ENABLE_THINKING=1; shift ;;
    --unfreeze-official-source) UNFREEZE_OFFICIAL_SOURCE=1; shift ;;
    --official-reader-path) OFFICIAL_READER_PATH="$2"; shift 2 ;;
    --skip-qa) SKIP_QA=1; shift ;;
    --save-reader) SAVE_READER=1; shift ;;
    --load-reader) LOAD_READER=1; shift ;;
    --resume) RESUME=1; shift ;;
    --partial-dir) PARTIAL_DIR="$2"; shift 2 ;;
    --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
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
if [[ -z "$PARTIAL_DIR" ]]; then
  PARTIAL_DIR="$OUTPUT_DIR/partial"
fi
if [[ -z "$BACKUP_DIR" ]]; then
  BACKUP_DIR="$OUTPUT_DIR/backup"
fi
RESUME_ARG=""
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARG="--resume"
fi

SCALE_ARG=""
if [[ -n "$SCALE" ]]; then
  SCALE_ARG="--scale $SCALE"
fi

MLP_ARG=""
if [[ "$BRIDGE_MLP" == "1" ]]; then
  MLP_ARG="$MLP_ARG --bridge-mlp"
fi
if [[ "$OUT_MLP" == "1" ]]; then
  MLP_ARG="$MLP_ARG --out-mlp"
fi

QA_PROMPT_ARGS=()
if [[ -n "$QA_PROMPT_TEMPLATE" ]]; then
  QA_PROMPT_ARGS+=(--qa-prompt-template "$QA_PROMPT_TEMPLATE")
fi
if [[ -n "$QA_BOOLQ_PROMPT_TEMPLATE" ]]; then
  QA_PROMPT_ARGS+=(--qa-boolq-prompt-template "$QA_BOOLQ_PROMPT_TEMPLATE")
fi
if [[ "$QA_CHAT_TEMPLATE" == "1" ]]; then
  QA_PROMPT_ARGS+=(--qa-chat-template)
fi
if [[ "$QA_CHAT_ENABLE_THINKING" == "1" ]]; then
  QA_PROMPT_ARGS+=(--qa-chat-enable-thinking)
fi

QA_SFT_ARGS=()
if [[ -n "$QA_SFT_FILE" ]]; then
  QA_SFT_ARGS+=(--qa-sft-file "$QA_SFT_FILE")
  QA_SFT_ARGS+=(--qa-sft-weight "$QA_SFT_WEIGHT")
  QA_SFT_ARGS+=(--qa-sft-max-len "$QA_SFT_MAX_LEN")
  QA_SFT_ARGS+=(--qa-sft-log-every "$QA_SFT_LOG_EVERY")
  if [[ "$QA_SFT_FULL_LOSS" == "1" ]]; then
    QA_SFT_ARGS+=(--qa-sft-full-loss)
  fi
fi
if [[ "$GATE_REG_WEIGHT" != "0" ]]; then
  QA_SFT_ARGS+=(--gate-reg-weight "$GATE_REG_WEIGHT")
fi
if [[ -n "$GATE_OVERRIDE" ]]; then
  QA_SFT_ARGS+=(--gate-override "$GATE_OVERRIDE")
fi
if [[ "$QA_SFT_WARMUP_STEPS" != "0" ]]; then
  QA_SFT_ARGS+=(--qa-sft-warmup-steps "$QA_SFT_WARMUP_STEPS")
fi
if [[ "$QA_SFT_LAZY" == "1" ]]; then
  QA_SFT_ARGS+=(--qa-sft-lazy --qa-sft-lazy-cache "$QA_SFT_LAZY_CACHE")
fi
if [[ "$QA_SFT_CHAT_TEMPLATE" == "1" ]]; then
  QA_SFT_ARGS+=(--qa-sft-chat-template)
fi
if [[ "$QA_SFT_CHAT_ENABLE_THINKING" == "1" ]]; then
  QA_SFT_ARGS+=(--qa-sft-chat-enable-thinking)
fi
if [[ "$UNFREEZE_OFFICIAL_SOURCE" == "1" ]]; then
  QA_SFT_ARGS+=(--unfreeze-official-source)
fi

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
  SAVE_READER_ARG=""
  if [[ "$SAVE_READER" == "1" ]]; then
    SAVE_READER_ARG="--save-reader ${OUTPUT_DIR}/reader-${C}-{mode}-seed{seed}.pt"
  fi
  LOAD_READER_ARG=""
  if [[ "$LOAD_READER" == "1" ]]; then
    LOAD_READER_ARG="--load-reader ${OUTPUT_DIR}/reader-${C}-{mode}-seed{seed}.pt"
  fi
  if [[ "$SKIP_QA" == "1" ]]; then
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --layer "$LAYER" \
      --device "$DEVICE" \
      $SCALE_ARG \
      $MLP_ARG \
      $SAVE_READER_ARG \
      $LOAD_READER_ARG \
      $RESUME_ARG \
      --partial-dir "$PARTIAL_DIR" \
      --backup-dir "$BACKUP_DIR" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --steps "$STEPS" \
      --seq-len "$SEQ_LEN" \
      --lr "$LR" \
      --seeds $SEEDS \
      --modes $MODES \
      "${QA_SFT_ARGS[@]}" \
      --output "$OUT"
  else
    "$PYTHON" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy "$TOKENS" \
      --rows-dir "$ROWS_DIR" \
      --model-dir "$MODEL_DIR" \
      --model "$MODEL" \
      --reader "$READER" \
      --layer "$LAYER" \
      --device "$DEVICE" \
      $SCALE_ARG \
      $MLP_ARG \
      $SAVE_READER_ARG \
      $LOAD_READER_ARG \
      $RESUME_ARG \
      --partial-dir "$PARTIAL_DIR" \
      --backup-dir "$BACKUP_DIR" \
      --official-reader-path "$OFFICIAL_READER_PATH" \
      --steps "$STEPS" \
      --seq-len "$SEQ_LEN" \
      --lr "$LR" \
      --seeds $SEEDS \
      --modes $MODES \
      --qa \
      --qa-exact-match \
      --qa-max-new-tokens "$MAX_NEW" \
      --qa-batch-size "$QA_BATCH_SIZE" \
      --qa-batch-max-tokens "$QA_BATCH_MAX_TOKENS" \
      "${QA_PROMPT_ARGS[@]}" \
      "${QA_SFT_ARGS[@]}" \
      --qa-file "$QA_FILE" \
      --output "$OUT"
  fi
done

echo "[phase1-matrix] all corpora done"
