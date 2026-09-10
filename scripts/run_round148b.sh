#!/usr/bin/env bash
# Round 148-B: reader-capacity upper bound.
#
# Train the official-source reader with key/value/norm/conv projections
# UNFROZEN (the PLE table stays frozen) on the same 6000-item mixed50 recipe
# as sft-large-mixed50, then evaluate the standard 1500 raw prompt with both
# generation and gold-answer NLL.
#
# Arms:
#   real    : real frozen PLE rows
#   control : shuffled PLE rows (content control)
#   no-reader baseline (generation + NLL)
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/round148b"
EVAL="$OUT/eval-raw"
LOG="$ROOT/logs/round148b.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$EVAL" "$ROOT/logs"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'
TRAIN_DIR="$ROOT/outputs/sft-unfrozen-mixed50"

log() { echo "=== [round148b] $* $(date -Is) ===" | tee -a "$LOG"; }

check_rows() {
  local n
  n="$(ls /dev/shm/qwen38-rows 2>/dev/null | wc -l)"
  if [[ "$n" -lt 128 ]]; then
    log "ERROR: /dev/shm/qwen38-rows has $n files; restore rows first"
    exit 1
  fi
}

run_eval() {
  # run_eval <name> <mode> <reader-or-empty>
  local name="$1" mode="$2" reader="$3"
  local out="$EVAL/phase1-$name.json"
  if [[ -f "$out" ]] && grep -q '"qa_gold"' "$out" 2>/dev/null; then
    log "skip eval $name"
    return 0
  fi
  local args=(
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy
    --rows-dir /dev/shm/qwen38-rows
    --model-dir "$ROOT/models/qwen38_ple"
    --model "$ROOT/models/Qwen3.5-0.8B"
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp
    --official-reader-path data/official_ple_reader.pt
    --steps 0 --seq-len 128 --lr 1e-4 --seeds 0 --modes "$mode"
    --qa --qa-exact-match --qa-gold-nll --qa-max-new-tokens 32
    --qa-batch-size 16 --qa-batch-max-tokens 2048
    --qa-prompt-template "$PROMPT"
    --qa-boolq-prompt-template "$BOOLQ_PROMPT"
    --qa-file data/qa-standard/eval.jsonl
    --output "$out"
  )
  if [[ -n "$reader" ]]; then args+=(--load-reader "$reader"); fi
  log "eval $name mode=$mode reader=$reader"
  "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED eval $name rc=$rc"
    exit 1
  fi
}

check_rows

# ------------------------------------------------------------------ training
if [[ ! -f "$TRAIN_DIR/DONE" ]]; then
  log "train unfrozen source reader: real + control, seed 0, 500 steps"
  env \
    OUTPUT_DIR="$TRAIN_DIR" CORPORA="PURE_WIKI" MODES="real control" \
    STEPS=500 SEEDS="0" LAYER=2 RESUME=1 \
    MAX_NEW=32 QA_BATCH_SIZE=16 QA_BATCH_MAX_TOKENS=2048 \
    QA_FILE=data/qa-standard/eval-600.jsonl \
    QA_SFT_FILE=data/qa-standard/train.jsonl \
    QA_SFT_WEIGHT=0.5 QA_SFT_MAX_LEN=512 QA_SFT_LOG_EVERY=100 \
    QA_SFT_WARMUP_STEPS=0 QA_SFT_LAZY=1 QA_SFT_LAZY_CACHE=128 \
    UNFREEZE_OFFICIAL_SOURCE=1 \
    bash scripts/run_phase2_diagnostic.sh >>"$LOG" 2>&1
  if [[ ! -f "$TRAIN_DIR/reader-PURE_WIKI-real-seed0.pt" ]]; then
    log "ERROR: unfrozen real reader missing"
    exit 1
  fi
  if [[ ! -f "$TRAIN_DIR/reader-PURE_WIKI-control-seed0.pt" ]]; then
    log "ERROR: unfrozen control reader missing"
    exit 1
  fi
  touch "$TRAIN_DIR/DONE"
else
  log "skip training (DONE)"
fi

# --------------------------------------------------------------- 1500 eval
run_eval "full" "no-reader" ""
run_eval "real-seed0" "real" "$TRAIN_DIR/reader-PURE_WIKI-real-seed0.pt"
run_eval "control-seed0" "control" "$TRAIN_DIR/reader-PURE_WIKI-control-seed0.pt"

# -------------------------------------------------------------- summaries
"$PY" scripts/summarize_standard_heldout.py \
  --raw-dir "$EVAL" --output "$OUT/eval-raw-summary.md" \
  --title "Round 148-B: unfrozen-source reader on standard 1500 (raw)" >>"$LOG" 2>&1
"$PY" scripts/summarize_gold_nll.py \
  --run no-reader="$EVAL/phase1-full.json" \
  --run real="$EVAL/phase1-real-seed0.json" \
  --run control="$EVAL/phase1-control-seed0.json" \
  --baseline no-reader \
  --pair real=control \
  --output "$OUT/gold-nll-raw.md" \
  --title "Round 148-B: unfrozen-source reader gold-answer NLL (raw)" >>"$LOG" 2>&1

for artifact in eval-raw-summary.md gold-nll-raw.md; do
  if [[ ! -s "$OUT/$artifact" ]]; then
    log "ERROR: missing artifact $artifact"
    exit 1
  fi
done

log "ROUND 148-B DONE"
echo done > "$OUT/DONE"
