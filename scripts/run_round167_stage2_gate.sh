#!/usr/bin/env bash
# Round 167 Stage 2.2: is the saturated read-out gate the cause of always-on
# injection damage, and does a per-dimension gate fix it?
#
# Pre-registration: docs/round-167-stage2-gate-selectivity-preregistration.md
#
# Design (mirrors round-162's real/control structure so arms are comparable):
#
#   for mode in scalar per_dim; for seed in 0 1 2:
#       train ONE reader with --gate-mode $mode          (mixed50, 500 steps)
#       evaluate it with --modes real control            (raw and chat templates)
#       record gate statistics
#
# so 6 trainings, 12 evaluations and 6 gate reports.
#
# UTILISATION (round-167 standing rule, >= 50%).  Concurrency depth is PER PHASE
# and measured, not assumed:
#
#   * training peaks near 5 GiB  -> 3 fit on a 24 GiB card (measured 91-99%)
#   * generation peaks near 13 GiB (KV cache + 248k-way logits) -> only 1 fits
#
# A single shared limit OOM'd stage B with "Process 89891 has 12.81 GiB in use".
# Utilisation is sampled CONTINUOUSLY across the queue and summarised at the end,
# because a probe taken between phases sees an idle GPU and always reads 0%.
#
# Usage:  bash scripts/run_round167_stage2_gate.sh
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
MODEL="$ROOT/models/Qwen3.5-0.8B"
DTYPE="${DTYPE:-float32}"
STEPS="${STEPS:-500}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-scalar per_dim}"
ITEMS="${ITEMS:-data/qa-standard/eval-600b.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-600}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-3}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-1}"
TARGET_UTIL="${TARGET_UTIL:-50}"
OUT="${OUT:-$ROOT/outputs/round167/stage2-gate}"
LOG="${LOG:-$ROOT/logs/round167-stage2-gate.log}"

export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [s2.2] $* $(date -Is) ===" | tee -a "$LOG"; }

LOCK_WAIT="${LOCK_WAIT:-3600}"
exec 9>/tmp/qwen35_heavy.lock
if ! flock -n 9; then
  log "another heavy run holds the lock; waiting up to ${LOCK_WAIT}s"
  flock -w "$LOCK_WAIT" 9 || log "WARNING: proceeding without the lock"
fi
log "lock held (or waived)"

UTIL_TRACE="$OUT/util-trace.csv"
: >"$UTIL_TRACE"
(
  while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
      >>"$UTIL_TRACE" 2>/dev/null
    sleep 3
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# The sampler is itself a background job, so the ceiling is limit + 1.
wait_for_slot() {
  local limit="$1"
  while [ "$(jobs -rp | wc -l)" -ge "$((limit + 1))" ]; do sleep 5; done
}

train_reader() {
  local mode="$1" seed="$2"
  local ckpt="$OUT/reader-$mode-seed$seed.pt"
  if [ -f "$ckpt" ]; then log "train $mode seed$seed: exists, skip"; return 0; fi
  PYTHONPATH=src "$PY" -u scripts/run_phase0.py \
    --reader official --layer 2 --bridge-mlp --out-mlp \
    --gate-mode "$mode" \
    --official-reader-path data/official_ple_reader.pt \
    --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda \
    --backbone-dtype "$DTYPE" --seeds "$seed" \
    --steps "$STEPS" --seq-len 128 --lr 1e-4 \
    --live-store --rows-dir "$ROWS" --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
    --qa-sft-file data/qa-standard/train.jsonl --qa-sft-weight 0.5 \
    --qa-sft-max-len 512 --qa-sft-log-every 100 --qa-sft-lazy --qa-sft-lazy-cache 128 \
    --modes real \
    --qa-exact-match --qa-max-new-tokens 32 --qa-batch-size 8 --qa-batch-max-tokens 4096 \
    --qa-prompt-template "$PROMPT" --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
    --qa-file "$ITEMS" --qa-max-items 60 \
    --save-reader "$ckpt" \
    --output "$OUT/train-$mode-seed$seed.json" >>"$LOG" 2>&1
  local rc=$?
  log "train $mode seed$seed rc=$rc"
  return $rc
}

eval_reader() {
  local mode="$1" seed="$2" tmpl="$3"
  local ckpt="$OUT/reader-$mode-seed$seed.pt"
  local out="$OUT/eval-$mode-seed$seed-$tmpl.json"
  if [ -f "$out" ] && grep -q '"answers"' "$out" 2>/dev/null; then
    log "eval $mode seed$seed [$tmpl]: exists, skip"; return 0
  fi
  local tflag=()
  [ "$tmpl" = "chat" ] && tflag=(--qa-chat-template)
  PYTHONPATH=src "$PY" -u scripts/run_phase0.py \
    --reader official --layer 2 --bridge-mlp --out-mlp \
    --gate-mode "$mode" \
    --official-reader-path data/official_ple_reader.pt \
    --load-reader "$ckpt" \
    --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda \
    --backbone-dtype "$DTYPE" --seeds "$seed" \
    --live-store --rows-dir "$ROWS" \
    --modes real control \
    "${tflag[@]}" \
    --qa-exact-match --qa-gold-nll --qa-gold-nll-spaced --qa-norm-stats \
    --qa-max-new-tokens 32 --qa-batch-size 4 --qa-batch-max-tokens 2048 \
    --qa-prompt-template "$PROMPT" --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
    --qa-file "$ITEMS" --qa-max-items "$MAX_ITEMS" \
    --output "$out" >>"$LOG" 2>&1
  local rc=$?
  log "eval $mode seed$seed [$tmpl] rc=$rc"
  return $rc
}

gate_stats() {
  local mode="$1" seed="$2"
  local ckpt="$OUT/reader-$mode-seed$seed.pt"
  local out="$OUT/gate-$mode-seed$seed.json"
  [ -f "$out" ] && { log "gate $mode seed$seed: exists, skip"; return 0; }
  # analyze_reader_gate.py takes --max-tokens (a prompt budget), not --max-items.
  PYTHONPATH=src "$PY" -u scripts/analyze_reader_gate.py \
    --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" \
    --reader-checkpoint "$ckpt" --layer 2 \
    --rows-dir "$ROWS" --qa-file "$ITEMS" --max-tokens 120 \
    --device cuda \
    --output "$out" >>"$LOG" 2>&1
  local rc=$?
  log "gate $mode seed$seed rc=$rc"
  return $rc
}

log "queue start modes='$MODES' seeds='$SEEDS' steps=$STEPS train_conc=$TRAIN_CONCURRENCY eval_conc=$EVAL_CONCURRENCY"

for mode in $MODES; do
  for seed in $SEEDS; do
    wait_for_slot "$TRAIN_CONCURRENCY"
    train_reader "$mode" "$seed" &
  done
done
wait
log "stage A (training) complete"

for mode in $MODES; do
  for seed in $SEEDS; do
    for tmpl in raw chat; do
      wait_for_slot "$EVAL_CONCURRENCY"
      eval_reader "$mode" "$seed" "$tmpl" &
    done
  done
done
wait
log "stage B (evaluation) complete"

for mode in $MODES; do
  for seed in $SEEDS; do
    wait_for_slot "$EVAL_CONCURRENCY"
    gate_stats "$mode" "$seed" &
  done
done
wait

kill "$SAMPLER" 2>/dev/null
PYTHONPATH=src "$PY" scripts/gpu_util_probe.py --trace "$UTIL_TRACE" \
  --require "$TARGET_UTIL" --json "$OUT/util-summary.json" >>"$LOG" 2>&1
log "utilisation summary written (see util-trace.csv / util-summary.json)"

log "queue DONE"
touch "$OUT/STAGE2GATE_DONE"
exec 9>&-
