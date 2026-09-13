#!/usr/bin/env bash
# Round 167 Stage 2.2: is the saturated read-out gate the cause of always-on
# injection damage, and does a per-dimension gate fix it?
#
# Pre-registration: docs/round-167-stage2-gate-selectivity-preregistration.md
#
# Design (mirrors round-162's real/control structure so the arms are comparable):
#
#   for mode in scalar per_dim; for seed in 0 1 2:
#       train ONE reader with --gate-mode $mode          (mixed50, 500 steps)
#       evaluate it with --modes real control            (raw and chat templates)
#       record gate statistics
#
# so 6 trainings, 12 evaluations and 6 gate reports.
#
# UTILISATION (round-167 standing rule, >= 50%).  A single arm measures ~23%
# utilisation at ~5 GiB of 24 GiB (OCCUPANCY_BOUND: the job cannot fill the card
# but several can), so trainings and evaluations run CONCURRENCY-deep rather
# than serially.  The queue probes before and after and refuses to be silent
# about a miss.
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
# Memory headroom: ~5 GiB per training job on a 24 GiB card -> 3 is safe.
CONCURRENCY="${CONCURRENCY:-3}"
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

# --- utilisation gate ------------------------------------------------------
probe() {
  local tag="$1"
  PYTHONPATH=src "$PY" scripts/gpu_util_probe.py --seconds 20 --require "$TARGET_UTIL" \
    --json "$OUT/util-$tag.json" >>"$LOG" 2>&1
  local rc=$?
  # A miss is recorded loudly but does not kill a half-finished queue: the
  # results are worth more than the gate, and round-167's rule is "diagnose",
  # not "abort".
  [ $rc -ne 0 ] && log "UTILISATION BELOW ${TARGET_UTIL}% at $tag (see util-$tag.json)"
  return 0
}

# Run up to $CONCURRENCY background jobs, waiting for the oldest when full.
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 5; done
}

train_reader() {
  local mode="$1" seed="$2"
  local ckpt="$OUT/reader-$mode-seed$seed.pt"
  if [ -f "$ckpt" ]; then
    log "train $mode seed$seed: exists, skip"
    return 0
  fi
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
    log "eval $mode seed$seed [$tmpl]: exists, skip"
    return 0
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
    --qa-max-new-tokens 32 --qa-batch-size 8 --qa-batch-max-tokens 4096 \
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
  # analyze_reader_gate.py takes --max-tokens (prompt budget), not --max-items,
  # and has no --backbone-dtype; it loads the model itself.
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

log "queue start modes='$MODES' seeds='$SEEDS' concurrency=$CONCURRENCY steps=$STEPS"
probe "start"

# --- Stage A: train 6 readers, CONCURRENCY at a time -----------------------
for mode in $MODES; do
  for seed in $SEEDS; do
    wait_for_slot
    train_reader "$mode" "$seed" &
  done
done
wait
log "stage A (training) complete"
probe "after-train"

# --- Stage B: evaluate each reader, raw + chat -----------------------------
for mode in $MODES; do
  for seed in $SEEDS; do
    for tmpl in raw chat; do
      wait_for_slot
      eval_reader "$mode" "$seed" "$tmpl" &
    done
  done
done
wait
log "stage B (evaluation) complete"
probe "after-eval"

# --- Stage C: gate statistics ---------------------------------------------
for mode in $MODES; do
  for seed in $SEEDS; do
    wait_for_slot
    gate_stats "$mode" "$seed" &
  done
done
wait

log "queue DONE"
touch "$OUT/STAGE2GATE_DONE"
exec 9>&-
