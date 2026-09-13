#!/usr/bin/env bash
# Round 167 Stage 1b: is the read-out collapse a property of the *initialisation
# and depth*, or of the architecture?
#
# Context.  Round 165A measured the read-out collapsing
# ``e_t`` PR 136.71 -> value_proj 14.39 -> branch_sum 2.68 -> c_t 1.02.
# Three independent results predict exactly that for this configuration:
#
#   * the official Engram config zero-initialises its ShortConv, and our reader
#     additionally zero-initialises ``out_proj``;
#   * arXiv 2510.06954: small initialisation drives condensation and then
#     asymptotic rank collapse in transformer training;
#   * ICML 2026 "The Implicit Bias of Depth": depth induces an *implicit
#     low-rank bias* because low-rank matrices propagate norm more efficiently
#     through successive multiplications.
#
# The production recipe trains a **zero-initialised 2-Linear MLP** read-out --
# the exact configuration all three predict will collapse.  So the collapse may
# be a training-recipe artifact rather than an architectural limit, and we have
# never tested a single alternative (round-167 TD-3a: ``zero_init_out`` used to
# be a literal in the library, so it was not merely untested but unreachable).
#
# Four variants, identical in every other respect (same corpus, steps, LR,
# seeds, backbone, layer):
#
#   prod           --out-mlp                    zero-init, 2 layers  (the existing arm)
#   nozero         --out-mlp --no-zero-init-out  real init, 2 layers
#   linear                                 zero-init, 1 layer
#   linear-nozero                 --no-zero-init-out  real init, 1 layer
#
# CRITICAL ORTHOGONALITY.  This queue is NOT the round-162 design: all four
# variants train on the same corpus, so there is no content manipulation here.
# It ships four readers and the PR/depth readings they produce.  Any output that
# leaks into the round-162 verdict tables is a mistake.
#
# Usage:  bash scripts/run_round167_stage1b.sh
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
MODEL="$ROOT/models/Qwen3.5-0.8B"
DTYPE="${DTYPE:-float32}"
STEPS="${STEPS:-500}"
OUT="${OUT:-$ROOT/outputs/round167/stage1b}"
LOG="${LOG:-$ROOT/logs/round167-stage1b.log}"
CORPUS="${CORPUS:-data/phase1/PURE_WIKI/tokens.npy}"

export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [stage1b] $* $(date -Is) ===" | tee -a "$LOG"; }

# Serialise against sibling heavy runs.
LOCK_WAIT="${LOCK_WAIT:-3600}"
exec 9>/tmp/qwen35_heavy.lock
if ! flock -n 9; then
  log "another heavy run holds the lock; waiting up to ${LOCK_WAIT}s"
  flock -w "$LOCK_WAIT" 9 || log "WARNING: proceeding without the lock"
fi
log "lock held (or waived)"

train_variant() {
  local name="$1"; shift
  local ckpt="$OUT/reader-$name-seed0.pt"
  if [[ -f "$ckpt" ]]; then
    log "variant $name: checkpoint exists, skipping training"
    return 0
  fi
  log "variant $name: training (extra flags: $*)"
  "$PY" -u scripts/run_phase0.py \
    --reader official --layer 2 --bridge-mlp \
    "$@" \
    --official-reader-path data/official_ple_reader.pt \
    --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda \
    --backbone-dtype "$DTYPE" --seeds 0 \
    --steps "$STEPS" --seq-len 128 --lr 1e-4 \
    --live-store --rows-dir "$ROWS" --tokens-npy "$CORPUS" \
    --modes real \
    --qa-exact-match --qa-max-new-tokens 32 --qa-batch-size 8 \
    --qa-batch-max-tokens 4096 \
    --qa-prompt-template "$PROMPT" --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
    --qa-file data/qa-standard/eval-600b.jsonl --qa-max-items 100 \
    --save-reader "$ckpt" \
    --output "$OUT/arm-$name.json" >>"$LOG" 2>&1
  local rc=$?
  log "variant $name: training rc=$rc"
  return $rc
}

train_variant prod          --out-mlp                       || exit 1
train_variant nozero        --out-mlp --no-zero-init-out    || exit 1
train_variant linear                                        || exit 1
train_variant linear-nozero          --no-zero-init-out     || exit 1

log "all four variants trained; measuring collapse provenance + effective depth"

# Collapse provenance: PR at e_t -> value_proj -> branch_sum -> c_t.
for name in prod nozero linear linear-nozero; do
  ckpt="$OUT/reader-$name-seed0.pt"
  prov="$OUT/provenance-$name.json"
  [[ -f "$prov" ]] && { log "provenance $name exists, skip"; continue; }
  log "provenance $name"
  "$PY" scripts/round165_collapse_provenance.py \
    --model "$MODEL" \
    --rows-dir "$ROWS" \
    --reader "$ckpt" \
    --qa-file data/qa-standard/eval-600b.jsonl \
    --max-items "${PROV_ITEMS:-200}" \
    --backbone-dtype "$DTYPE" \
    --offsets 0,12 \
    --output "$prov" >>"$LOG" 2>&1 || log "provenance $name FAILED (rc=$?)"
done

# Effective depth per variant (the Stage 1a instrument, applied to each read-out).
for name in prod nozero linear linear-nozero; do
  ckpt="$OUT/reader-$name-seed0.pt"
  ed="$OUT/effective-depth-$name.json"
  [[ -f "$ed" ]] && { log "depth $name exists, skip"; continue; }
  log "effective depth $name"
  "$PY" scripts/round167_effective_depth.py \
    --model "$MODEL" \
    --rows-dir "$ROWS" \
    --reader "$ckpt" \
    --qa-file data/qa-standard/eval-600b.jsonl \
    --max-items "${ED_ITEMS:-200}" \
    --backbone-dtype "$DTYPE" \
    --output "$ed" >>"$LOG" 2>&1 || log "depth $name FAILED (rc=$?)"
done

log "queue DONE"
touch "$OUT/STAGE1B_DONE"
exec 9>&-
