#!/usr/bin/env bash
# Round 168 Stage 1.5c: the marginal-advantage (margin) distribution.
#
# Pre-registration: docs/round-168-stage1.5c-preregistration.md
#
# Three domain-matched pairs (count model <- same-domain training stream):
#
#   wiki  PURE_WIKI  -> wikitext-heldout-decon
#   stem  PURE_STEM  -> PURE_STEM-heldout-decon
#   code  PURE_CODE  -> PURE_CODE-heldout-decon
#
# Phases:
#   A  counts    CPU only.  Deliberately overlapped with whatever holds the GPU
#                (round 167 stage 2.2 when this first ran), because it needs none
#                of it -- that is the cheapest possible utilisation win.
#   B  backbone  GPU minutes, no training.  Memory-gated one at a time: two
#                concurrent 0.8B fp32 forwards plus stage 2.2's 12.6 GiB does not
#                fit in 24 GiB.
#   C  analyze   CPU only.
#
# The utilisation sampler runs ONLY across phase B.  Sampling the whole queue
# would average the CPU-bound phases in at 0% and report a misleading number
# (the round-167 playbook's "a probe between phases always reads 0%").
#
# Usage:  bash scripts/run_round168_margin.sh
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
MODEL="${MODEL:-$ROOT/models/Qwen3.5-0.8B}"
DTYPE="${DTYPE:-float32}"
LAYER="${LAYER:-2}"
CHUNK_TOKENS="${CHUNK_TOKENS:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
TARGET_UTIL="${TARGET_UTIL:-50}"
OUT="${OUT:-$ROOT/outputs/round168/margin}"
LOG="${LOG:-$ROOT/logs/round168-margin.log}"
DOMAINS="${DOMAINS:-wiki:PURE_WIKI:wikitext-heldout-decon stem:PURE_STEM:PURE_STEM-heldout-decon code:PURE_CODE:PURE_CODE-heldout-decon}"
# Which phases to run.  Phase A is CPU-only and idempotent, so it is often run
# on its own first (STAGES=counts) while a GPU job still holds the card.
STAGES="${STAGES:-counts backbone analyze fusion}"

export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

log() { echo "=== [r168-1.5c] $* $(date -Is) ===" | tee -a "$LOG"; }

run_fusion() {
  local tag="$1"
  PYTHONPATH=src "$PY" -u scripts/round168_fusion_probe.py \
    --tag "$tag" --workdir "$OUT" --n-perm "${N_PERM:-16}" >>"$LOG" 2>&1
  local rc=$?
  log "fusion $tag rc=$rc"
  return $rc
}

# The heavy lock is taken for PHASE B ONLY.  Phase A is pure numpy on the CPU and
# touching no GPU state, so serialising it behind whatever holds the card would
# buy nothing; the lock exists to stop two jobs sharing 24 GiB of VRAM.
LOCK_WAIT="${LOCK_WAIT:-7200}"
take_lock() {
  exec 9>/tmp/qwen35_heavy.lock
  if ! flock -n 9; then
    log "another heavy run holds the lock; waiting up to ${LOCK_WAIT}s"
    flock -w "$LOCK_WAIT" 9 || log "WARNING: proceeding without the lock"
  fi
  log "lock held (or waived)"
}
release_lock() { exec 9>&-; }

UTIL_TRACE="$OUT/util-trace.csv"
SAMPLER=""

start_sampler() {
  : >"$UTIL_TRACE"
  (
    while true; do
      nvidia-smi --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits >>"$UTIL_TRACE" 2>/dev/null
      sleep 3
    done
  ) &
  SAMPLER=$!
  log "utilisation sampler started (pid $SAMPLER)"
}

stop_sampler() {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  SAMPLER=""
}

# A bare `wait` also waits on the infinite sampler and deadlocks the queue
# (round 167, first launch of stage 2.2).  Track the PIDs we mean to join.
JOB_PIDS=()
launch() { "$@" & JOB_PIDS+=("$!"); }
wait_all() {
  local p
  for p in "${JOB_PIDS[@]:-}"; do wait "$p" 2>/dev/null; done
  JOB_PIDS=()
}

wait_for_mem() {
  local need="${1:-8000}" free
  while :; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
    [ "${free:-0}" -ge "$need" ] && break
    sleep 5
  done
}

run_stage() {
  local stage="$1" tag="$2" train="$3" evaltok="$4"
  PYTHONPATH=src "$PY" -u scripts/round168_margin_distribution.py \
    --stage "$stage" --tag "$tag" --workdir "$OUT" \
    --train-npy "$train" --eval-npy "$evaltok" \
    --model "$MODEL" --backbone-dtype "$DTYPE" --layer "$LAYER" \
    --chunk-tokens "$CHUNK_TOKENS" >>"$LOG" 2>&1
  local rc=$?
  log "$stage $tag rc=$rc"
  return $rc
}

log "queue start domains='$DOMAINS' model=$MODEL dtype=$DTYPE layer=$LAYER"

want() { [[ " $STAGES " == *" $1 "* ]]; }

if want counts; then
  for spec in $DOMAINS; do
    IFS=: read -r tag traindir evaldir <<<"$spec"
    log "phase A (counts) $tag: $traindir -> $evaldir"
    run_stage counts "$tag" "data/phase1/$traindir/tokens.npy" \
      "data/phase1/$evaldir/tokens.npy"
  done
  log "phase A (counts) complete"
fi

if want backbone; then
  take_lock
  start_sampler
  for spec in $DOMAINS; do
    IFS=: read -r tag traindir evaldir <<<"$spec"
    wait_for_mem "$MIN_FREE_MIB"
    log "phase B (backbone) $tag"
    run_stage backbone "$tag" "data/phase1/$traindir/tokens.npy" \
      "data/phase1/$evaldir/tokens.npy"
  done
  stop_sampler
  release_lock
  log "phase B (backbone) complete"

  PYTHONPATH=src "$PY" scripts/gpu_util_probe.py --trace "$UTIL_TRACE" \
    --json "$OUT/util-summary.json" >>"$LOG" 2>&1
  PYTHONPATH=src "$PY" scripts/gpu_util_probe.py --trace "$UTIL_TRACE" \
    --require "$TARGET_UTIL" >>"$LOG" 2>&1 \
    || log "WARNING: phase B utilisation below ${TARGET_UTIL}% (see util-summary.json)"
fi

if want analyze; then
  for spec in $DOMAINS; do
    IFS=: read -r tag traindir evaldir <<<"$spec"
    log "phase C (analyze) $tag"
    run_stage analyze "$tag" "data/phase1/$traindir/tokens.npy" \
      "data/phase1/$evaldir/tokens.npy"
  done
fi

# Phase D is Stage 1.5e.  It needs no GPU and no new forward pass: it consumes
# the two .npz files the earlier phases wrote.
if want fusion; then
  for spec in $DOMAINS; do
    IFS=: read -r tag traindir evaldir <<<"$spec"
    log "phase D (fusion / complementarity) $tag"
    run_fusion "$tag"
  done
fi

log "queue DONE"
touch "$OUT/STAGE15C_DONE"
