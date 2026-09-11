#!/usr/bin/env bash
# Round 162 follow-up queue: waits for the 0.8B six-arm queue, then runs, in
# priority order, the measurements that decide how the arm comparison should be
# read.  Each step is skippable and resumable; every step logs its own header.
#
#   1. reader_contribution_similarity.py -- do the three corpora even change the
#      vector injected at the answer position?  A null format result is only
#      interpretable next to this number.
#   2. run_round162_reader_crosseval.sh -- is each reader specialised on its own
#      corpus?  (manipulation strength)
#   3. run_round162_format_prior.sh with NO_SFT=1 -- the decisive 1/2/3
#      comparison in the regime where the QA SFT mix cannot dilute the corpus
#      signal (secondary regime, reported as such).
#   4. run_round162_format_prior.sh with BACKBONE=4B on a balanced 200-per-task
#      subset -- the backbone where round-156 actually saw the scaffolding.
#
# Usage:
#   setsid nohup bash scripts/round162_followup.sh > <log> 2>&1 < /dev/null &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT08="$ROOT/outputs/round162-0.8B"
OUT08_NOSFT="$ROOT/outputs/round162-0.8B-nosft"
OUT4B="$ROOT/outputs/round162-4B"
LOGDIR="$ROOT/logs"
FOLLOW_LOG="$LOGDIR/round162-followup.log"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

log() { echo "=== [r162-followup] $* $(date -Is) ===" | tee -a "$FOLLOW_LOG"; }

# --- wait for the main 0.8B queue ------------------------------------------
WAIT_MAX="${WAIT_MAX:-7200}"
waited=0
while ! grep -q "queue DONE" "$LOGDIR/round162-0.8B.log" 2>/dev/null; do
  if [[ "$waited" -ge "$WAIT_MAX" ]]; then
    log "gave up waiting for the 0.8B queue after ${WAIT_MAX}s"
    exit 1
  fi
  sleep 30
  waited=$((waited + 30))
done
log "0.8B queue finished after ${waited}s wait"

MODEL08="$ROOT/models/Qwen3.5-0.8B"

# --- 1. contribution similarity -------------------------------------------
if [[ ! -f "$OUT08/contribution-similarity.json" ]]; then
  log "contribution similarity (0.8B)"
  "$PY" scripts/with_rusage.py "$OUT08/stats-followup.jsonl" \
    "$PY" -u scripts/reader_contribution_similarity.py \
    --model "$MODEL08" --model-dir "$ROOT/models/qwen38_ple" \
    --rows-dir "$ROOT/qwen38-rows" --device cuda --backbone-dtype float32 \
    --official-reader-path data/official_ple_reader.pt \
    --qa-file data/qa-standard/eval.jsonl --max-items 192 \
    --reader "wiki=$OUT08/reader-wiki-seed0.pt" \
    --reader "code=$OUT08/reader-code-seed0.pt" \
    --reader "stem=$OUT08/reader-stem-seed0.pt" \
    --output "$OUT08/contribution-similarity.json" >>"$FOLLOW_LOG" 2>&1 \
    || log "WARNING: contribution similarity failed"
else
  log "skip contribution similarity (exists)"
fi

# --- 2. reader cross-eval (manipulation strength) -------------------------
log "reader cross-eval (0.8B)"
OUTDIR="$OUT08" LOGFILE="$LOGDIR/round162-0.8B-crosseval.log" \
  bash scripts/run_round162_reader_crosseval.sh >>"$FOLLOW_LOG" 2>&1 \
  || log "WARNING: cross-eval failed"

# --- 3. 4B replication on a balanced 600-item subset ----------------------
if [[ -f "$OUT08/STOP_4B" ]]; then
  log "STOP_4B present; not starting the 4B replication"
elif [[ -f "$OUT4B/arm-no-reader.json" ]]; then
  log "skip 4B replication (complete)"
else
  log "4B replication (600 balanced items)"
  # Conservative batching: the 4B needs bf16 to fit the 24GB card, and round-152
  # already OOM'd once with concurrent fp32 work, so the batch budget here is
  # deliberately small (2 items / 1024 tokens).
  BACKBONE=4B ITEMS_FILE=data/qa-standard/eval-600b.jsonl \
    BATCH_SIZE=2 BATCH_TOKENS=1024 \
    OUTDIR="$OUT4B" LOGFILE="$LOGDIR/round162-4B.log" \
    bash scripts/run_round162_format_prior.sh >>"$FOLLOW_LOG" 2>&1 \
    || log "WARNING: 4B replication failed"
fi

# --- 4. corpus-only (no QA SFT) decisive comparison at 0.8B ---------------
log "0.8B corpus-only variant (NO_SFT=1)"
if [[ -f "$OUT08_NOSFT/arm-no-reader.json" ]]; then
  log "skip corpus-only variant (complete)"
else
  NO_SFT=1 OUTDIR="$OUT08_NOSFT" LOGFILE="$LOGDIR/round162-0.8B-nosft.log" \
    bash scripts/run_round162_format_prior.sh >>"$FOLLOW_LOG" 2>&1 \
    || log "WARNING: corpus-only variant failed"
fi

log "follow-up queue DONE"
