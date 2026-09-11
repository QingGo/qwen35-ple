#!/usr/bin/env bash
# Round 162 follow-up #2: the decisive NO-SFT comparison, run small.
#
# Why this exists
# ---------------
# The headline 0.8B result (§6 of docs/round-162-format-prior.md) is a null on
# content-dependence, and its known weakness is a **ceiling effect**: under the
# round-152 recipe (reader trained 1:1 on corpus windows and QA answer-only SFT)
# every injecting arm collapses to a 2-token terse answer, so "the three corpora
# produce the same format" could mean "there is no room for a difference".
#
# The single NO-SFT arm that did finish (reader trained on corpus windows only,
# no QA SFT) shows the regime is genuinely different -- mean 13.95 generated
# tokens vs 2.55, 96.9% space_lead, i.e. no longer saturated.  The full six-arm
# NO-SFT queue costs ~24 min/arm (longer generations make decoding the
# bottleneck), so this runs only the arms the verdict needs -- 1 (wiki),
# 2 (code), 3 (stem) and 6 (no-reader) -- on the balanced 200-per-task subset,
# which is the cheapest design that still has a zero-injection reference.
#
# This is a SECONDARY regime and must be reported as such: it is a reduced arm
# set on a reduced item set, not the pre-registered design.
#
# Usage:
#   setsid nohup bash scripts/round162_followup2.sh > <log> 2>&1 < /dev/null &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
LOGDIR="$ROOT/logs"
FOLLOW_LOG="$LOGDIR/round162-followup2.log"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

log() { echo "=== [r162-followup2] $* $(date -Is) ===" | tee -a "$FOLLOW_LOG"; }

WAIT_MAX="${WAIT_MAX:-10800}"
waited=0
while ! grep -q "queue DONE" "$LOGDIR/round162-4B.log" 2>/dev/null; do
  if [[ "$waited" -ge "$WAIT_MAX" ]]; then
    log "gave up waiting for the 4B queue after ${WAIT_MAX}s"
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
done
log "4B queue finished after ${waited}s wait"

# --- 0. 4B contribution-level magnitude control ---------------------------
# The 0.8B arms carry a contribution-norm measurement (§4.2); the 4B arms only
# got the parameter-level audit, which leaves "were the three 4B readers
# injecting the same magnitude?" answered by proxy.  This closes that gap with
# the same instrument before the GPU is handed to the next queue.
OUT4B="$ROOT/outputs/round162-4B"
if [[ ! -f "$OUT4B/contribution-similarity.json" ]]; then
  log "4B contribution similarity"
  "$ROOT/venv/bin/python" scripts/with_rusage.py "$OUT4B/stats-followup.jsonl" \
    "$ROOT/venv/bin/python" -u scripts/reader_contribution_similarity.py \
    --model "$ROOT/models/Qwen3.5-4B" --model-dir "$ROOT/models/qwen38_ple" \
    --rows-dir "$ROOT/qwen38-rows" --device cuda --backbone-dtype bfloat16 \
    --official-reader-path data/official_ple_reader.pt \
    --qa-file data/qa-standard/eval-600b.jsonl --max-items 192 \
    --reader "wiki=$OUT4B/reader-wiki-seed0.pt" \
    --reader "code=$OUT4B/reader-code-seed0.pt" \
    --reader "stem=$OUT4B/reader-stem-seed0.pt" \
    --output "$OUT4B/contribution-similarity.json" >>"$FOLLOW_LOG" 2>&1 \
    || log "WARNING: 4B contribution similarity failed"
else
  log "skip 4B contribution similarity (exists)"
fi

OUT="$ROOT/outputs/round162-0.8B-nosft600"
if [[ -f "$OUT/arm-no-reader.json" ]]; then
  log "skip: decisive NO-SFT subset already complete"
  exit 0
fi

log "decisive NO-SFT subset: arms 1/2/3 + 6, 600 balanced items"
NO_SFT=1 \
ARMS="wiki code stem no-reader" \
ITEMS_FILE=data/qa-standard/eval-600b.jsonl \
OUTDIR="$OUT" \
LOGFILE="$LOGDIR/round162-0.8B-nosft600.log" \
  bash scripts/run_round162_format_prior.sh >>"$FOLLOW_LOG" 2>&1 \
  || log "WARNING: decisive NO-SFT subset failed"

log "follow-up #2 DONE"
