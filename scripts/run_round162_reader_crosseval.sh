#!/usr/bin/env bash
# Round 162 manipulation-strength check: does each trained reader actually
# specialise on its own corpus?
#
# A null result on the decisive 1-vs-2-vs-3 format comparison is only
# interpretable if the manipulation bit.  Training three readers on three
# different corpora and then finding their *generations* identical could mean
# (a) content really does not matter, or (b) the readers never specialised.
# This script separates the two by evaluating every reader on every corpus's
# held-out windows: if reader-code has the lowest next-token loss on CODE while
# reader-wiki has the lowest on WIKI, the manipulation demonstrably bit.
#
# No training happens here (``--load-reader`` + ``--steps 0``); the run only
# computes ``val_loss`` on the requested corpus split, so it is cheap.
#
# Usage:
#   BACKBONE=0.8B bash scripts/run_round162_reader_crosseval.sh
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
BACKBONE="${BACKBONE:-0.8B}"

case "$BACKBONE" in
  0.8B) MODEL="$ROOT/models/Qwen3.5-0.8B"; DTYPE="float32" ;;
  4B)   MODEL="$ROOT/models/Qwen3.5-4B";   DTYPE="bfloat16" ;;
  *)    echo "unknown BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

OUT="${OUTDIR:-$ROOT/outputs/round162-${BACKBONE}}"
LOG="${LOGFILE:-$ROOT/logs/round162-${BACKBONE}-crosseval.log}"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

log() { echo "=== [r162x-$BACKBONE] $* $(date -Is) ===" | tee -a "$LOG"; }

LOCK_WAIT="${LOCK_WAIT:-3600}"
exec 9>/tmp/qwen35_heavy.lock
if ! flock -n 9; then
  log "waiting up to ${LOCK_WAIT}s for /tmp/qwen35_heavy.lock"
  flock -w "$LOCK_WAIT" 9 || log "WARNING: proceeding without the lock (GPU-only work)"
fi

for READER in wiki code stem; do
  CKPT="$OUT/reader-$READER-seed0.pt"
  if [[ ! -f "$CKPT" ]]; then
    log "missing $CKPT; run the main queue first"
    exit 1
  fi
  for CORPUS in PURE_WIKI PURE_CODE PURE_STEM; do
    OUTJ="$OUT/crosseval-${READER}-on-${CORPUS}.json"
    if [[ -f "$OUTJ" ]] && grep -q '"val_loss"' "$OUTJ" 2>/dev/null; then
      log "skip $READER on $CORPUS (complete)"
      continue
    fi
    log "reader=$READER on $CORPUS"
    "$PY" scripts/with_rusage.py "$OUT/stats-crosseval.jsonl" "$PY" -u scripts/run_phase0.py \
      --live-store --tokens-npy "data/phase1/$CORPUS/tokens.npy" \
      --rows-dir "$ROWS" \
      --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda \
      --backbone-dtype "$DTYPE" \
      --reader official --layer 2 --bridge-mlp --out-mlp \
      --official-reader-path data/official_ple_reader.pt \
      --steps 0 --seq-len 128 --lr 1e-4 --seeds 0 --modes real \
      --load-reader "$CKPT" \
      --output "$OUTJ" >>"$LOG" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      log "FAILED $READER on $CORPUS rc=$rc"
      exit 1
    fi
    LOSS="$("$PY" - "$OUTJ" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d['results'][0]['val_loss']:.6f}")
PYEOF
)"
    log "reader=$READER on $CORPUS val_loss=$LOSS"
  done
done

log "cross-eval DONE"
exec 9>&-
