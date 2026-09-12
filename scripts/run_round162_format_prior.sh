#!/usr/bin/env bash
# Round 162: is the PLE format shift a content-dependent PRIOR or a layer-2
# PERTURBATION artifact?
#
# Six arms, same frozen backbone, same prompts, same reader architecture and
# same training budget.  Only the corpus that trains the reader changes in
# arms 1/2/3; arm 4 reuses arm 1's reader with shuffled rows; arms 5/6 inject
# nothing.
#
#   1 wiki      reader trained on PURE_WIKI   -> evaluated with real rows
#   2 code      reader trained on PURE_CODE   -> evaluated with real rows
#   3 stem      reader trained on PURE_STEM   -> evaluated with real rows
#   4 wiki-shuf arm 1's reader                -> evaluated with SHUFFLED rows
#   5 ple-off   reader present, contribution identically zero
#   6 no-reader no reader at all
#
# Pre-registration lives in docs/round-162-format-prior.md section 0 and was
# written before any number in this queue existed.
#
# Usage (on the remote box):
#   bash scripts/run_round162_format_prior.sh                 # 0.8B, 1500 items
#   BACKBONE=4B ITEMS_FILE=data/qa-standard/eval-600b.jsonl \
#     bash scripts/run_round162_format_prior.sh                # 4B replication
#
# Every step is resumable: an arm whose output JSON already carries a
# `qa_exact` block with `answers` is skipped.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
BACKBONE="${BACKBONE:-0.8B}"
ITEMS_FILE="${ITEMS_FILE:-data/qa-standard/eval.jsonl}"
MAX_ITEMS="${MAX_ITEMS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_TOKENS="${BATCH_TOKENS:-4096}"
STEPS="${STEPS:-500}"
DTYPE_08="${DTYPE_08:-float32}"

case "$BACKBONE" in
  0.8B) MODEL="$ROOT/models/Qwen3.5-0.8B"; DTYPE="$DTYPE_08" ;;
  2B)   MODEL="$ROOT/models/Qwen3.5-2B";   DTYPE="bfloat16" ;;
  4B)   MODEL="$ROOT/models/Qwen3.5-4B";   DTYPE="bfloat16" ;;
  *)    echo "unknown BACKBONE=$BACKBONE (use 0.8B, 2B or 4B)" >&2; exit 2 ;;
esac

TAG="r162-${BACKBONE}"
OUT="${OUTDIR:-$ROOT/outputs/round162-${BACKBONE}}"
LOG="${LOGFILE:-$ROOT/logs/round162-${BACKBONE}.log}"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [$TAG] $* $(date -Is) ===" | tee -a "$LOG"; }

QA_COMMON=(
  --qa-exact-match --qa-gold-nll --qa-gold-nll-spaced --qa-norm-stats
  --qa-max-new-tokens 32 --qa-batch-size "$BATCH_SIZE"
  --qa-batch-max-tokens "$BATCH_TOKENS"
  --qa-prompt-template "$PROMPT" --qa-boolq-prompt-template "$BOOLQ_PROMPT"
  --qa-file "$ITEMS_FILE"
)
if [[ "$MAX_ITEMS" != "0" ]]; then QA_COMMON+=(--qa-max-items "$MAX_ITEMS"); fi

SFT_COMMON=(
  --qa-sft-file data/qa-standard/train.jsonl --qa-sft-weight 0.5
  --qa-sft-max-len 512 --qa-sft-log-every 100 --qa-sft-lazy --qa-sft-lazy-cache 128
)
# NO_SFT=1 drops the QA SFT mix, so the reader is trained on corpus windows only.
# This is the *maximal* corpus signal variant: with --qa-sft-weight 0.5 half of
# every arm's training steps are identical across arms, which dilutes exactly
# the manipulation under test.  The round-152 recipe (weight 0.5) stays the
# headline for comparability with round-156; the corpus-only run is the
# secondary regime and is reported as such.
if [[ "${NO_SFT:-0}" == "1" ]]; then SFT_COMMON=(); fi
READER_COMMON=(
  --reader official --layer 2 --bridge-mlp --out-mlp
  --official-reader-path data/official_ple_reader.pt
  --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda
  --backbone-dtype "$DTYPE" --seeds 0
  --steps "$STEPS" --seq-len 128 --lr 1e-4
)
LIVE=(
  --live-store --rows-dir "$ROWS"
)

peak_gpu() {
  # max of the sampled GPU memory.used values (MiB)
  awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m+0}' "$1" 2>/dev/null || echo 0
}

watch_gpu() {
  local f="$1"
  : >"$f"
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$f" 2>/dev/null
    sleep 3
  done
}

# ARMS selects a subset of the six arms (space-separated).  When a subset is
# requested the queue is no longer the full pre-registered design, so the
# resulting JSONs must never be reported as such; it exists to make the
# decisive 1/2/3 + 5-or-6 comparison affordable in a secondary regime.
ARMS="${ARMS:-wiki code stem wiki-shuf ple-off no-reader}"

run_arm() {
  # run_arm <name> <corpus-tokens-npy|-> <mode> [extra flags...]
  local name="$1" corpus="$2" mode="$3"
  shift 3
  if [[ " $ARMS " != *" $name "* ]]; then
    log "skip arm $name (not in ARMS='$ARMS')"
    return 0
  fi
  local out="$OUT/arm-$name.json"
  if [[ -f "$out" ]] && grep -q '"answers"' "$out" 2>/dev/null; then
    log "skip arm $name (complete)"
    return 0
  fi
  local args=("${READER_COMMON[@]}" --modes "$mode" "${QA_COMMON[@]}")
  if [[ "$corpus" != "-" ]]; then
    args+=("${LIVE[@]}" --tokens-npy "$corpus")
  fi
  if [[ $# -gt 0 ]]; then args+=("$@"); fi
  args+=(--output "$out")

  local mon="$OUT/.gpu.$name"
  watch_gpu "$mon" & local monpid=$!
  log "arm $name start (mode=$mode corpus=${corpus##*/})"
  "$PY" scripts/with_rusage.py "$OUT/stats.jsonl" \
    "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  kill "$monpid" 2>/dev/null
  wait "$monpid" 2>/dev/null
  local gpu; gpu=$(peak_gpu "$mon")
  rm -f "$mon"
  echo "arm=$name wall_peak_gpu_mib=$gpu" >>"$OUT/timings.txt"
  log "arm $name rc=$rc peak_gpu=${gpu}MiB"
  if [[ $rc -ne 0 ]]; then
    log "FAILED arm $name rc=$rc; aborting queue"
    return 1
  fi
  return 0
}

# Serialise against any sibling heavy run (round-156 lesson: two big jobs on one
# box silently compete for the GPU).  The wait is bounded: this queue needs only
# the GPU, so if the holder is demonstrably CPU-only (nvidia-smi reports no
# compute processes) the wait is waived after LOCK_WAIT seconds and the fact is
# logged loudly.  A silent indefinite block would be worse than a documented
# concurrent run.
LOCK_WAIT="${LOCK_WAIT:-3600}"
exec 9>/tmp/qwen35_heavy.lock
if ! flock -n 9; then
  log "another heavy run holds /tmp/qwen35_heavy.lock; waiting up to ${LOCK_WAIT}s"
  if ! flock -w "$LOCK_WAIT" 9; then
    log "WARNING: lock still held after ${LOCK_WAIT}s -- proceeding WITHOUT it"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$LOG"
    log "  justification: the GPU shows no compute process, so the holder is CPU-only;"
    log "  this deviation is recorded in docs/round-162-format-prior.md section 8."
  fi
fi
log "lock held (or waived)"

log "queue start backbone=$BACKBONE dtype=$DTYPE items=$ITEMS_FILE steps=$STEPS batch=$BATCH_SIZE/$BATCH_TOKENS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

WIKI_CKPT="$OUT/reader-wiki-seed0.pt"

# --- arms 1..3: train on one corpus each, then evaluate with real rows ------
run_arm wiki "data/phase1/PURE_WIKI/tokens.npy" real \
  "${SFT_COMMON[@]}" --save-reader "$WIKI_CKPT" || exit 1
run_arm code "data/phase1/PURE_CODE/tokens.npy" real \
  "${SFT_COMMON[@]}" --save-reader "$OUT/reader-code-seed0.pt" || exit 1
run_arm stem "data/phase1/PURE_STEM/tokens.npy" real \
  "${SFT_COMMON[@]}" --save-reader "$OUT/reader-stem-seed0.pt" || exit 1

# --- arm 6: no reader at all (the zero-injection reference) -----------------
# Placed directly after the decisive trio so that a queue cut short still
# carries the 1-vs-2-vs-3 contrast AND the reference it is compared against.
run_arm no-reader "-" no-reader || exit 1

# --- arm 4: arm 1's reader, shuffled rows at matched magnitude --------------
run_arm wiki-shuf "data/phase1/PURE_WIKI/tokens.npy" control \
  --load-reader "$WIKI_CKPT" || exit 1

# --- arm 5: reader present, contribution identically zero -------------------
run_arm ple-off "-" real --ple-off --load-reader "$WIKI_CKPT" || exit 1

log "queue DONE"
exec 9>&-
