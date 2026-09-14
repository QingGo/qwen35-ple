#!/usr/bin/env bash
# Round 169 overnight queue: finish the B1 capacity curve, then replicate B1 on
# the other two domains, then write a roll-up.
#
# Design rules, each learned the hard way in this session:
#
#  * MEMORY GUARD.  The container's cap is /sys/fs/cgroup/memory.max = 120 GiB,
#    NOT the host's 1007 GB that `free` reports, and page cache counts against it.
#    Five concurrent jobs already OOM-killed the control arm once (rc=137) while
#    its log looked perfectly normal.  Every launch waits for headroom first.
#  * ONE HEAVY GPU JOB AT A TIME.  2 trainers is 19 GiB of 24 GiB and leaves no
#    room for an eval; 3 is over.  Serial is a deliberate choice, not laziness.
#  * EVERY rc IS RECORDED.  A killed job stops mid-epoch with no error line, so
#    "did the product appear" is not enough.
#  * DISK IS RECYCLED.  Each domain's E0 snapshot is ~6.5 GB and is deleted the
#    moment that domain's training is done; only the 1.24 GB bank is kept.
#  * EXPLORATORY WORK IS NAMED AS SUCH.  The pre-registration froze wiki only
#    (PURE_WIKI -> wikitext-heldout-decon).  code/stem are a replication of the
#    frozen rule on other domains and CANNOT revise the wiki verdict.
set -uo pipefail

REPO=${REPO:-/root/autodl-tmp/qwen35-ple/repo}
PY=${PY:-/root/autodl-tmp/qwen35-ple/venv/bin/python}
LOGD=${LOGD:-/root/autodl-tmp/qwen35-ple/logs}
OUT=$REPO/outputs/round169
EXPL=$OUT/exploratory
ROWS=/root/autodl-tmp/qwen35-ple/qwen38-rows
MARGIN=/root/autodl-tmp/qwen35-ple/outputs/round168/margin
MODEL=/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
READER_DIR=/root/autodl-tmp/qwen35-ple/outputs/round162-0.8B-nosft600
MEM_CAP_GB=${MEM_CAP_GB:-85}
DONE_MARK=$OUT/OVERNIGHT_DONE
FAIL_MARK=$OUT/OVERNIGHT_FAILED

mkdir -p "$OUT" "$EXPL" "$LOGD"
cd "$REPO" || exit 1
export PYTHONPATH=src
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -f "$DONE_MARK" "$FAIL_MARK"
FAILED=()
STEPS=()

say() { echo "[$(date +%H:%M:%S)] $*"; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }

# Never launch into a nearly-full cgroup.  The cap is unknown to the kernel
# until it kills something, so this has to be enforced by us.
wait_mem() {
  local cap=${1:-$MEM_CAP_GB} n=0 cur
  while :; do
    cur=$(mem_gb); cur=${cur:-0}
    [ "$cur" -le "$cap" ] && return 0
    n=$((n + 1))
    [ $((n % 10)) -eq 1 ] && say "  memory guard: ${cur} GB > ${cap} GB, waiting"
    sleep 30
  done
}

# Launch a GPU job, wait for it, record rc. Never bare `wait`.
run_gpu() {
  local name=$1; shift
  local log=$LOGD/overnight-$name.log
  wait_mem
  say "START $name  (mem $(mem_gb) GB, gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')%)"
  local t0 t1 rc
  t0=$(date +%s)
  { echo "=== $name start $(date -Is) ==="; "$@"; rc=$?; echo "=== $name rc=$rc ==="; } > "$log" 2>&1
  t1=$(date +%s)
  say "END   $name rc=$rc wall=$((t1 - t0))s -> $log"
  STEPS+=("$name rc=$rc wall=$((t1 - t0))s")
  [ "$rc" -ne 0 ] && FAILED+=("$name")
  return 0
}

train_bank() {  # snapshot_dir train_tokens reader keep_index arm epochs lr out
  "$PY" scripts/round169_train_rows.py \
    --snapshot-dir "$1" --train-tokens "$2" --model "$MODEL" \
    --reader-kind saved --reader "$3" --keep-index "$4" \
    --arm "$5" --epochs "$6" --lr "$7" --out "$8"
}

eval_bank() {  # snapshot_dir bank keep_index tokens positions counts reader out
  "$PY" scripts/round169_eval_rows.py \
    --snapshot-dir "$1" --trained-bank "$2" --keep-index "$3" \
    --tokens "$4" --positions "$5" --counts-npz "$6" \
    --rows-dir "$ROWS" --model "$MODEL" \
    --reader-kind saved --reader "$7" --layer 2 --chunk 1024 --out "$8"
}

# Wait until no other round169 GPU job is running.  Two trainers plus an eval do
# not fit: 2x9.5 GiB of the 24 GiB card plus an eval is over, and the cgroup has
# already killed a control arm once.  The lr=1e-4 arm and its eval are still in
# flight when this queue starts, so it waits for them rather than racing them.
wait_quiet() {
  local n=0 busy
  while :; do
    busy=$(pgrep -cf '[r]ound169_(train|eval)_rows' || true); busy=${busy:-0}
    [ "$busy" -eq 0 ] && return 0
    n=$((n + 1))
    [ $((n % 5)) -eq 1 ] && say "  waiting for $busy in-flight round169 job(s) to finish"
    sleep 60
  done
}

##############################################################################
say "=== overnight queue starting ==="
say "memory cap enforced at ${MEM_CAP_GB} GB of $(awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.max) GB"
df -h /root/autodl-tmp | tail -1
wait_quiet
say "no in-flight round169 jobs; starting the queue"

# ---- STEP 1: wiki lr=1e-3, 1 epoch (completes the pre-registered capacity curve)
# The 4-epoch run showed harm accumulated over exposure; if 1 epoch harms far
# less, the harm is a function of accumulated uninformed movement.
KEEP=$OUT/keep-wiki-eval.npy
REAL_R=$READER_DIR/../round162-0.8B-nosft/reader-wiki-seed0.pt
WIKI_E0=$OUT/E0.npy
if [ ! -s "$OUT/E-real-e1-lr1e-3.npy" ]; then
  run_gpu wiki-e1-lr1e-3 train_bank "$OUT" data/phase1/PURE_WIKI/tokens.npy \
    "$REAL_R" "$KEEP" real 1 1e-3 "$OUT/E-real-e1-lr1e-3.npy"
  run_gpu wiki-e1-eval eval_bank "$OUT" "$OUT/E-real-e1-lr1e-3.npy" "$KEEP" \
    data/phase1/wikitext-heldout-decon/tokens.npy outputs/round168/rung3/positions-wiki.npy \
    "$MARGIN/wiki-counts.npz" "$REAL_R" "$OUT/eval-real-e1-lr1e-3.json"
else
  say "SKIP wiki-e1 (bank already present)"
fi

##############################################################################
# ---- recycle the 6.5 GB wiki snapshot: every wiki arm is finished by now
if [ -s "$WIKI_E0" ]; then
  say "recycling wiki E0 snapshot ($(du -h "$WIKI_E0" | cut -f1))"
  rm -f "$WIKI_E0"
fi
df -h /root/autodl-tmp | tail -1

##############################################################################
# ---- STEPS 2-3: EXPLORATORY replication on code and stem
# Does ROWS_HARMED generalise?  R(K) says the two domains differ a lot: code's
# rows are the best (R=0.69-0.80) and stem's the worst but read-out-saturated.
# If code's rows survive retraining and stem's do not, the effect is about the
# rows; if both are harmed, it is about the frozen reader.
for spec in "code:PURE_CODE:reader-code-seed0.pt" "stem:PURE_STEM:reader-stem-seed0.pt"; do
  tag=${spec%%:*}; rest=${spec#*:}; corpus=${rest%%:*}; rfile=${rest##*:}
  say "=== EXPLORATORY domain: $tag ==="
  SNAPDIR=$EXPL/$tag
  KEEPX=$EXPL/keep-$tag-eval.npy
  READER=$READER_DIR/$rfile
  EVAL=data/phase1/${corpus}-heldout-decon/tokens.npy
  POS=outputs/round168/rung3/positions-$tag.npy
  COUNTS=$MARGIN/$tag-counts.npz
  BANK=$EXPL/E-real-e4-lr1e-3-$tag.npy

  if [ ! -s "$SNAPDIR/E0.npy" ]; then
    wait_mem
    say "  snapshot $tag"
    { "$PY" scripts/round169_row_snapshot.py \
        --train-tokens "data/phase1/$corpus/tokens.npy" --rows-dir "$ROWS" \
        --out-dir "$SNAPDIR" --chunk 50000; rc=$?; } > "$LOGD/overnight-snap-$tag.log" 2>&1
    STEPS+=("snap-$tag rc=${rc:-?}")
    [ "${rc:-1}" -ne 0 ] && { FAILED+=("snap-$tag"); continue; }
  fi

  if [ ! -s "$KEEPX" ]; then
    { "$PY" scripts/round169_keep_index.py \
        --snapshot-codes "$SNAPDIR/trigram-codes.npy" \
        --tokens "$EVAL" --positions "$POS" --out "$KEEPX" \
        --meta-out "$EXPL/keep-$tag.meta.json"; rc=$?; } > "$LOGD/overnight-keep-$tag.log" 2>&1
    STEPS+=("keep-$tag rc=${rc:-?}")
    [ "${rc:-1}" -ne 0 ] && { FAILED+=("keep-$tag"); continue; }
  fi

  if [ ! -s "$BANK" ]; then
    run_gpu "train-$tag" train_bank "$SNAPDIR" "data/phase1/$corpus/tokens.npy" \
      "$READER" "$KEEPX" real 4 1e-3 "$BANK"
  else
    say "  SKIP train-$tag (bank present)"
  fi

  if [ -s "$BANK" ] && [ ! -s "$EXPL/eval-real-$tag.json" ]; then
    run_gpu "eval-$tag" eval_bank "$SNAPDIR" "$BANK" "$KEEPX" "$EVAL" "$POS" \
      "$COUNTS" "$READER" "$EXPL/eval-real-$tag.json"
  fi

  # recycle the ~6.5 GB snapshot the moment its domain is done (keep the 1.24 GB bank)
  [ -d "$SNAPDIR" ] && { say "  recycling $tag snapshot"; rm -rf "$SNAPDIR"; }
  df -h /root/autodl-tmp | tail -1
done

##############################################################################
say "=== writing summary from the artifacts (mechanical, not prose) ==="
{ "$PY" scripts/round169_summarize.py --root "$OUT" --exploratory "$EXPL" \
    --out "$OUT/OVERNIGHT_SUMMARY.md"; rc=$?; } > "$LOGD/overnight-summary.log" 2>&1
STEPS+=("summary rc=${rc:-?}")
[ "${rc:-1}" -ne 0 ] && FAILED+=("summary")
say "SUMMARY:"; head -60 "$OUT/OVERNIGHT_SUMMARY.md" 2>/dev/null

say "=== writing step roll-up ==="
{
  echo "# Round 169 overnight roll-up (step log)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_overnight.sh"
  echo
  echo "## Steps"
  for s in "${STEPS[@]}"; do echo "* $s"; done
  echo
  echo "## Failures"
  if [ ${#FAILED[@]} -eq 0 ]; then echo "* none"; else for f in "${FAILED[@]}"; do echo "* $f"; done; fi
  echo
  echo "## Result files present"
  for f in "$OUT"/eval-*.json "$EXPL"/eval-*.json; do [ -s "$f" ] && echo "* $f"; done
} > "$OUT/OVERNIGHT_ROLLUP.md" 2>/dev/null

say "ROLLUP:"; cat "$OUT/OVERNIGHT_ROLLUP.md" 2>/dev/null | head -40

# A done-marker is written either way; the collector reads the failure list, so a
# partial night still gets pulled and shut down rather than being left running.
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "OK" > "$DONE_MARK"
  say "*** OVERNIGHT DONE (all steps ok) ***"
else
  printf '%s\n' "${FAILED[@]}" > "$FAIL_MARK"
  echo "OK_WITH_FAILURES" > "$DONE_MARK"
  say "*** OVERNIGHT DONE with ${#FAILED[@]} failure(s): ${FAILED[*]} ***"
fi
say "queue finished; the local collector pulls artifacts and then shuts the box down"
exit 0
