#!/usr/bin/env bash
# Round 168 Stage 1.5g -- the aligned rung-3 probe, three domains x two K.
#
# Why all six run at once instead of serially
# -------------------------------------------
# The smoke run (20k wiki positions) measured: wall 64s, of which row fetching
# was 3.4s, i.e. ~5%.  `user` time was 4m52s for 64s of wall, so the probe is
# CPU-bound on the ridge/MLP fits, not on the 48 GiB row table.  The box has 128
# cores, so six independent runs get ~21 cores each.  Serial execution would
# leave the box idle for no reason -- the round-167 utilisation rule: if a job
# is not I/O-bound and cores are free, launch the independent arms together.
#
# The thread pools are pinned because six processes each defaulting to all 128
# cores is oversubscription, which is slower than pinning, not faster.
#
# Usage:
#   bash scripts/run_round168_rung3.sh            # all six
#   TAGS="wiki" KS="5000" bash scripts/run_round168_rung3.sh
set -uo pipefail

REPO=${REPO:-/root/autodl-tmp/qwen35-ple/repo}
PY=${PY:-/root/autodl-tmp/qwen35-ple/venv/bin/python}
LOGDIR=${LOGDIR:-/root/autodl-tmp/qwen35-ple/logs}
OUT=$REPO/outputs/round168/rung3
TAGS=${TAGS:-"wiki code stem"}
KS=${KS:-"5000 1000"}

mkdir -p "$OUT" "$LOGDIR"
cd "$REPO" || exit 1
export PYTHONPATH=src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

eval_for() {
  case "$1" in
    wiki) echo "data/phase1/wikitext-heldout-decon/tokens.npy" ;;
    code) echo "data/phase1/PURE_CODE-heldout-decon/tokens.npy" ;;
    stem) echo "data/phase1/PURE_STEM-heldout-decon/tokens.npy" ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}

# Each domain must use the train stream its OWN Stage 1.5c counts used, otherwise
# the candidate set K and the explicit-count baseline would not be the same
# reference and R would be a ratio of two different measurements.
train_for() {
  case "$1" in
    wiki) echo "data/phase1/PURE_WIKI/tokens.npy" ;;
    code) echo "data/phase1/PURE_CODE/tokens.npy" ;;
    stem) echo "data/phase1/PURE_STEM/tokens.npy" ;;
  esac
}

done_marker() { grep -q '"stage": "complete"' "$1" 2>/dev/null; }

PIDS=()
NAMES=()
run_one() {
  local tag=$1 k=$2
  local log="$LOGDIR/rung3-$tag-k$k.log"
  local out="$OUT/probe-$tag-k$k.json"
  if done_marker "$out"; then
    echo "[$(date +%H:%M:%S)] SKIP   $tag K=$k (already complete)"
    return 0
  fi
  {
    local t0 t1 rc
    t0=$(date +%s)
    echo "=== $tag K=$k start $(date -Is) ==="
    "$PY" scripts/probe_table_next_token.py \
      --eval-tokens "$(eval_for "$tag")" \
      --aligned-positions "outputs/round168/rung3/positions-$tag.npy" \
      --train-tokens "$(train_for "$tag")" \
      --topk "$k" \
      --mlp-hidden 512 \
      --max-alloc-mb 8192 \
      --state-file "outputs/round168/rung3/state-$tag-k$k" \
      --out "$out"
    rc=$?
    t1=$(date +%s)
    echo "=== $tag K=$k rc=$rc wall=$((t1 - t0))s end $(date -Is) ==="
  } > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$tag-k$k")
  echo "[$(date +%H:%M:%S)] LAUNCH $tag K=$k pid=$! -> $log"
}

# Only the PIDs we launched are joined.  A bare `wait` also waits on the
# sampler below, which never exits on its own -- the deadlock documented in
# docs/gpu-utilisation-playbook.md section 2.5.
wait_all() {
  local i p
  for i in "${!PIDS[@]}"; do
    p=${PIDS[$i]}
    if wait "$p"; then
      echo "[$(date +%H:%M:%S)] OK     ${NAMES[$i]}"
    else
      echo "[$(date +%H:%M:%S)] FAIL   ${NAMES[$i]} (rc=$?) -- see $LOGDIR/rung3-${NAMES[$i]}.log"
    fi
  done
  PIDS=()
  NAMES=()
}

# Continuous sampling, not a probe between phases: a probe between phases sees
# an idle box and reports 0%, which is noise rather than a measurement.
STOP=$(mktemp)
sampler() {
  local f="$LOGDIR/rung3-util.log"
  while [ ! -f "$STOP" ]; do
    local n load
    n=$(pgrep -cf "[p]robe_table_next_token" || true)
    load=$(cut -d' ' -f1-3 /proc/loadavg)
    printf '%s procs=%s load=%s\n' "$(date +%H:%M:%S)" "${n:-0}" "$load" >> "$f"
    sleep 30
  done
}
sampler &
SAMPLER_PID=$!

echo "[$(date +%H:%M:%S)] launching $TAGS x K=$KS  (threads/job=$OMP_NUM_THREADS, cores=$(nproc))"
for tag in $TAGS; do
  for k in $KS; do
    run_one "$tag" "$k"
  done
done

wait_all

touch "$STOP"
wait "$SAMPLER_PID" 2>/dev/null

echo
echo "[$(date +%H:%M:%S)] queue DONE"
for tag in $TAGS; do
  for k in $KS; do
    f="$OUT/probe-$tag-k$k.json"
    if done_marker "$f"; then
      printf '  COMPLETE %-10s K=%-5s\n' "$tag" "$k"
    else
      printf '  MISSING  %-10s K=%-5s\n' "$tag" "$k"
    fi
  done
done
