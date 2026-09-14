#!/usr/bin/env bash
# Round 169, EXPLORATORY: the R(K) curve for ladder rung 3.
#
# This does NOT touch the frozen Stage 1.5g verdict.  Section 3 of
# docs/round-168-stage1.5g-preregistration.md decides on K=5000 (primary) and
# K=1000 (sensitivity); the extra K values here only map the shape of R(K) and
# are reported as an exploratory curve.
#
# The question it answers: is the 0.51-0.74 shortfall a limit of the MEASUREMENT
# (smaller candidate sets weaken the explicit-count baseline, so R should rise
# toward 1 as K falls) or is the row content genuinely degraded (R plateaus below
# 1 and does not care about K)?
#
# CPU + row-table I/O only; no GPU.  Nine independent runs, launched together.
set -uo pipefail

REPO=${REPO:-/root/autodl-tmp/qwen35-ple/repo}
PY=${PY:-/root/autodl-tmp/qwen35-ple/venv/bin/python}
LOGDIR=${LOGDIR:-/root/autodl-tmp/qwen35-ple/logs}
OUT=$REPO/outputs/round169/rk-curve
TAGS=${TAGS:-"wiki code stem"}
KS=${KS:-"200 500 20000"}

mkdir -p "$OUT" "$LOGDIR"
cd "$REPO" || exit 1
export PYTHONPATH=src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

eval_for() {
  case "$1" in
    wiki) echo "data/phase1/wikitext-heldout-decon/tokens.npy" ;;
    code) echo "data/phase1/PURE_CODE-heldout-decon/tokens.npy" ;;
    stem) echo "data/phase1/PURE_STEM-heldout-decon/tokens.npy" ;;
  esac
}
train_for() {
  case "$1" in
    wiki) echo "data/phase1/PURE_WIKI/tokens.npy" ;;
    code) echo "data/phase1/PURE_CODE/tokens.npy" ;;
    stem) echo "data/phase1/PURE_STEM/tokens.npy" ;;
  esac
}
done_marker() { grep -q '"stage": "complete"' "$1" 2>/dev/null; }

PIDS=(); NAMES=()
run_one() {
  local tag=$1 k=$2
  local log="$LOGDIR/r169-rk-$tag-k$k.log"
  local out="$OUT/probe-$tag-k$k.json"
  if done_marker "$out"; then echo "[$(date +%H:%M:%S)] SKIP   $tag K=$k"; return 0; fi
  {
    local t0 t1 rc
    t0=$(date +%s)
    echo "=== $tag K=$k start $(date -Is) ==="
    "$PY" scripts/probe_table_next_token.py \
      --eval-tokens "$(eval_for "$tag")" \
      --aligned-positions "outputs/round168/rung3/positions-$tag.npy" \
      --train-tokens "$(train_for "$tag")" \
      --topk "$k" --mlp-hidden 512 --max-alloc-mb 8192 \
      --state-file "$OUT/state-$tag-k$k" \
      --out "$out"
    rc=$?
    t1=$(date +%s)
    echo "=== $tag K=$k rc=$rc wall=$((t1 - t0))s end $(date -Is) ==="
  } > "$log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$tag-k$k")
  echo "[$(date +%H:%M:%S)] LAUNCH $tag K=$k pid=$!"
}

for tag in $TAGS; do for k in $KS; do run_one "$tag" "$k"; done; done

# Join only the PIDs we launched (never a bare `wait`: the playbook's deadlock).
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "[$(date +%H:%M:%S)] OK   ${NAMES[$i]}"
  else echo "[$(date +%H:%M:%S)] FAIL ${NAMES[$i]} -- see $LOGDIR/r169-rk-${NAMES[$i]}.log"; fi
done

echo "[$(date +%H:%M:%S)] queue DONE"
for tag in $TAGS; do for k in $KS; do
  f="$OUT/probe-$tag-k$k.json"
  done_marker "$f" && printf '  COMPLETE %-6s K=%-6s\n' "$tag" "$k" \
                   || printf '  MISSING  %-6s K=%-6s\n' "$tag" "$k"
done; done
