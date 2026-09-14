#!/usr/bin/env bash
# Round 169 Stage B1: train the row bank for the real and null arms.
#
# Two arms, run CONCURRENTLY: each step needs ~5 GiB on the GPU (batch 512 with a
# frozen 0.8B backbone and a backward through it), so two fit a 24 GiB card.  The
# round-167 utilisation rule applies -- if the arms are independent and the
# device is not saturated, launch them together.
#
# bf16 saving is deliberately not used; see --keep-index in round169_train_rows.py.
set -uo pipefail

REPO=${REPO:-/root/autodl-tmp/qwen35-ple/repo}
PY=${PY:-/root/autodl-tmp/qwen35-ple/venv/bin/python}
LOGDIR=${LOGDIR:-/root/autodl-tmp/qwen35-ple/logs}
OUT=$REPO/outputs/round169
READER=/root/autodl-tmp/qwen35-ple/outputs/round162-0.8B-nosft/reader-wiki-seed0.pt
MODEL=/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
EPOCHS=${EPOCHS:-4}
LR=${LR:-1e-3}

mkdir -p "$OUT" "$LOGDIR"
cd "$REPO" || exit 1
export PYTHONPATH=src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIDS=()
NAMES=()
run_arm() {
  local arm=$1
  local log="$LOGDIR/r169-train-$arm.log"
  local out="$OUT/E-$arm-e$EPOCHS-lr$LR.npy"
  if [ -s "$out" ]; then
    echo "[$(date +%H:%M:%S)] SKIP   $arm (already present)"
    return 0
  fi
  {
    local t0 t1 rc
    t0=$(date +%s)
    echo "=== arm=$arm epochs=$EPOCHS lr=$LR start $(date -Is) ==="
    "$PY" scripts/round169_train_rows.py \
      --snapshot-dir outputs/round169 \
      --train-tokens data/phase1/PURE_WIKI/tokens.npy \
      --model "$MODEL" \
      --reader-kind saved --reader "$READER" \
      --keep-index outputs/round169/keep-wiki-eval.npy \
      --arm "$arm" --epochs "$EPOCHS" --lr "$LR" \
      --out "$out"
    rc=$?
    t1=$(date +%s)
    echo "=== arm=$arm rc=$rc wall=$((t1 - t0))s end $(date -Is) ==="
  } > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$arm")
  echo "[$(date +%H:%M:%S)] LAUNCH $arm pid=$! -> $log"
}

wait_all() {
  local i p
  for i in "${!PIDS[@]}"; do
    p=${PIDS[$i]}
    if wait "$p"; then
      echo "[$(date +%H:%M:%S)] OK     ${NAMES[$i]}"
    else
      echo "[$(date +%H:%M:%S)] FAIL   ${NAMES[$i]} rc=$? -- see $LOGDIR/r169-train-${NAMES[$i]}.log"
    fi
  done
  PIDS=(); NAMES=()
}

run_arm real
sleep 20   # stagger the model loads; two 3.2 GB loads at once spikes peak memory
run_arm shuf
wait_all

echo
echo "[$(date +%H:%M:%S)] queue DONE"
for arm in real shuf; do
  f="$OUT/E-$arm-e$EPOCHS-lr$LR.npy"
  if [ -s "$f" ]; then printf '  COMPLETE %-6s %s\n' "$arm" "$(du -h "$f" | cut -f1)"
  else printf '  MISSING  %-6s\n' "$arm"; fi
done
