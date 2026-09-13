#!/usr/bin/env bash
# Round 167 Stage 1a: effective depth under PLE on/off, three backbones.
#
# Pre-registration: docs/round-167-stage1-effective-depth-preregistration.md
# The readers are the *no-SFT* ones for all three backbones so the comparison is
# consistent (2B only exists in that regime), i.e. the unsaturated configuration
# in which round-166 saw the table content reach the output at all.
#
# Usage:  bash scripts/run_round167_stage1a.sh
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="${REPO:-$ROOT/repo}"
PY="${PY:-$ROOT/venv/bin/python}"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
QA="${QA:-data/qa-standard/eval-600b.jsonl}"
ITEMS="${ITEMS:-200}"
# Batched forwards.  At batch 1 the GPU sat at ~9% (an item's prompt is ~30
# tokens); batching is validated numerically equivalent to the batch-1 path
# (top-5 overlap identical, KL to 4e-5).  32 fits comfortably in 24 GiB.
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"
OUT="${OUT:-$ROOT/outputs/round167}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$OUT" "$LOGDIR"

# name|model-dir|backbone-dtype|reader|batch-tokens
# The 4B backbone (hidden 2560, 32 layers) needs a much smaller token budget:
# 8192 tokens OOMs it, matching the round-162 4B convention of 4 items / 2048
# tokens.
CASES=(
  "0.8B|Qwen3.5-0.8B|float32|round162-0.8B-nosft600|8192"
  "2B|Qwen3.5-2B|bfloat16|round166-2B-nosft|8192"
  "4B|Qwen3.5-4B|bfloat16|round166-4B-nosft|2048"
)

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for spec in "${CASES[@]}"; do
  IFS='|' read -r name model dtype readerdir batch_tokens <<<"$spec"
  out="$OUT/effective-depth-$name.json"
  log="$LOGDIR/round167-stage1a-$name.log"
  if [ -f "$out" ]; then
    echo "[stage1a] $name: already present, skipping ($out)"
    continue
  fi
  echo "[stage1a] $name -> $out"
  PYTHONPATH=src "$PY" scripts/round167_effective_depth.py \
    --model "$ROOT/models/$model" \
    --rows-dir "$ROWS" \
    --reader "$ROOT/outputs/$readerdir/reader-wiki-seed0.pt" \
    --qa-file "$QA" \
    --max-items "$ITEMS" \
    --batch-size "$BATCH_SIZE" \
    --batch-tokens "$batch_tokens" \
    --backbone-dtype "$dtype" \
    --output "$out" >"$log" 2>&1
  rc=$?
  echo "[stage1a] $name exit=$rc"
  if [ $rc -ne 0 ]; then
    echo "[stage1a] $name FAILED; tail of $log:"
    tail -20 "$log"
    exit $rc
  fi
done
echo "[stage1a] done"
touch "$OUT/STAGE1A_DONE"
