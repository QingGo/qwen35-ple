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
OUT="${OUT:-$ROOT/outputs/round167}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$OUT" "$LOGDIR"

# name|model-dir|backbone-dtype|reader
CASES=(
  "0.8B|Qwen3.5-0.8B|float32|round162-0.8B-nosft600"
  "2B|Qwen3.5-2B|bfloat16|round166-2B-nosft"
  "4B|Qwen3.5-4B|bfloat16|round166-4B-nosft"
)

cd "$REPO"
for spec in "${CASES[@]}"; do
  IFS='|' read -r name model dtype readerdir <<<"$spec"
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
