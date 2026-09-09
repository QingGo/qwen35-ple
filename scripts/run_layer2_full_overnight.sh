#!/usr/bin/env bash
# Robust layer2 full diagnostic runner: retries until all corpora JSONs exist.
# Intended to be launched with nohup on the remote 4090.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
OUT="${OUTPUT_DIR:-$ROOT/outputs/phase2-diagnostic-layer2}"
LOG="${LOG:-$ROOT/logs/phase2-layer2-full.log}"
mkdir -p "$OUT" "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export ROOT="$ROOT"
export OUTPUT_DIR="$OUT"
export ROWS_DIR="${ROWS_DIR:-/dev/shm/qwen38-rows}"
export PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
export MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen38_ple}"
export MODEL="${MODEL:-$ROOT/models/Qwen3.5-0.8B}"
export LAYER="${LAYER:-2}"
export CORPORA="${CORPORA:-PURE_WIKI PURE_CODE FW_STEM}"
export STEPS="${STEPS:-500}"
export SEEDS="${SEEDS:-0 1 2}"
export RESUME=1
export QA_BATCH_SIZE="${QA_BATCH_SIZE:-16}"
export QA_BATCH_MAX_TOKENS="${QA_BATCH_MAX_TOKENS:-2048}"
echo "=== [layer2-full] start $(date -Is) ===" | tee -a "$LOG"
for attempt in 1 2 3 4 5; do
  echo "=== [layer2-full] attempt $attempt $(date -Is) ===" | tee -a "$LOG"
  bash scripts/run_phase2_diagnostic.sh >>"$LOG" 2>&1
  rc=$?
  echo "=== [layer2-full] attempt $attempt rc=$rc $(date -Is) ===" | tee -a "$LOG"
  missing=0
  for c in $CORPORA; do
    if ! grep -q '"summary"' "$OUT/phase1-$c.json" 2>/dev/null; then missing=1; fi
  done
  if [[ $missing -eq 0 ]]; then
    echo "=== [layer2-full] ALL DONE $(date -Is) ===" | tee -a "$LOG"
    echo "done" > "$OUT/DONE"
    exit 0
  fi
  sleep 30
done
echo "=== [layer2-full] FAILED after retries $(date -Is) ===" | tee -a "$LOG"
echo "failed" > "$OUT/FAILED"
exit 1
