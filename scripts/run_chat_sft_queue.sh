#!/usr/bin/env bash
# Decisive chat-template experiment:
#   train the reader on chat-template QA SFT data and evaluate with the chat
#   template, so format/instruction-following is controlled by the protocol.
#
# Waits for the current large-SFT queue to finish, then runs:
#   1. sft-chat-mixed50 (6000 QA items, 3 seeds, eval-600 chat)
#   2. sft-chat-mixed50-warmup250 (two-stage)
#   3. full 1500-item chat held-out eval of the best reader
#   4. oracle/report
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/logs/chat-sft-queue.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/outputs"

log() { echo "=== [chat-sft] $* $(date -Is) ===" | tee -a "$LOG"; }
run_logged() {
  local logfile="$1"; shift
  echo "--- [$(date -Is)] $*" >>"$logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  echo "--- [$(date -Is)] rc=$rc" >>"$logfile"
  return $rc
}
matrix_done() {
  local dir="$1" corpus="$2"
  [[ -f "$dir/phase1-$corpus.json" ]] && grep -q '"summary"' "$dir/phase1-$corpus.json"
}

# Wait for the current large-SFT queue.
for i in $(seq 1 480); do
  if [[ -f "$ROOT/outputs/LARGE_SFT_DONE" ]]; then log "large-SFT queue DONE detected"; break; fi
  if (( i % 10 == 1 )); then log "waiting for large-SFT queue (iteration $i)"; fi
  sleep 60
done

run_chat_config() {
  # run_chat_config <name> <warmup>
  local name="$1" warmup="$2"
  local out="$ROOT/outputs/$name"
  mkdir -p "$out"
  if [[ -f "$out/DONE" ]]; then log "skip $name"; return 0; fi
  log "chat SFT $name warmup=$warmup"
  for attempt in 1 2 3; do
    run_logged "$ROOT/logs/$name.log" env \
      OUTPUT_DIR="$out" CORPORA="PURE_WIKI" MODES="real control no-reader" \
      STEPS=500 SEEDS="0 1 2" LAYER=2 RESUME=1 \
      MAX_NEW=32 QA_BATCH_SIZE=16 QA_BATCH_MAX_TOKENS=2048 \
      QA_FILE=data/qa-standard/eval-600.jsonl \
      QA_SFT_FILE=data/qa-standard/train.jsonl \
      QA_SFT_WEIGHT=0.5 QA_SFT_MAX_LEN=512 QA_SFT_LOG_EVERY=100 \
      QA_SFT_LAZY=1 QA_SFT_LAZY_CACHE=128 \
      QA_SFT_CHAT_TEMPLATE=1 QA_CHAT_TEMPLATE=1 \
      QA_SFT_WARMUP_STEPS="$warmup" \
      bash scripts/run_phase2_diagnostic.sh
    if matrix_done "$out" PURE_WIKI; then break; fi
    log "chat SFT $name attempt $attempt failed; retrying"
    sleep 20
  done
  if matrix_done "$out" PURE_WIKI; then
    run_logged "$ROOT/logs/$name.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$out/phase1-PURE_WIKI.json" --protocol v2 \
        --output "$out/summary-v2.json" --markdown "$out/summary-v2.md"
    run_logged "$ROOT/logs/$name.log" \
      "$PY" scripts/check_phase2_gates.py \
        --files "$out/phase1-PURE_WIKI.json" --protocol v2 --metric auto \
        --output "$out/gates-v2.json" --markdown "$out/gates-v2.md"
    touch "$out/DONE"
  fi
}

run_chat_config "sft-chat-mixed50" "0"
run_chat_config "sft-chat-mixed50-warmup250" "250"

# Full 1500-item chat held-out eval of the best chat-SFT reader.
FULL="$ROOT/outputs/qa-standard-eval-chat-large"
mkdir -p "$FULL"
if ! matrix_done "$FULL" full; then
  log "full chat eval: no-reader"
  run_logged "$ROOT/logs/chat-sft-full.log" \
    "$PY" -u scripts/run_phase0.py \
      --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
      --rows-dir /dev/shm/qwen38-rows \
      --model-dir "$ROOT/models/qwen38_ple" \
      --model "$ROOT/models/Qwen3.5-0.8B" \
      --reader official --layer 2 --device cuda --bridge-mlp --out-mlp \
      --official-reader-path data/official_ple_reader.pt \
      --steps 0 --seq-len 128 --lr 1e-4 --seeds 0 --modes no-reader \
      --qa --qa-exact-match --qa-max-new-tokens 32 \
      --qa-batch-size 16 --qa-batch-max-tokens 2048 \
      --qa-chat-template --qa-file data/qa-standard/eval.jsonl \
      --resume --partial-dir "$FULL/partial" --backup-dir "$FULL/backup" \
      --output "$FULL/phase1-full.json"
  for seed in 0 1 2; do
    ckpt="$ROOT/outputs/sft-chat-mixed50/reader-PURE_WIKI-real-seed$seed.pt"
    if [[ -f "$ckpt" ]]; then
      log "full chat eval: real seed=$seed"
      run_logged "$ROOT/logs/chat-sft-full.log" \
        "$PY" -u scripts/run_phase0.py \
          --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
          --rows-dir /dev/shm/qwen38-rows \
          --model-dir "$ROOT/models/qwen38_ple" \
          --model "$ROOT/models/Qwen3.5-0.8B" \
          --reader official --layer 2 --device cuda --bridge-mlp --out-mlp \
          --official-reader-path data/official_ple_reader.pt \
          --load-reader "$ckpt" \
          --steps 0 --seq-len 128 --lr 1e-4 --seeds "$seed" --modes real \
          --qa --qa-exact-match --qa-max-new-tokens 32 \
          --qa-batch-size 16 --qa-batch-max-tokens 2048 \
          --qa-chat-template --qa-file data/qa-standard/eval.jsonl \
          --resume --partial-dir "$FULL/partial-seed$seed" \
          --backup-dir "$FULL/backup-seed$seed" \
          --output "$FULL/phase1-real-seed$seed.json"
    fi
  done
  touch "$FULL/DONE"
fi

# Oracle + report.
ORACLE="$ROOT/outputs/oracle-analysis"
mkdir -p "$ORACLE"
for dir in "$ROOT"/outputs/sft-chat-* "$FULL"; do
  [[ -d "$dir" ]] || continue
  name="$(basename "$dir")"
  if compgen -G "$dir/phase1-*.json" >/dev/null; then
    run_logged "$ROOT/logs/chat-sft-queue.log" \
      "$PY" scripts/analyze_oracle_upper_bound.py \
        --files "$dir/phase1-*.json" --protocol v2 --metric auto \
        --output "$ORACLE/$name.json" --markdown "$ORACLE/$name.md"
  fi
done

run_logged "$ROOT/logs/chat-sft-queue.log" \
  "$PY" scripts/build_overnight_report.py \
    --root "$ROOT/outputs" --output "$ROOT/outputs/OVERNIGHT_REPORT.md"

log "CHAT SFT QUEUE DONE"
echo done > "$ROOT/outputs/CHAT_SFT_DONE"
