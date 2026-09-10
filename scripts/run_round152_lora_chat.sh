#!/usr/bin/env bash
# Round 152 secondary protocol: re-evaluate the trained LoRA arms on the chat
# template (the pre-registered secondary endpoint), reusing the saved reader
# checkpoints and LoRA adapters.  No retraining: read-only evaluation.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/round152chat"
LOG="$ROOT/logs/round152-lora-chat.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

log() { echo "=== [lora-chat] $* $(date -Is) ===" | tee -a "$LOG"; }

run_arm() {
  local name="$1" mode="$2"
  local reader="$ROOT/outputs/round152/adapter-$name-seed0"
  [[ -f "$reader" ]] || reader="$ROOT/outputs/round152/adapter-$name-seed0.pt"
  local adapter="$ROOT/outputs/round152/adapter-$name-seed0.lora"
  local out="$OUT/$name.json"
  if [[ -f "$out" ]] && grep -q '"qa_gold"' "$out" 2>/dev/null; then
    log "skip $name (complete)"
    return 0
  fi
  if [[ ! -f "$reader" || ! -d "$adapter" ]]; then
    log "SKIP $name: missing reader ($reader) or adapter ($adapter)"
    return 1
  fi
  local args=(
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy
    --rows-dir /dev/shm/qwen38-rows
    --model-dir "$ROOT/models/qwen38_ple"
    --model "$ROOT/models/Qwen3.5-0.8B"
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp
    --official-reader-path data/official_ple_reader.pt
    --lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.0
    --load-lora-adapter "$adapter"
    --load-reader "$reader"
    --steps 0 --seq-len 128 --seeds 0 --modes "$mode"
    --qa --qa-exact-match --qa-gold-nll --qa-max-new-tokens 32
    --qa-batch-size 8 --qa-batch-max-tokens 2048
    --qa-chat-template
    --qa-file data/qa-standard/eval.jsonl
    --output "$out"
  )
  log "run $name mode=$mode"
  "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED $name rc=$rc"
    return 1
  fi
  log "finished $name"
  return 0
}

run_arm lora-nople real
run_arm lora-real real
run_arm lora-control control

GEN=()
for n in lora-nople lora-real lora-control; do
  [[ -f "$OUT/$n.json" ]] && GEN+=("--run" "$n=$OUT/$n.json")
done
if (( ${#GEN[@]} )); then
  "$PY" scripts/summarize_arms.py "${GEN[@]}" \
    --pair lora-real=lora-control --pair lora-real=lora-nople \
    --title "LoRA row (chat template): backbone LoRA x frozen PLE" \
    --output "$OUT/arms-summary.md" --json-output "$OUT/arms-summary.json" >>"$LOG" 2>&1
  log "summary rc=$?"
fi
log "LORA CHAT DONE"
echo done > "$OUT/DONE"
