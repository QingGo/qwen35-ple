#!/usr/bin/env bash
# Round 152: LoRA row of the 0.8B x frozen-PLE 2x2 interaction test.
#
# Arms (51.2B PLE table frozen; LoRA adapters + reader trainable):
#   lora-nople   : backbone + LoRA, PLE forced off     (--gate-override 0.0)
#   lora-real    : backbone + LoRA + real PLE reader
#   lora-control : backbone + LoRA + shuffled PLE reader
#
# Same recipe as round149-fullft (500 steps, mixed50, lr 1e-4) so the only
# difference between rows is the adaptation method.  A LoRA *and* a full-FT arm
# both failing to keep open QA alive would mean the recipe is at fault.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/round152"
LOG="$ROOT/logs/round152-lora.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [round152] $* $(date -Is) ===" | tee -a "$LOG"; }

check_rows() {
  local n
  n="$(ls /dev/shm/qwen38-rows 2>/dev/null | wc -l)"
  if [[ "$n" -lt 128 ]]; then
    log "ERROR: /dev/shm/qwen38-rows has $n files; run ensure_qwen38_rows.sh first"
    exit 1
  fi
}

run_arm() {
  # run_arm <name> <mode> <gate_override_or_empty>
  local name="$1" mode="$2" gate="$3"
  local out="$OUT/$name.json"
  if [[ -f "$out" ]] && grep -q '"qa_gold"' "$out" 2>/dev/null; then
    log "skip $name (complete)"
    return 0
  fi
  local args=(
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy
    --rows-dir /dev/shm/qwen38-rows
    --model-dir "$ROOT/models/qwen38_ple"
    --model "$ROOT/models/Qwen3.5-0.8B"
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp
    --official-reader-path data/official_ple_reader.pt
    --lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.0
    --steps 500 --seq-len 128 --lr 1e-4 --seeds 0 --modes "$mode"
    --qa-sft-file data/qa-standard/train.jsonl
    --qa-sft-weight 0.5 --qa-sft-max-len 512 --qa-sft-log-every 100
    --qa-sft-lazy --qa-sft-lazy-cache 128
    --qa --qa-exact-match --qa-gold-nll --qa-max-new-tokens 32
    --qa-batch-size 8 --qa-batch-max-tokens 2048
    --qa-prompt-template "$PROMPT"
    --qa-boolq-prompt-template "$BOOLQ_PROMPT"
    --qa-file data/qa-standard/eval.jsonl
    --save-reader "$OUT/adapter-$name-seed{seed}"
  )
  if [[ -n "$gate" ]]; then args+=(--gate-override "$gate"); fi
  args+=(--output "$out")
  log "run $name mode=$mode gate=${gate:-none}"
  "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED $name rc=$rc; aborting queue"
    exit 1
  fi
  log "finished $name"
}

check_rows

# no-PLE first: if the gentle adaptation cannot keep open QA alive either, the
# recipe (not PLE) is the problem and the interaction test must not be run.
run_arm "lora-nople" "no-reader" "0.0"
run_arm "lora-real" "real" ""
run_arm "lora-control" "control" ""

ARGS=()
for name in lora-nople lora-real lora-control; do
  [[ -f "$OUT/$name.json" ]] && ARGS+=("--run" "$name=$OUT/$name.json")
done
"$PY" scripts/summarize_gold_nll.py "${ARGS[@]}" \
  --baseline lora-nople \
  --pair lora-real=lora-control \
  --output "$OUT/gold-nll-raw.md" \
  --title "LoRA row: backbone LoRA x frozen PLE (gold-answer NLL, raw)" >>"$LOG" 2>&1

for artifact in lora-nople.json lora-real.json lora-control.json gold-nll-raw.md; do
  if [[ ! -s "$OUT/$artifact" ]]; then
    log "ERROR: missing artifact $artifact"
    exit 1
  fi
done

log "ROUND 152 DONE"
echo done > "$OUT/DONE"
