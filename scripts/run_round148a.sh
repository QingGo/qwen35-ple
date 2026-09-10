#!/usr/bin/env bash
# Round 148-A: format-insensitive gold-answer NLL, head/order ablation, and the
# oracle-routing probe.
#
# Deliverables (under $ROOT/outputs/round148a):
#   gold-nll-raw.md / gold-nll-chat.md   paired NLL tables
#   features-raw.npz / features-chat.npz layer-2 hidden states
#   probe-raw.{json,md} / probe-chat.{json,md}
#
# This is a diagnostic round: it does not train any reader.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/round148a"
LOG="$ROOT/logs/round148a.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [round148a] $* $(date -Is) ===" | tee -a "$LOG"; }

check_rows() {
  local n
  n="$(ls /dev/shm/qwen38-rows 2>/dev/null | wc -l)"
  if [[ "$n" -lt 128 ]]; then
    log "ERROR: /dev/shm/qwen38-rows has $n files; run ensure_qwen38_rows.sh first"
    exit 1
  fi
}

run_nll() {
  # run_nll <name> <mode> <reader-or-empty> <chat:0|1> <mask-or-empty> <seed>
  local name="$1" mode="$2" reader="$3" chat="$4" mask="$5" seed="$6"
  local out="$OUT/$name.json"
  if [[ -f "$out" ]] && grep -q '"qa_gold"' "$out" 2>/dev/null; then
    log "skip $name"
    return 0
  fi
  local args=(
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy
    --rows-dir /dev/shm/qwen38-rows
    --model-dir "$ROOT/models/qwen38_ple"
    --model "$ROOT/models/Qwen3.5-0.8B"
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp
    --official-reader-path data/official_ple_reader.pt
    --steps 0 --seq-len 128 --lr 1e-4 --seeds "$seed" --modes "$mode"
    --qa-gold-nll --qa-file data/qa-standard/eval.jsonl
    --qa-prompt-template "$PROMPT"
    --qa-boolq-prompt-template "$BOOLQ_PROMPT"
  )
  if [[ "$chat" == "1" ]]; then args+=(--qa-chat-template); fi
  if [[ -n "$reader" ]]; then args+=(--load-reader "$reader"); fi
  if [[ -n "$mask" ]]; then args+=(--qa-head-mask "$mask"); fi
  args+=(--output "$out")
  log "run $name mode=$mode chat=$chat mask=$mask seed=$seed"
  "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED $name rc=$rc; aborting queue"
    exit 1
  fi
}

READ_RAW="$ROOT/outputs/sft-large-mixed50/reader-PURE_WIKI-real-seed"
CTRL_RAW="$ROOT/outputs/sft-large-mixed50/reader-PURE_WIKI-control-seed"
READ_CHAT="$ROOT/outputs/sft-chat-mixed50/reader-PURE_WIKI-real-seed"
CTRL_CHAT="$ROOT/outputs/sft-chat-mixed50/reader-PURE_WIKI-control-seed"

check_rows

# ---------------------------------------------------------------- raw prompt
run_nll "nll-raw-no-reader" "no-reader" "" "0" "" "0"
for seed in 0 1 2; do
  [[ -f "${READ_RAW}${seed}.pt" ]] && run_nll "nll-raw-real-seed${seed}" "real" "${READ_RAW}${seed}.pt" "0" "" "$seed"
  [[ -f "${CTRL_RAW}${seed}.pt" ]] && run_nll "nll-raw-control-seed${seed}" "control" "${CTRL_RAW}${seed}.pt" "0" "" "$seed"
done
if [[ -f "${READ_RAW}0.pt" ]]; then
  run_nll "nll-raw-real-seed0-keep2gram" "real" "${READ_RAW}0.pt" "0" "2gram" "0"
  run_nll "nll-raw-real-seed0-keep3gram" "real" "${READ_RAW}0.pt" "0" "3gram" "0"
fi

# --------------------------------------------------------------- chat prompt
run_nll "nll-chat-no-reader" "no-reader" "" "1" "" "0"
for seed in 0 1 2; do
  [[ -f "${READ_CHAT}${seed}.pt" ]] && run_nll "nll-chat-real-seed${seed}" "real" "${READ_CHAT}${seed}.pt" "1" "" "$seed"
  [[ -f "${CTRL_CHAT}${seed}.pt" ]] && run_nll "nll-chat-control-seed${seed}" "control" "${CTRL_CHAT}${seed}.pt" "1" "" "$seed"
done
if [[ -f "${READ_CHAT}0.pt" ]]; then
  run_nll "nll-chat-real-seed0-keep2gram" "real" "${READ_CHAT}0.pt" "1" "2gram" "0"
  run_nll "nll-chat-real-seed0-keep3gram" "real" "${READ_CHAT}0.pt" "1" "3gram" "0"
fi

# --------------------------------------------------------------- NLL report
RAW_RUNS=("nll-raw-no-reader=$OUT/nll-raw-no-reader.json")
CHAT_RUNS=("nll-chat-no-reader=$OUT/nll-chat-no-reader.json")
for seed in 0 1 2; do
  [[ -f "$OUT/nll-raw-real-seed${seed}.json" ]] && RAW_RUNS+=("nll-raw-real-seed${seed}=$OUT/nll-raw-real-seed${seed}.json")
  [[ -f "$OUT/nll-raw-control-seed${seed}.json" ]] && RAW_RUNS+=("nll-raw-control-seed${seed}=$OUT/nll-raw-control-seed${seed}.json")
  [[ -f "$OUT/nll-chat-real-seed${seed}.json" ]] && CHAT_RUNS+=("nll-chat-real-seed${seed}=$OUT/nll-chat-real-seed${seed}.json")
  [[ -f "$OUT/nll-chat-control-seed${seed}.json" ]] && CHAT_RUNS+=("nll-chat-control-seed${seed}=$OUT/nll-chat-control-seed${seed}.json")
done
[[ -f "$OUT/nll-raw-real-seed0-keep2gram.json" ]] && RAW_RUNS+=("nll-raw-real-seed0-keep2gram=$OUT/nll-raw-real-seed0-keep2gram.json")
[[ -f "$OUT/nll-raw-real-seed0-keep3gram.json" ]] && RAW_RUNS+=("nll-raw-real-seed0-keep3gram=$OUT/nll-raw-real-seed0-keep3gram.json")
[[ -f "$OUT/nll-chat-real-seed0-keep2gram.json" ]] && CHAT_RUNS+=("nll-chat-real-seed0-keep2gram=$OUT/nll-chat-real-seed0-keep2gram.json")
[[ -f "$OUT/nll-chat-real-seed0-keep3gram.json" ]] && CHAT_RUNS+=("nll-chat-real-seed0-keep3gram=$OUT/nll-chat-real-seed0-keep3gram.json")

raw_args=(); for spec in "${RAW_RUNS[@]}"; do raw_args+=(--run "$spec"); done
chat_args=(); for spec in "${CHAT_RUNS[@]}"; do chat_args+=(--run "$spec"); done
"$PY" scripts/summarize_gold_nll.py "${raw_args[@]}" --baseline nll-raw-no-reader \
  --pair nll-raw-real-seed0=nll-raw-control-seed0 \
  --pair nll-raw-real-seed1=nll-raw-control-seed1 \
  --pair nll-raw-real-seed2=nll-raw-control-seed2 \
  --pair nll-raw-real-seed0-keep2gram=nll-raw-real-seed0 \
  --pair nll-raw-real-seed0-keep3gram=nll-raw-real-seed0 \
  --output "$OUT/gold-nll-raw.md" --title "Gold-answer NLL: raw prompt (format-insensitive)" >>"$LOG" 2>&1
"$PY" scripts/summarize_gold_nll.py "${chat_args[@]}" --baseline nll-chat-no-reader \
  --pair nll-chat-real-seed0=nll-chat-control-seed0 \
  --pair nll-chat-real-seed1=nll-chat-control-seed1 \
  --pair nll-chat-real-seed2=nll-chat-control-seed2 \
  --pair nll-chat-real-seed0-keep2gram=nll-chat-real-seed0 \
  --pair nll-chat-real-seed0-keep3gram=nll-chat-real-seed0 \
  --output "$OUT/gold-nll-chat.md" --title "Gold-answer NLL: chat template (format-insensitive)" >>"$LOG" 2>&1

# ------------------------------------------------------- hidden-state features
"$PY" scripts/extract_layer_hidden.py \
  --model "$ROOT/models/Qwen3.5-0.8B" --device cuda --layer 2 \
  --qa-file data/qa-standard/eval.jsonl \
  --prompt-template "$PROMPT" --boolq-prompt-template "$BOOLQ_PROMPT" \
  --output "$OUT/features-raw.npz" >>"$LOG" 2>&1
"$PY" scripts/extract_layer_hidden.py \
  --model "$ROOT/models/Qwen3.5-0.8B" --device cuda --layer 2 \
  --qa-file data/qa-standard/eval.jsonl --chat-template \
  --output "$OUT/features-chat.npz" >>"$LOG" 2>&1

# ------------------------------------------------------ oracle-routing probe
"$PY" scripts/train_oracle_routing_probe.py \
  --features "$OUT/features-raw.npz" --protocol raw \
  --no-reader "$ROOT/outputs/qa-standard-eval-large-mixed50/phase1-full.json" \
  --real "$ROOT/outputs/qa-standard-eval-large-mixed50/phase1-real-seed0.json" \
  --real "$ROOT/outputs/qa-standard-eval-large-mixed50/phase1-real-seed1.json" \
  --real "$ROOT/outputs/qa-standard-eval-large-mixed50/phase1-real-seed2.json" \
  --output-json "$OUT/probe-raw.json" --output-md "$OUT/probe-raw.md" >>"$LOG" 2>&1
"$PY" scripts/train_oracle_routing_probe.py \
  --features "$OUT/features-chat.npz" --protocol chat \
  --no-reader "$ROOT/outputs/qa-standard-eval-chat-large/phase1-full.json" \
  --real "$ROOT/outputs/qa-standard-eval-chat-large/phase1-real-seed0.json" \
  --real "$ROOT/outputs/qa-standard-eval-chat-large/phase1-real-seed1.json" \
  --real "$ROOT/outputs/qa-standard-eval-chat-large/phase1-real-seed2.json" \
  --output-json "$OUT/probe-chat.json" --output-md "$OUT/probe-chat.md" >>"$LOG" 2>&1

for artifact in gold-nll-raw.md gold-nll-chat.md features-raw.npz features-chat.npz probe-raw.json probe-chat.json; do
  if [[ ! -s "$OUT/$artifact" ]]; then
    log "ERROR: missing artifact $artifact"
    exit 1
  fi
done

log "ROUND 148-A DONE"
echo done > "$OUT/DONE"
