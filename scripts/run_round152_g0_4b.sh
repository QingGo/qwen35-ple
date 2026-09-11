#!/usr/bin/env bash
# Round 152-G0: scale/space matching test.
#
# Qwen3.5-4B has hidden 2560, exactly the PLE source hidden (16 heads x 160),
# while the 0.8B backbone is 1024.  If the frozen graft fails only because of
# the hidden/scale mismatch, the frozen 4B backbone should show a real-vs-control
# content effect that the 0.8B row never produced.
#
# Arms (frozen backbone, 51.2B PLE table frozen, train reader only):
#   g0-no-reader : backbone alone
#   g0-real      : frozen reader (official source projections)
#   g0-control   : shuffled PLE rows
#
# Eval: standard 1500 raw prompt, generation + gold-answer NLL.
#
# The checkpoint is stored in bf16 (9.3GB for 4.2B params), so bf16 loading is
# lossless w.r.t. the released weights and 2x cheaper than the fp32 conversion
# the 0.8B path uses.  fp32 + concurrent work already OOM'd on the 24GB card.
#
# The frozen no-reader arm is NOT rerun: it does not depend on the backbone
# width and is already measured (round148a nll-raw-no-reader, mean 8.7429).
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/outputs/round152g0"
LOG="$ROOT/logs/round152-g0.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

log() { echo "=== [g0] $* $(date -Is) ===" | tee -a "$LOG"; }

check_rows() {
  local n
  n="$(ls /dev/shm/qwen38-rows 2>/dev/null | wc -l)"
  if [[ "$n" -lt 128 ]]; then
    log "ERROR: /dev/shm/qwen38-rows has $n files; run ensure_qwen38_rows.sh first"
    exit 1
  fi
}

run_arm() {
  # run_arm <name> <mode> <reader_ckpt_or_empty> [extra_flags...]
  local name="$1" mode="$2" reader="$3"
  shift 3
  local out="$OUT/$name.json"
  if [[ -f "$out" ]] && grep -q '"qa_gold"' "$out" 2>/dev/null; then
    log "skip $name (complete)"
    return 0
  fi
  local args=(
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy
    --rows-dir /dev/shm/qwen38-rows
    --model-dir "$ROOT/models/qwen38_ple"
    --model "$ROOT/models/Qwen3.5-4B"
    --backbone-dtype bfloat16
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp
    --official-reader-path data/official_ple_reader.pt
    --steps 0 --seq-len 128 --lr 1e-4 --seeds 0 --modes "$mode"
    --qa --qa-exact-match --qa-gold-nll --qa-max-new-tokens 32
    --qa-batch-size 1 --qa-batch-max-tokens 256
    --qa-prompt-template "$PROMPT"
    --qa-boolq-prompt-template "$BOOLQ_PROMPT"
    --qa-file data/qa-standard/eval.jsonl
  )
  if [[ -n "$reader" ]]; then args+=(--load-reader "$reader"); fi
  if [[ $# -gt 0 ]]; then args+=("$@"); fi
  args+=(--output "$out")
  log "run $name mode=$mode reader=${reader:-none}"
  "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED $name rc=$rc; aborting queue"
    exit 1
  fi
  log "finished $name"
}

check_rows

# The frozen 0.8B readers were trained for hidden 1024, and the official source
# reader projects into 2560, so the 4B backbone needs its own reader.  Train one
# frozen reader on the 4B backbone first (corpus + QA SFT mix, no backbone
# adaptation), then evaluate it against its own control.
TRAIN_REAL="$OUT/reader-4b-real-seed0.pt"
TRAIN_CTRL="$OUT/reader-4b-control-seed0.pt"

train_reader() {
  # train_reader <mode> <output_ckpt>
  local mode="$1" ckpt="$2"
  if [[ -f "$ckpt" ]]; then
    log "skip reader training $ckpt (exists)"
    return 0
  fi
  log "training 4B frozen reader mode=$mode"
  "$PY" -u scripts/run_phase0.py \
    --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
    --rows-dir /dev/shm/qwen38-rows \
    --model-dir "$ROOT/models/qwen38_ple" \
    --model "$ROOT/models/Qwen3.5-4B" \
    --backbone-dtype bfloat16 \
    --reader official --layer 2 --device cuda --bridge-mlp --out-mlp \
    --official-reader-path data/official_ple_reader.pt \
    --steps 500 --seq-len 128 --lr 1e-4 --seeds 0 --modes "$mode" \
    --qa-sft-file data/qa-standard/train.jsonl \
    --qa-sft-weight 0.5 --qa-sft-max-len 512 --qa-sft-log-every 100 \
    --qa-sft-lazy --qa-sft-lazy-cache 128 \
    --save-reader "$ckpt" \
    --output "$OUT/train-$mode.json" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED reader training mode=$mode rc=$rc; aborting queue"
    exit 1
  fi
}

train_reader real "$TRAIN_REAL"
train_reader control "$TRAIN_CTRL"

run_arm "g0-real" "real" "$TRAIN_REAL"
run_arm "g0-control" "control" "$TRAIN_CTRL"

# Round-152b follow-up: the real-vs-control contrast above cannot distinguish
# "real rows carry content" from "shuffled rows poison the residual stream".
# One extra arm settles it:
#   g0-nople  same trained real reader, PLE contribution suppressed (--ple-off)
#             -> isolates the PLE rows from the reader/QA-SFT machinery
# A `--modes no-reader` arm would be numerically identical (the contribution is
# zero either way), so it is deliberately not run: it would cost another ~35min
# of GPU to reproduce the same numbers.
run_arm "g0-nople" "real" "$TRAIN_REAL" --ple-off

# Reference: the frozen 0.8B row (round148a) for the same prompt protocol.
# NOTE (round-152b): the baseline run MUST be appended to the same array handed
# to summarize_gold_nll.py.  An earlier revision collected it into a separate
# `ARGS` array that was never used, so the summarizer exited with
# "baseline ... is not one of the runs", never wrote gold-nll-raw.md, and the
# artifact gate below aborted an otherwise successful G0 run (the finisher then
# idled for hours waiting on a DONE marker that could never appear).
FROZEN_REF="$ROOT/outputs/round148a/nll-raw-no-reader.json"
ALL_ARMS=(g0-real g0-control g0-nople)
# Two argument sets on purpose: the frozen 0.8B reference only has gold-NLL rows,
# so feeding it to summarize_arms.py renders a bogus all-`nan` EM row.
GEN_ARGS=()
GOLD_ARGS=()
for name in "${ALL_ARMS[@]}"; do
  if [[ -s "$OUT/$name.json" ]]; then
    GEN_ARGS+=("--run" "$name=$OUT/$name.json")
    GOLD_ARGS+=("--run" "$name=$OUT/$name.json")
  fi
done
if [[ -s "$FROZEN_REF" ]]; then
  GOLD_ARGS+=("--run" "frozen-0.8b-no-reader=$FROZEN_REF")
fi

PAIRS=("--pair" "g0-real=g0-control")
for name in g0-nople; do
  [[ -s "$OUT/$name.json" ]] && PAIRS+=("--pair" "g0-real=$name")
done

"$PY" scripts/summarize_arms.py "${GEN_ARGS[@]}" \
  "${PAIRS[@]}" \
  --title "G0 scale/space: frozen Qwen3.5-4B (hidden 2560) + frozen PLE" \
  --output "$OUT/arms-summary.md" --json-output "$OUT/arms-summary.json" >>"$LOG" 2>&1

"$PY" scripts/summarize_gold_nll.py "${GOLD_ARGS[@]}" \
  --baseline frozen-0.8b-no-reader \
  "${PAIRS[@]}" \
  --output "$OUT/gold-nll-raw.md" \
  --title "G0 scale/space: frozen Qwen3.5-4B (hidden 2560) + frozen PLE" >>"$LOG" 2>&1

for artifact in g0-real.json g0-control.json g0-nople.json gold-nll-raw.md arms-summary.md; do
  if [[ ! -s "$OUT/$artifact" ]]; then
    log "ERROR: missing artifact $artifact"
    exit 1
  fi
done

log "G0 DONE"
echo done > "$OUT/DONE"

# Hand off to the finisher, which validates every artifact before it will stop
# the instance.  Running it from here means one evaluator of the exit criteria,
# started exactly once, at the moment the work is actually finished.
if [[ "${G0_NO_FINISH:-0}" != "1" ]]; then
  bash "$ROOT/round152_finish.sh" >/dev/null 2>&1 &
  log "started finisher"
fi
