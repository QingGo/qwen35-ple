#!/usr/bin/env bash
# Next-stage queue: standard held-out QA, larger SFT data, two-stage curriculum.
#
# Starts from the overnight result that answer-only QA SFT mixed with corpus
# next-token loss fixes BoolQ and improves TriviaQA.  This queue checks whether
# that recipe generalizes to a 1500-item standard held-out set and scales to
# 6000 SFT items.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/logs/large-sft-queue.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/outputs"

log() { echo "=== [large-sft] $* $(date -Is) ===" | tee -a "$LOG"; }
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

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

# ---------------------------------------------------------------------------
# 1. Immediate generalization of the 88-item SFT readers to the 1500-item
#    standard held-out eval.  No training.
# ---------------------------------------------------------------------------
STD="$ROOT/outputs/qa-standard-eval"
mkdir -p "$STD"
run_eval_arm() {
  # run_eval_arm <name> <reader-or-empty> <modes>
  local name="$1" reader="$2" modes="$3"
  local out="$STD/$name"
  mkdir -p "$out"
  if matrix_done "$out" eval; then return 0; fi
  local load_args=()
  if [[ -n "$reader" ]]; then load_args=(--load-reader "$reader"); fi
  log "standard eval $name modes=$modes reader=$reader"
  run_logged "$ROOT/logs/qa-standard-eval-$name.log" \
    "$PY" -u scripts/run_phase0.py \
      --live-store \
      --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
      --rows-dir /dev/shm/qwen38-rows \
      --model-dir "$ROOT/models/qwen38_ple" \
      --model "$ROOT/models/Qwen3.5-0.8B" \
      --reader official --layer 2 --device cuda \
      --bridge-mlp --out-mlp \
      --official-reader-path data/official_ple_reader.pt \
      "${load_args[@]}" \
      --steps 0 --seq-len 128 --lr 1e-4 \
      --seeds 0 --modes "$modes" \
      --qa --qa-exact-match --qa-max-new-tokens 32 \
      --qa-batch-size 16 --qa-batch-max-tokens 2048 \
      --qa-prompt-template "$PROMPT" \
      --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
      --qa-file data/qa-standard/eval.jsonl \
      --resume --partial-dir "$out/partial" --backup-dir "$out/backup" \
      --output "$out/phase1-eval.json"
  if matrix_done "$out" eval; then
    run_logged "$ROOT/logs/qa-standard-eval-$name.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$out/phase1-eval.json" --protocol v2 \
        --output "$out/summary-v2.json" --markdown "$out/summary-v2.md"
    run_logged "$ROOT/logs/qa-standard-eval-$name.log" \
      "$PY" scripts/check_phase2_gates.py \
        --files "$out/phase1-eval.json" --protocol v2 --metric auto \
        --output "$out/gates-v2.json" --markdown "$out/gates-v2.md"
    touch "$out/DONE"
  fi
}

run_eval_arm "no-reader" "" "no-reader"
run_eval_arm "sft-mixed50" "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" "real"
run_eval_arm "sft-mixed50-purecode" "$ROOT/outputs/sft-mixed50-purecode/reader-PURE_CODE-real-seed0.pt" "real"
run_eval_arm "sft-mixed25" "$ROOT/outputs/sft-mixed25/reader-PURE_WIKI-real-seed0.pt" "real"

# ---------------------------------------------------------------------------
# 2. Large-data mixed50 and two-stage warmup on PURE_WIKI.
# ---------------------------------------------------------------------------
run_large_config() {
  # run_large_config <name> <weight> <warmup>
  local name="$1" weight="$2" warmup="$3"
  local out="$ROOT/outputs/$name"
  mkdir -p "$out"
  if [[ -f "$out/DONE" ]]; then log "skip $name (DONE)"; return 0; fi
  log "large SFT $name weight=$weight warmup=$warmup"
  for attempt in 1 2 3; do
    run_logged "$ROOT/logs/$name.log" env \
      OUTPUT_DIR="$out" CORPORA="PURE_WIKI" MODES="real control no-reader" \
      STEPS=500 SEEDS="0 1 2" LAYER=2 RESUME=1 \
      MAX_NEW=32 QA_BATCH_SIZE=16 QA_BATCH_MAX_TOKENS=2048 \
      QA_FILE=data/qa-standard/eval-600.jsonl \
      QA_SFT_FILE=data/qa-standard/train.jsonl \
      QA_SFT_WEIGHT="$weight" QA_SFT_MAX_LEN=512 QA_SFT_LOG_EVERY=100 \
      QA_SFT_WARMUP_STEPS="$warmup" QA_SFT_LAZY=1 QA_SFT_LAZY_CACHE=128 \
      bash scripts/run_phase2_diagnostic.sh
    if matrix_done "$out" PURE_WIKI; then break; fi
    log "large SFT $name attempt $attempt failed; retrying"
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

run_large_config "sft-large-mixed50" "0.5" "0"
run_large_config "sft-large-mixed50-warmup250" "0.5" "250"

# ---------------------------------------------------------------------------
# 3. Full 1500-item held-out eval of the best large-data readers.
# ---------------------------------------------------------------------------
FULL="$ROOT/outputs/qa-standard-eval-large-mixed50"
mkdir -p "$FULL"
if ! matrix_done "$FULL" full; then
  log "full standard eval: no-reader"
  run_logged "$ROOT/logs/qa-standard-full.log" \
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
      --qa-prompt-template "$PROMPT" \
      --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
      --qa-file data/qa-standard/eval.jsonl \
      --resume --partial-dir "$FULL/partial" --backup-dir "$FULL/backup" \
      --output "$FULL/phase1-full.json"
  for seed in 0 1 2; do
    ckpt="$ROOT/outputs/sft-large-mixed50/reader-PURE_WIKI-real-seed$seed.pt"
    if [[ -f "$ckpt" ]]; then
      log "full standard eval: real seed=$seed"
      run_logged "$ROOT/logs/qa-standard-full.log" \
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
          --qa-prompt-template "$PROMPT" \
          --qa-boolq-prompt-template "$BOOLQ_PROMPT" \
          --qa-file data/qa-standard/eval.jsonl \
          --resume --partial-dir "$FULL/partial-seed$seed" \
          --backup-dir "$FULL/backup-seed$seed" \
          --output "$FULL/phase1-real-seed$seed.json"
    fi
  done
  touch "$FULL/DONE"
fi

# ---------------------------------------------------------------------------
# 4. Oracle analysis + report.
# ---------------------------------------------------------------------------
ORACLE="$ROOT/outputs/oracle-analysis"
mkdir -p "$ORACLE"
for dir in "$STD"/* "$ROOT"/outputs/sft-large-* "$FULL"; do
  [[ -d "$dir" ]] || continue
  name="$(basename "$dir")"
  if compgen -G "$dir/phase1-*.json" >/dev/null; then
    run_logged "$ROOT/logs/large-sft-queue.log" \
      "$PY" scripts/analyze_oracle_upper_bound.py \
        --files "$dir/phase1-*.json" --protocol v2 --metric auto \
        --output "$ORACLE/$name.json" --markdown "$ORACLE/$name.md"
  fi
done

run_logged "$ROOT/logs/large-sft-queue.log" \
  "$PY" scripts/build_overnight_report.py \
    --root "$ROOT/outputs" --output "$ROOT/outputs/OVERNIGHT_REPORT.md"

log "LARGE SFT QUEUE DONE"
echo "done" > "$ROOT/outputs/LARGE_SFT_DONE"
