#!/usr/bin/env bash
# Overnight SFT / upper-bound queue for the pure-PLE grafting question.
#
# The layer2 corpus-only diagnostic (run_layer2_full_overnight.sh) must finish
# first.  This script then runs, in order:
#
#   1. layer2 baseline QA-only rerun on the held-out 62-item QA eval set;
#   2. oracle-context upper bound (gold answer in the prompt);
#   3. answer-only QA SFT configs (QA-only / 75% / 50% / 25% QA mix);
#   4. a mixed50 generalization check on PURE_CODE;
#   5. gate statistics on BoolQ vs TriviaQA prompts;
#   6. oracle routing analysis on every completed matrix.
#
# Everything is retried and logged; completed steps are skipped on rerun.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
L2="$ROOT/outputs/phase2-diagnostic-layer2"
LOG="$ROOT/logs/sft-overnight.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/outputs"
cd "$REPO"

log() { echo "=== [sft-overnight] $* $(date -Is) ===" | tee -a "$LOG"; }
run_logged() {
  # run_logged <logfile> <command...>
  local logfile="$1"; shift
  echo "--- [$(date -Is)] $*" >>"$logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  echo "--- [$(date -Is)] rc=$rc" >>"$logfile"
  return $rc
}
matrix_done() {
  # matrix_done <dir> <corpus>
  local dir="$1" corpus="$2"
  [[ -f "$dir/phase1-$corpus.json" ]] && grep -q '"summary"' "$dir/phase1-$corpus.json"
}

# ---------------------------------------------------------------------------
# 0. Wait for the layer2 corpus-only diagnostic.
# ---------------------------------------------------------------------------
if [[ "${WAIT_L2:-1}" == "1" ]]; then
  for i in $(seq 1 480); do
    if [[ -f "$L2/DONE" ]]; then log "layer2 diagnostic DONE detected"; break; fi
    if [[ -f "$L2/FAILED" ]]; then log "layer2 diagnostic FAILED; continuing with available corpora"; break; fi
    if (( i % 10 == 1 )); then log "waiting for layer2 diagnostic (iteration $i)"; fi
    sleep 60
  done
fi

# ---------------------------------------------------------------------------
# 1. Baseline QA-only rerun on the held-out 62-item eval set.
# ---------------------------------------------------------------------------
BASE="$ROOT/outputs/baseline-layer2-eval62"
mkdir -p "$BASE"
if ! matrix_done "$BASE" PURE_WIKI; then
  log "baseline eval62: loading layer2 readers and evaluating on qa.eval.jsonl"
  for attempt in 1 2 3; do
    run_logged "$ROOT/logs/baseline-eval62.log" \
      "$PY" -u scripts/run_phase0.py \
        --live-store \
        --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
        --rows-dir /dev/shm/qwen38-rows \
        --model-dir "$ROOT/models/qwen38_ple" \
        --model "$ROOT/models/Qwen3.5-0.8B" \
        --reader official --layer 2 --device cuda \
        --bridge-mlp --out-mlp \
        --official-reader-path data/official_ple_reader.pt \
        --load-reader "$L2/reader-PURE_WIKI-{mode}-seed{seed}.pt" \
        --steps 0 --seq-len 128 --lr 1e-4 \
        --seeds 0 1 2 --modes real control no-reader \
        --qa --qa-exact-match --qa-max-new-tokens 96 \
        --qa-batch-size 16 --qa-batch-max-tokens 2048 \
        --qa-prompt-template $'Question: {question}\nAnswer:' \
        --qa-boolq-prompt-template $'Question: {question}\nAnswer with one word, Yes or No:' \
        --qa-file data/phase1/kb-wiki/qa.eval.jsonl \
        --resume \
        --partial-dir "$BASE/partial" --backup-dir "$BASE/backup" \
        --output "$BASE/phase1-PURE_WIKI.json"
    if matrix_done "$BASE" PURE_WIKI; then break; fi
    log "baseline eval62 attempt $attempt failed; retrying with --resume"
    sleep 20
  done
fi
if matrix_done "$BASE" PURE_WIKI; then
  run_logged "$ROOT/logs/baseline-eval62.log" \
    "$PY" scripts/summarize_phase1_matrix.py \
      --files "$BASE/phase1-PURE_WIKI.json" --protocol v2 \
      --output "$BASE/summary-v2.json" --markdown "$BASE/summary-v2.md"
  run_logged "$ROOT/logs/baseline-eval62.log" \
    "$PY" scripts/check_phase2_gates.py \
      --files "$BASE/phase1-PURE_WIKI.json" --protocol v2 --metric auto \
      --output "$BASE/gates-v2.json" --markdown "$BASE/gates-v2.md"
  touch "$BASE/DONE"
fi

# ---------------------------------------------------------------------------
# 2. Oracle-context upper bound.
# ---------------------------------------------------------------------------
CTX="$ROOT/outputs/oracle-context"
mkdir -p "$CTX"
run_logged "$ROOT/logs/oracle-context.log" \
  "$PY" scripts/make_oracle_context_qa.py \
    --input data/phase1/kb-wiki/qa.eval.jsonl \
    --output data/qa-oracle-answer-eval62.json --mode answer
run_logged "$ROOT/logs/oracle-context.log" \
  "$PY" scripts/make_oracle_context_qa.py \
    --input data/qa-expanded-150.json \
    --output data/qa-oracle-answer-150.json --mode answer
for spec in "eval62:data/phase1/kb-wiki/qa.eval.jsonl:data/qa-oracle-answer-eval62.json" \
            "150:data/qa-expanded-150.json:data/qa-oracle-answer-150.json"; do
  IFS=: read -r name plain oracle <<<"$spec"
  out="$CTX/$name"
  mkdir -p "$out"
  if ! matrix_done "$out" PURE_WIKI; then
    log "oracle context $name"
    run_logged "$ROOT/logs/oracle-context.log" \
      "$PY" -u scripts/run_phase0.py \
        --live-store \
        --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
        --rows-dir /dev/shm/qwen38-rows \
        --model-dir "$ROOT/models/qwen38_ple" \
        --model "$ROOT/models/Qwen3.5-0.8B" \
        --reader official --layer 2 --device cuda \
        --bridge-mlp --out-mlp \
        --official-reader-path data/official_ple_reader.pt \
        --load-reader "$L2/reader-PURE_WIKI-{mode}-seed{seed}.pt" \
        --steps 0 --seq-len 128 --lr 1e-4 \
        --seeds 0 --modes real no-reader \
        --qa --qa-exact-match --qa-max-new-tokens 96 \
        --qa-batch-size 16 --qa-batch-max-tokens 2048 \
        --qa-prompt-template $'Question: {question}\nAnswer:' \
        --qa-boolq-prompt-template $'Question: {question}\nAnswer with one word, Yes or No:' \
        --qa-file "$oracle" \
        --resume \
        --partial-dir "$out/partial" --backup-dir "$out/backup" \
        --output "$out/phase1-PURE_WIKI.json"
  fi
  if matrix_done "$out" PURE_WIKI; then
    run_logged "$ROOT/logs/oracle-context.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$out/phase1-PURE_WIKI.json" --protocol v2 \
        --output "$out/summary-v2.json" --markdown "$out/summary-v2.md"
    touch "$out/DONE"
  fi
done

# ---------------------------------------------------------------------------
# 3. Answer-only QA SFT configs.
# ---------------------------------------------------------------------------
# name:qa_sft_weight:corpora:gate_reg_weight
CONFIGS=(
  "sft-only:1.0:PURE_WIKI:0"
  "sft-mixed75:0.75:PURE_WIKI:0"
  "sft-mixed50:0.5:PURE_WIKI:0"
  "sft-mixed25:0.25:PURE_WIKI:0"
  "sft-mixed50-gate01:0.5:PURE_WIKI:0.01"
)
for spec in "${CONFIGS[@]}"; do
  IFS=: read -r name weight corpora gate_reg <<<"$spec"
  gate_reg="${gate_reg:-0}"
  out="$ROOT/outputs/$name"
  mkdir -p "$out"
  if [[ -f "$out/DONE" ]]; then log "skip $name (DONE)"; continue; fi
  log "SFT config $name weight=$weight corpora=$corpora"
  for attempt in 1 2 3; do
    run_logged "$ROOT/logs/$name.log" env \
      OUTPUT_DIR="$out" CORPORA="$corpora" MODES="real control no-reader" \
      STEPS=500 SEEDS="0 1 2" LAYER=2 RESUME=1 \
      QA_FILE=data/phase1/kb-wiki/qa.eval.jsonl \
      QA_SFT_FILE=data/phase1/kb-wiki/qa.train.jsonl \
      QA_SFT_WEIGHT="$weight" QA_SFT_MAX_LEN=512 QA_SFT_LOG_EVERY=100 \
      GATE_REG_WEIGHT="$gate_reg" \
      bash scripts/run_phase2_diagnostic.sh
    if matrix_done "$out" "${corpora%% *}"; then break; fi
    log "SFT $name attempt $attempt failed; retrying"
    sleep 20
  done
  first_corpus="${corpora%% *}"
  if matrix_done "$out" "$first_corpus"; then
    run_logged "$ROOT/logs/$name.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$out"/phase1-*.json --protocol v2 \
        --output "$out/summary-v2.json" --markdown "$out/summary-v2.md"
    run_logged "$ROOT/logs/$name.log" \
      "$PY" scripts/check_phase2_gates.py \
        --files "$out"/phase1-*.json --protocol v2 --metric auto \
        --output "$out/gates-v2.json" --markdown "$out/gates-v2.md"
    touch "$out/DONE"
  fi
done

# ---------------------------------------------------------------------------
# 4. Generalization check: mixed50 on PURE_CODE.
# ---------------------------------------------------------------------------
GEN="$ROOT/outputs/sft-mixed50-purecode"
mkdir -p "$GEN"
if [[ ! -f "$GEN/DONE" ]]; then
  log "generalization check: mixed50 on PURE_CODE"
  for attempt in 1 2 3; do
    run_logged "$ROOT/logs/sft-mixed50-purecode.log" env \
      OUTPUT_DIR="$GEN" CORPORA="PURE_CODE" MODES="real control no-reader" \
      STEPS=500 SEEDS="0 1 2" LAYER=2 RESUME=1 \
      QA_FILE=data/phase1/kb-wiki/qa.eval.jsonl \
      QA_SFT_FILE=data/phase1/kb-wiki/qa.train.jsonl \
      QA_SFT_WEIGHT=0.5 QA_SFT_MAX_LEN=512 QA_SFT_LOG_EVERY=100 \
      bash scripts/run_phase2_diagnostic.sh
    if matrix_done "$GEN" PURE_CODE; then break; fi
    log "mixed50 PURE_CODE attempt $attempt failed; retrying"
    sleep 20
  done
  if matrix_done "$GEN" PURE_CODE; then
    run_logged "$ROOT/logs/sft-mixed50-purecode.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$GEN/phase1-PURE_CODE.json" --protocol v2 \
        --output "$GEN/summary-v2.json" --markdown "$GEN/summary-v2.md"
    touch "$GEN/DONE"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Gate statistics on BoolQ vs TriviaQA prompts.
# ---------------------------------------------------------------------------
GATE="$ROOT/outputs/gate-stats"
mkdir -p "$GATE"
gate_one() {
  # gate_one <name> <reader-checkpoint> <qa-file>
  local name="$1" ckpt="$2" qa="$3"
  if [[ -f "$GATE/$name.json" ]]; then return 0; fi
  run_logged "$ROOT/logs/gate-stats.log" \
    "$PY" scripts/analyze_reader_gate.py \
      --model "$ROOT/models/Qwen3.5-0.8B" \
      --reader-checkpoint "$ckpt" \
      --rows-dir /dev/shm/qwen38-rows \
      --model-dir "$ROOT/models/qwen38_ple" \
      --layer 2 --qa-file "$qa" --max-tokens 512 \
      --output "$GATE/$name.json" --markdown "$GATE/$name.md"
}
gate_one "layer2-real-seed0" "$L2/reader-PURE_WIKI-real-seed0.pt" data/qa-expanded-150.json
gate_one "layer2-control-seed0" "$L2/reader-PURE_WIKI-control-seed0.pt" data/qa-expanded-150.json
if [[ -f "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" ]]; then
  gate_one "sft-mixed50-real-seed0" "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" data/phase1/kb-wiki/qa.eval.jsonl
fi
if [[ -f "$ROOT/outputs/sft-mixed50-gate01/reader-PURE_WIKI-real-seed0.pt" ]]; then
  gate_one "sft-mixed50-gate01-real-seed0" "$ROOT/outputs/sft-mixed50-gate01/reader-PURE_WIKI-real-seed0.pt" data/phase1/kb-wiki/qa.eval.jsonl
fi

# ---------------------------------------------------------------------------
# 6. Oracle routing analysis on every completed matrix.
# ---------------------------------------------------------------------------
ORACLE="$ROOT/outputs/oracle-analysis"
mkdir -p "$ORACLE"
oracle_one() {
  # oracle_one <name> <glob>
  local name="$1" pattern="$2"
  run_logged "$ROOT/logs/oracle-analysis.log" \
    "$PY" scripts/analyze_oracle_upper_bound.py \
      --files $pattern --protocol v2 --metric auto \
      --output "$ORACLE/$name.json" --markdown "$ORACLE/$name.md"
}
if compgen -G "$L2/phase1-*.json" >/dev/null; then
  oracle_one "layer2-corpus-only" "$L2/phase1-*.json"
fi
if compgen -G "$BASE/phase1-*.json" >/dev/null; then
  oracle_one "layer2-baseline-eval62" "$BASE/phase1-*.json"
fi
for name in sft-only sft-mixed75 sft-mixed50 sft-mixed25 sft-mixed50-gate01 sft-mixed50-purecode; do
  if compgen -G "$ROOT/outputs/$name/phase1-*.json" >/dev/null; then
    oracle_one "$name" "$ROOT/outputs/$name/phase1-*.json"
  fi
done

run_logged "$ROOT/logs/overnight-report.log" \
  "$PY" scripts/build_overnight_report.py \
    --root "$ROOT/outputs" --output "$ROOT/outputs/OVERNIGHT_REPORT.md"

log "ALL QUEUE STEPS DONE"
echo "done" > "$ROOT/outputs/SFT_OVERNIGHT_DONE"
