#!/usr/bin/env bash
# Causal gate ablation: force the PLE reader gate to 0 / 0.5 / 1 and re-run QA.
#
# This isolates whether the BoolQ regression is caused by the learned gate
# opening, independently of the SFT results.  It waits for the main overnight
# queue to finish, then runs the ablation on the layer2 real seed0 reader and
# (if present) the best SFT reader.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
L2="$ROOT/outputs/phase2-diagnostic-layer2"
LOG="$ROOT/logs/gate-ablation.log"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/outputs"
cd "$REPO"

log() { echo "=== [gate-ablation] $* $(date -Is) ===" | tee -a "$LOG"; }
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

# Wait for the main queue (up to ~10h).
for i in $(seq 1 600); do
  if [[ -f "$ROOT/outputs/SFT_OVERNIGHT_DONE" ]]; then log "main queue DONE"; break; fi
  if (( i % 10 == 1 )); then log "waiting for main queue (iteration $i)"; fi
  sleep 60
done

run_ablation() {
  # run_ablation <name> <reader-checkpoint-or-empty> <override>
  local name="$1" ckpt="$2" override="$3"
  local out="$ROOT/outputs/gate-ablation-$name"
  mkdir -p "$out"
  if matrix_done "$out" PURE_WIKI; then log "skip $name (done)"; return 0; fi
  log "gate ablation $name override=$override reader=$ckpt"
  local load_args=()
  if [[ -n "$ckpt" ]]; then
    load_args=(--load-reader "$ckpt")
  fi
  run_logged "$ROOT/logs/gate-ablation-$name.log" \
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
      --gate-override "$override" \
      --steps 0 --seq-len 128 --lr 1e-4 \
      --seeds 0 --modes real \
      --qa --qa-exact-match --qa-max-new-tokens 96 \
      --qa-batch-size 16 --qa-batch-max-tokens 2048 \
      --qa-prompt-template $'Question: {question}\nAnswer:' \
      --qa-boolq-prompt-template $'Question: {question}\nAnswer with one word, Yes or No:' \
      --qa-file data/qa-expanded-150.json \
      --resume --partial-dir "$out/partial" --backup-dir "$out/backup" \
      --output "$out/phase1-PURE_WIKI.json"
  if matrix_done "$out" PURE_WIKI; then
    run_logged "$ROOT/logs/gate-ablation-$name.log" \
      "$PY" scripts/summarize_phase1_matrix.py \
        --files "$out/phase1-PURE_WIKI.json" --protocol v2 \
        --output "$out/summary-v2.json" --markdown "$out/summary-v2.md"
    run_logged "$ROOT/logs/gate-ablation-$name.log" \
      "$PY" scripts/check_phase2_gates.py \
        --files "$out/phase1-PURE_WIKI.json" --protocol v2 --metric auto \
        --output "$out/gates-v2.json" --markdown "$out/gates-v2.md"
    touch "$out/DONE"
  fi
}

for override in 0.0 0.5 1.0; do
  run_ablation "layer2-real-${override}" "$L2/reader-PURE_WIKI-real-seed0.pt" "$override"
done
if [[ -f "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" ]]; then
  for override in 0.0 0.5 1.0; do
    run_ablation "sft-mixed50-real-${override}" \
      "$ROOT/outputs/sft-mixed50/reader-PURE_WIKI-real-seed0.pt" "$override"
  done
fi

# Oracle analysis for the ablation outputs.
for dir in "$ROOT"/outputs/gate-ablation-*; do
  [[ -d "$dir" ]] || continue
  name="$(basename "$dir")"
  if compgen -G "$dir/phase1-*.json" >/dev/null; then
    run_logged "$ROOT/logs/gate-ablation.log" \
      "$PY" scripts/analyze_oracle_upper_bound.py \
        --files "$dir/phase1-*.json" --protocol v2 --metric auto \
        --output "$ROOT/outputs/oracle-analysis/$name.json" \
        --markdown "$ROOT/outputs/oracle-analysis/$name.md"
  fi
done

run_logged "$ROOT/logs/gate-ablation.log" \
  "$PY" scripts/build_overnight_report.py \
    --root "$ROOT/outputs" --output "$ROOT/outputs/OVERNIGHT_REPORT.md"
log "GATE ABLATION DONE"
echo "done" > "$ROOT/outputs/GATE_ABLATION_DONE"
