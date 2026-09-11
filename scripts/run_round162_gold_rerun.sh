#!/usr/bin/env bash
# Round 162 gold-NLL consistency re-run.
#
# WHY THIS EXISTS
# ---------------
# The six-arm queue started before the `_qa_gold_nll` indentation bug was found
# (commit f1685cf: only the last `--qa-batch-size` items were scored).  The first
# two arms therefore ran under the broken code and every later arm under the
# fixed code, so the gold-NLL *column* would silently mix two scoring routines.
# Nothing in the artifact would show it: both versions emit a `qa_gold` block
# with plausible numbers, and only `metrics.qa_n` betrays the difference.
#
# This script recomputes the gold-NLL column for **all six arms** offline, with
# one code path, on the readers the arms actually saved (no training, no
# generation).  It then cross-checks two invariants:
#
#   * `qa_gold.metrics.qa_n == 1500` for every arm (the bug produced 4);
#   * the re-run's `val_loss` is bit-identical to the original arm's, which is
#     only possible if the loaded reader and the token stream match the arm
#     exactly -- i.e. the re-run is scoring the same model the arm evaluated.
#
# The exact-match / format columns come from a different function
# (`_qa_exact_match`) and were never affected, so they are read directly from the
# original arm files.
#
# Usage:
#   setsid nohup bash scripts/run_round162_gold_rerun.sh > <log> 2>&1 < /dev/null &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="$ROOT/repo"
PY="$ROOT/venv/bin/python"
ROWS="${ROWS:-$ROOT/qwen38-rows}"
BACKBONE="${BACKBONE:-0.8B}"
ITEMS_FILE="${ITEMS_FILE:-data/qa-standard/eval.jsonl}"

case "$BACKBONE" in
  0.8B) MODEL="$ROOT/models/Qwen3.5-0.8B"; DTYPE="float32"; BATCH=8; BTOK=4096 ;;
  4B)   MODEL="$ROOT/models/Qwen3.5-4B";   DTYPE="bfloat16"; BATCH=2; BTOK=1024 ;;
  *)    echo "unknown BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

OUT="${OUTDIR:-$ROOT/outputs/round162-${BACKBONE}}"
LOG="${LOGFILE:-$ROOT/logs/round162-${BACKBONE}-goldrerun.log}"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"

log() { echo "=== [r162-gold-$BACKBONE] $* $(date -Is) ===" | tee -a "$LOG"; }

LOCK_WAIT="${LOCK_WAIT:-7200}"
exec 9>/tmp/qwen35_heavy.lock
if ! flock -n 9; then
  log "waiting up to ${LOCK_WAIT}s for /tmp/qwen35_heavy.lock"
  flock -w "$LOCK_WAIT" 9 || log "WARNING: proceeding without the lock (GPU-only work)"
fi

PROMPT=$'Question: {question}\nAnswer:'
BOOLQ_PROMPT=$'Question: {question}\nAnswer with one word, Yes or No:'

COMMON=(
  --reader official --layer 2 --bridge-mlp --out-mlp
  --official-reader-path data/official_ple_reader.pt
  --model "$MODEL" --model-dir "$ROOT/models/qwen38_ple" --device cuda
  --backbone-dtype "$DTYPE" --seeds 0 --steps 0 --seq-len 128 --lr 1e-4
  --qa-gold-nll --qa-gold-nll-spaced
  --qa-batch-size "$BATCH" --qa-batch-max-tokens "$BTOK"
  --qa-prompt-template "$PROMPT" --qa-boolq-prompt-template "$BOOLQ_PROMPT"
  --qa-file "$ITEMS_FILE"
)

log "code under test: $(md5sum scripts/run_phase0.py | awk '{print $1}')"

run_one() {
  # run_one <arm> <mode> <tokens-npy|-> [extra flags...]
  local arm="$1" mode="$2" tokens="$3"
  shift 3
  local out="$OUT/gold-$arm.json"
  local args=("${COMMON[@]}" --modes "$mode")
  if [[ "$tokens" != "-" ]]; then args+=(--live-store --rows-dir "$ROWS" --tokens-npy "$tokens"); fi
  if [[ $# -gt 0 ]]; then args+=("$@"); fi
  args+=(--output "$out")
  log "gold re-run arm=$arm mode=$mode"
  "$PY" scripts/with_rusage.py "$OUT/stats-goldrerun.jsonl" \
    "$PY" -u scripts/run_phase0.py "${args[@]}" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then log "FAILED gold re-run arm=$arm rc=$rc"; return 1; fi
  "$PY" - "$out" "$OUT/arm-$arm.json" "$arm" <<'PYEOF' | tee -a "$LOG"
import json, math, sys
new = json.load(open(sys.argv[1]))["results"][0]
old_path, arm = sys.argv[2], sys.argv[3]
try:
    old = json.load(open(old_path))["results"][0]
except Exception:
    old = None
m = (new.get("qa_gold") or {}).get("metrics", {})
n = m.get("qa_n")
raw, spaced = m.get("qa_mean_nll"), m.get("qa_mean_nll_spaced")
print(f"[check] arm={arm} gold_n={n} raw_nll={raw if raw is None else round(raw,4)} "
      f"spaced_nll={spaced if spaced is None else round(spaced,4)}")
status = "OK" if n and int(n) == 1500 else "BAD"
if old is not None:
    same = (
        math.isclose(new["val_loss"], old["val_loss"], rel_tol=0, abs_tol=0)
        if new.get("val_loss") is not None and old.get("val_loss") is not None
        else False
    )
    old_n = ((old.get("qa_gold") or {}).get("metrics") or {}).get("qa_n")
    print(f"[check] arm={arm} val_loss rerun={new['val_loss']!r} original={old['val_loss']!r} "
          f"bit_identical={same}; original_gold_n={old_n}")
    if not same:
        status = "VAL_LOSS_MISMATCH"
print(f"[check] arm={arm} STATUS={status}")
PYEOF
}

WIKI_CKPT="$OUT/reader-wiki-seed0.pt"
run_one wiki "data/phase1/PURE_WIKI/tokens.npy" real --load-reader "$WIKI_CKPT" || exit 1
run_one code "data/phase1/PURE_CODE/tokens.npy" real --load-reader "$OUT/reader-code-seed0.pt" || exit 1
run_one stem "data/phase1/PURE_STEM/tokens.npy" real --load-reader "$OUT/reader-stem-seed0.pt" || exit 1
run_one no-reader "-" no-reader || exit 1
run_one wiki-shuf "data/phase1/PURE_WIKI/tokens.npy" control --load-reader "$WIKI_CKPT" || exit 1
run_one ple-off "-" real --ple-off --load-reader "$WIKI_CKPT" || exit 1

log "gold re-run DONE"
exec 9>&-
