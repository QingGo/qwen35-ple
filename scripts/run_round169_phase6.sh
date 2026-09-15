#!/usr/bin/env bash
# Round 169 phase 6 -- the top-1 gate, then the soft-shrinkage A/B.
#
# TWO EXPERIMENTS, IN THE ORDER THEIR FALSIFICATION POWER DEMANDS.
#
# 1. THE GATE.  Every number this project has is mean NLL, a log quantity.  The
#    injection's gain is +1.925 nats per position at backbone surprisal >= 8 --
#    a 6.9x probability ratio -- but at 8 nats the base probability is 3.4e-4, so
#    the realised token is the wrong argmax either way.  A 6.9x improvement on a
#    0.03% probability is a log-loss improvement and may be nothing else.  The
#    eval now records per-position top-1 for all three arms, so one eval settles
#    whether the gain reaches a decision.  The verdict rule is fixed in
#    scripts/round169_top1_verdict.py before this runs: GAIN iff the paired
#    frozen-minus-none top-1 delta on surprisal >= 2 is positive with t >= 3,
#    FLAT otherwise.  A FLAT verdict stops the NLL-optimisation line.
#
# 2. THE A/B THE DERIVATION PREDICTS.  docs/round-170-embedding-optimizer-
#    derivation.md derives the optimal per-row step weight as c/(c+k) from a
#    James-Stein posterior mean, and the hard mask already running is its k -> 0
#    limit.  The derivation says the smooth form should beat the hard one at the
#    same threshold.  So: one arm, identical in every respect to the running
#    --min-train-count 10 arm except that the low-count rows are shrunk instead
#    of frozen.  If soft does not beat hard, the isotropy assumption in the
#    derivation is doing more work than the data supports.
#
# Guards are the same four as every phase before this one.
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/qwen35-ple}
REPO=${REPO:-$ROOT/repo}
PY=${PY:-$ROOT/venv/bin/python}
LOGD=${LOGD:-$ROOT/logs}
OUT=$REPO/outputs/round169
ROWS=$ROOT/qwen38-rows
MARGIN=$ROOT/outputs/round168/margin
MODEL=$ROOT/models/Qwen3.5-0.8B
READER=$ROOT/outputs/round162-0.8B-nosft/reader-wiki-seed0.pt
MEM_CAP_GB=${MEM_CAP_GB:-85}
LR=1e-4
EPOCHS=4
K_SOFT=10
TAU=2.0
T_MIN=3.0

DONE_MARK=$OUT/PHASE6_DONE
FAIL_MARK=$OUT/PHASE6_FAILED
START_MARK=$OUT/PHASE6_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase6-queue.log}
FAILFILE=$OUT/.phase6-failures
STEPSFILE=$OUT/.phase6-steps
PHASE5_DONE=$OUT/PHASE5_DONE

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
is_number() { awk -v a="$1" 'BEGIN{exit !(a ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/)}'; }
better_than() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 > b+0)}'; }
delta_of() {
  [ -s "$1" ] || return 1
  "$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["delta"]["mean"])))' "$1" 2>/dev/null
}

wait_mem() {
  local cap=${1:-$MEM_CAP_GB} n=0 cur
  while :; do
    cur=$(mem_gb); cur=${cur:-0}
    [ "$cur" -le "$cap" ] && return 0
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && say "  memory guard: ${cur} GB > ${cap} GB, waiting"
    sleep 30
  done
}

wait_quiet() {
  local n=0 busy
  while :; do
    busy=$(pgrep -cf '[r]ound169_(train|eval)_rows' || true); busy=${busy:-0}
    [ "$busy" -eq 0 ] && return 0
    n=$((n + 1)); [ $((n % 5)) -eq 1 ] && say "  waiting for $busy in-flight round169 job(s)"
    sleep 30
  done
}

run_gpu() {  # name cmd...
  local name=$1; shift
  local log=$LOGD/phase6-$name.log t0 t1 rc
  wait_mem; wait_quiet
  say "START $name  (mem $(mem_gb) GB, gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')%)"
  t0=$(date +%s)
  { echo "=== $name start $(date -Is) ==="; "$@"; rc=$?; echo "=== $name rc=$rc ==="; } > "$log" 2>&1
  t1=$(date +%s)
  say "END   $name rc=$rc wall=$((t1 - t0))s -> $log"
  step "$name rc=$rc wall=$((t1 - t0))s"
  [ "$rc" -ne 0 ] && fail "$name"
  return 0
}

eval_bank() {  # bank out
  "$PY" scripts/round169_eval_rows.py \
    --snapshot-dir "$OUT" --trained-bank "$1" \
    --keep-index "$OUT/keep-wiki-eval.npy" \
    --tokens data/phase1/wikitext-heldout-decon/tokens.npy \
    --positions outputs/round168/rung3/positions-wiki.npy \
    --counts-npz "$MARGIN/wiki-counts.npz" \
    --rows-dir "$ROWS" --model "$MODEL" \
    --reader-kind saved --reader "$READER" --layer 2 --chunk 1024 --out "$2"
}

train_soft() {  # min_count power out
  "$PY" scripts/round169_train_rows.py \
    --snapshot-dir "$OUT" --train-tokens data/phase1/PURE_WIKI/tokens.npy \
    --model "$MODEL" --reader-kind saved --reader "$READER" \
    --keep-index "$OUT/keep-wiki-eval.npy" --arm real \
    --epochs "$EPOCHS" --lr "$LR" --min-train-count "$1" --row-scale-power "$2" --out "$3"
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); gate tau=$TAU t_min=$T_MIN; soft k=$K_SOFT" > "$START_MARK"
say "=== phase 6 (top-1 gate, then soft shrinkage) starting ==="

# ---- wait for arm A ---------------------------------------------------------
n=0
while [ ! -s "$PHASE5_DONE" ]; do
  n=$((n + 1))
  if [ "$n" -gt 4 ] && ! pgrep -f '[r]un_round169_phase5.sh' > /dev/null 2>&1; then
    say "phase 5 is gone without writing its marker; continuing"
    step "phase5-marker-missing"
    break
  fi
  [ $((n % 20)) -eq 1 ] && say "  waiting for PHASE5_DONE (in-flight: $(pgrep -cf '[r]ound169_(train|eval)_rows' || echo 0))"
  sleep 30
done
wait_quiet
say "phase 5 finished; $(tail -1 "$LOGD/phase5-queue.log" 2>/dev/null)"

# ---- 1. THE GATE ------------------------------------------------------------
# One eval on the reference bank.  It re-derives delta (must reproduce -0.00272)
# AND writes per-position top-1 for all three arms, which is what the gate reads.
REF_BANK=$OUT/E-real-e4-lr1e-4.npy
GATE_JSON=$OUT/eval-real-lr1e-4-top1.json
if [ ! -s "$REF_BANK" ]; then
  say "FAILED: the reference bank is absent; the gate cannot run"
  fail "no-reference-bank"
else
  if [ -s "$GATE_JSON" ] && [ -s "${GATE_JSON%.json}.deltas.npz" ]; then
    say "SKIP top1-gate eval (already present)"
  else
    run_gpu top1-gate eval_bank "$REF_BANK" "$GATE_JSON"
  fi
  REC="${GATE_JSON%.json}.deltas.npz"
  if [ -s "$REC" ]; then
    if "$PY" scripts/round169_top1_verdict.py --record "$REC" --tau "$TAU" --t-min "$T_MIN" \
         --out-json "$OUT/gate-top1.json" --out-md "$OUT/PHASE6_GATE.md" \
         > "$LOGD/phase6-gate.log" 2>&1; then
      VERDICT=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["verdict"])' \
                  "$OUT/gate-top1.json" 2>/dev/null)
      say "*** TOP-1 GATE VERDICT: ${VERDICT:-UNKNOWN} (tau=$TAU, t_min=$T_MIN) ***"
      step "top1-gate verdict=${VERDICT:-UNKNOWN}"
      grep -E '^\| (frozen|trained) ' "$OUT/PHASE6_GATE.md" 2>/dev/null | while read -r l; do say "  $l"; done
    else
      say "gate tool failed; see $LOGD/phase6-gate.log"
      fail "gate-tool"
    fi
  else
    say "no per-position record from the gate eval; the gate cannot be read"
    fail "gate-record-missing"
  fi
fi

# ---- 2. SOFT SHRINKAGE AT THE SAME THRESHOLD --------------------------------
# Identical to arm A (--min-train-count 10) except that low-count rows are shrunk
# by c/(c+k) instead of frozen.  The derivation predicts soft > hard.
SOFT_BANK=$OUT/E-real-e4-lr1e-4-soft$K_SOFT.npy
SOFT_JSON=$OUT/eval-real-lr1e-4-soft$K_SOFT.json
if [ ! -s "$SOFT_BANK" ]; then
  run_gpu "train-soft$K_SOFT" train_soft "$K_SOFT" 1.0 "$SOFT_BANK"
else
  say "SKIP train-soft$K_SOFT (bank present)"
fi
if [ -s "$SOFT_BANK" ] && [ ! -s "$SOFT_JSON" ]; then
  run_gpu "eval-soft$K_SOFT" eval_bank "$SOFT_BANK" "$SOFT_JSON"
fi

D_REF=$(delta_of "$OUT/eval-real-lr1e-4.json")
D_HARD=$(delta_of "$OUT/eval-real-lr1e-4-mask10.json")
D_SOFT=$(delta_of "$SOFT_JSON")
say "A/B  lr=1e-4 baseline=$D_REF   hard(>=10)=$D_HARD   soft(k=$K_SOFT)=$D_SOFT"
if is_number "$D_SOFT" && is_number "$D_HARD"; then
  if better_than "$D_SOFT" "$D_HARD"; then
    say "  soft BEATS hard at the same threshold -- the derivation's qualitative claim holds"
    step "soft-vs-hard SOFT_WINS"
  else
    say "  soft does NOT beat hard -- the isotropy assumption is doing the work, not the data"
    step "soft-vs-hard HARD_WINS_OR_TIE"
  fi
else
  fail "ab-not-readable"
fi
df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 6 roll-up (top-1 gate + soft shrinkage)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase6.sh"
  echo
  echo "## 1. The top-1 gate"
  echo
  if [ -s "$OUT/PHASE6_GATE.md" ]; then cat "$OUT/PHASE6_GATE.md"; else echo "*(gate did not run)*"; fi
  echo
  echo "## 2. Soft shrinkage vs the hard mask"
  echo
  echo "Same threshold ($K_SOFT), same lr, same epochs, same reader and stream."
  echo "The only difference is that rows below the threshold are shrunk by c/(c+k) instead of frozen."
  echo
  echo "| arm | delta |"
  echo "|---|---|"
  echo "| lr=1e-4 (no gating) | ${D_REF:-?} |"
  echo "| hard mask >= $K_SOFT | ${D_HARD:-?} |"
  echo "| soft shrink k=$K_SOFT | ${D_SOFT:-?} |"
  echo
  echo "Prediction (docs/round-170-embedding-optimizer-derivation.md): soft > hard."
  echo
  echo "## Steps"
  if [ -s "$STEPSFILE" ]; then sed 's/^/* /' "$STEPSFILE"; else echo "* (none)"; fi
  echo
  echo "## Failures"
  if [ "$(nfail)" = "0" ]; then echo "* none"; else sort -u "$FAILFILE" | sed 's/^/* /'; fi
} > "$OUT/PHASE6_ROLLUP.md" 2>/dev/null

say "ROLLUP:"; head -30 "$OUT/PHASE6_ROLLUP.md" 2>/dev/null

if [ "$(nfail)" = "0" ]; then
  echo "OK" > "$DONE_MARK"; say "*** PHASE 6 DONE (all steps ok) ***"
else
  sort -u "$FAILFILE" > "$FAIL_MARK"
  echo "OK_WITH_FAILURES" > "$DONE_MARK"
  say "*** PHASE 6 DONE with $(nfail) failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
fi
exit 0
