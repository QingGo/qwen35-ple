#!/usr/bin/env bash
# Round 169 phase 5 -- row selection: train only the rows with enough evidence.
#
# THE AXIS CHANGED.  The lr ladder closed the learning-rate axis: delta is
# monotone in kept_delta_rms and never changes sign, so the best result available
# there is zero (not "training fails" but "the best training can do is equal not
# training").  What is left is not a better optimiser but a narrower update:
#
#   SparseAdam is scale-invariant per coordinate, so a row's FIRST update is
#   lr * sign(g) no matter how weak the evidence is.  A trigram seen once takes a
#   full-size step in the direction of one noisy example, replacing a corpus-level
#   prior with a one-sample estimate.  That is the measured mechanism behind
#   "rare contexts lose, frequent contexts gain".
#
# A better optimiser cannot fix that -- it would drive the training loss lower and
# move the rows further, and the harm is monotone in how far the rows move.  A
# narrower update can.  The hard mask below is the limit of evidence-scaled
# shrinkage: rows whose trigram occurred fewer than N times keep their frozen E0
# value exactly.
#
# THE PREDICTION IS AN IDENTITY, NOT A FORECAST.  Held rows contribute exactly
# zero to the paired delta, and because ``trigram_codes`` is an injection rather
# than a hash each distinct trigram owns its row -- so holding low-count rows
# cannot change the trajectory of high-count ones.  The aggregate is therefore
# fixed in advance by the frozen band table, and the threshold-1 row of that
# table already reproduced the measured -0.00272 to ten decimals before any of
# this ran.  See docs/round-169-phase5-row-selection-preregistration.md.
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

# Frozen in the pre-registration, read from reband-lr1e-4.json (a phase-4
# artifact that predates every phase-5 run).
PRED_A=0.0074002256     # threshold 10
PRED_B=0.0069824346     # threshold 50
TOL=0.0002

DONE_MARK=$OUT/PHASE5_DONE
FAIL_MARK=$OUT/PHASE5_FAILED
START_MARK=$OUT/PHASE5_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase5-queue.log}
FAILFILE=$OUT/.phase5-failures
STEPSFILE=$OUT/.phase5-steps
PHASE4_DONE=$OUT/PHASE4_DONE

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
is_number() { awk -v a="$1" 'BEGIN{exit !(a ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/)}'; }
within() { awk -v a="$1" -v p="$2" -v t="$3" 'BEGIN{d=a-p; if(d<0)d=-d; exit !(d<=t)}'; }
is_positive() { awk -v a="$1" 'BEGIN{exit !(a>0)}'; }

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
  local log=$LOGD/phase5-$name.log t0 t1 rc
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

train_masked() {  # min_count out
  "$PY" scripts/round169_train_rows.py \
    --snapshot-dir "$OUT" --train-tokens data/phase1/PURE_WIKI/tokens.npy \
    --model "$MODEL" --reader-kind saved --reader "$READER" \
    --keep-index "$OUT/keep-wiki-eval.npy" --arm real \
    --epochs "$EPOCHS" --lr "$LR" --min-train-count "$1" --out "$2"
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

delta_of() {
  [ -s "$1" ] || return 1
  "$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["delta"]["mean"])))' "$1" 2>/dev/null
}

run_arm() {  # min_count tag
  local n=$1 tag=$2
  local bank=$OUT/E-real-e4-lr1e-4-$tag.npy ej=$OUT/eval-real-lr1e-4-$tag.json
  if [ -s "$bank" ]; then
    say "SKIP train $tag (bank present)"
  else
    run_gpu "train-$tag" train_masked "$n" "$bank"
  fi
  if [ -s "$bank" ] && [ ! -s "$ej" ]; then
    run_gpu "eval-$tag" eval_bank "$bank" "$ej"
  fi
  [ -s "$bank" ] || fail "bank-$tag"
  delta_of "$ej"
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); pred A=$PRED_A B=$PRED_B tol=$TOL" > "$START_MARK"
say "=== phase 5 (row selection) starting ==="

# ---- wait for the records the prediction was read from ---------------------
n=0
while [ ! -s "$PHASE4_DONE" ]; do
  n=$((n + 1))
  if [ "$n" -gt 4 ] && ! pgrep -f '[r]un_round169_phase4.sh' > /dev/null 2>&1; then
    say "phase 4 is gone without writing its marker; the predictions are still frozen, continuing"
    step "phase4-marker-missing"
    break
  fi
  [ $((n % 20)) -eq 1 ] && say "  waiting for PHASE4_DONE (in-flight: $(pgrep -cf '[r]ound169_(train|eval)_rows' || echo 0))"
  sleep 30
done
wait_quiet
say "phase 4 finished; $(tail -1 "$LOGD/phase4-queue.log" 2>/dev/null)"

# ---- the pre-registration must hold before the GPU is spent ----------------
if [ ! -s "$OUT/reband-lr1e-4.json" ]; then
  say "FAILED: reband-lr1e-4.json is absent, so the frozen prediction cannot be checked"
  echo "no-frozen-prediction" > "$FAIL_MARK"; echo "FAILED" > "$DONE_MARK"; exit 1
fi
say "frozen predictions: A(>=10)=$PRED_A  B(>=50)=$PRED_B  tolerance=$TOL"

# ---- arm A ------------------------------------------------------------------
DA=$(run_arm 10 "mask10")
if ! is_number "$DA"; then
  fail "arm-A-no-delta"
  say "arm A produced no usable delta (got '$DA'); stopping"
  echo "arm-a-failed" > "$FAIL_MARK"
else
  say "ARM A (>=10)  delta=$DA   predicted=$PRED_A"
  if within "$DA" "$PRED_A" "$TOL"; then
    say "  identity HOLDS (|delta - predicted| <= $TOL)"
    if is_positive "$DA"; then
      say "  and delta is POSITIVE: this is the first training arm to beat frozen rows"
    else
      say "  but delta is not positive, so the decomposition is right and the claim still fails"
      fail "arm-A-not-positive"
    fi
    # ---- arm B: only when A landed on its identity --------------------------
    DB=$(run_arm 50 "mask50")
    if ! is_number "$DB"; then
      fail "arm-B-no-delta"
      say "arm B produced no usable delta (got '$DB')"
    else
      say "ARM B (>=50)  delta=$DB   predicted=$PRED_B"
      if within "$DB" "$PRED_B" "$TOL"; then
        say "  identity HOLDS at a second threshold"
      else
        say "  MISMATCH at the second threshold: the decomposition has a hidden variable"
        fail "arm-B-mismatch"
      fi
    fi
  else
    say "  MISMATCH: the decomposition is not an identity. Skipping arm B."
    say "  (measured $DA vs predicted $PRED_A; tolerance $TOL)"
    fail "arm-A-mismatch"
  fi
fi

# ---- re-band every arm that has a per-position record ----------------------
say "=== re-banding ==="
shopt -s nullglob
for rec in "$OUT"/*mask*.deltas.npz; do
  tag=$(basename "$rec" .deltas.npz)
  { "$PY" scripts/round169_reband.py --record "$rec" --snapshot-dir "$OUT" --tag "$tag" \
      --out-json "$OUT/reband-$tag.json" --out-md "$LOGD/reband-$tag.md"; rc=$?; } \
    > "$LOGD/phase5-reband-$tag.log" 2>&1
  step "reband-$tag rc=${rc:-?}"
  [ "${rc:-1}" -ne 0 ] && fail "reband-$tag"
done
shopt -u nullglob

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 5 roll-up (row selection)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase5.sh"
  echo
  echo "delta = frozen - trained; POSITIVE means training the rows helped."
  echo "The predictions are identities: held rows contribute exactly zero and rows do not mix."
  echo "Pre-registration: docs/round-169-phase5-row-selection-preregistration.md"
  echo
  echo "| arm | measured | predicted | within $TOL |"
  echo "|---|---|---|---|"
  for pair in "mask10:$PRED_A" "mask50:$PRED_B"; do
    tag=${pair%%:*}; pred=${pair##*:}
    j=$OUT/eval-real-lr1e-4-$tag.json
    if [ -s "$j" ]; then
      d=$(delta_of "$j")
      if within "$d" "$pred" "$TOL"; then ok=YES; else ok=NO; fi
      echo "| $tag | $d | $pred | $ok |"
    else
      echo "| $tag | (not run) | $pred | - |"
    fi
  done
  echo
  echo "Ceiling, stated in advance: +0.0074 nats is 4.2% of what injection buys (0.17593)."
  echo
  echo "## Steps"
  if [ -s "$STEPSFILE" ]; then sed 's/^/* /' "$STEPSFILE"; else echo "* (none)"; fi
  echo
  echo "## Failures"
  if [ "$(nfail)" = "0" ]; then echo "* none"; else sort -u "$FAILFILE" | sed 's/^/* /'; fi
} > "$OUT/PHASE5_ROLLUP.md" 2>/dev/null

say "ROLLUP:"; cat "$OUT/PHASE5_ROLLUP.md" 2>/dev/null | head -24

if [ "$(nfail)" = "0" ]; then
  echo "OK" > "$DONE_MARK"; say "*** PHASE 5 DONE (all steps ok) ***"
else
  sort -u "$FAILFILE" > "$FAIL_MARK"
  echo "OK_WITH_FAILURES" > "$DONE_MARK"
  say "*** PHASE 5 DONE with $(nfail) failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
exit 0
