#!/usr/bin/env bash
# Round 169 phase 8 -- one eval that opens two doors that have both been shut.
#
# No training.  One forward-pass evaluation, three arms, ~15 minutes, and it
# answers two questions that the last four phases could not.
#
# DOOR 1: a gate that can actually be deployed.
#
# The project's oracle gates say the value of the memory is being left on the
# table: injecting everywhere gives +0.175931 nats, injecting only where it helps
# (the sign oracle) gives +0.355047, 2.02x.  Half the value is lost to not knowing
# when to trust the memory.  But every gate variable measured so far is either an
# ORACLE or worthless:
#
#   * surprisal of the realised token -- an oracle.  You cannot compute it without
#     already knowing what happened, so the tau=2.25 gate (1.18x) is not runnable;
#   * the row's training count -- genuinely observable, and measured at <= 1.00x
#     at every threshold, because the injection's gain is POSITIVE in every count
#     band (even trigrams seen once gain +0.0949).
#
# The deployable variables are the model's own confidences, available at the
# moment the decision has to be made.  This eval records four of them per arm --
# entropy, top-1 probability, top1-minus-top2 log-margin, and the argmax id -- so
# that a gate can be trained AND scored offline at zero GPU cost from one run.
#
# DOOR 2: the regime that has never been measured.
#
# `score` in every previous eval is restricted to positions whose trigram has an
# E0 row, i.e. that appeared in training -- 443,460 of 1,152,326 aligned
# positions, 39%.  The other 61% are NOT ignored: they receive a row fetched from
# the shard table, so the injection acts there, and that action feeds the
# autoregressive state of every later position.  But no position in that regime
# has ever been scored.  The gate record covers all aligned positions, so the
# unseen-trigram injection effect becomes readable for the first time.
#
# That is the difference between "the memory helps where it has an entry" and
# "the memory helps", and it is the difference Paper B turns on.
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

# The physics must not move.  These are the phase-7 primary means on the same
# bank and stream; the NLL path was refactored (cross_entropy -> log_softmax +
# gather) to get the features, so the run has to prove it computes the same thing.
REF_NONE=2.744994
REF_FROZEN=2.569063
TOL=1e-5

DONE_MARK=$OUT/PHASE8_DONE
FAIL_MARK=$OUT/PHASE8_FAILED
START_MARK=$OUT/PHASE8_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase8-queue.log}
FAILFILE=$OUT/.phase8-failures
STEPSFILE=$OUT/.phase8-steps

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
close() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }

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
  local log=$LOGD/phase8-$name.log t0 t1 rc
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

eval_gate() {  # bank out gate_record
  "$PY" scripts/round169_eval_rows.py \
    --snapshot-dir "$OUT" --trained-bank "$1" \
    --keep-index "$OUT/keep-wiki-eval.npy" \
    --tokens data/phase1/wikitext-heldout-decon/tokens.npy \
    --positions outputs/round168/rung3/positions-wiki.npy \
    --counts-npz "$MARGIN/wiki-counts.npz" \
    --rows-dir "$ROWS" --model "$MODEL" \
    --reader-kind saved --reader "$READER" --layer 2 --chunk 1024 \
    --out "$2" --gate-record "$3"
}

# ---- guard A: the script on this box can do what this phase assumes ---------
# Phase 6 spent 669 s of GPU because the eval script on the box was 100 minutes
# stale and the artifact came back one field short.  One grep costs a second.
eval_ready() {
  grep -q 'gate_record' scripts/round169_eval_rows.py 2>/dev/null &&
    grep -q 'logmargin' scripts/round169_eval_rows.py 2>/dev/null
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); gate features + all-position record" > "$START_MARK"
say "=== phase 8 (deployable gate features + the unseen-trigram regime) starting ==="

if ! eval_ready; then
  say "FAILED guard A: scripts/round169_eval_rows.py on this box has no --gate-record."
  say "  Fix: QWEN35_RSYNC=1 scripts/ssh_autodl.sh scripts/round169_eval_rows.py <dest>/scripts/"
  fail "stale-eval-script"
  echo "stale-eval-script" > "$FAIL_MARK"
  echo "FAILED guard A" > "$DONE_MARK"
  exit 1
fi
say "guard A ok: eval script $(md5sum scripts/round169_eval_rows.py | cut -c1-8)"

BANK=$OUT/E-real-e4-lr1e-4.npy
JSON=$OUT/eval-real-lr1e-4-gate.json
GREC=$OUT/eval-real-lr1e-4-gate.positions.npz

if [ ! -s "$BANK" ]; then
  say "FAILED: the reference bank is absent"
  fail "no-reference-bank"
  echo "no-reference-bank" > "$FAIL_MARK"
  echo "FAILED" > "$DONE_MARK"
  exit 1
fi

if [ -s "$JSON" ] && [ -s "$GREC" ]; then
  say "SKIP gate eval (both artifacts already present)"
  step "gate-eval cached"
else
  run_gpu gate-eval eval_gate "$BANK" "$JSON" "$GREC"
fi

# ---- guard B: the artifact must carry what the offline analysis needs ------
if [ -s "$GREC" ]; then
  MISSING=$("$PY" - "$GREC" <<'PYEOF'
import sys
import numpy as np
need = {"score", "nll_none", "nll_frozen", "has_train_row", "ent_none", "ent_frozen",
        "maxp_none", "maxp_frozen", "logmargin_none", "logmargin_frozen",
        "top1_none", "top1_frozen"}
z = np.load(sys.argv[1])
missing = sorted(need - set(z.files))
print(",".join(missing))
PYEOF
)
  if [ -n "$MISSING" ]; then
    say "FAILED guard B: the gate record is missing $MISSING"
    fail "gate-record-missing-fields"
  else
    say "guard B ok: the gate record carries every field the gate study needs"
    step "gate-record complete"
  fi
else
  say "the gate record was not written"
  fail "gate-record-absent"
fi

# ---- guard C: the physics did not move -------------------------------------
if [ -s "$JSON" ]; then
  NONE=$("$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["mean_nll"]["none_pure_backbone"])))' "$JSON" 2>/dev/null)
  FROZ=$("$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["mean_nll"]["frozen_rows"])))' "$JSON" 2>/dev/null)
  say "reproduction: none=$NONE (ref $REF_NONE)  frozen=$FROZ (ref $REF_FROZEN)"
  if close "$NONE" "$REF_NONE" "$TOL" && close "$FROZ" "$REF_FROZEN" "$TOL"; then
    say "guard C ok: the refactored NLL path reproduces phase 7 to $TOL"
    step "physics-reproduced"
  else
    say "FAILED guard C: the refactored NLL path does NOT reproduce phase 7."
    say "  The features are not free -- they changed the number. Report, do not use."
    fail "physics-moved"
  fi
else
  say "no primary JSON to check"
  fail "primary-json-absent"
fi

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 8 roll-up (gate features + the unseen-trigram regime)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase8.sh"
  echo
  echo "One eval, no training. Two doors:"
  echo
  echo "1. **A deployable gate.** Every gate variable measured so far is either an"
  echo "   oracle (surprisal of the realised token) or worthless (row training count,"
  echo "   measured at <= 1.00x because the gain is positive in every count band)."
  echo "   This run records the model's own confidences -- entropy, top-1 probability,"
  echo "   the top1-minus-top2 log-margin, and the argmax -- for all three arms, so a"
  echo "   gate can be trained and scored offline at zero GPU cost."
  echo
  echo "2. **The regime never measured.** Previous evals scored only positions whose"
  echo "   trigram appeared in training (443,460 of 1,152,326 aligned, 39%). The other"
  echo "   61% still receive a shard-table row and still feed the autoregressive state."
  echo "   The gate record covers all aligned positions."
  echo
  echo "Provenance:"
  echo
  echo '```text'
  md5sum scripts/round169_eval_rows.py scripts/run_round169_phase8.sh 2>/dev/null
  echo '```'
  echo
  echo "## Guards"
  echo
  echo '```text'
  [ -s "$STEPSFILE" ] && cat "$STEPSFILE"
  echo '```'
  echo
  echo "## Headline"
  echo
  if [ -s "$JSON" ]; then
    "$PY" - "$JSON" "$GREC" <<'PYEOF'
import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
m = d["mean_nll"]
g = d["injection_effect_vs_pure_backbone"]["frozen_minus_none"]
print(f"- primary (39% of positions, trigram seen in training):")
print(f"  - none {m['none_pure_backbone']:.6f} -> frozen {m['frozen_rows']:.6f}")
print(f"  - injection gain {g['mean']:+.6f} nats, t {g['t']:+.1f}")
try:
    z = np.load(sys.argv[2])
    n = z["score"].size
    row = z["has_train_row"].astype(bool)
    gain = z["nll_none"].astype(np.float64) - z["nll_frozen"].astype(np.float64)
    print(f"- gate record: {n:,} aligned positions, {int((~row).sum()):,} with no trainable row")
    print(f"  - SEEN     trigram: n {int(row.sum()):,}  gain {gain[row].mean():+.6f}")
    print(f"  - UNSEEN   trigram: n {int((~row).sum()):,}  gain {gain[~row].mean():+.6f}")
except Exception as exc:
    print(f"- gate record unreadable: {exc}")
PYEOF
  else
    echo "*(no primary JSON)*"
  fi
  echo
  echo "## Failures"
  echo
  if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE"; else echo "(none)"; fi
} > "$OUT/PHASE8_ROLLUP.md"

NFAIL=$(nfail)
if [ "$NFAIL" -gt 0 ]; then
  echo "failures=$NFAIL" > "$FAIL_MARK"
  say "*** PHASE 8 DONE with $NFAIL failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
else
  rm -f "$FAIL_MARK"
  say "*** PHASE 8 DONE (all steps ok) ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
echo "done $(date -Is); failures=$NFAIL" > "$DONE_MARK"
exit 0
