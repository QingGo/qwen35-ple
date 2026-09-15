#!/usr/bin/env bash
# Round 169 phase 7 -- the top-1 gate, re-run after phase 6 spent 11 minutes
# producing a record the gate could not read.
#
# WHAT WENT WRONG IN PHASE 6, because this whole phase exists to repair it:
#
# Phase 6's gate eval ran to completion (rc=0, 669 s) against a copy of
# scripts/round169_eval_rows.py that was 100 minutes stale -- the version from
# 19:09, which predates the commit that records per-position top-1.  The eval
# therefore wrote a perfectly good record with eight fields and no `hit_*`
# arrays, and the gate tool refused it:
#
#     ValueError: record is missing hit_none / hit_frozen
#
# The refusal was correct and it is the reason this cost 11 minutes instead of
# the whole line's conclusion.  Two defences failed to catch it and one did:
#
#   * phase 6 skipped the eval only if the JSON and npz EXISTED.  They did exist
#     -- carrying the wrong fields.  Existence is not content.
#   * nothing checked that the script on the box was the script the phase was
#     written against.  A queue uploads its own file and assumes the rest of the
#     tree came with it; rsync had not been re-run for this one.
#   * the gate tool checked the contract and refused.  That is what a consumer
#     guard is for, and it is the only reason the failure was loud.
#
# So phase 7 adds the two missing guards, both cheap and both before the GPU:
#
#   guard A  the eval script on this box must contain the top-1 capability.
#            Fails in one second rather than eleven minutes.
#   guard B  a cached record is only reused if it CARRIES the hit arrays.  A
#            record that cannot answer the question is not a cache hit.
#
# And it reads one extra gate for free: phase 6's eval of the soft-shrinkage arm
# runs after the script was repaired, so that record does carry top-1 and the
# same verdict tool can be pointed at it at the cost of a few seconds.
#
# Guards are otherwise the same four as every phase before this one.
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
K_SOFT=10
TAU=2.0
T_MIN=3.0

DONE_MARK=$OUT/PHASE7_DONE
FAIL_MARK=$OUT/PHASE7_FAILED
START_MARK=$OUT/PHASE7_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase7-queue.log}
FAILFILE=$OUT/.phase7-failures
STEPSFILE=$OUT/.phase7-steps
PHASE6_DONE=$OUT/PHASE6_DONE

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
is_number() { awk -v a="$1" 'BEGIN{exit !(a ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/)}'; }

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
  local log=$LOGD/phase7-$name.log t0 t1 rc
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

# ---- guard A: the script on this box can answer the question ---------------
eval_script_ready() {
  grep -q 'hit_frozen' scripts/round169_eval_rows.py 2>/dev/null
}

# ---- guard B: a cached record must CARRY the hit arrays --------------------
record_has_hits() {
  [ -s "$1" ] || return 1
  "$PY" -c 'import sys, numpy as np
z = np.load(sys.argv[1])
need = {"hit_none", "hit_frozen", "hit_trained"}
raise SystemExit(0 if need <= set(z.files) else 1)' "$1" > /dev/null 2>&1
}

read_gate() {  # record md-label
  local rec=$1 label=$2
  "$PY" scripts/round169_top1_verdict.py --record "$rec" --tau "$TAU" --t-min "$T_MIN" \
    --out-json "$OUT/gate-$label.json" --out-md "$OUT/PHASE7_GATE_$label.md" \
    > "$LOGD/phase7-gate-$label.log" 2>&1
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); gate tau=$TAU t_min=$T_MIN; soft k=$K_SOFT" > "$START_MARK"
say "=== phase 7 (top-1 gate, on a repair) starting ==="

# ---- guard A, before anything expensive ------------------------------------
if ! eval_script_ready; then
  say "FAILED guard A: scripts/round169_eval_rows.py on this box has no top-1 recording."
  say "  Phase 6 spent 669 s of GPU on this exact mismatch. Refusing to repeat it."
  say "  Fix: QWEN35_RSYNC=1 scripts/ssh_autodl.sh scripts/round169_eval_rows.py <dest>/scripts/"
  fail "stale-eval-script"
  echo "stale-eval-script" > "$FAIL_MARK"
  echo "FAILED guard A" > "$DONE_MARK"
  exit 1
fi
say "guard A ok: the eval script records top-1 ($(md5sum scripts/round169_eval_rows.py | cut -c1-8))"

# ---- wait for phase 6 -------------------------------------------------------
n=0
while [ ! -s "$PHASE6_DONE" ]; do
  n=$((n + 1))
  if [ "$n" -gt 4 ] && ! pgrep -f '[r]un_round169_phase6.sh' > /dev/null 2>&1; then
    say "phase 6 is gone without writing its marker; continuing"
    step "phase6-marker-missing"
    break
  fi
  [ $((n % 20)) -eq 1 ] && say "  waiting for PHASE6_DONE (in-flight: $(pgrep -cf '[r]ound169_(train|eval)_rows' || echo 0))"
  sleep 30
done
wait_quiet
say "phase 6 finished; $(tail -1 "$LOGD/phase6-queue.log" 2>/dev/null)"

# ---- 1. the gate, on a record that can actually answer it -------------------
REF_BANK=$OUT/E-real-e4-lr1e-4.npy
GATE_JSON=$OUT/eval-real-lr1e-4-top1.json
REC=${GATE_JSON%.json}.deltas.npz

if [ ! -s "$REF_BANK" ]; then
  say "FAILED: the reference bank is absent; the gate cannot run"
  fail "no-reference-bank"
else
  if record_has_hits "$REC"; then
    say "SKIP gate eval (a cached record carrying top-1 is present)"
    step "gate-eval cached"
  else
    if [ -e "$REC" ] || [ -e "$GATE_JSON" ]; then
      say "the cached gate record exists but does NOT carry top-1; discarding and re-running"
      step "gate-eval stale-record-discarded"
      rm -f "$REC" "$GATE_JSON"
    fi
    run_gpu top1-gate eval_bank "$REF_BANK" "$GATE_JSON"
  fi

  if record_has_hits "$REC"; then
    if read_gate "$REC" top1; then
      VERDICT=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["verdict"])' \
                  "$OUT/gate-top1.json" 2>/dev/null)
      say "*** TOP-1 GATE VERDICT: ${VERDICT:-UNKNOWN} (tau=$TAU, t_min=$T_MIN) ***"
      step "top1-gate verdict=${VERDICT:-UNKNOWN}"
      grep -E '^\| (frozen|trained) ' "$OUT/PHASE7_GATE_top1.md" 2>/dev/null | while read -r l; do say "  $l"; done
    else
      say "gate tool failed; see $LOGD/phase7-gate-top1.log"
      fail "gate-tool"
    fi
  else
    say "guard B STILL failing after a fresh eval: the record has no hit arrays"
    say "  This means the eval script is current but the recording is not reached."
    fail "gate-record-no-hits"
  fi
fi

# ---- 2. the same gate on the soft-shrinkage arm, at the cost of seconds -----
SOFT_REC=$OUT/eval-real-lr1e-4-soft$K_SOFT.deltas.npz
SOFT_VERDICT=absent
if record_has_hits "$SOFT_REC"; then
  if read_gate "$SOFT_REC" soft$K_SOFT; then
    SOFT_VERDICT=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["verdict"])' \
                     "$OUT/gate-soft$K_SOFT.json" 2>/dev/null)
    say "*** SOFT(k=$K_SOFT) TOP-1 GATE VERDICT: ${SOFT_VERDICT:-UNKNOWN} ***"
    step "soft-gate verdict=${SOFT_VERDICT:-UNKNOWN}"
  else
    say "soft gate tool failed; see $LOGD/phase7-gate-soft$K_SOFT.log"
    fail "soft-gate-tool"
  fi
else
  say "no top-1 in the soft$K_SOFT record (it was evaluated by the stale script); skipped"
  say "  A re-eval is 11 minutes; phase 7 does not spend it without a reason to look."
fi

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 7 roll-up (the top-1 gate, re-run)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase7.sh"
  echo
  echo "Phase 6's gate eval produced a record without the \`hit_*\` arrays because"
  echo "the copy of \`scripts/round169_eval_rows.py\` on the box was 100 minutes stale."
  echo "The gate tool refused it, correctly. This phase re-ran the eval against the"
  echo "repaired script and adds the guard that should have caught it first."
  echo
  echo "Provenance:"
  echo
  echo '```text'
  md5sum scripts/round169_eval_rows.py scripts/round169_top1_verdict.py scripts/run_round169_phase7.sh 2>/dev/null
  echo '```'
  echo
  echo "## 1. The gate, reference arm (lr=1e-4)"
  echo
  if [ -s "$OUT/PHASE7_GATE_top1.md" ]; then cat "$OUT/PHASE7_GATE_top1.md"; else echo "*(gate did not run)*"; fi
  echo
  echo "## 2. The gate, soft-shrinkage arm (k=$K_SOFT)"
  echo
  if [ -s "$OUT/PHASE7_GATE_soft$K_SOFT.md" ]; then
    cat "$OUT/PHASE7_GATE_soft$K_SOFT.md"
  else
    echo "*(no top-1 in that record; not read)*"
  fi
  echo
  echo "## 3. Steps"
  echo
  echo '```text'
  [ -s "$STEPSFILE" ] && cat "$STEPSFILE"
  echo '```'
  echo
  echo "## 4. Failures"
  echo
  if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE"; else echo "(none)"; fi
} > "$OUT/PHASE7_ROLLUP.md"

NFAIL=$(nfail)
if [ "$NFAIL" -gt 0 ]; then
  echo "failures=$NFAIL" > "$FAIL_MARK"
  say "*** PHASE 7 DONE with $NFAIL failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
else
  rm -f "$FAIL_MARK"
  say "*** PHASE 7 DONE (all steps ok) ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
echo "done $(date -Is); failures=$NFAIL; soft-verdict=$SOFT_VERDICT" > "$DONE_MARK"
exit 0
