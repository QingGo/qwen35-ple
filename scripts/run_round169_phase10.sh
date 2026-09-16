#!/usr/bin/env bash
# Round 169 phase 10 -- how much of the HEADLINE +0.176 is memory at all?
#
# Phase 9 shuffled the rows injected at unseen trigrams and kept 62% of the gain.
# That was read as "the unseen gain is mostly a generic effect".  It is worse than
# that: the `none` arm has NO READER AT ALL, so `frozen minus none` -- the +0.175931
# every round of this project has quoted -- contains BOTH the memory retrieval and
# whatever the reader contributes as a plain extra layer.  If the generic component
# is structural, a large part of the headline was never memory.
#
# This phase applies the same fill to the SEEN block, which is where the headline
# lives.  The primary record already carries per-position `nll_none` and
# `nll_frozen` for all 443,460 scored positions, so the decomposition gets a PAIRED
# test at no extra cost -- which the phase-9 unseen numbers did not have.
#
#   shuffle scope=all   permute every injected row within its own block.  Both
#                       marginal distributions survive exactly; only the
#                       trigram-to-row correspondence dies.  This is the number
#                       that says how much of +0.176 is retrieval.
#   zero    scope=all   no content anywhere.  Reader(h, 0) is still a nonzero
#                       contribution, which is why phase 9 found it HARMFUL:
#                       it isolates "the reader as an extra layer" from "the row".
#
# Both arms are independent and one eval leaves the card under half used, so they
# run together on a VRAM-and-memory headroom check.
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
PER_JOB_GB=${PER_JOB_GB:-22}
VRAM_NEED_MB=${VRAM_NEED_MB:-6000}
MODES="shuffle zero"

REF_NONE=2.744994
REF_SEEN_REAL=0.175931
TOL=1e-5
REF_JSON=$OUT/eval-real-lr1e-4-gate.json
REF_REC=$OUT/eval-real-lr1e-4-gate.deltas.npz

DONE_MARK=$OUT/PHASE10_DONE
FAIL_MARK=$OUT/PHASE10_FAILED
START_MARK=$OUT/PHASE10_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase10-queue.log}
FAILFILE=$OUT/.phase10-failures
STEPSFILE=$OUT/.phase10-steps

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
vram_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
close() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }

wait_headroom() {
  local need=${1:-$PER_JOB_GB} n=0 used
  while :; do
    used=$(mem_gb); used=${used:-0}
    [ $((used + need)) -le "$MEM_CAP_GB" ] && return 0
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && say "  headroom guard: ${used} GB + ${need} GB > ${MEM_CAP_GB} GB, waiting"
    sleep 30
  done
}

wait_vram() {
  local need=${1:-$VRAM_NEED_MB} n=0 free
  while :; do
    free=$(vram_free_mb); free=${free:-0}
    [ "$free" -ge "$need" ] && return 0
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && say "  vram guard: ${free} MiB free < ${need} MiB, waiting"
    sleep 20
  done
}

PIDS=""

launch_fill() {  # mode
  local mode=$1
  local json=$OUT/eval-real-lr1e-4-fillall-$mode.json
  local grec=$OUT/eval-real-lr1e-4-fillall-$mode.positions.npz
  if [ -s "$json" ] && [ -s "$grec" ]; then
    say "SKIP fillall-$mode (both artifacts present)"
    step "fillall-$mode skipped"
    return 0
  fi
  wait_headroom; wait_vram
  say "LAUNCH fillall-$mode  (mem $(mem_gb) GB, vram free $(vram_free_mb) MiB)"
  (
    { echo "=== fillall-$mode start $(date -Is) ==="
      "$PY" scripts/round169_eval_rows.py \
        --snapshot-dir "$OUT" --trained-bank "$OUT/E-real-e4-lr1e-4.npy" \
        --keep-index "$OUT/keep-wiki-eval.npy" \
        --tokens data/phase1/wikitext-heldout-decon/tokens.npy \
        --positions outputs/round168/rung3/positions-wiki.npy \
        --counts-npz "$MARGIN/wiki-counts.npz" \
        --rows-dir "$ROWS" --model "$MODEL" \
        --reader-kind saved --reader "$READER" --layer 2 --chunk 1024 \
        --unseen-fill "$mode" --fill-scope all \
        --out "$json" --gate-record "$grec"
      echo "=== fillall-$mode rc=$? ==="
    } > "$LOGD/phase10-fillall-$mode.log" 2>&1
  ) &
  PIDS="$PIDS $!"
}

eval_ready() {
  grep -q 'fill_scope' scripts/round169_eval_rows.py 2>/dev/null &&
    grep -q 'fill_injection' scripts/round169_eval_rows.py 2>/dev/null
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
# A re-run must clear the previous run's roll-up.  Phase 10's first attempt
# failed in 58 s; the retry did not remove PHASE10_ROLLUP.md, and a watcher
# read the dead run's table as the new result.  Existence is not currency.
rm -f "$OUT/PHASE10_ROLLUP.md"
echo "started $(date -Is); fills(all scope): $MODES" > "$START_MARK"
say "=== phase 10 (how much of the headline is memory?) starting ==="

if ! eval_ready; then
  say "FAILED guard A: scripts/round169_eval_rows.py on this box has no --fill-scope."
  fail "stale-eval-script"
  echo "stale-eval-script" > "$FAIL_MARK"
  echo "FAILED guard A" > "$DONE_MARK"
  exit 1
fi
say "guard A ok: eval script $(md5sum scripts/round169_eval_rows.py | cut -c1-8)"
if [ ! -s "$REF_REC" ]; then
  say "FAILED: the reference per-position record is absent, so no paired test is possible"
  fail "no-reference-record"
  echo "no-reference-record" > "$FAIL_MARK"
  echo "FAILED" > "$DONE_MARK"
  exit 1
fi

T0=$(date +%s)
for m in $MODES; do launch_fill "$m"; done
say "all fills launched; waiting"
for p in $PIDS; do wait "$p" || true; done
say "all fills finished in $(( $(date +%s) - T0 ))s"

for m in $MODES; do
  if [ -s "$OUT/eval-real-lr1e-4-fillall-$m.json" ] && [ -s "$OUT/eval-real-lr1e-4-fillall-$m.positions.npz" ]; then
    step "fillall-$m ok"
  else
    fail "fillall-$m-incomplete"
    say "fillall-$m produced incomplete artifacts; see $LOGD/phase10-fillall-$m.log"
  fi
done

# ---- guard B: the none arm cannot depend on the fill -----------------------
BAD=0
CHECKED=0
for m in $MODES; do
  f=$OUT/eval-real-lr1e-4-fillall-$m.json
  [ -s "$f" ] || continue
  v=$("$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["mean_nll"]["none_pure_backbone"])))' "$f" 2>/dev/null)
  say "  none arm in fillall-$m: $v (reference $REF_NONE)"
  CHECKED=$((CHECKED + 1))
  close "$v" "$REF_NONE" "$TOL" || BAD=1
done
if [ "$CHECKED" -eq 0 ]; then
  # A guard that passes vacuously is worse than no guard: on this phase's first
  # run every arm had already died, the loop body never executed, BAD stayed 0,
  # and guard B reported "ok: the runs are comparable" about zero runs.
  say "FAILED guard B: no arm produced a JSON, so nothing was checked"
  fail "guard-b-checked-nothing"
elif [ "$BAD" -eq 0 ]; then
  say "guard B ok ($CHECKED arms checked): the none arm is identical -- the runs are comparable"
  step "none-arm-invariant"
else
  say "FAILED guard B: the none arm moved between runs"
  fail "none-arm-moved"
fi

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 10 roll-up (how much of the headline is memory?)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase10.sh"
  echo
  echo "The \`none\` arm has no reader, so \`frozen minus none\` = **memory retrieval +"
  echo "the reader acting as an extra layer**.  Phase 9 found the unseen-regime gain"
  echo "kept 62% under a shuffle, which put the generic component on the table."
  echo "These arms apply the same fill to the SEEN block, where +${REF_SEEN_REAL} lives,"
  echo "and the primary record gives a PAIRED test on all 443,460 scored positions."
  echo
  echo '```text'
  md5sum scripts/round169_eval_rows.py scripts/run_round169_phase10.sh 2>/dev/null
  echo '```'
  echo
  echo "## The decomposition"
  echo
  "$PY" - "$OUT" "$REF_SEEN_REAL" <<'PYEOF'
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
real = float(sys.argv[2])


def fmt(v, spec="+.6f"):
    return "*(missing)*" if v is None else format(v, spec)


def gain_of(path):
    d = json.load(open(path))
    return -float(d["injection_effect_vs_pure_backbone"]["frozen_minus_none"]["mean"])


ref_rec = out / "eval-real-lr1e-4-gate.deltas.npz"
base = np.load(ref_rec)
base_gain = (base["nll_none"].astype(np.float64) - base["nll_frozen"].astype(np.float64))
base_score = base["score"]

print("| arm | SEEN gain | vs real | paired diff vs real | SE | t | content share |")
print("|---|---|---|---|---|---|---|")
print(f"| real | {real:+.6f} | 1.00x | - | - | - | 100% |")
for m in ("shuffle", "zero"):
    f = out / f"eval-real-lr1e-4-fillall-{m}.json"
    rec = out / f"eval-real-lr1e-4-fillall-{m}.deltas.npz"
    if not f.exists() or not rec.exists():
        print(f"| {m} | *(missing)* | | | | | |")
        continue
    g = gain_of(f)
    z = np.load(rec)
    if not np.array_equal(z["score"], base_score):
        print(f"| {m} | {g:+.6f} | | *(SCORED SETS DIFFER -- not comparable)* | | | |")
        continue
    gz = (z["nll_none"].astype(np.float64) - z["nll_frozen"].astype(np.float64))
    d = base_gain - gz                      # how much better the real rows are
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean() / se) if se > 0 else float("nan")
    content = d.mean() / real if real else float("nan")
    print(f"| {m} | {g:+.6f} | {g / real:.2f}x | {d.mean():+.6f} | {se:.6f} | {t:+.1f} | "
          f"{content:.1%} |")
print()
print("`paired diff vs real` is the part of the headline that ONLY the correct row")
print("buys: positive means the real rows beat the fill on the same positions.")
print("`content share` is that difference as a fraction of the headline gain.")
PYEOF
  echo
  echo "## Guards"
  echo
  echo '```text'
  [ -s "$STEPSFILE" ] && cat "$STEPSFILE"
  echo '```'
  echo
  echo "## Failures"
  echo
  if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE"; else echo "(none)"; fi
} > "$OUT/PHASE10_ROLLUP.md"

NFAIL=$(nfail)
if [ "$NFAIL" -gt 0 ]; then
  echo "failures=$NFAIL" > "$FAIL_MARK"
  say "*** PHASE 10 DONE with $NFAIL failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
else
  rm -f "$FAIL_MARK"
  say "*** PHASE 10 DONE (all steps ok) ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
echo "done $(date -Is); failures=$NFAIL" > "$DONE_MARK"
exit 0
