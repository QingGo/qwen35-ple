#!/usr/bin/env bash
# Round 169 phase 9 -- is the unseen-regime gain the ROW'S CONTENT or just a bias?
#
# Phase 8 found the injection still gains +0.041296 nats on the 690,303 positions
# whose trigram never appeared in training.  Those rows come from the PRETRAINED
# shard table, and the number has two readings that mean very different things:
#
#   (a) CONTENT.  The row carries real information about this trigram, and the
#       round-162 reader extracts it even though our fine-tuning stream never
#       taught it anything about this trigram.  That generalises.
#   (b) BIAS.    Any vector pushed through the reader shifts the logits
#       favourably, and which trigram it describes is irrelevant.  That is noise
#       wearing a result's clothes, and Paper B cannot stand on it.
#
# Three fills of the unseen block separate them.  The seen block is never touched,
# so every trained-row result is unaffected.
#
#   real     the pretrained shard row (phase 8: +0.041296)
#   zero     nothing at all.  If the gain SURVIVES here, the row was never the
#            source and (b) is answered immediately.
#   shuffle  the very rows being injected, permuted among themselves.  The
#            marginal distribution is preserved EXACTLY and only the
#            trigram-to-row correspondence is destroyed, so:
#                real > shuffle          -> content
#                shuffle ~ zero          -> a wrong row is neutral
#                shuffle < zero          -> a wrong row is actively harmful
#   mean     one constant, content-free vector.
#
# WHY THREE AT ONCE.  They are fully independent -- different outputs, the same
# read-only inputs -- and one eval leaves the card far under half used (playbook
# section 3.1, measured at 22-38% GPU on this workload).  Running them serially is
# the exact waste that section was written about, so each launch takes a HEADROOM
# check (bytes, not process count) and then all three run together.
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
MODES="zero mean shuffle"

# The `none` arm injects nothing, so it cannot depend on the fill.  It must be
# bit-for-bit the same number in all three runs and equal to phase 8's, or the
# runs are not comparable and every difference below is uninterpretable.
REF_NONE=2.744994
REF_UNSEEN_REAL=0.041296
TOL=1e-5

DONE_MARK=$OUT/PHASE9_DONE
FAIL_MARK=$OUT/PHASE9_FAILED
START_MARK=$OUT/PHASE9_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase9-queue.log}
FAILFILE=$OUT/.phase9-failures
STEPSFILE=$OUT/.phase9-steps

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }
close() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }

# The playbook rule: admit on MEMORY HEADROOM, never on process count.
wait_headroom() {
  local need=${1:-$PER_JOB_GB} n=0 used
  while :; do
    used=$(mem_gb); used=${used:-0}
    [ $((used + need)) -le "$MEM_CAP_GB" ] && return 0
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && say "  headroom guard: ${used} GB used + ${need} GB needed > ${MEM_CAP_GB} GB, waiting"
    sleep 30
  done
}

vram_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# The binding constraint under concurrency is VRAM, not host memory: this eval
# holds a 0.8B fp32 model plus a (chunk x vocab) logits block per job.  Guarding
# only the cgroup would let three launches through and OOM the third on the card.
wait_vram() {
  local need=${1:-6000} n=0 free
  while :; do
    free=$(vram_free_mb); free=${free:-0}
    [ "$free" -ge "$need" ] && return 0
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && say "  vram guard: ${free} MiB free < ${need} MiB needed, waiting"
    sleep 20
  done
}

PIDS=""

launch_fill() {  # mode
  local mode=$1
  local json=$OUT/eval-real-lr1e-4-fill-$mode.json
  if [ -s "$json" ]; then
    say "SKIP fill-$mode (JSON present)"
    step "fill-$mode skipped"
    return 0
  fi
  wait_headroom
  wait_vram
  say "LAUNCH fill-$mode  (mem $(mem_gb) GB, vram free $(vram_free_mb) MiB, gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')%)"
  (
    { echo "=== fill-$mode start $(date -Is) ==="
      "$PY" scripts/round169_eval_rows.py \
        --snapshot-dir "$OUT" --trained-bank "$OUT/E-real-e4-lr1e-4.npy" \
        --keep-index "$OUT/keep-wiki-eval.npy" \
        --tokens data/phase1/wikitext-heldout-decon/tokens.npy \
        --positions outputs/round168/rung3/positions-wiki.npy \
        --counts-npz "$MARGIN/wiki-counts.npz" \
        --rows-dir "$ROWS" --model "$MODEL" \
        --reader-kind saved --reader "$READER" --layer 2 --chunk 1024 \
        --unseen-fill "$mode" --out "$json"
      echo "=== fill-$mode rc=$? ==="
    } > "$LOGD/phase9-fill-$mode.log" 2>&1
  ) &
  PIDS="$PIDS $!"
}

# ---- guard A: the script on this box can do what this phase assumes ---------
eval_ready() {
  grep -q 'unseen_fill' scripts/round169_eval_rows.py 2>/dev/null &&
    grep -q 'unseen_regime' scripts/round169_eval_rows.py 2>/dev/null
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); fills: $MODES; concurrent, headroom ${PER_JOB_GB} GB" > "$START_MARK"
say "=== phase 9 (content vs bias at unseen trigrams) starting ==="

if ! eval_ready; then
  say "FAILED guard A: scripts/round169_eval_rows.py on this box has no --unseen-fill."
  fail "stale-eval-script"
  echo "stale-eval-script" > "$FAIL_MARK"
  echo "FAILED guard A" > "$DONE_MARK"
  exit 1
fi
say "guard A ok: eval script $(md5sum scripts/round169_eval_rows.py | cut -c1-8); OMP threads 8 x 3 concurrent"
say "admission is by MEMORY HEADROOM (${PER_JOB_GB} GB/job, cap ${MEM_CAP_GB} GB), not by process count"

T0=$(date +%s)
for m in $MODES; do launch_fill "$m"; done
say "all fills launched; waiting"
for p in $PIDS; do wait "$p" || true; done
T1=$(date +%s)
say "all fills finished in $((T1 - T0))s"

for m in $MODES; do
  if [ -s "$OUT/eval-real-lr1e-4-fill-$m.json" ]; then
    step "fill-$m ok"
  else
    fail "fill-$m-no-json"
    say "fill-$m produced no JSON; see $LOGD/phase9-fill-$m.log"
  fi
done

# ---- guard B: the comparison must be legitimate ----------------------------
# `none` injects nothing, so it is independent of the fill BY CONSTRUCTION.  Three
# runs that disagree on it are not three runs of the same experiment, and no
# difference between their frozen arms could be attributed to the fill.
if [ -s "$OUT/eval-real-lr1e-4-fill-zero.json" ]; then
  BAD=0
  CHECKED=0
  for m in $MODES; do
    f=$OUT/eval-real-lr1e-4-fill-$m.json
    [ -s "$f" ] || continue
    v=$("$PY" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["mean_nll"]["none_pure_backbone"])))' "$f" 2>/dev/null)
    say "  none arm in fill-$m: $v (reference $REF_NONE)"
    CHECKED=$((CHECKED + 1))
    close "$v" "$REF_NONE" "$TOL" || BAD=1
  done
  if [ "$CHECKED" -eq 0 ]; then
    # A guard that passes vacuously is worse than no guard: when every arm has
    # already died the loop body never runs, BAD stays 0, and guard B reports
    # "the runs are comparable" about zero runs.  That is what phase 10's first
    # attempt did.
    say "FAILED guard B: no arm produced a JSON, so nothing was checked"
    fail "guard-b-checked-nothing"
  elif [ "$BAD" -eq 0 ]; then
    say "guard B ok ($CHECKED arms checked): the none arm is identical in all fills -- the runs are comparable"
    step "none-arm-invariant"
  else
    say "FAILED guard B: the none arm moved between fills, so the fill is not the only variable"
    fail "none-arm-moved"
  fi
fi

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 9 roll-up (is the unseen-regime gain content or bias?)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase9.sh"
  echo
  echo "Phase 8 measured **+${REF_UNSEEN_REAL} nats** over the 690,303 positions whose"
  echo "trigram never appeared in training, where the injected row comes from the"
  echo "PRETRAINED shard table.  These fills decide what that number is:"
  echo
  echo '| fill | what it injects | meaning if the gain survives |'
  echo '|---|---|---|'
  echo '| `real` | the pretrained shard row | content |'
  echo '| `zero` | nothing | the row was never the source |'
  echo '| `shuffle` | the same rows, permuted | a generic perturbation |'
  echo '| `mean` | their average | one content-free vector |'
  echo
  echo "Provenance:"
  echo
  echo '```text'
  md5sum scripts/round169_eval_rows.py scripts/run_round169_phase9.sh 2>/dev/null
  echo '```'
  echo
  echo "## The comparison"
  echo
  "$PY" - "$OUT" "$REF_UNSEEN_REAL" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
real = float(sys.argv[2])
# The real arm's n and frac-positive come from phase 8's gate record, where they
# were measured; they are not None, and formatting None here is what emptied this
# table on the first run of this phase.
rows = [("real (phase 8)", real, 690303, None, 0.516)]
for m in ("zero", "mean", "shuffle"):
    f = out / f"eval-real-lr1e-4-fill-{m}.json"
    if not f.exists():
        rows.append((m, None, None, None, None))
        continue
    d = json.load(open(f))
    u = d.get("unseen_regime") or {}
    rows.append((m, u.get("injection_gain"), u.get("n"), u.get("mean_nll_frozen"),
                 u.get("frac_positive")))
print("| fill | unseen gain (nats) | vs real | n | frac positive |")
print("|---|---|---|---|---|")
def col(v, spec):
    return "*(missing)*" if v is None else format(v, spec)


for name, g, n, mf, fp in rows:
    if g is None:
        print(f"| {name} | *(missing)* | | | |")
        continue
    rel = "1.00x" if name.startswith("real") else (f"{g / real:.2f}x" if real else "-")
    print(f"| {name} | {g:+.6f} | {rel} | {col(n, ',')} | {col(fp, '.3f')} |")
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
} > "$OUT/PHASE9_ROLLUP.md"

NFAIL=$(nfail)
if [ "$NFAIL" -gt 0 ]; then
  echo "failures=$NFAIL" > "$FAIL_MARK"
  say "*** PHASE 9 DONE with $NFAIL failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
else
  rm -f "$FAIL_MARK"
  say "*** PHASE 9 DONE (all steps ok) ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
echo "done $(date -Is); failures=$NFAIL" > "$DONE_MARK"
exit 0
