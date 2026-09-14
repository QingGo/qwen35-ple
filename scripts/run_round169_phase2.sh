#!/usr/bin/env bash
# Round 169 phase 2 -- the learning-rate ladder, and why it replaced the
# code/stem replication that phase 1 was queued to run next.
#
# PHASE 1's READING WAS WRONG, AND ITS OWN LOG SAID SO.
#
# B1 compared rows frozen against rows retrained at lr=1e-3 and measured
# delta = -0.33127, which read as "training the rows destroys them".  The
# lr=1e-4 arm landed later the same night and shows the optimiser had simply
# diverged:
#
#   lr=1e-3   mean_loss 3.055 -> 3.165 -> 4.141 -> 3.471   DIVERGING
#             delta = -0.33127            kept_delta_rms = 1.822e-02
#   lr=1e-4   mean_loss 2.974 -> 2.956 -> 2.937 -> 2.918   converging
#             delta = -0.00272            kept_delta_rms = 2.397e-03
#
# So -0.33127 is the damage of a diverged optimiser, not a property of the rows.
# At lr=1e-4 the aggregate cost is 1.5% of what injection buys, and the
# per-frequency table FLIPS SIGN: contexts seen >=10 times come out BETTER after
# training (+0.0215 in the 200+ band, t=+8.3) while the 1-9 bands come out
# slightly worse.  The live question is therefore no longer "do the rows
# survive retraining" but "is there a learning rate at which the aggregate delta
# is POSITIVE".
#
# SIGN CONVENTION.  delta = frozen - trained, so a POSITIVE delta means training
# the rows HELPED.  Every number below is in that convention.  The whole ladder
# exists to find a positive point; nothing here is allowed to re-read a negative
# delta as a success.
#
# WHY code/stem ARE DEFERRED.  Replicating a diverged configuration on two more
# domains buys less than mapping the curve that decides whether PLE rows should
# be trained at all.  They are deferred, not cancelled, and the pre-registration
# remains wiki-only -- so no verdict below can revise the frozen B1 rule.
#
# GUARDS (the same four as phase 1; each was paid for once already):
#   * MEMORY.  The cap is /sys/fs/cgroup/memory.max = 120 GiB, not the host's
#     1007 GB that `free` reports, and page cache counts against it.
#   * ONE HEAVY GPU JOB AT A TIME.  The queue is stopped before the ladder
#     starts, and every launch waits for quiet as well as for headroom.
#   * EVERY rc IS RECORDED.  A killed trainer stops mid-epoch with no error line.
#   * E0 IS NOT RECYCLED.  Every point on the ladder is scored against the same
#     frozen snapshot, so the 6.83 GB E0 is hardlinked out of reach of phase 1's
#     rm -f rather than copied (a copy would not fit in the 11 GB of free disk).
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/qwen35-ple}
REPO=${REPO:-$ROOT/repo}
PY=${PY:-$ROOT/venv/bin/python}
LOGD=${LOGD:-$ROOT/logs}
OUT=$REPO/outputs/round169
EXPL=$OUT/exploratory
ROWS=$ROOT/qwen38-rows
MARGIN=$ROOT/outputs/round168/margin
MODEL=$ROOT/models/Qwen3.5-0.8B
READER=$ROOT/outputs/round162-0.8B-nosft/reader-wiki-seed0.pt
E0_SAFE=$ROOT/E0-wiki-keep.npy
MEM_CAP_GB=${MEM_CAP_GB:-85}
EPOCHS=${EPOCHS:-4}

DONE_MARK=$OUT/PHASE2_DONE
FAIL_MARK=$OUT/PHASE2_FAILED
START_MARK=$OUT/PHASE2_STARTED
EP1_JSON=$OUT/eval-real-e1-lr1e-3.json

# lr=1e-4, 4 epochs, real arm, same reader and same held-out stream.  Measured,
# not assumed: this is the reference the ladder has to beat to be interesting.
REF_LR_TAG=lr1e-4
REF_DELTA=-0.0027166083372815774

# say() writes to the log FILE and never to stdout, and fail()/step() write to
# files rather than to shell arrays.  This is a correctness requirement, not a
# style choice, and it cost this run its stopping rule:
#
#   D1=$(run_point 3.162e-4 lr3.162e-4)
#
# captures EVERYTHING the call prints.  With say() on stdout, D1 held the
# progress text ("... START train-lr3.162e-4 (mem 52 GB, gpu 0%) ... END ...")
# followed by the number, and the frozen rule below then evaluated
# ``awk -v a="$D1" 'BEGIN{... a+0 ...}'`` -- where awk reads a leading "[0" as
# 0.  Zero beats every negative delta, so the ladder climbed to a second point
# after a first point that had already lost to lr=1e-4.  The run only stopped
# because a human read the delta and killed the arm.
#
# The same subshell silently discarded FAILED+=() from inside run_point, so the
# killed arm was reported as "all steps ok".  Both failures are one bug: state
# and logging that live in a subshell.
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase2-queue.log}
FAILFILE=$OUT/.phase2-failures
STEPSFILE=$OUT/.phase2-steps

say() { echo "[$(date +%H:%M:%S)] $*" >> "$QUEUE_LOG"; }
step() { echo "$1" >> "$STEPSFILE"; }
fail() { echo "$1" >> "$FAILFILE"; }
nfail() { if [ -s "$FAILFILE" ]; then sort -u "$FAILFILE" | wc -l | tr -d ' '; else echo 0; fi; }
mem_gb() { awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current; }

wait_mem() {
  local cap=${1:-$MEM_CAP_GB} n=0 cur
  while :; do
    cur=$(mem_gb); cur=${cur:-0}
    [ "$cur" -le "$cap" ] && return 0
    n=$((n + 1))
    [ $((n % 10)) -eq 1 ] && say "  memory guard: ${cur} GB > ${cap} GB, waiting"
    sleep 30
  done
}

wait_quiet() {
  local n=0 busy
  while :; do
    busy=$(pgrep -cf '[r]ound169_(train|eval)_rows' || true); busy=${busy:-0}
    [ "$busy" -eq 0 ] && return 0
    n=$((n + 1))
    [ $((n % 5)) -eq 1 ] && say "  waiting for $busy in-flight round169 job(s)"
    sleep 30
  done
}

run_gpu() {  # name cmd...
  local name=$1; shift
  local log=$LOGD/phase2-$name.log t0 t1 rc
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

train_bank() {  # lr out
  "$PY" scripts/round169_train_rows.py \
    --snapshot-dir "$OUT" --train-tokens data/phase1/PURE_WIKI/tokens.npy \
    --model "$MODEL" --reader-kind saved --reader "$READER" \
    --keep-index "$OUT/keep-wiki-eval.npy" --arm real \
    --epochs "$EPOCHS" --lr "$1" --out "$2"
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

# delta = frozen - trained; prints nothing (and fails) if the json is unusable.
delta_of() {
  [ -s "$1" ] || return 1
  "$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]))["delta"]["mean"];print(repr(float(d)))' "$1" 2>/dev/null
}

# A decision rule must refuse a non-numeric input, not coerce it.  awk maps any
# leading non-number to 0, and 0 wins against every negative delta, so a leaking
# log line silently flips "stop" into "climb".
is_number() { awk -v a="$1" 'BEGIN{exit !(a ~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/)}'; }
better_than() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 > b+0)}'; }

run_point() {  # lr tag -> prints ONLY the delta on stdout (empty on failure)
  local lr=$1 tag=$2
  local bank=$OUT/E-real-e4-$tag.npy ej=$OUT/eval-real-$tag.json
  if [ -s "$bank" ]; then
    say "SKIP train $tag (bank present)"
  else
    run_gpu "train-$tag" train_bank "$lr" "$bank"
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
mkdir -p "$OUT" "$EXPL" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is); reference $REF_LR_TAG delta=$REF_DELTA; epochs=$EPOCHS" > "$START_MARK"

say "=== phase 2 (lr ladder) starting ==="

# ---- 1. wait for phase 1's step 1 (wiki, 1 epoch, lr=1e-3) to land ----------
n=0
while [ ! -s "$EP1_JSON" ]; do
  n=$((n + 1))
  if [ $((n % 10)) -eq 1 ]; then
    say "  waiting for $(basename "$EP1_JSON") (in-flight: $(pgrep -cf '[r]ound169_(train|eval)_rows' || echo 0))"
  fi
  if [ "$n" -gt 240 ]; then
    say "FAILED: phase 1 step 1 never produced $EP1_JSON"
    echo "ep1-timeout" > "$FAIL_MARK"; echo "FAILED" > "$DONE_MARK"; exit 1
  fi
  sleep 30
done
say "phase 1 step 1 landed: $(delta_of "$EP1_JSON" || echo 'unparseable') nats (frozen - trained)"

# ---- 2. stop phase 1 BEFORE waiting for anything ---------------------------
# The order here was backwards on the first run and it deadlocked.  With
# wait_quiet first, any job the queue starts while we wait becomes a job we wait
# for forever -- and the only thing that can stop it is the pkill queued behind
# the wait.  It happened: the queue launched a 4-epoch code arm three seconds
# before this script noticed step 1 had landed, and the log repeated "waiting for
# 1 in-flight job(s)" with no error for ten minutes of wasted GPU.
#
# So: kill the producers first, then wait for their orphans to drain.
# The queue deletes E0 in this window, which is why E0 was hardlinked out of its
# reach before this script started.  A partial snapshot it began is removed
# below; those artifacts are rebuilt when code/stem are un-deferred.
if pgrep -f '[r]un_round169_overnight.sh' > /dev/null 2>&1; then
  say "stopping the phase 1 queue (code/stem deferred: a diverged configuration is not worth replicating)"
  pkill -f '[r]un_round169_overnight.sh' 2>/dev/null || true
fi
pkill -f '[r]ound169_row_snapshot' 2>/dev/null || true
pkill -f '[r]ound169_train_rows' 2>/dev/null || true
pkill -f '[r]ound169_eval_rows' 2>/dev/null || true
sleep 5
wait_quiet
for d in "$EXPL/code" "$EXPL/stem"; do
  [ -d "$d" ] && { say "  clearing partial snapshot $d"; rm -rf "$d"; }
done
rm -f "$OUT/OVERNIGHT_FAILED"
say "queue stopped; queue log tail:"; tail -3 "$LOGD/overnight-queue.log" 2>/dev/null

# ---- 3. make sure the frozen snapshot is in place --------------------------
if [ ! -s "$OUT/E0.npy" ]; then
  if [ -s "$E0_SAFE" ]; then
    say "restoring E0 from the hardlink (phase 1 recycled the name, not the data)"
    ln "$E0_SAFE" "$OUT/E0.npy" || cp "$E0_SAFE" "$OUT/E0.npy"
  else
    say "FAILED: no E0.npy and no $E0_SAFE -- every point needs the same frozen snapshot"
    echo "e0-missing" > "$FAIL_MARK"; echo "FAILED" > "$DONE_MARK"; exit 1
  fi
fi
say "E0 in place: $(stat -c '%s bytes nlink=%h' "$OUT/E0.npy")"
df -h /root/autodl-tmp | tail -1

# ---- 4. the ladder ---------------------------------------------------------
# Geometric midpoint of 1e-4 and 1e-3.  The 1e-4 arm already shows the frequent
# bands gaining while the rare bands lose slightly; whatever the aggregate does
# here decides whether a second, higher point is worth the GPU time.
D1=$(run_point 3.1622776601683795e-4 lr3.162e-4)
if ! is_number "$D1"; then
  fail "point-lr3.162e-4"
  say "lr=3.162e-4 produced no usable delta (got '$D1'); the ladder cannot continue"
else
  say "LADDER lr=3.162e-4  delta=$D1   (lr=1e-4 was $REF_DELTA; positive delta means training helped)"
  if better_than "$D1" "$REF_DELTA"; then
    # Non-monotone: raising lr bought something, so the peak is above 1e-4 and
    # the next point brackets it from the other side.
    say "curve is improving with lr -- climbing to the next point"
    D2=$(run_point 5.623413251903491e-4 lr5.623e-4)
    if is_number "$D2"; then
      say "LADDER lr=5.623e-4  delta=$D2"
    else
      fail "point-lr5.623e-4"
      say "lr=5.623e-4 produced no usable delta (got '$D2')"
    fi
  else
    say "delta did not improve on lr=1e-4, so the curve is monotone over this range:"
    say "the best point on the lr axis is at or below 1e-4, where it is still $REF_DELTA."
    say "climbing further would only add divergence damage; stopping the ladder here."
  fi
fi

# ---- 5. roll-up ------------------------------------------------------------
say "=== summary over every artifact present ==="
{ "$PY" scripts/round169_summarize.py --root "$OUT" --exploratory "$EXPL" \
    --out "$OUT/OVERNIGHT_SUMMARY.md"; rc=$?; } > "$LOGD/phase2-summary.log" 2>&1
step "summary rc=${rc:-?}"
[ "${rc:-1}" -ne 0 ] && fail "summary"

{
  echo "# Round 169 phase 2 roll-up (lr ladder)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase2.sh"
  echo
  echo "delta = frozen - trained; POSITIVE means training the rows helped."
  echo "Reference: lr=1e-4, $EPOCHS epochs -> $REF_DELTA (aggregate cost 1.5% of injection gain)."
  echo "lr=1e-3, $EPOCHS epochs -> -0.33127, but its mean_loss rose every epoch (3.055 -> 4.141): diverged."
  echo "lr=1e-3, 1 epoch    -> -0.26370: 80% of the damage is in the first pass."
  echo
  echo "## Ladder"
  for tag in lr1e-4 lr3.162e-4 lr5.623e-4; do
    j=$OUT/eval-real-$tag.json
    if [ -s "$j" ]; then
      echo "* $tag delta=$(delta_of "$j" || echo '?')  $( [ -s "$OUT/E-real-e4-$tag.npy" ] && stat -c 'bank=%s bytes' "$OUT/E-real-e4-$tag.npy" )"
    fi
  done
  echo
  echo "## Steps"
  if [ -s "$STEPSFILE" ]; then sed 's/^/* /' "$STEPSFILE"; else echo "* (none recorded)"; fi
  echo
  echo "## Failures"
  if [ "$(nfail)" = "0" ]; then echo "* none"; else sort -u "$FAILFILE" | sed 's/^/* /'; fi
  echo
  echo "## Result files present"
  for f in "$OUT"/eval-*.json; do [ -s "$f" ] && echo "* $f"; done
} > "$OUT/PHASE2_ROLLUP.md" 2>/dev/null

say "ROLLUP:"; head -40 "$OUT/PHASE2_ROLLUP.md" 2>/dev/null

if [ "$(nfail)" = "0" ]; then
  echo "OK" > "$DONE_MARK"
  say "*** PHASE 2 DONE (all steps ok) ***"
else
  sort -u "$FAILFILE" > "$FAIL_MARK"
  echo "OK_WITH_FAILURES" > "$DONE_MARK"
  say "*** PHASE 2 DONE with $(nfail) failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
fi
say "the local collector pulls artifacts and then shuts the box down"
exit 0
