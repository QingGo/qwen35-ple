#!/usr/bin/env bash
# Round 169 phase 3 -- re-score the two already-known lr points so their band
# tables can be corrected, then re-band every point that has a record.
#
# WHY.  B1's published band table stratifies by ``context_counts`` from the
# round-168 margin artifact, and that variable turned out to be one token behind
# the row it claims to describe: measured on the exact scored set,
# ``context_counts[t]`` equals the training count of the trigram ending at
# ``t - 1`` in 297,998 / 297,998 non-zero cases, while the row injected at ``t``
# is the trigram ending at ``t`` (see
# docs/round-169-band-variable-off-by-one.md).  The PRIMARY numbers -- delta, R,
# t -- do not band anything and are untouched.  The band table is not, and it is
# the table that carries the mechanism claim.
#
# The correction needs no retraining: ``trigram-train-count.npy`` holds the exact
# count of every snapshot row's trigram, and the per-position record carries
# ``snapshot_index``.  It does need per-position deltas, which the lr=1e-3 and
# lr=1e-4 evaluations predate -- hence two re-scores at ~11 min each.
#
# This script deliberately writes to NEW output names.  The original
# eval-real.json / eval-real-lr1e-4.json are the record of what was measured
# first and are never overwritten.
#
# Guards are the same four as phases 1 and 2.
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

DONE_MARK=$OUT/PHASE3_DONE
FAIL_MARK=$OUT/PHASE3_FAILED
START_MARK=$OUT/PHASE3_STARTED
PHASE2_DONE=$OUT/PHASE2_DONE

FAILED=()
STEPS=()

say() { echo "[$(date +%H:%M:%S)] $*"; }
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
  local log=$LOGD/phase3-$name.log t0 t1 rc
  wait_mem; wait_quiet
  say "START $name  (mem $(mem_gb) GB, gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')%)"
  t0=$(date +%s)
  { echo "=== $name start $(date -Is) ==="; "$@"; rc=$?; echo "=== $name rc=$rc ==="; } > "$log" 2>&1
  t1=$(date +%s)
  say "END   $name rc=$rc wall=$((t1 - t0))s -> $log"
  STEPS+=("$name rc=$rc wall=$((t1 - t0))s")
  [ "$rc" -ne 0 ] && FAILED+=("$name")
  return 0
}

rescore() {  # bank out
  "$PY" scripts/round169_eval_rows.py \
    --snapshot-dir "$OUT" --trained-bank "$1" \
    --keep-index "$OUT/keep-wiki-eval.npy" \
    --tokens data/phase1/wikitext-heldout-decon/tokens.npy \
    --positions outputs/round168/rung3/positions-wiki.npy \
    --counts-npz "$MARGIN/wiki-counts.npz" \
    --rows-dir "$ROWS" --model "$MODEL" \
    --reader-kind saved --reader "$READER" --layer 2 --chunk 1024 --out "$2"
}

reband() {  # record tag
  "$PY" scripts/round169_reband.py --record "$1" --snapshot-dir "$OUT" --tag "$2" \
    --out-json "$OUT/reband-$2.json" --out-md "$LOGD/reband-$2.md"
}

finish() {  # rc
  local rc=$1
  {
    echo "# Round 169 phase 3 roll-up (band-variable correction)"
    echo
    echo "Generated $(date -Is) by scripts/run_round169_phase3.sh"
    echo
    echo "The band variable in the published table is one token behind the injected row;"
    echo "the primary delta/R are not banded and are unaffected."
    echo "See docs/round-169-band-variable-off-by-one.md"
    echo
    echo "## Steps"
    for s in "${STEPS[@]}"; do echo "* $s"; done
    echo
    echo "## Failures"
    if [ ${#FAILED[@]} -eq 0 ]; then echo "* none"; else for f in "${FAILED[@]}"; do echo "* $f"; done; fi
    echo
    echo "## Per-position records present"
    for f in "$OUT"/*.deltas.npz; do [ -s "$f" ] && echo "* $f"; done
  } > "$OUT/PHASE3_ROLLUP.md" 2>/dev/null
  if [ ${#FAILED[@]} -eq 0 ]; then
    echo "OK" > "$DONE_MARK"
    say "*** PHASE 3 DONE (all steps ok) ***"
  else
    printf '%s\n' "${FAILED[@]}" > "$FAIL_MARK"
    echo "OK_WITH_FAILURES" > "$DONE_MARK"
    say "*** PHASE 3 DONE with ${#FAILED[@]} failure(s): ${FAILED[*]} ***"
  fi
  say "the local collector pulls artifacts and then shuts the box down"
  exit "$rc"
}

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK"
echo "started $(date -Is)" > "$START_MARK"
say "=== phase 3 (band-variable correction) starting ==="

# ---- 1. wait for the ladder -------------------------------------------------
# Wait for the marker, or for phase 2 to disappear without writing one.  A plain
# iteration cap cannot express this: the ladder legitimately runs for hours, so
# any cap short enough to be a safety net would fire while it is still working,
# and any cap long enough to cover the ladder would not be a safety net at all.
n=0
while [ ! -s "$PHASE2_DONE" ]; do
  n=$((n + 1))
  if [ "$n" -gt 4 ] && ! pgrep -f '[r]un_round169_phase2.sh' > /dev/null 2>&1; then
    say "phase 2 is gone without writing its marker; correcting whatever exists"
    STEPS+=("phase2-died-without-marker")
    FAILED+=("phase2-died-without-marker")
    break
  fi
  if [ $((n % 20)) -eq 1 ]; then
    say "  waiting for PHASE2_DONE (in-flight: $(pgrep -cf '[r]ound169_(train|eval)_rows' || echo 0))"
  fi
  sleep 30
done
wait_quiet
say "ladder finished; $(tail -1 "$LOGD/phase2-queue.log" 2>/dev/null)"

# ---- 2. re-score the two points that predate the per-position record --------
for spec in "lr1e-3:E-real-e4-lr1e-3.npy:eval-real-lr1e-3-rescored.json" \
            "lr1e-4:E-real-e4-lr1e-4.npy:eval-real-lr1e-4-rescored.json"; do
  tag=${spec%%:*}; rest=${spec#*:}; bank=${rest%%:*}; outj=${rest##*:}
  if [ ! -s "$OUT/$bank" ]; then
    say "SKIP $tag: $bank is gone (it cannot be rebuilt without another 80 min of GPU)"
    FAILED+=("bank-missing-$tag")
    continue
  fi
  if [ -s "$OUT/$outj" ]; then
    say "SKIP $tag: $outj already present"
  else
    run_gpu "rescore-$tag" rescore "$OUT/$bank" "$OUT/$outj"
  fi
done

# ---- 3. re-band every point that has a record -------------------------------
say "=== re-banding every point that has a per-position record ==="
shopt -s nullglob
for rec in "$OUT"/*.deltas.npz; do
  tag=$(basename "$rec" .deltas.npz)
  { reband "$rec" "$tag"; rc=$?; } > "$LOGD/phase3-reband-$tag.log" 2>&1
  STEPS+=("reband-$tag rc=${rc:-?}")
  [ "${rc:-1}" -ne 0 ] && FAILED+=("reband-$tag")
done
shopt -u nullglob

if [ -s "$LOGD/reband-lr3.162e-4.md" ]; then
  say "=== corrected band table (lr=3.162e-4) ==="
  head -30 "$LOGD/reband-lr3.162e-4.md"
fi
df -h /root/autodl-tmp | tail -1

finish 0
