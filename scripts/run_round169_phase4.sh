#!/usr/bin/env bash
# Round 169 phase 4 -- produce the per-position records the band correction needs.
#
# WHY THIS EXISTS.  Phase 3 re-scored the two pre-record arms successfully (both
# rc=0) and then had nothing to re-band: every arm's ``.deltas.npz`` was missing,
# and each JSON carried
#
#     "per_position_record_error": "IndexError: index 1152889 is out of bounds
#                                   for axis 0 with size 1152889"
#
# The record is written from ``codes[score]`` and ``j_all[score]``, but those
# arrays are indexed by TRIGRAM and have length T-2 -- ``codes[i]`` describes the
# trigram ending at ``i + 2``, which is where the snapshot's own ``+ 2`` comes
# from.  ``context_count`` and ``rowid`` are indexed by stream position and do
# not take the offset.  One array in the tuple needed ``- 2`` and the other two
# did not, and the try/except that protects the primary result (correctly)
# turned the IndexError into an empty artifact instead of a crash.
#
# So the expensive numbers were never at risk and are all present; what is
# missing is only the artifact that answers "which band".  This phase re-runs the
# four arms whose band tables we quote, with the offset fixed, and then re-bands
# them.
#
# Nothing is overwritten unless it is already broken: the two ``-rescored`` JSONs
# from phase 3 exist only as failures, so they are legitimately replaced.  The
# ladder point's original JSON is left alone and a ``-rescored`` name is used.
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

DONE_MARK=$OUT/PHASE4_DONE
FAIL_MARK=$OUT/PHASE4_FAILED
START_MARK=$OUT/PHASE4_STARTED
QUEUE_LOG=${QUEUE_LOG:-$LOGD/phase4-queue.log}
FAILFILE=$OUT/.phase4-failures
STEPSFILE=$OUT/.phase4-steps

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
  local log=$LOGD/phase4-$name.log t0 t1 rc
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

##############################################################################
cd "$REPO" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOGD"
rm -f "$DONE_MARK" "$FAIL_MARK" "$FAILFILE" "$STEPSFILE"
echo "started $(date -Is)" > "$START_MARK"
say "=== phase 4 (per-position records) starting ==="

# tag : bank : out-json.  The four band tables this project quotes.
SPECS=(
  "lr1e-3:E-real-e4-lr1e-3.npy:eval-real-lr1e-3-rescored.json"
  "lr1e-4:E-real-e4-lr1e-4.npy:eval-real-lr1e-4-rescored.json"
  "lr3.162e-4:E-real-e4-lr3.162e-4.npy:eval-real-lr3.162e-4-rescored.json"
  "ep1-lr1e-3:E-real-e1-lr1e-3.npy:eval-real-e1-lr1e-3-rescored.json"
)

for spec in "${SPECS[@]}"; do
  tag=${spec%%:*}; rest=${spec#*:}; bank=${rest%%:*}; outj=${rest##*:}
  rec="${outj%.json}.deltas.npz"
  if [ ! -s "$OUT/$bank" ]; then
    say "SKIP $tag: $bank is absent"
    fail "bank-missing-$tag"
    continue
  fi
  # A record that exists AND is non-empty AND whose JSON carries no error key is
  # already good; anything else is re-run.  This makes the phase restartable.
  if [ -s "$OUT/$rec" ] && ! grep -q per_position_record_error "$OUT/$outj" 2>/dev/null; then
    say "SKIP $tag: $rec already present and clean"
    step "$tag already-good"
    continue
  fi
  run_gpu "rescore-$tag" rescore "$OUT/$bank" "$OUT/$outj"
done

say "=== re-banding every arm that has a per-position record ==="
shopt -s nullglob
for rec in "$OUT"/*.deltas.npz; do
  tag=$(basename "$rec" .deltas.npz)
  { reband "$rec" "$tag"; rc=$?; } > "$LOGD/phase4-reband-$tag.log" 2>&1
  step "reband-$tag rc=${rc:-?}"
  [ "${rc:-1}" -ne 0 ] && fail "reband-$tag"
done
shopt -u nullglob

# A losing arm whose record still failed is the thing this phase exists to catch,
# so say so explicitly rather than letting an empty glob look like success.
nrec=$(ls "$OUT"/*.deltas.npz 2>/dev/null | wc -l | tr -d ' ')
say "per-position records produced: $nrec"
[ "$nrec" -eq 0 ] && fail "no-records-produced"

df -h /root/autodl-tmp | tail -1

{
  echo "# Round 169 phase 4 roll-up (per-position records)"
  echo
  echo "Generated $(date -Is) by scripts/run_round169_phase4.sh"
  echo
  echo "Phase 3 produced no record because the record indexed per-trigram arrays"
  echo "with stream positions; see docs/round-169-band-variable-off-by-one.md"
  echo
  echo "## Steps"
  if [ -s "$STEPSFILE" ]; then sed 's/^/* /' "$STEPSFILE"; else echo "* (none)"; fi
  echo
  echo "## Failures"
  if [ "$(nfail)" = "0" ]; then echo "* none"; else sort -u "$FAILFILE" | sed 's/^/* /'; fi
  echo
  echo "## Records"
  for f in "$OUT"/*.deltas.npz; do [ -s "$f" ] && echo "* $(basename "$f") ($(stat -c %s "$f") bytes)"; done
  echo
  echo "## Corrected band tables"
  for f in "$OUT"/reband-*.json; do [ -s "$f" ] && echo "* $(basename "$f")"; done
} > "$OUT/PHASE4_ROLLUP.md" 2>/dev/null

say "ROLLUP:"; head -30 "$OUT/PHASE4_ROLLUP.md" 2>/dev/null

if [ "$(nfail)" = "0" ]; then
  echo "OK" > "$DONE_MARK"; say "*** PHASE 4 DONE (all steps ok) ***"
else
  sort -u "$FAILFILE" > "$FAIL_MARK"
  echo "OK_WITH_FAILURES" > "$DONE_MARK"
  say "*** PHASE 4 DONE with $(nfail) failure(s): $(sort -u "$FAILFILE" | tr '\n' ' ') ***"
fi
exit 0
