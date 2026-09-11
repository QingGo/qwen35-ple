#!/usr/bin/env bash
# Round-162 finisher: wait for the experiment queues, verify the artifacts, then
# stop the instance.
#
# Why this file exists rather than an ad-hoc shutdown: the instance bills by the
# hour and nothing else will stop it.  Round 155 taught the two failure modes
# that matter here, and both are handled below:
#
#   * a background waiter started from tmux is killed by SIGHUP the moment the
#     session that started it is torn down, so it dies before writing its first
#     log line and the instance runs on.  Everything here is launched with
#     `setsid nohup` and its own log.
#   * waiting on a marker with no liveness check burns hours next to a dead
#     producer.  If no queue process is alive and the marker is missing, waiting
#     longer cannot help, so stop waiting.
#
# Exit criteria are the artifacts, not the markers.
set -uo pipefail
ROOT=${ROOT:-/root/autodl-tmp/qwen35-ple}
LOG=$ROOT/logs/round162-finish.log
FOLLOW_LOG=$ROOT/logs/round162-followup.log
MANIFEST=$ROOT/outputs/round162-FINISH-MANIFEST.txt
WAIT_MINUTES=${WAIT_MINUTES:-330}
SHUTDOWN_ON_FAILURE=${SHUTDOWN_ON_FAILURE:-1}

log() { echo "=== [r162-finish] $* $(date -Is) ===" >>"$LOG"; }

producer_alive() {
  pgrep -f '[r]ound162_followup\.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_round162_format_prior\.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_round162_reader_crosseval\.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_round162_gold_rerun\.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_phase0\.py' >/dev/null 2>&1 && return 0
  return 1
}

log "started; waiting up to ${WAIT_MINUTES}m for the follow-up queue"
waited=0
while (( waited < WAIT_MINUTES )); do
  if grep -q "follow-up queue DONE" "$FOLLOW_LOG" 2>/dev/null; then
    log "follow-up DONE marker seen after ${waited}m"
    break
  fi
  if ! producer_alive; then
    log "no queue process is alive and no DONE marker after ${waited}m; stop waiting"
    break
  fi
  sleep 60
  waited=$((waited + 1))
done
log "wait phase over after ${waited}m"

# --- artifact validation ---------------------------------------------------
# Checked rather than assumed: each list must be complete for its group to count.
check_group() {
  local label="$1" dir="$2" ; shift 2
  local missing=() present=0
  for arm in "$@"; do
    if [[ -s "$dir/arm-$arm.json" ]]; then present=$((present + 1)); else missing+=("arm-$arm.json"); fi
  done
  printf '%s\t%s\t%d/%d\t%s\n' "$label" "$dir" "$present" "$#" "${missing[*]:-none}" >>"$MANIFEST"
  [[ ${#missing[@]} -eq 0 ]]
}

: >"$MANIFEST"
{
  echo "round-162 finish manifest  $(date -Is)"
  echo "group	directory	present	missing"
} >>"$MANIFEST"

ok=1
check_group "0.8B"     "$ROOT/outputs/round162-0.8B"       wiki code stem wiki-shuf ple-off no-reader || ok=0
check_group "4B"       "$ROOT/outputs/round162-4B"         wiki code stem wiki-shuf ple-off no-reader || ok=0
check_group "0.8B-nosft" "$ROOT/outputs/round162-0.8B-nosft" wiki code stem wiki-shuf ple-off no-reader || ok=0

for extra in "$ROOT/outputs/round162-0.8B/contribution-similarity.json"; do
  if [[ -s "$extra" ]]; then
    echo "extra	$extra	present" >>"$MANIFEST"
  else
    echo "extra	$extra	MISSING" >>"$MANIFEST"
    ok=0
  fi
done

# The gold-NLL column is only trustworthy if every arm scored every item; the
# f1685cf bug produced qa_n = 4 for a 1500-item eval and looked like a number.
for f in "$ROOT"/outputs/round162-*/*.json; do
  [[ -s "$f" ]] || continue
  "$ROOT/venv/bin/python" - "$f" >>"$MANIFEST" 2>/dev/null <<'PY' || true
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as exc:                      # noqa: BLE001
    print(f"unreadable\t{p}\t{exc}")
else:
    rows = d.get("results") or [{}]
    gold = (rows[0].get("qa_gold") or {}).get("metrics") or {}
    ex = (rows[0].get("qa_exact") or {}).get("metrics") or {}
    print(f"counts\t{p}\tgold_n={gold.get('qa_n')}\tem_n={ex.get('qa_n')}")
PY
done

log "manifest written to $MANIFEST"

if [[ "$ok" != "1" ]]; then
  log "ARTIFACT CHECK FAILED: at least one expected arm is missing"
fi

if [[ "${FINISH_NO_SHUTDOWN:-0}" == "1" ]]; then
  log "FINISH_NO_SHUTDOWN=1: leaving the instance running"
  exit 0
fi

if [[ "$ok" == "1" ]]; then
  log "ALL EXPECTED ARTIFACTS PRESENT; shutting down"
elif [[ "$SHUTDOWN_ON_FAILURE" == "1" ]] && ! producer_alive; then
  # Unattended: nobody is coming to read a refusal, and the disk persists, so
  # stop paying and leave the manifest as the record.
  log "artifacts incomplete but no producer is alive; shutting down anyway (recorded)"
else
  log "artifacts incomplete and a producer is still alive; NOT shutting down"
  exit 1
fi

sync
shutdown -h now
