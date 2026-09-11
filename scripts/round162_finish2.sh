#!/usr/bin/env bash
# Wait for the round-162 follow-up #2 queue, PULL the artifacts, then shut the
# instance down.
#
# Why this is one script rather than "poll from the shell, then pull, then
# shut down": on 2026-09-12 the remote finisher shut the instance down at 01:46
# while two 4B arms had finished but had not been pulled.  They survived on
# /root/autodl-tmp, but only by luck -- the ordering guarantee we actually want
# is "artifacts are on the local disk BEFORE the box goes away", and that is
# only enforceable if the pull and the shutdown are in the same process.
#
# Usage: setsid nohup bash scripts/round162_finish2.sh > <log> 2>&1 < /dev/null &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"
REMOTE_ROOT=/root/autodl-tmp/qwen35-ple
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_POLLS=${MAX_POLLS:-90}          # 90 * 60s = 90 min ceiling

log() { echo "=== [r162-finish2] $* $(date -Is) ==="; }

remote() { bash "$HERE/ssh_autodl.sh" "$@"; }
pull() {
  QWEN35_RSYNC=1 bash "$HERE/ssh_autodl.sh" -az --exclude='*.pt' --exclude='*-partial*' \
    "root@connect.nmb1.seetacloud.com:$REMOTE_ROOT/outputs/$1/" "$REPO_DIR/outputs/$1/"
}

cd "$REPO_DIR"

# --- 1. wait for the producer ------------------------------------------------
polls=0
while (( polls < MAX_POLLS )); do
  last=$(remote "tail -1 $REMOTE_ROOT/logs/round162-followup2.log 2>/dev/null" 2>/dev/null | tr -d '\r')
  alive=$(remote "pgrep -cf '[r]ound162_followup2' 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r' | tail -1)
  log "poll=$polls alive=$alive last=${last:0:90}"

  if [[ "$last" == *"follow-up #2 DONE"* ]]; then
    log "DONE marker seen"
    break
  fi
  # A dead producer with no DONE marker cannot make progress; stop waiting.
  if [[ "$alive" == "0" ]]; then
    log "producer is gone and no DONE marker; stopping the wait"
    break
  fi
  sleep "$POLL_SECONDS"
  polls=$((polls + 1))
done

# --- 2. PULL BEFORE ANYTHING ELSE -------------------------------------------
log "pulling artifacts"
pull round162-4B                 || log "WARNING: 4B pull failed"
pull round162-0.8B-nosft600      || log "WARNING: nosft600 pull failed"
pull round162-0.8B-nosft         || log "WARNING: 0.8B-nosft pull failed"
pull round162-0.8B               || log "WARNING: 0.8B pull failed"

# --- 3. verify the pull landed ----------------------------------------------
for d in round162-4B round162-0.8B-nosft600; do
  n=$(ls outputs/$d/arm-*.json 2>/dev/null | wc -l | tr -d ' ')
  log "local outputs/$d arms=$n"
done

# The decisive artifact is the zero-injection arm of the no-SFT subset: without
# it the subset has no reference and the pull bought nothing.
if [[ ! -s outputs/round162-0.8B-nosft600/arm-no-reader.json ]]; then
  log "DECISIVE ARM MISSING (nosft600/arm-no-reader.json); NOT shutting down"
  log "leaving the instance up so the queue can be re-run or inspected"
  exit 1
fi
log "decisive arm present locally; safe to shut down"

# --- 4. shut the instance down ----------------------------------------------
remote "shutdown -h now" 2>/dev/null || true
log "shutdown issued"
