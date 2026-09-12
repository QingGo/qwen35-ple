#!/usr/bin/env bash
# Wait for a remote queue, PULL its artifacts, then shut the instance down.
#
# Why the pull and the shutdown are one process
# ---------------------------------------------
# On 2026-09-12 the round-162 finisher shut the instance down at 01:46 while two
# 4B arms had finished but had never been pulled.  They survived on
# /root/autodl-tmp, but only because the data disk outlives a shutdown, and the
# report written in the meantime called them lost.  The ordering we actually
# want is "the artifacts are on the local disk BEFORE the box goes away", and
# that is only enforceable if the pull and the shutdown live in the same
# process -- a shell session that polls, then pulls, then shuts down by hand
# loses the race the moment anything goes wrong.
#
# It refuses to shut down unless every file in REQUIRED_FILES is present
# locally, so a truncated pull leaves the instance up for inspection instead of
# destroying the only copy.
#
# Usage: setsid nohup bash scripts/round162_finish2.sh > <log> 2>&1 < /dev/null &
#
# Environment:
#   PRODUCER_PATTERN  pgrep -f pattern for the queue process
#   DONE_MARKER/DONE_LOG  string meaning "finished", and the remote log to grep
#   PULL_DIRS         dirs under outputs/ to pull
#   REQUIRED_FILES    local paths that must exist before shutdown is allowed
#   MAX_POLLS/POLL_SECONDS  90 * 60s = 90 min ceiling by default
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"
REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/qwen35-ple}
REMOTE_HOST=${REMOTE_HOST:-root@connect.nmb1.seetacloud.com}

PRODUCER_PATTERN=${PRODUCER_PATTERN:-'[r]un_round162_format_prior'}
DONE_MARKER=${DONE_MARKER:-'queue DONE'}
DONE_LOG=${DONE_LOG:-$REMOTE_ROOT/logs/round162-0.8B-nosft600b.log}
PULL_DIRS=${PULL_DIRS:-'round162-0.8B-nosft600'}
REQUIRED_FILES=${REQUIRED_FILES:-'outputs/round162-0.8B-nosft600/arm-wiki-shuf.json outputs/round162-0.8B-nosft600/arm-ple-off.json'}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_POLLS=${MAX_POLLS:-90}

# `date -Is` is a GNU extension and errors on BSD/macOS, where this runs.
log() { echo "=== [r162-finish2] $* $(date +%Y-%m-%dT%H:%M:%S%z) ==="; }

remote() { bash "$HERE/ssh_autodl.sh" "$@"; }
pull() {
  QWEN35_RSYNC=1 bash "$HERE/ssh_autodl.sh" -az --exclude='*.pt' --exclude='*-partial*' \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$1/" "$REPO_DIR/outputs/$1/"
}

cd "$REPO_DIR"

# --- 1. wait for the producer ------------------------------------------------
polls=0
while (( polls < MAX_POLLS )); do
  last=$(remote "tail -1 $DONE_LOG 2>/dev/null" 2>/dev/null | tr -d '\r')
  alive=$(remote "pgrep -cf \"$PRODUCER_PATTERN\" 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r' | tail -1)
  log "poll=$polls alive=$alive last=${last:0:90}"

  if [[ "$last" == *"$DONE_MARKER"* ]]; then
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
for d in $PULL_DIRS; do
  pull "$d" || log "WARNING: pull of $d failed"
done

# --- 3. verify the pull landed ----------------------------------------------
missing=0
for f in $REQUIRED_FILES; do
  if [[ -s "$f" ]]; then
    log "present: $f ($(wc -c <"$f" | tr -d ' ') bytes)"
  else
    log "MISSING: $f"
    missing=1
  fi
done

if (( missing )); then
  log "required artifacts are NOT local; refusing to shut down"
  log "leaving the instance up so the queue can be re-run or inspected"
  exit 1
fi
log "all required artifacts are local; safe to shut down"

# --- 4. shut the instance down ----------------------------------------------
remote "shutdown -h now" 2>/dev/null || true
log "shutdown issued"
