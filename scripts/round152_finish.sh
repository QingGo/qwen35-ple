#!/usr/bin/env bash
# Round-152 finisher: regenerate reports, then stop the instance ONLY when every
# expected experiment succeeded.
#
# History (why this script looks paranoid):
#   * v1 trusted the chain's DONE marker and shut the instance down even though
#     G0 had failed instantly -> wasted a night of GPU time.
#   * v2 gated on the G0 marker with a 12h `sleep 60` loop and no liveness check.
#     G0 *succeeded* but its artifact gate aborted on a summarizer bug, so the
#     marker was never written; v2 slept for hours next to a dead producer while
#     the instance billed.  That is the bug this version fixes: when the thing we
#     are waiting for is no longer running, waiting longer cannot help.
#
# Exit criteria are the artifact checks below, not any marker.
set -uo pipefail
ROOT=${ROOT:-/root/autodl-tmp/qwen35-ple}
PY=$ROOT/venv/bin/python
LOG=$ROOT/logs/round152-finish.log
WAIT_MINUTES=${WAIT_MINUTES:-90}
log() { echo "=== [finish] $* $(date -Is) ===" >>"$LOG"; }

# Is any producer that could still write the G0 marker alive?
producer_alive() {
  pgrep -f '[r]un_phase0\.py' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_round152_g0_4b\.sh' >/dev/null 2>&1 && return 0
  return 1
}

# Wait for the G0 marker, but give up as soon as the producer is gone.
waited=0
while (( waited < WAIT_MINUTES )); do
  if [[ -f "$ROOT/outputs/round152g0/DONE" ]]; then
    log "G0 marker seen after ${waited}m"
    break
  fi
  if ! producer_alive; then
    log "G0 producer is gone and no DONE marker exists after ${waited}m; stop waiting"
    break
  fi
  sleep 60
  waited=$((waited + 1))
done
[[ -f "$ROOT/outputs/round152g0/DONE" ]] || log "proceeding WITHOUT the G0 marker (timeout or dead producer)"

# Validate from the scripts' own DONE markers plus the artifacts themselves.  The
# chain's status files are not trustworthy: they were written with $? read after
# a helper call, so a failed run could be recorded as rc=0.
ok=1
[[ -f "$ROOT/outputs/round152/DONE" ]] || { log "REFUSING SHUTDOWN: lora row DONE marker missing"; ok=0; }
[[ -f "$ROOT/outputs/round152g0/DONE" ]] || { log "REFUSING SHUTDOWN: g0 DONE marker missing"; ok=0; }
for f in lora-nople lora-real lora-control; do
  [[ -s "$ROOT/outputs/round152/$f.json" ]] || { log "REFUSING SHUTDOWN: missing $f.json"; ok=0; }
done
for f in g0-real g0-control g0-nople; do
  [[ -s "$ROOT/outputs/round152g0/$f.json" ]] || { log "REFUSING SHUTDOWN: missing $f.json"; ok=0; }
done
[[ -s "$ROOT/outputs/round152/arms-summary.md" ]] || { log "REFUSING SHUTDOWN: missing LoRA summary"; ok=0; }
[[ -s "$ROOT/outputs/round152g0/gold-nll-raw.md" ]] || { log "REFUSING SHUTDOWN: missing G0 gold NLL summary"; ok=0; }
[[ -s "$ROOT/outputs/round152g0/arms-summary.md" ]] || { log "REFUSING SHUTDOWN: missing G0 arms summary"; ok=0; }

if [[ "$ok" != "1" ]]; then
  log "ARTIFACT CHECK FAILED: at least one expected artifact is missing or failed"
  mkdir -p "$ROOT/outputs/chain"
  echo "finish-blocked $(date -Is)" > "$ROOT/outputs/chain/FINISH_BLOCKED"
  # An unattended run has nobody to read a "refusing to shut down" message, and
  # the instance keeps billing while it waits for a human who is asleep.  When
  # the producer is already gone, no further artifact can appear, so the only
  # useful action left is to stop paying and leave a loud marker.  Set
  # SHUTDOWN_ON_FAILURE=0 when a human is actively debugging.
  if [[ "${SHUTDOWN_ON_FAILURE:-1}" == "1" ]] && ! producer_alive; then
    log "producer is dead and no further artifact can appear; SHUTTING DOWN ANYWAY (failure recorded)"
    sync
    shutdown -h now
  fi
  exit 1
fi

log "regenerating reports"
cd "$ROOT/repo" || exit 1
"$PY" "$ROOT/repo/scripts/bench_engram_edge.py" \
  --rows-dir "$ROOT/qwen38-rows" --label nvme-persistent --cold \
  --json "$ROOT/outputs/g3/nvme-persistent.json" \
  --markdown "$ROOT/outputs/g3/nvme-persistent.md" >>"$LOG" 2>&1
log "g3 nvme rc=$?"
"$PY" "$ROOT/repo/scripts/bench_engram_edge.py" \
  --rows-dir /dev/shm/qwen38-rows --label shm \
  --json "$ROOT/outputs/g3/shm.json" \
  --markdown "$ROOT/outputs/g3/shm.md" >>"$LOG" 2>&1
log "g3 shm rc=$?"

LORA_RUNS=()
for name in lora-nople lora-real lora-control; do
  f="$ROOT/outputs/round152/$name.json"
  [[ -f "$f" ]] && LORA_RUNS+=("--run" "$name=$f")
done
if (( ${#LORA_RUNS[@]} )); then
  "$PY" "$ROOT/repo/scripts/summarize_arms.py" "${LORA_RUNS[@]}" \
    --pair lora-real=lora-control --pair lora-real=lora-nople \
    --title "LoRA row: backbone LoRA x frozen PLE (generation + gold NLL)" \
    --output "$ROOT/outputs/round152/arms-summary.md" \
    --json-output "$ROOT/outputs/round152/arms-summary.json" >>"$LOG" 2>&1
  log "lora summary rc=$?"
fi

# G0: summarise EVERY arm that exists, so regenerating here cannot silently drop
# the g0-nople control that the round-152b conclusion depends on.
G0_RUNS=()
G0_PAIRS=("--pair" "g0-real=g0-control")
for name in g0-real g0-control g0-nople; do
  f="$ROOT/outputs/round152g0/$name.json"
  if [[ -f "$f" ]]; then
    G0_RUNS+=("--run" "$name=$f")
    [[ "$name" != "g0-real" ]] && G0_PAIRS+=("--pair" "g0-real=$name")
  fi
done
G0_FROZEN="$ROOT/outputs/round148a/nll-raw-no-reader.json"
G0_GOLD=("${G0_RUNS[@]}")
if [[ -s "$G0_FROZEN" ]]; then
  G0_GOLD+=("--run" "frozen-0.8b-no-reader=$G0_FROZEN")
fi
if (( ${#G0_RUNS[@]} )); then
  "$PY" "$ROOT/repo/scripts/summarize_arms.py" "${G0_RUNS[@]}" \
    "${G0_PAIRS[@]}" \
    --title "G0 scale/space: frozen Qwen3.5-4B (hidden 2560) + frozen PLE" \
    --output "$ROOT/outputs/round152g0/arms-summary.md" \
    --json-output "$ROOT/outputs/round152g0/arms-summary.json" >>"$LOG" 2>&1
  log "g0 summary rc=$?"
  if [[ -s "$G0_FROZEN" ]]; then
    "$PY" "$ROOT/repo/scripts/summarize_gold_nll.py" "${G0_GOLD[@]}" \
      --baseline frozen-0.8b-no-reader "${G0_PAIRS[@]}" \
      --output "$ROOT/outputs/round152g0/gold-nll-raw.md" \
      --title "G0 scale/space: frozen Qwen3.5-4B (hidden 2560) + frozen PLE" >>"$LOG" 2>&1
    log "g0 gold nll rc=$?"
  else
    log "skipping g0 gold NLL: frozen 0.8B reference $G0_FROZEN is missing"
  fi
fi

if [[ "${FINISH_NO_SHUTDOWN:-0}" == "1" ]]; then
  log "FINISH_NO_SHUTDOWN=1: reports regenerated, leaving the instance running"
  exit 0
fi

log "ALL ARTIFACTS PRESENT; SHUTTING DOWN INSTANCE"
sync
shutdown -h now
