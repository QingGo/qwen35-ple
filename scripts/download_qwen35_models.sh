#!/usr/bin/env bash
# Download Qwen3.5-4B and Qwen3.5-2B into the persistent model directory.
#
# Both models share the Qwen3.8-Flash-Next tokenizer (vocab 248320).
# Qwen3.5-4B has hidden_size=2560, exactly matching the frozen PLE source
# space; Qwen3.5-2B (hidden 2048) is the intermediate scale point.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/qwen35-ple}"
REPO="${REPO:-$ROOT/repo}"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/logs/download-qwen35-models.log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
mkdir -p "$MODELS_DIR" "$ROOT/logs"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true

log() { echo "=== [qwen35-download] $* $(date -Is) ===" | tee -a "$LOG"; }

check_space() {
  local required_gb="${1:-20}"
  local avail_kb
  avail_kb="$(df -Pk "$MODELS_DIR" | awk 'NR==2 {print $4}')"
  if (( avail_kb < required_gb * 1024 * 1024 )); then
    log "ERROR: need ${required_gb} GiB free under $MODELS_DIR, have $((avail_kb / 1024 / 1024)) GiB"
    exit 1
  fi
}

download_model() {
  # download_model <repo> <local_dir> <min_free_gb>
  local repo="$1" dir="$2" min_free="$3"
  if [[ -f "$dir/config.json" ]] && compgen -G "$dir/*.safetensors" >/dev/null; then
    log "skip $repo (already present at $dir)"
    return 0
  fi
  check_space "$min_free"
  log "download $repo -> $dir (ModelScope first, HF fallback)"
  "$PY" "$REPO/scripts/download_modelscope_model.py" \
    --model "$repo" --local-dir "$dir" --max-workers 3 >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "ERROR: download $repo failed rc=$rc"
    exit 1
  fi
  if [[ ! -f "$dir/config.json" ]]; then
    log "ERROR: $dir/config.json missing after download"
    exit 1
  fi
  log "done $repo ($(du -sh "$dir" | awk '{print $1}'))"
}

check_space 16
download_model "Qwen/Qwen3.5-4B" "$MODELS_DIR/Qwen3.5-4B" 10
download_model "Qwen/Qwen3.5-2B" "$MODELS_DIR/Qwen3.5-2B" 5
log "ALL DOWNLOADS DONE"
