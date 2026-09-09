#!/usr/bin/env bash
# Ensure a valid EngramDB Store-I qwen38-rows directory exists.
#
# Resolution order:
#   1. requested --rows-dir is already valid -> use it;
#   2. --persistent-dir is valid -> copy it into --rows-dir (e.g. /dev/shm);
#   3. neither is valid -> extract from ModelScope into --persistent-dir,
#      then copy into --rows-dir.
#
# This is the post-reboot path after the AutoDL data disk is expanded:
# keep the canonical 51.2GB rows on the persistent data disk, and optionally
# materialize them in /dev/shm for fast training/QA reads.
#
# Usage:
#   bash scripts/ensure_qwen38_rows.sh \
#     --rows-dir /dev/shm/qwen38-rows \
#     --persistent-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
#     --download-dir /root/autodl-tmp/qwen35-ple/tmp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROWS_DIR="/dev/shm/qwen38-rows"
PERSISTENT_DIR="/root/autodl-tmp/qwen35-ple/qwen38-rows"
DOWNLOAD_DIR="/root/autodl-tmp/qwen35-ple/tmp"
PYTHON="${PYTHON:-python3}"
EXPECTED_SHARDS=128
SHARD_BYTES=400001920
FORCE_EXTRACT=0
KEEP_SOURCE=0
WRITE_MANIFEST=1
MIN_FREE_GB=58

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows-dir) ROWS_DIR="$2"; shift 2 ;;
    --persistent-dir) PERSISTENT_DIR="$2"; shift 2 ;;
    --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --expected-shards) EXPECTED_SHARDS="$2"; shift 2 ;;
    --shard-bytes) SHARD_BYTES="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --force-extract) FORCE_EXTRACT=1; shift ;;
    --keep-source) KEEP_SOURCE=1; shift ;;
    --no-manifest) WRITE_MANIFEST=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

VERIFY_SCRIPT="$REPO_ROOT/scripts/verify_qwen38_rows.py"
EXTRACT_SCRIPT="$REPO_ROOT/scripts/download_qwen38_fp8_rows.py"
if [[ ! -f "$VERIFY_SCRIPT" ]]; then
  echo "[ensure-rows] missing $VERIFY_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$EXTRACT_SCRIPT" ]]; then
  echo "[ensure-rows] missing $EXTRACT_SCRIPT" >&2
  exit 1
fi

verify_dir() {
  local dir="$1"
  "$PYTHON" "$VERIFY_SCRIPT" \
    --rows-dir "$dir" \
    --expected-shards "$EXPECTED_SHARDS" \
    --shard-bytes "$SHARD_BYTES" \
    --quiet >/dev/null 2>&1
}

write_manifest() {
  local dir="$1"
  if [[ "$WRITE_MANIFEST" != "1" ]]; then
    return 0
  fi
  mkdir -p "$dir"
  "$PYTHON" "$VERIFY_SCRIPT" \
    --rows-dir "$dir" \
    --expected-shards "$EXPECTED_SHARDS" \
    --shard-bytes "$SHARD_BYTES" \
    --write-manifest "$dir/manifest.json" \
    --quiet >/dev/null
}

copy_rows() {
  local src="$1"
  local dst="$2"
  if [[ "$src" == "$dst" ]]; then
    return 0
  fi
  echo "[ensure-rows] copying rows: $src -> $dst"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --progress "$src/" "$dst/"
  else
    cp -a "$src/." "$dst/"
  fi
}

check_free_space() {
  local dir="$1"
  local label="$2"
  mkdir -p "$dir"
  local avail_kb
  avail_kb="$(df -Pk "$dir" | awk 'NR==2 {print $4}')"
  local required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if (( avail_kb < required_kb )); then
    echo "[ensure-rows] ERROR: $label has only $((avail_kb / 1024 / 1024)) GiB free;" >&2
    echo "[ensure-rows] need at least ${MIN_FREE_GB} GiB for extraction/copy" >&2
    exit 1
  fi
}

echo "[ensure-rows] rows_dir=$ROWS_DIR"
echo "[ensure-rows] persistent_dir=$PERSISTENT_DIR"
echo "[ensure-rows] download_dir=$DOWNLOAD_DIR"

if [[ "$FORCE_EXTRACT" != "1" ]] && verify_dir "$ROWS_DIR"; then
  echo "[ensure-rows] requested rows dir is valid: $ROWS_DIR"
  write_manifest "$ROWS_DIR"
  echo "[ensure-rows] ready"
  exit 0
fi

if [[ "$FORCE_EXTRACT" != "1" && "$ROWS_DIR" != "$PERSISTENT_DIR" ]] \
  && verify_dir "$PERSISTENT_DIR"; then
  check_free_space "$ROWS_DIR" "rows dir"
  copy_rows "$PERSISTENT_DIR" "$ROWS_DIR"
  verify_dir "$ROWS_DIR"
  write_manifest "$ROWS_DIR"
  echo "[ensure-rows] ready (copied from persistent dir)"
  exit 0
fi

echo "[ensure-rows] no valid rows found; extracting from ModelScope"
check_free_space "$PERSISTENT_DIR" "persistent dir"
mkdir -p "$DOWNLOAD_DIR"

EXTRACT_ARGS=(
  --out-dir "$PERSISTENT_DIR"
  --download-dir "$DOWNLOAD_DIR"
  --expected-shards "$EXPECTED_SHARDS"
)
if [[ "$KEEP_SOURCE" == "1" ]]; then
  EXTRACT_ARGS+=(--keep-source)
fi
"$PYTHON" "$EXTRACT_SCRIPT" "${EXTRACT_ARGS[@]}"

if ! verify_dir "$PERSISTENT_DIR"; then
  echo "[ensure-rows] ERROR: extracted persistent rows failed verification" >&2
  "$PYTHON" "$VERIFY_SCRIPT" \
    --rows-dir "$PERSISTENT_DIR" \
    --expected-shards "$EXPECTED_SHARDS" \
    --shard-bytes "$SHARD_BYTES" >&2 || true
  exit 1
fi
write_manifest "$PERSISTENT_DIR"

if [[ "$ROWS_DIR" != "$PERSISTENT_DIR" ]]; then
  check_free_space "$ROWS_DIR" "rows dir"
  copy_rows "$PERSISTENT_DIR" "$ROWS_DIR"
  if ! verify_dir "$ROWS_DIR"; then
    echo "[ensure-rows] ERROR: copied rows failed verification: $ROWS_DIR" >&2
    exit 1
  fi
  write_manifest "$ROWS_DIR"
fi

echo "[ensure-rows] ready"
