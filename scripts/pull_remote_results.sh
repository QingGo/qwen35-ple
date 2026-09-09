#!/usr/bin/env bash
# Pull remote Phase 2 diagnostic results back to the local Mac.
#
# Usage:
#   bash scripts/pull_remote_results.sh \
#     --host root@connect.nmb1.seetacloud.com \
#     --port 19236 \
#     --remote-dir /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic \
#     --local-dir artifacts/phase2-diagnostic
set -euo pipefail

HOST="${HOST:-root@connect.nmb1.seetacloud.com}"
PORT="${PORT:-19236}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic}"
LOCAL_DIR="${LOCAL_DIR:-artifacts/phase2-diagnostic}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --local-dir) LOCAL_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOCAL_DIR"
echo "[pull-results] $HOST:$REMOTE_DIR -> $LOCAL_DIR"
if command -v rsync >/dev/null 2>&1; then
  rsync -av --progress -e "ssh -p $PORT $SSH_OPTS" \
    "$HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
else
  scp -P "$PORT" $SSH_OPTS -r "$HOST:$REMOTE_DIR/." "$LOCAL_DIR/"
fi
echo "[pull-results] done"
