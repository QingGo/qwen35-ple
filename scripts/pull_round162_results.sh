#!/usr/bin/env bash
# Pull the round-162 arm artifacts from the AutoDL box and run the analysis.
#
# Only the small JSON/text artifacts are pulled; reader checkpoints (~200 MB
# each) stay on the remote box where `audit_reader_checkpoints.py` reads them.
#
# Usage:
#   bash scripts/pull_round162_results.sh                 # 0.8B
#   BACKBONE=4B bash scripts/pull_round162_results.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE="${BACKBONE:-0.8B}"
SUFFIX="${SUFFIX:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/qwen35-ple}"
LOCAL_DIR="${LOCAL_DIR:-$REPO/outputs/round162}"
REMOTE_DIR="$REMOTE_ROOT/outputs/round162-${BACKBONE}${SUFFIX}"

mkdir -p "$LOCAL_DIR"
echo "[r162-pull] $REMOTE_DIR -> $LOCAL_DIR"
QWEN35_RSYNC=1 bash "$REPO/scripts/ssh_autodl.sh" -a \
  --include='arm-*.json' --include='crosseval-*.json' \
  --include='stats*.jsonl' --include='timings.txt' \
  --exclude='*' \
  "root@${QWEN35_HOST:-connect.nmb1.seetacloud.com}:$REMOTE_DIR/" "$LOCAL_DIR/"

shopt -s nullglob
ARMS=()
for f in "$LOCAL_DIR"/arm-wiki.json "$LOCAL_DIR"/arm-code.json "$LOCAL_DIR"/arm-stem.json \
         "$LOCAL_DIR"/arm-wiki-shuf.json "$LOCAL_DIR"/arm-ple-off.json; do
  [[ -f "$f" ]] && ARMS+=("--arm" "$(basename "$f" .json | sed 's/^arm-//')=$f")
done
# any extra arms (e.g. corpus-only variant pulled into a subdirectory)
for f in "$LOCAL_DIR"/arm-*.json; do
  base="$(basename "$f" .json | sed 's/^arm-//')"
  case "$base" in wiki|code|stem|wiki-shuf|ple-off) continue ;; esac
  ARMS+=("--arm" "$base=$f")
done

if [[ ${#ARMS[@]} -eq 0 ]]; then
  echo "[r162-pull] no arm JSONs found yet"
  exit 1
fi

echo "[r162-pull] analysing ${#ARMS[@]} arms"
python3 "$REPO/scripts/analyze_round162.py" "${ARMS[@]}" \
  --out "$LOCAL_DIR/analysis.json" \
  --markdown "$LOCAL_DIR/analysis.md"
