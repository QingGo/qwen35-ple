#!/usr/bin/env bash
# Build/verify a Store-P view for qwen35-ple using the EngramDB CLI.
#
# Usage:
#   scripts/build_table_assets.sh [ROWS_DIR] [N_GRAMS] [OUT_DIR] [VIEW_NAME]
#
# Environment:
#   ENGRAMDB_CLI  path to engramdb CLI (default: ../EngramDB/target/{release,debug}/engramdb)
#   SLOT_BYTES    view slot width (default 2560; use 4096 for page-aligned inference)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENGRAMDB_CLI="${ENGRAMDB_CLI:-}"
if [[ -z "$ENGRAMDB_CLI" ]]; then
  for candidate in \
    "$REPO_ROOT/../EngramDB/target/release/engramdb" \
    "$REPO_ROOT/../EngramDB/target/debug/engramdb"; do
    if [[ -x "$candidate" ]]; then
      ENGRAMDB_CLI="$candidate"
      break
    fi
  done
fi
if [[ -z "$ENGRAMDB_CLI" || ! -x "$ENGRAMDB_CLI" ]]; then
  echo "error: engramdb CLI not found; set ENGRAMDB_CLI or build EngramDB first" >&2
  exit 1
fi

ROWS_DIR="${1:-data/rows}"
N_GRAMS="${2:-20000}"
OUT_DIR="${3:-data/views}"
VIEW_NAME="${4:-qwen-ple.view.bin}"
SLOT_BYTES="${SLOT_BYTES:-2560}"
VIEW_FILE="$OUT_DIR/$VIEW_NAME"
KEYS_FILE="$OUT_DIR/${VIEW_NAME%.bin}.keys.txt"

mkdir -p "$OUT_DIR"

echo "building Store-P view:"
echo "  rows_dir = $ROWS_DIR"
echo "  n_grams  = $N_GRAMS"
echo "  slot     = $SLOT_BYTES"
echo "  output   = $VIEW_FILE"

"$ENGRAMDB_CLI" view build \
  "$ROWS_DIR" "$N_GRAMS" "$VIEW_FILE" "$KEYS_FILE" \
  --slot "$SLOT_BYTES" \
  --verify

echo "verifying manifest:"
python3 - "$OUT_DIR/${VIEW_NAME%.bin}.manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert manifest["grans"] > 0
assert manifest["heads"] == 16
assert manifest["slot_bytes"] in (2560, 4096), manifest["slot_bytes"]
assert manifest["rows"] == manifest["grans"] * manifest["heads"]
assert manifest["record_bytes"] == 2560
print(f"OK: grans={manifest['grans']} slot={manifest['slot_bytes']} rows={manifest['rows']} source={manifest.get('source')}")
PY
