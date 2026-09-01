#!/usr/bin/env bash
# WSL reproducible environment for qwen35-ple PLE experiments.
#
# This is the V131 starting point: it pins the package versions and records the
# paths used by the WSL benchmarks.  Re-run after a fresh WSL setup to validate
# that Store-P / SlotIndex / live-store paths still work.
#
# Usage:
#   bash scripts/wsl_repro.sh [--full]
#
# --full also runs the qwen35-ple test suite and the access-order benchmark
# smoke (when a view is available).

set -euo pipefail

# --- WSL-specific paths (adjust if your layout differs) -----------------------
QWEEN_DIR="${QWEEN_DIR:-/home/zeng/qwen35-ple}"
ENGRAMDB_DIR="${ENGRAMDB_DIR:-/home/zeng/EngramDB}"
ROWS_DIR="${ROWS_DIR:-/home/zeng/qwen38-rows}"
VENV="${VENV:-$QWEEN_DIR/.venv}"
PY_VER="${PY_VER:-3.12}"
ENGRAMDB_VERSION="${ENGRAMDB_VERSION:-0.2.11}"

echo "[wsl-repro] qwen35-ple = $QWEEN_DIR"
echo "[wsl-repro] engramdb   = $ENGRAMDB_DIR"
echo "[wsl-repro] rows       = $ROWS_DIR"
echo "[wsl-repro] python     = $VENV"

# --- Python environment ------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[wsl-repro] creating venv at $VENV ..."
  python3 -m venv "$VENV"
fi

# Use uv if available for fast, pinned installs; fall back to pip.
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" \
    "engramdb-python==${ENGRAMDB_VERSION}" \
    "pytest>=8" "ruff>=0.6"
else
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install \
    "engramdb-python==${ENGRAMDB_VERSION}" \
    "pytest>=8" "ruff>=0.6"
fi

# --- Source trees ------------------------------------------------------------
if [[ -d "$ENGRAMDB_DIR" ]]; then
  echo "[wsl-repro] EngramDB present; python path will use source when needed."
fi
if [[ ! -d "$QWEEN_DIR" ]]; then
  echo "[wsl-repro] ERROR: qwen35-ple repo not found at $QWEEN_DIR" >&2
  exit 1
fi

# --- Minimal validation ------------------------------------------------------
cd "$QWEEN_DIR"
echo "[wsl-repro] running live-store smoke tests ..."
PYTHONPATH="src:$ENGRAMDB_DIR/python" "$VENV/bin/python" -m pytest \
  tests/test_live_store.py \
  tests/test_slot_index.py \
  -q

if [[ "${1:-}" == "--full" ]]; then
  echo "[wsl-repro] running full qwen35-ple pytest ..."
  PYTHONPATH="src:$ENGRAMDB_DIR/python" "$VENV/bin/python" -m pytest -q
fi

echo "[wsl-repro] WSL_ENV_OK"
