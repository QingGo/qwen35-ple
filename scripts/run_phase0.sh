#!/bin/bash
# Phase 0 one-command runner.
#
# Usage from repo root:
#   bash scripts/run_phase0.sh --features data/ple-adapter-features-20k --steps 20 --seeds 0 1 2
#
# On a normal installed environment (WSL/GPU), this should just use `python`.
# On the local Intel Mac dev box it falls back to the known compatibility paths.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x "/Users/zeng/miniconda3/envs/qwen3-tts/bin/python" ]; then
  PYTHON_BIN="/Users/zeng/miniconda3/envs/qwen3-tts/bin/python"
fi

# Local compatibility path (kept for this Intel Mac).
LOCAL_PYTHONPATH="src:../EngramDB/python:/tmp/tf53:/tmp/extra"
# If the caller already has a working package install, prefer that.
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${LOCAL_PYTHONPATH}"
else
  export PYTHONPATH="${LOCAL_PYTHONPATH}"
fi

echo "[phase0] python=$PYTHON_BIN"
echo "[phase0] PYTHONPATH=$PYTHONPATH"
exec "$PYTHON_BIN" scripts/run_phase0.py "$@"
