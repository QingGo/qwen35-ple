#!/usr/bin/env bash
# Single-mix helper for background WSL runs.
# Usage: bash scripts/run_mix_one_wrapper.sh M1
set -euo pipefail

MIX="${1:-M1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG="/home/zeng/mix-${MIX}.log"
echo "[wrapper] start ${MIX} $(date)" > "${LOG}"
bash scripts/run_mix_batch.sh \
  --pyenv .venv/bin/python \
  --mixes "${MIX}" \
  --seeds 0 \
  --output-dir outputs \
  >> "${LOG}" 2>&1
echo "[wrapper] done ${MIX} $(date)" >> "${LOG}"
