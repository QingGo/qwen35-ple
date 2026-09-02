#!/usr/bin/env bash
# Background wrapper for M2-M5 (skip M1, already finished).
set -euo pipefail

cd /home/zeng/qwen35-ple
LOG="/home/zeng/mix-rest.log"
echo "[wrapper] start M2-M5 $(date)" > "${LOG}"
bash scripts/run_mix_batch.sh \
  --pyenv .venv/bin/python \
  --mixes M2 M3 M4 M5 \
  --seeds 0 \
  --output-dir outputs \
  >> "${LOG}" 2>&1
echo "[wrapper] done M2-M5 $(date)" >> "${LOG}"
