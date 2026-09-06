#!/usr/bin/env bash
# Run per-item CAP-1 + formal benchmark eval for Purified OPSD seeds.
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:vendor/peft-mora/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="${PYTHON:-.venv/bin/python}"
$PY scripts/run_cap1_formal_eval.py --output outputs/cap1-formal-base.json
$PY scripts/run_cap1_formal_eval.py --adapter outputs/cap1-purified-mora-80 --output outputs/cap1-formal-s0.json
$PY scripts/run_cap1_formal_eval.py --adapter outputs/cap1-purified-mora-80-s1 --output outputs/cap1-formal-s1.json
$PY scripts/run_cap1_formal_eval.py --adapter outputs/cap1-purified-mora-80-s2 --output outputs/cap1-formal-s2.json
