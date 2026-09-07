#!/usr/bin/env bash
# Sensitivity sweep for PLE evidence: n-gram order and memory bank size.
# Each run uses run_ple_evidence_p0.py and writes JSON under outputs/.
set -e
cd /home/zeng/qwen35-ple
COMMON="--model data/models/Qwen3.5-0.8B --code-root /home/zeng/qwen35-ple --wiki-path data/sources/wikitext.jsonl --train-frac 0.8 --context-len 32 --max-per-doc-code 4 --max-per-doc-wiki 3 --max-eval-positions 120 --calib-frac 0.5 --seed 0 --device cuda"
for ord in 2 3 4 5; do
  .venv/bin/python scripts/run_ple_evidence_p0.py $COMMON --max-order "${ord}" --output "outputs/sens-order-${ord}.json" > "outputs/sens-order-${ord}.log" 2>&1
  echo "ORDER_${ord}_DONE" >> "outputs/sens-order-${ord}.log"
done
for pair in "40 80" "120 240" "300 600"; do
  set -- ${pair}
  cf=$1
  wd=$2
  .venv/bin/python scripts/run_ple_evidence_p0.py $COMMON --max-code-files "${cf}" --max-wiki-docs "${wd}" --max-order 4 --output "outputs/sens-mem-c${cf}-w${wd}.json" > "outputs/sens-mem-c${cf}-w${wd}.log" 2>&1
  echo "MEM_${cf}_${wd}_DONE" >> "outputs/sens-mem-c${cf}-w${wd}.log"
done
echo ALL_SENSITIVITY_DONE > /mnt/c/Users/minam/job_status_sensitivity.txt
