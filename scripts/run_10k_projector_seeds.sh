#!/usr/bin/env bash
# Run 3 seeds of PLE projector training on the 10k dataset with truncated eval.
set -e
cd /home/zeng/qwen35-ple
for seed in 0 1 2; do
  .venv/bin/python scripts/train_ple_projector.py \
    --model data/models/Qwen3.5-0.8B \
    --code-root /home/zeng/qwen35-ple \
    --code-corpus data/code-corpus.jsonl \
    --wiki-path data/sources/wikitext.jsonl \
    --dataset data/ple-projector-dataset-10k.jsonl \
    --max-train-samples 7000 \
    --max-eval-samples 300 \
    --steps 100 \
    --batch-size 8 \
    --device cuda \
    --seed "${seed}" \
    --output "outputs/ple-projector-10k-seed${seed}.json" \
    > "outputs/ple-projector-10k-seed${seed}.log" 2>&1
  echo "DONE seed${seed}"
done
