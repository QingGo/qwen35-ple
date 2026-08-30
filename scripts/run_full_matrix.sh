#!/bin/bash
set -e
cd /Users/zeng/code/qwen35-ple
source /Users/zeng/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-tts
export PYTHONPATH=/tmp/tf53:/tmp/extra:src:/Users/zeng/code/EngramDB/python:/Users/zeng/.cache/uv/archive-v0/cCb0wBZ_lLyWmI1X6tadR:/Users/zeng/.cache/uv/archive-v0/IBggXHYTHPSZR0XCTWA3f

for spec in "1 1 0" "8 1 0" "1 4 0" "8 4 0" "1 4 1" "8 4 1"; do
  set -- $spec
  layer=$1; branches=$2; sc=$3
  for mode in real control; do
    out="outputs/matrix_l${layer}_b${branches}_s${sc}_${mode}.json"
    echo "=== START layer=$layer branches=$branches short=$sc mode=$mode ===" >> /tmp/matrix.log
    if [ "$sc" = "1" ]; then
      python scripts/run_ple_adapter.py --mode "$mode" --features data/ple-adapter-features-20k --steps 40 --seq-len 128 --lr 1e-4 --layer "$layer" --branches "$branches" --short-conv --output "$out" >> /tmp/matrix.log 2>&1
    else
      python scripts/run_ple_adapter.py --mode "$mode" --features data/ple-adapter-features-20k --steps 40 --seq-len 128 --lr 1e-4 --layer "$layer" --branches "$branches" --output "$out" >> /tmp/matrix.log 2>&1
    fi
    echo "=== DONE layer=$layer branches=$branches short=$sc mode=$mode ===" >> /tmp/matrix.log
  done
done
echo "MATRIX ALL DONE" >> /tmp/matrix.log
