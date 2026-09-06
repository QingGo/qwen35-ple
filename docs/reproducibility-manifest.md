# Reproducibility Manifest

> 更新：2026-09-06  
> 目的：让当前有限资源实验可被复现，并作为后续顶刊/顶会 artifact 的基础。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA GTX 1070 8GB |
| 系统 | Windows + WSL2 Ubuntu |
| Python | 3.13（远程 venv） |
| PyTorch | 2.6.0+cu124 |
| Transformers | 0.8B Qwen3.5 本地权重 |
| PEFT | 0.20.0 + vendor/peft-mora/src（MoRA） |
| 数据目录 | `data/` |
| 输出目录 | `outputs/` |

---

## 2. 核心数据

- `data/purified-opsd-train.jsonl`：Purified OPSD 训练集，138 条；
- `data/cap1-rag-distill-eval39.jsonl`：CAP-1 held-out，39 条；
- `data/formal-benchmarks/*.jsonl`：GSM8K-like / MATH-like / HumanEval-like / MBPP-like；
- `data/sources/wikitext.jsonl`：PLE/BM25 同域 wiki 语料；
- `configs/ngram-fusion-router.json`：per-task PLE 融合参数。

---

## 3. 已跑实验

### 3.1 Purified OPSD 多 seed

```bash
PYTHONPATH=src:vendor/peft-mora/src \
python scripts/run_lora_distill.py \
  --model data/models/Qwen3.5-0.8B \
  --data data/purified-opsd-train.jsonl \
  --steps 80 --seed {0,1,2} --use-mora \
  --output outputs/cap1-purified-mora-80-s{0,1,2}
```

### 3.2 LoRA / QLoRA 对照

```bash
python scripts/run_lora_distill.py ... --steps 80 --seed 0 \
  --output outputs/cap1-purified-lora-80
python scripts/run_lora_distill.py ... --steps 80 --seed 0 --use-qlora \
  --output outputs/cap1-purified-qlora-80
```

### 3.3 CAP-1 + 正式基准 per-item 评测

```bash
python scripts/run_cap1_formal_eval.py \
  --adapter outputs/cap1-purified-mora-80-s0 \
  --output outputs/cap1-formal-s0.json
```

### 3.4 PLE / BM25 / n-gram / 混合基线

```bash
python scripts/run_ple_baseline_ablation.py \
  --model data/models/Qwen3.5-0.8B --device cuda \
  --max-eval-positions 160 --seed 0 \
  --output outputs/ple-baseline-hybrid-s0.json
```

### 3.5 多源系统消融

```bash
python scripts/run_multisource_ablation.py \
  --model data/models/Qwen3.5-0.8B \
  --adapter outputs/cap1-purified-mora-80 \
  --seed 0 --output outputs/ms-purified-mora-s0.json
```

---

## 4. 固定随机性

- 所有训练脚本通过 `--seed` 固定 torch / random；
- 评测脚本通过 `--seed` 固定采样；
- 分析脚本 `scripts/analyze_seed_peritem.py` 使用固定 bootstrap seed。

---

## 5. 关键输出文件

| 实验 | 输出 |
|---|---|
| Purified MoRA seeds | `outputs/cap1-purified-mora-80-s{0,1,2}` |
| Purified LoRA/QLoRA | `outputs/cap1-purified-lora-80` / `cap1-purified-qlora-80` |
| Per-item 评测 | `outputs/cap1-formal-*.json` |
| Seed 分析 | `outputs/seed-adapter-analysis.md` |
| PLE 基线 | `outputs/ple-baseline-ablation*.json` |
| PLE 混合 | `outputs/ple-baseline-hybrid-s*.json` |
| 多源系统 | `outputs/ms-purified-mora-s*.json` |

---

## 6. 已知限制

- 正式基准为本地生成 “like” 数据，不是官方真实基准；
- CPU 100 tok/s 尚未达成；
- 当前只有单卡 8GB 下的 80 步小训练；
- PLE 结论为 3-seed 小样本局部任务证据。
