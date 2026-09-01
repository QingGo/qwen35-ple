# Phase 0 实验基座协议（2026-08-30）

> 目标：把“真实 PLE vs shuffled control vs no-reader”的判定从临时脚本升级为
> **一条命令可复现、固定分割、多 seed、可选 QA 信号**的正式实验协议。

---

## 1. 核心设计

### 1.1 固定 train/val 分割

- 输入：预计算好的 `tokens.npy` + `e_t.npy`（当前使用 `data/ple-adapter-features-20k`）。
- 分割：按 token 位置切分，默认前 90% 训练、后 10% 验证。
- 验证集**绝不参与训练采样**。
- 可选 `--val-frac` 调整。

### 1.2 三线对照

| 线 | 含义 |
|---|---|
| `no-reader` | 原始 Qwen3.5 backbone，无 PLE 注入 |
| `real` | 从训练集学习 reader，使用真实 e_t |
| `control` | 从训练集学习 reader，使用逐 token 随机置换的 e_t |

### 1.3 多 seed

- `--seeds 0 1 2`
- 每个 seed 独立初始化 reader、独立随机训练窗口、独立 control 排列。
- 汇总输出 `mean ± std`。

### 1.4 QA 信号（双口径）

默认包含 TriviaQA / NQ / BoolQ 风格的小型题目。支持两种口径：

1. **log-likelihood（`--qa`）**
   - 计算答案 token 的平均 loss。
   - 低 loss = 模型更可能生成该答案。
2. **exact-match（`--qa-exact-match`）**
   - 用贪心生成逐 token 解码，检查生成文本中是否出现标准答案。
   - real/control 在每一步都会重新读取当前 token 序列的 PLE `e_t` 并注入 reader。
   - no-reader 走同一条贪心生成循环，但不注入 PLE。
   - 可用 `--qa-file` 传入外部 JSON 题目列表。

---

## 2. 一条命令

### 本地 Intel Mac

```bash
bash scripts/run_phase0.sh \
  --features data/ple-adapter-features-20k \
  --steps 20 \
  --seq-len 128 \
  --seeds 0 1 2 \
  --modes no-reader real control \
  --qa \
  --output outputs/phase0.json
```

### WSL / GPU / 正常 Python 环境

```bash
python scripts/run_phase0.py \
  --features /path/to/features \
  --steps 100 \
  --seq-len 256 \
  --seeds 0 1 2 \
  --modes no-reader real control \
  --qa \
  --qa-exact-match \
  --qa-max-new-tokens 16 \
  --output outputs/phase0.json
```

---

## 3. 输出结构

```json
{
  "config": { ... },
  "summary": {
    "no-reader": { "val_loss_mean": ..., "val_loss_std": ..., "val_ppl_mean": ... },
    "real":     { ... },
    "control":  { ... }
  },
  "results": [ ... ]
}
```

---

## 4. Phase 0 Gate

**通过条件：**

- 三线 + ≥3 seeds 可由一条命令生成；
- 输出包含：
  - no-reader baseline
  - real
  - control
  - 每个 seed 的 val loss / ppl
- 数据分割固定、验证集不参与训练。

**注意：Phase 0 本身不要求 real 超过 baseline。**
Phase 0 只负责建立可信协议；科学判定在 Phase 2 的 1M–5M token 实验中完成。

---

## 5. 环境与可复现性

### 当前本地环境路径（临时）

- Python：`/Users/zeng/miniconda3/envs/qwen3-tts/bin/python`
- EngramDB：`../EngramDB/python`
- transformers：`/tmp/tf53`
- peft：`/tmp/extra`

这些是本机临时兼容路径，仅用于开发。

### WSL / GPU 推荐路径

1. 正常安装：
   ```bash
   uv sync --all-groups
   ```
2. 确认可导入：
   ```bash
   python -c "import engramdb, torch, transformers; print('ok')"
   ```
3. 直接运行 `scripts/run_phase0.py`。

---

## 6. 当前进度

- [x] Phase 0 PPL 三线 + 多 seed harness：`scripts/run_phase0.py`
- [x] 一条命令 wrapper：`scripts/run_phase0.sh`
- [x] 最小 QA log-likelihood：TriviaQA / NQ / BoolQ 风格
- [x] QA exact-match 生成式评测：`--qa-exact-match`（含 live PLE 逐步注入）
- [x] 本地 smoke 已跑通（1 seed / 1 step / no-reader + real）
- [x] `QwenEngramReader`（忠实官方/engram-peft 风格 gating）已加入 `src/qwen35_ple/reader.py`
- [x] Phase 1 live-precomputed 数值一致性 gate 已通过（`run_live_vs_precomputed.py` 当前路径 max_abs_diff=0）
- [ ] 3-seed 正式 Phase 0 报告
- [ ] WSL/GPU 通道实机验证
- [ ] 旧预计算 e_t 文件需重新生成，或后续直接使用 live Store 训练
