# Round 53：P1 记忆接口原型实测与门禁判定

> 日期：2026-09-04
> 状态：P1 门禁未通过
> 结论：exact bank + TokenMem cross-attention + distribution memory/router 在冻结 backbone 下仍不能把 rare PLE 信号转为 real>control。按 round-50 停止条件，转向 RAG / 蒸馏 / 更语义化记忆，不进入大规模 MoRA/GaLore/RL。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| Backbone | Qwen3.5-0.8B（frozen） |
| PLE | Qwen3.8-Flash-Next 真实 FP8 表 |
| Bank | `data/ple-books-160k`，161,296 tokens |
| Bank entries | 347,439（2-gram 70,887；3-gram 126,889；4-gram 149,663） |
| Control bank | 相同 exact keys，PLE 值整体 shuffle |
| Memory module | TokenMemCrossAttention + MemoryLogitHead + MemoryRouter |
| Training | 100 steps，seq_len=64，lr=1e-4，layer=8，GPU，next-token CE |
| Evaluation | `data/rare-kb-v1.json`，270 题（rare 182 / common 88），rows-dir 真表 e_t fallback |
| Metric | answer-token 平均 logprob，第一答案 token hit |

## 2. 结果

### 2.1 real 训练 checkpoint

| 分组 | n | real logprob | control logprob | real−control | se | t | real−control 胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rare | 182 | -3.937210 | -3.937341 | +0.000131 | 0.000183 | +0.71 | 60/182 |
| common | 88 | -7.522085 | -7.521146 | -0.000939 | 0.000836 | -1.12 | 13/88 |
| all | 270 | -5.105614 | -5.105396 | -0.000218 | 0.000300 | -0.73 | 73/270 |

### 2.2 control 训练 checkpoint

| 分组 | n | real logprob | control logprob | real−control | se | t |
|---|---:|---:|---:|---:|---:|---:|
| rare | 182 | -3.932926 | -3.932970 | +0.000044 | 0.000218 | +0.20 |
| common | 88 | -7.523049 | -7.521981 | -0.001068 | 0.000937 | -1.14 |
| all | 270 | -5.103040 | -5.102721 | -0.000318 | 0.000340 | -0.94 |

### 2.3 first-token hit

- 所有条件下 rare/common first-token hit 均接近 0：
  - real=0.0055（rare）、0.0（common）；
  - control 相同；
  - 说明 logprob 的微弱差异没有变成可观测的 decoding 收益。

### 2.4 memory module 的总体效应

- real 相对 no-memory 的 answer-logprob 提升很大：
  - rare：+0.311；
  - common：+0.718。
- 但 control 几乎得到同样提升。
- 因此 P1 memory interface 学到的是“一个通用的 distribution shift / scaling 效应”，不是“真实 PLE n-gram 内容特有的因果增益”。

---

## 3. 判定

P1 门禁要求：

```text
rare knowledge: real > control
```

实测：

```text
real-ckpt:   rare real−control = +0.000131, t = 0.71   → 不显著
control-ckpt: rare real−control = +0.000044, t = 0.20   → 不显著
common:      real−control 为负，t ≈ -1.1                → 无正信号
first-token: 0 差异                                      → 无 decoding 收益
```

结论：

> **P1 门禁未通过。**

这与 Phase A 的纯特征结果一致：PLE 有极弱条件信息，但当前 frozen backbone + 小记忆接口无法把它转化为 rare 知识任务上的 real>control。

## 4. 停止条件触发

按 `docs/round-50-systematic-plan.md`：

> 如果 exact bank + cross-attention + router 后 rare real−control 仍不显著，则进入 RAG / 蒸馏 / 更语义化记忆的转向路线。

因此：

- **不进入大规模 MoRA / GaLore / backbone adaptation / RL**；
- **不把当前 PLE 视为已证明可用的知识增强记忆**；
- 转向：
  1. 同口径 RAG baseline（用外部检索器把知识直接给 backbone）；
  2. OPD / Purified OPSD 蒸馏（学生从 Qwen3.8-Flash-Next 获取能力，不依赖 PLE 表内容）；
  3. 更语义化记忆 / 混合记忆方案（如果仍要保留 PLE 作为局部语言先验）。

## 5. 证据与可复现性

- 产物位于 WSL：
  - `data/exact-ple-bank.npz`
  - `data/exact-ple-bank-control.npz`
  - `outputs/p1-memory-real.pt`
  - `outputs/p1-memory-control.pt`
  - `outputs/p1-memory-eval.json`
  - `outputs/p1-memory-eval-control-ckpt.json`
- 代码与说明：
  - `docs/round-52-p1-memory-prototype.md`
  - `scripts/build_exact_ple_bank.py`
  - `scripts/train_p1_memory.py`
  - `scripts/eval_p1_memory.py`
- 局限：
  - 100 步小规模训练；
  - bank 来自单一书籍语料；
  - 未做 3-seed 显著性；
  - 未做 layer / head / 4-gram 单独消融；
  - 但即使当前最便宜的接口已经看不到 real>control，继续放大前需要先出现正向证据。

---

## 6. 下一步

1. 构建 RAG baseline：
   - 同一 QA 集；
   - 用外部检索（BM25/embedding）提供证据；
   - 与 real/control/no-memory 同口径对比。
2. 开始 OPD / Purified OPSD：
   - 先用离线 teacher 数据；
   - 再 on-policy；
   - 不把 PLE real>control 作为前提。
3. 如果 RAG/蒸馏证明“能力提升不依赖 PLE”，则把 PLE 降级为可选局部语言先验。
