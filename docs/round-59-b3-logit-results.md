# Round 59：B3 logit-space 直接记忆下界实测

> 日期：2026-09-04
> 状态：B3 下界已测量，结论支持 PLE 信息不足
> 目标：验证“即使绕过 hidden Jacobian，直接在 logits 层加 PLE 记忆头，是否仍能出现 real>control”。

---

## 1. 动机

Round 56 提出了 B3 下界：

- 不改 hidden；
- 直接把记忆特征映射成 logit 偏移；
- 这是“logit-level 最优修正”的可实现下界。

如果 B3 仍不能产生 real>control，就能更严格地说：

> 瓶颈不在 hidden 注入通道，而在 PLE 信息本身。

## 2. 实验设置

| 项 | 值 |
|---|---|
| Backbone | Qwen3.5-0.8B（frozen） |
| Memory head | `PureLogitMemoryModule`：base logits + scale × MLP(memory feature) |
| Bank | PLE exact bank，347,439 entries |
| Training | 200 steps，seq_len=64，lr=1e-4，GPU |
| Eval | rare-kb 270 题（rare 182 / common 88） |
| Metric | answer-token average logprob + first-token hit |

## 3. 结果

### 3.1 单种子（seed 0）汇总

| 分组 | no-memory logprob | real logprob | control logprob | real−control | t | first-hit real |
|---|---:|---:|---:|---:|---:|---:|
| rare | -4.248 | -4.192 | -4.191 | -0.00101 | -0.73 | 4/182 |
| common | -8.241 | -8.257 | -8.251 | -0.00578 | -0.94 | 0/88 |
| all | -5.549 | -5.517 | -5.514 | -0.00256 | -1.17 | 4/270 |

### 3.2 3-seed 汇总

| 分组 | 各 seed real−control | mean | se |
|---|---:|---:|---:|
| rare | -0.00101 / -0.00134 / -0.00116 | **-0.00117** | 0.000078 |
| common | -0.00578 / +0.00368 / -0.00825 | -0.00345 | 0.00297 |
| all | -0.00256 / +0.00030 / -0.00347 | -0.00191 | 0.00093 |

3 个 seed 的 rare real−control 全部为负，且 se 很小：

> **不存在可检测的 PLE real>control 正信号，反而有微弱的负倾向。**

### 3.3 解读

- real 相对 no-memory 在 rare 上略好：
  - logprob +0.05 左右；
  - first-token hit 从 1/182 提升到 4/182（seed 0）。
- 但 real 相对 control 在 3-seed 下全部为负或接近 0：
  - rare mean = −0.00117；
  - 说明 logit-space PLE 头学到的主要是 **记忆向量的通用分布效应**，不是 exact n-gram 内容的因果增益。
- 连直接改 logits 都无法放大 PLE 的 real>control，因此：

> **PLE 信息不足的结论进一步加强。**

## 4. 与 B4/P1 对比

| 方法 | 通道 | rare real−control | 结论 |
|---|---|---|---|
| P1 hidden injection（B4） | hidden + cross-attention + router | +0.00013 | 不显著 |
| B3 logit-space direct | base logits + memory logits | −0.00101 | 不显著/微负 |
| RAG | 输入上下文 | +0.851 | 显著 |

结论：

- 不是 hidden 通道拖累 PLE；
- 也不是 logit 层无法表达；
- 而是 **PLE 本身的 task-level 条件信息不足**。

## 5. 对计划的影响

- PLE-Final 的启动条件进一步收紧：
  - 必须出现 **B3 logit-space real>control 显著为正**；
  - 或者出现全新的、信息量更大的 PLE 用法；
- 当前证据不支持继续投入大规模 PLE backbone adaptation；
- 主路径继续：
  - R1：多任务评测
  - D1：RAG 产品化
  - D2：教师蒸馏 / OPD / Purified OPSD

## 6. 产物

- `scripts/train_b3_logit_memory.py`
- `scripts/eval_b3_logit_memory.py`
- WSL outputs：
  - `outputs/b3-logit-real.pt`
  - `outputs/b3-logit-eval.json`
