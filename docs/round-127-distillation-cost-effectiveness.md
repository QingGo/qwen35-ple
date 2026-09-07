# Round 127：蒸馏/OPD 的算力配置、步数与成本收益分析

> 日期：2026-09-08
> 问题：蒸馏需要什么算力？OPD 需要什么配置？蒸馏是否比全量训练更少步数？
> 如果按租赁 GPU 成本，哪种方法性价比最高？

---

## 1. 先区分“训练学生”和“生成教师数据”

这是关键：

```text
教师推理成本
≠
学生训练成本
```

蒸馏可以拆成：

1. **教师数据生成**：用大模型生成 teacher text / logits；
2. **学生训练**：用这些数据训练 0.8B + adapter。

### 1.1 如果使用现有高质量文本作为 teacher

这是最便宜的：

```text
不需要大 teacher 在线推理
直接用已有 CoT / solution / 验证过的答案
```

我们已有的 CAP-1 / Purified OPSD 路线就接近这种。

### 1.2 如果需要 Qwen3.8-Flash-Next 作为 teacher

Qwen3.8-Flash-Next 的规模：

```text
总参数约 176B
激活参数约 6B
MoE + PLE
```

要在本地跑 teacher，最低配置参考：

```text
8GB GPU + 约 48GB RAM（GGUF/CPU offload）
速度约 34–35 tok/s
```

也就是说：

```text
本地 8GB 单卡可以跑，但很慢；
更适合在云端高 RAM 机器上离线生成 teacher 数据。
```

---

## 2. OPD / Purified OPSD 的算力配置

### 2.1 OPD 流程

```text
1. student 采样 rollout
2. teacher 给逐 token 分布
3. 计算 KL / CE 损失
4. 更新 student
```

### 2.2 学生侧最低配置

- Qwen3.5-0.8B；
- 只训练 LoRA / MoRA / adapter；
- **GTX 1070 8GB 可以跑**；
- 我们已经在 1070 上跑过 CAP-1 LoRA / QLoRA / MoRA。

### 2.3 教师侧配置

OPD 真正贵的是 teacher rollout：

| 方案 | 配置 | 速度 |
|---|---|---|
| 离线 teacher text | 不需要大模型 | 无限快 |
| 本地 Qwen3.8-Flash-Next | 8GB GPU + 48GB RAM | 约 35 tok/s |
| 云端 A100 / H100 | 80GB 显存 | 快 |
| API teacher | 按 token 计费 | 中等 |

### 2.4 如果只有 1070

最优做法：

```text
1070 只做 student 训练；
teacher 输出在云端/API 一次性生成；
保存成 JSONL / npz；
本地反复训练。
```

不需要在 1070 上实时跑 teacher。

---

## 3. 蒸馏是否比全量训练更少步数？

### 3.1 是，但要看目标

| 方案 | 通常数据/步数 | 说明 |
|---|---|---|
| 全量 CPT 让模型学会用 PLE | 0.2–20B tokens | 非常贵 |
| 小 adapter 蒸馏 | 100–1000 步 | 1070 可跑 |
| OPD / Purified OPSD | 几千到几万步 | 比 CPT 少，但采样成本高 |
| 离线缓存 + reader | 20M tokens 特征缓存 | 比全量 CPT 便宜很多 |

所以：

```text
对于“让 0.8B 获得某类能力”，蒸馏通常比全量预训练/CPT 少几个数量级。
```

### 3.2 OPD 不一定比 SFT 少步数

OPD 的优势不是“步数少”，而是：

```text
on-policy 采样减少分布失配
teacher 分布比奖励更稳定
比 RL 便宜
```

但 OPD 需要采样 rollout，所以：

```text
每一步成本可能比 SFT 高；
但比 RL 低。
```

### 3.3 真正的成本公式

```text
总成本 ≈ 教师生成成本 + 学生训练成本 + 验证/过滤成本
```

Purified OPSD 多了一个：

```text
验证/过滤成本
```

它可以防止蒸馏退化，通常值得花。

---

## 4. 按租赁 GPU 成本排序

以下为粗略排序，价格只是量级估计。

### 4.1 最省钱：本地 1070 + 离线数据

```text
成本：电费
能力：LoRA/MoRA 蒸馏、RAG self-distill、reader-only
```

适合：

- 小规模验证；
- CAP-1 / Purified OPSD；
- 离线 teacher text。

### 4.2 第二省钱：1070 + 一次性云 teacher 生成

```text
本地 1070 训练学生：约 0 元
云端生成 teacher：按 token 计费或租几小时
```

适合：

- 想用 Qwen3.8-Flash-Next 做真 logit/OPD；
- 不想买高 RAM 机器。

### 4.3 第三：4090 云租用做 target-side reader

```text
20M tokens 只训 reader ≈ 1–2 小时
4090 租金大约 $0.2–0.5/小时
≈ $0.5–2
```

这是“低成本高信息”的 sweet spot。

### 4.4 更贵：A100 / H100 全量 CPT

```text
70M tokens 全量训练可能 10–20 小时
A100 租金约 $1–3/小时
≈ $20–60
```

如果数据量到 1B tokens：

```text
成本可能到几百美元。
```

### 4.5 最贵：持续 RL / 大规模 OPD

```text
需要大量 rollout + 验证 + teacher
成本通常比蒸馏高几个数量级。
```

---

## 5. 性价比结论

| 方法 | 性价比 | 适合场景 |
|---|---|---|
| 本地 1070 + 离线 teacher-text / RAG self-distill | 最高 | 小模型能力验证、论文小实验 |
| 1070 学生 + 云端一次性 teacher 数据 | 高 | OPD / Purified OPSD |
| 4090 云租用 + 离线缓存 + reader-only | 高 | 20M 级 target-side reader |
| 4090 云租用 + 直接训练 | 中 | 小规模全量/适配 |
| A100/H100 全量 CPT | 低-中 | 正式大训练 |
| 持续 RL / 大规模 OPD | 最低 | 追求极限能力，但预算高 |

---

## 6. 最推荐路线

```text
第一步：1070 本地跑
  - RAG self-distill
  - Purified OPSD
  - 离线 teacher text LoRA/MoRA
  - 成本 ≈ 电费

第二步：如果确实要 Qwen3.8-Flash-Next teacher
  - 租 4090 或高 RAM 机器一次性生成 teacher 数据
  - 回 1070 训练学生
  - 成本可能只有几美元

第三步：如果要做 20M target-side reader
  - 4090 租 1–2 小时
  - 用离线缓存 + reader-only
  - 成本可能 $1–2

第四步：只有小规模验证有效后，再考虑 A100/H100 全量训练
```

### 一句话

> **蒸馏通常比全量 CPT 便宜几个数量级；
> 最划算的是“本地 1070 训练学生 + 云端一次性生成教师数据 + 只训小 adapter/reader”。**
