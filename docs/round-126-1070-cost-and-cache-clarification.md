# Round 126：冻结/适配实验回顾、1070 成本估算与两阶段缓存法说明

> 日期：2026-09-08
> 目的：回答三个问题：
> 1. 冻结 backbone、LoRA/变种我们是不是试过？为什么效果不好？
> 2. 用 GTX 1070 需要多少时间？
> 3. “两阶段离线缓存法”到底怎么运作？

---

## 1. 我们试过什么，为什么还不够

### 1.1 冻结 backbone + reader/projector

我们确实试过：

- 冻结 0.8B；
- 只训练 PLE Projector / MLP reader；
- 用 hidden state + memory features 生成 scale/bias；
- 也在 hidden-state 注入、residual reader、MLP reader 上做过实验。

结果：

```text
局部低熵任务：有真实信号，10k 5 seeds 显著。
知识/QA/实体 value：几乎无效。
```

原因不是“冻结本身错了”，而是：

1. **训练量太小**：10k 样本，远小于 20M；
2. **通道太窄**：只用 7 个手工特征 + hidden → scale/bias，没有充分对齐 PLE 行与 0.8B 隐藏空间；
3. **没有把 PLE 当作真正的“外部条件输入”**，而是当作 logit prior；
4. **0.8B 的 24 层没有机会被训练去利用这个外部信号**。

### 1.2 LoRA / QLoRA / MoRA

我们也试过并在某些任务上有效：

```text
CAP-1 LoRA / QLoRA / MoRA：
- held-out logprob 提升
- code-output / arithmetic 提升
- knowledge 一般
```

但为什么没有解决“嫁接 PLE 提升通用能力”？

因为：

- LoRA/MoRA 是**参数化能力**，它们不一定让模型学会“读取外部 PLE”；
- 它们训练数据很少（几十到几百条），不足以学会复杂 memory-reading；
- 它们和 PLE 的联合优化很浅，没有形成“PLE 行 → 0.8B 层 → 新语义”的完整通路；
- PLE 本身也不是一个通用知识库，只靠 adapter 无法把它变成 RAG 或蒸馏。

所以：

```text
LoRA/MoRA 是能力通道，
不是外部记忆读取通道。
```

---

## 2. GTX 1070 单卡时间估算

1070 是 Pascal 架构，8GB 显存，无 Tensor Core，bf16 支持弱，FP16 也没有现代卡那样加速。

### 2.1 吞吐假设

0.8B 模型，1070 单卡：

| 模式 | 保守 | 乐观 |
|---|---:|---:|
| 只 forward（缓存特征/推理） | 300 tok/s | 1,000 tok/s |
| reader 训练（冻结 backbone） | 150–400 tok/s | 600 tok/s |
| 全量 CPT / 全参数训练 | 100–300 tok/s | 500 tok/s |

### 2.2 20M target tokens

只 forward / 只训 reader：

```text
@300 tok/s = 18.5 小时
@500 tok/s = 11.1 小时
@800 tok/s = 6.9 小时
@1000 tok/s = 5.6 小时
```

结论：

```text
1070 上 20M tokens 大约 6–19 小时。
```

### 2.3 70M tokens（50M source + 20M target）

只 forward：

```text
@300 tok/s = 64.8 小时
@500 tok/s = 38.9 小时
@800 tok/s = 24.3 小时
@1000 tok/s = 19.4 小时
```

结论：

```text
1070 上 70M tokens 大约 20–65 小时。
```

### 2.4 如果全量训练

```text
70M tokens @ 200 tok/s ≈ 97 小时
70M tokens @ 100 tok/s ≈ 194 小时
```

结论：

```text
全量 CPT 在 1070 上可能要 4–8 天，基本不现实。
```

### 2.5 如果只需要验证 2M–5M tokens

更合理的第一步：

```text
2M tokens @ 500 tok/s ≈ 1.1 小时
5M tokens @ 500 tok/s ≈ 2.8 小时
```

结论：

```text
1070 可以先做 2M–5M 的小规模验证。
```

---

## 3. “两阶段离线缓存法”详细说明

### 3.1 普通做法（我们现在）

```text
每个 training step：
  1. 取一个 batch；
  2. 跑 0.8B forward + backward；
  3. 更新 reader；
  4. 下一 batch 再跑一遍 0.8B。
```

缺点：

```text
0.8B 被反复运行；
1070 上很慢；
换 reader 结构要重新跑大模型。
```

### 3.2 两阶段缓存法

#### 阶段 A：一次性提取特征

用冻结 0.8B 对语料跑一次 forward，不更新任何参数：

```text
for each token position:
    save:
      - hidden_state h
      - PLE memory features m
      - target token y
      - context / task label
```

这个阶段只做 forward，是唯一贵的一步。

#### 阶段 B：在缓存上训练 reader

之后训练 reader 时：

```text
不再运行 0.8B；
直接从磁盘读 h, m, y；
训练小 MLP / projector / low-rank reader。
```

可以：

```text
- 跑很多 epoch；
- 快速换结构；
- 在 CPU 或小 GPU 上训练；
- 反复实验而不重跑大模型。
```

### 3.3 存什么、需要多大空间

如果存：

```text
20M tokens
hidden = 1024 维，fp16 => 2 字节
= 20M × 1024 × 2 ≈ 40 GB
```

如果只存 7 个 memory features + target：

```text
约几百 MB 到几 GB。
```

所以：

```text
建议先只存少量层 hidden + features；
或者存压缩/quantized hidden；
或者先只存 2M–5M tokens。
```

### 3.4 能不能同时训练 backbone？

不能。

缓存法适合：

```text
只训练 target-side reader / projector，
让 0.8B 读取一个固定外部信号。
```

如果要训练 backbone 来“适应 PLE”，需要保留完整 backprop，那就不能只用缓存，必须做全量或部分全量训练。

这正是为什么：

```text
冻结 backbone + 大规模 reader 训练，
是 1070 上最现实的第一阶段；
全量 CPT 放到有更强 GPU 时再做。
```

---

## 4. 结论

- 我们确实试过冻结 backbone、LoRA/QLoRA/MoRA；
- 它们不是“完全没用”，而是：
  - 训练量太小；
  - 通道太窄；
  - 没有真正形成“PLE → 0.8B 多层处理”的完整通路；
- 1070 上：
  - 20M tokens 只训 reader：约 6–19 小时；
  - 70M tokens 只训 reader：约 20–65 小时；
  - 全量 CPT：基本不现实；
- 最可行路径：
  - 1070 先做 2M–5M 离线缓存；
  - 训练 reader；
  - 用 real/control 和真实任务判断是否值得继续；
  - 如果局部任务有效但知识任务仍无效，则转 RAG / distillation。
