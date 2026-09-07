# Round 128：OPD 的 rollout、teacher logits、数据量与时间估算

> 日期：2026-09-08
> 问题：离线 teacher-text 我们目前有没有？
> OPD 是不是“本地 rollout → 云端 teacher logits → 下载训练”？
> 获取 logits 需要多少时间？数据量多大？

---

## 1. 我们目前有的离线 teacher-text

有，但规模很小，而且不是 Qwen3.8-Flash-Next 生成的：

| 文件 | 行数 | 大小 | 说明 |
|---|---:|---:|---|
| `data/teacher-distill-smoke.jsonl` | 30 | 26KB | teacher-text smoke |
| `data/sources/distilled_corpus_400k_with_cot-filtered.jsonl` | 2,326 | 7.2MB | 已有 CoT/答案文本 |
| `data/cap1-rag-distill-train20.jsonl` | 20 | 100KB | RAG self-distill |
| `data/cap1-rag-distill-160.jsonl` | 160 | 404KB | RAG self-distill |
| `data/cap1-rag-distill-200.jsonl` | 200 | 484KB | RAG self-distill |
| `data/cap1-rag-distill-eval39.jsonl` | 39 | 80KB | held-out |

结论：

```text
我们有“离线 teacher-text”，
但是小规模 + 主要是已有数据/自蒸馏，
不是真正的 Qwen3.8-Flash-Next teacher logits。
```

---

## 2. OPD 的正确流程

你的理解基本正确，但需要精确一点：

```text
1. student 本地生成 rollout：
   y ~ π_θ(y | x)

2. teacher 对 rollout 的每个 token 给分布：
   π_teacher(· | x, y_<t)

3. 损失：
   KL(π_teacher || π_student)
   或 CE(teacher_argmax, student)

4. 反向传播更新 student
```

注意：

```text
OPD 用的是 teacher 的逐 token 分布（logits / logprobs），
不是只给一个标量分数。
```

如果你只用 teacher 的 text：

```text
那更接近 SFT / teacher-text distillation，
不是严格 OPD。
```

---

## 3. 流程上确实可以这样分工

```text
本地 1070：
  1. student 生成 rollout
  2. 保存 rollout text

云端/高 RAM 机器：
  3. teacher 对这些 rollout 做 forward
  4. 保存 teacher logits / logprobs
  5. 上传或打包

本地 1070：
  6. 下载 teacher logits
  7. 训练 student / adapter
```

但实际瓶颈是数据量。

---

## 4. logits 数据量估算

词表大小：

```text
Qwen tokenizer vocab ≈ 248,320
```

### 4.1 全量 logits

```text
每个 token：
248,320 × 4 bytes ≈ 1 MB

1M tokens ≈ 1 TB
20M tokens ≈ 20 TB
```

结论：

```text
保存全量 logits 基本不可行。
```

### 4.2 top-k logits

只保存 teacher top-100：

```text
每个 token：
100 × 4 bytes ≈ 400 B

1M tokens ≈ 400 MB
20M tokens ≈ 8 GB
```

结论：

```text
可行，但仍较大。
```

### 4.3 只保存 teacher logprob（当前生成 token）

```text
每个 token 1 个 float ≈ 4 B
1M tokens ≈ 4 MB
20M tokens ≈ 80 MB
```

结论：

```text
非常小，但只能做 CE / logprob 型蒸馏，不是完全 KL。
```

### 4.4 只保存 teacher text

```text
约等于训练数据本身。
1M tokens 文本可能几 MB 到几十 MB。
```

这就是我们目前有的“离线 teacher-text”。

---

## 5. 获取 logits 需要多少时间

### 5.1 如果 teacher 是 Qwen3.8-Flash-Next，在本地 8GB+48GB RAM

```text
约 35 tok/s
20M tokens ≈ 159 小时 ≈ 6.6 天
```

太慢，不建议。

### 5.2 如果 teacher 在 A100 / H100 上

假设前向吞吐：

```text
1,000–5,000 tok/s
```

```text
1M tokens ≈ 3–17 分钟
20M tokens ≈ 1–6 小时
```

结论：

```text
20M token teacher scoring 在 A100/H100 上约 1–6 小时。
```

### 5.3 如果使用 API teacher

看 API 是否返回：

- 只返回 logprobs：很快，数据很小；
- 返回完整 logits：通常不允许或不划算；
- 按 token 计费：20M tokens 可能需要一定费用。

---

## 6. 1070 本地 rollout 需要多少时间

先要生成 student rollout：

```text
1070 生成 0.8B，约 300–800 tok/s
```

```text
1M rollout tokens ≈ 0.3–0.9 小时
20M rollout tokens ≈ 7–18 小时
```

所以：

```text
如果做 20M tokens，
本地 rollout 和云端 teacher scoring 都会是小时级。
```

建议先做：

```text
100k–1M tokens 的 OPD 小规模验证。
```

---

## 7. 最可行的数据方案

| 方案 | 数据量 | 时间 | 可行性 |
|---|---|---:|---|
| 全量 logits | 20M ≈ 20TB | 小时级 + 存储巨大 | 不可行 |
| top-100 logits | 20M ≈ 8GB | 小时级 | 可行 |
| 生成 token logprob | 20M ≈ 80MB | 小时级 | 很可行 |
| teacher text | 很小 | 快 | 最容易，但是 SFT 而非严格 OPD |
| 只对困难 token 存 top-k | 更小 | 快 | 高性价比 |

---

## 8. 推荐的第一步 OPD 验证

```text
1. 本地 1070 生成 100k–500k rollout tokens
2. 云端 A100/H100 或 API 获取 top-k logprobs
3. 只保存生成 token 的 logprob + top-k
4. 本地训练 LoRA/MoRA
5. 看 held-out 是否比 teacher-text SFT 更好
```

这个量级：

```text
本地 rollout：20–60 分钟
云端 teacher：10–30 分钟
数据量：几十 MB 到几百 MB
总成本：很低
```

---

## 9. 总结

- 我们有“离线 teacher-text”，但是小规模、非 Qwen3.8 真实 logits；
- OPD 确实可以“本地 rollout → 云端 teacher logits → 下载训练”；
- **不要保存全量 logits**，应该保存：
  - top-k logits；
  - 或只保存生成 token 的 logprob；
  - 或只保存困难 token 的分布；
- 20M tokens 全量 logits 不现实；
- 20M tokens top-k/ logprob 数据量可接受；
- 1070 单卡做 20M rollout 约 7–18 小时；
- 建议先做 100k–500k tokens 的 OPD 小规模验证。
