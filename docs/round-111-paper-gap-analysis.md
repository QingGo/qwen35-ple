# Round 111：距离完整论文工作还差什么？

> 日期：2026-09-06
> 状态：系统性差距分析，基于多轮 web 调研和当前实验证据
> 问题：我们是否已经具备一篇完整论文？如果还不具备，差哪些关键环节？

---

## 1. 当前证据快照

### 1.1 PLE / 外部记忆

- PLE 作为 n-gram 可寻址外部记忆；
- code 局部续写：3 seed 上 projector 对 fixed 的平均 NLL 改进 +0.069；
- 1k 数据 seed0：projector 比 fixed 改进 +0.208，hit 0.782 vs 0.724；
- code 是主要优势，name 仍为短板；
- 开放生成：
  - 直接开 raw projector 会退化；
  - 加 learned token policy 后大幅缓解，但仍有长尾泄漏。

### 1.2 Purified OPSD / Adapter

- CAP-1 held-out 多 seed 正提升；
- 正式风格基准（GSM8K-like / MATH-like / HumanEval-like / MBPP-like）只有部分 seed 正；
- LoRA/QLoRA/MoRA 的正式基准表现不稳定。

### 1.3 工程

- RAG + PLE + adapter + router 已能跑通；
- CPU fp32 约 2.24 tok/s，距离 100 tok/s 还很远；
- 已有 CI、lint、manifest、大量实验脚本。

---

## 2. 这篇论文可能是什么

根据现有证据，有三条可选论文路线：

| 路线 | 主张 | 现实性 |
|---|---|---|
| A：系统/边界论文 | “小模型 + 可审计 n-gram 外部记忆 + 学习型投影器能做什么、不能做什么” | 最现实 |
| B：方法论文 | “PLE Projector + token-level learned policy 改进局部低熵续写” | 中低，需要更强正收益 |
| C：规模化方法论文 | “Memory Grafting 式外部记忆在 0.8B 上可替代部分预训练” | 最冒险，缺乏 5M–20M token 规模证据 |

现有证据最支持 **A**，并可作为 **B** 的“诚实边界版”。

---

## 3. 关键差距：按“审稿人会不会拒”排序

### 3.1 真实公开基准（最高风险）

当前所有正式风格基准都是仓库内合成的：

```text
GSM8K-like / MATH-like / HumanEval-like / MBPP-like
```

审稿人会直接问：

> “为什么不用真实 GSM8K、HumanEval、NaturalQuestions？”

需要至少替换/补充：

| 任务 | 建议基准 |
|---|---|
| 代码 | HumanEval、MBPP 官方子集 |
| 数学 | GSM8K、MATH 子集 |
| 知识 QA | NaturalQuestions、TriviaQA、PopQA |
| 长上下文记忆 | LongBench、LongMemEval、MemGym、RULER |
| 多跳推理 | HotpotQA、MuSiQue |
| 可信记忆 | MemTrapBench、needle、real/control |

如果拿不到官方数据，必须在论文中明确写：

> 本工作使用 synthetic proxy，结论只适用于同域局部续写，不宣称通用能力。

### 3.2 强基线

当前主要是：

```text
base / BM25 RAG / n-gram retrieval / PLE fusion / LoRA / QLoRA / MoRA / Purified MoRA
```

还缺少外部记忆和混合记忆领域的标准强基线：

```text
kNN-LM
NGM
MemSFT / TokenMem
Memory Grafting 的 reader（能复现则直接对比）
Memory Layers at Scale
RAG + dense / hybrid retriever
RAG + adapter 的完整系统
```

至少需要一个任务上证明：

```text
PLE > BM25
PLE > kNN-LM
PLE > n-gram LM
PLE > RAG
PLE > no memory
```

如果证明不了，论文应当定位为“边界/负结果”。

### 3.3 统计严谨性

已有：

- 3 seed；
- paired rows；
- paired t-test（正常近似）。

还缺：

- 5–10 seed；
- Wilcoxon / bootstrap；
- 95% CI；
- effect size；
- 多重比较校正；
- 每任务样本量足够；
- 1k 数据的多 seed（目前只有 seed0）。

### 3.4 联合系统评测

当前组件基本分开评估：

- PLE 局部任务；
- Purified OPSD 能力任务；
- RAG 检索；
- PLE Projector；
- token policy。

但还没有一个统一实验：

```text
base
+RAG
+PLE
+Purified MoRA
+RAG + PLE + Purified MoRA
```

在同一批真实任务、同一口径、同一 seed 上比较。

这是“系统论文”的核心表。

### 3.5 规模化曲线

外部记忆论文必须回答容量问题：

| 维度 | 需要的数据点 |
|---|---|
| 记忆规模 | 100 / 1k / 10k / 100k 文档 |
| 数据规模 | 100 / 1k / 10k / 100k 训练 token |
| 模型规模 | 0.8B / 1.5B / 4B（如可行） |
| 上下文长度 | 8 / 32 / 128 / 512 |
| n-gram order | 2 / 3 / 4 |
| reader / projector 容量 | linear / MLP-64 / MLP-256 |

当前只有：

- 100 samples × 3 seed；
- 1k samples × 1 seed；
- 10k dataset 已构建，未训练。

### 3.6 生成质量评估

当前大部分指标是 teacher-forced logprob。

已有文献明确警告：

- kNN-LM 不改进开放生成；
- 困惑度/对数概率不能完全代表生成质量；
- RAG 还需要 faithfulness / answer relevance / context relevance。

需要增加：

- 重复率；
- 结构正确性；
- exact match / pass@k；
- LLM-as-judge 与人工一致性；
- 真实生成案例。

### 3.7 可复现与 artifact

已有：

- CI；
- repro manifest；
- seed 固定；
- 大量脚本。

还需要：

- 公开模型 adapter；
- 公开记忆表 / 数据集（至少样例/哈希）；
- 训练和评测容器/环境锁定；
- artifact 评估清单；
- 不同硬件上的运行时间。

### 3.8 效率 / 部署

当前 CPU fp32 约 2 tok/s，和项目目标 100 tok/s 差距巨大。

如果论文以“低资源可部署”为卖点，必须提供：

```text
CPU tok/s
内存占用
检索延迟
PLE 矩阵大小
量化后精度损失
KV cache / 编译优化
```

否则只能去掉“产品化”主张，只谈科学边界。

### 3.9 理论-实验闭环

仓库已有大量数学推导：

- 条件互信息；
- Blackwell 序；
- log opinion pool；
- 支撑集校准；
- Hedge 后悔界；
- rate-distortion。

但缺少实验对应：

- 验证 log-linear 族是否真的逼近最优；
- 验证 Blackwell 序在真实数据上；
- 展示支撑集校准的 NLL 曲线；
- 展示记忆规模-性能率失真曲线；
- 展示 router 后悔界。

对于 TMLR/ML 会议，这是加分项；对于 ACL 类，可能不是必需。

---

## 4. 最可能的“最小完整论文”

一个现实且可完成的最小论文可以是：

> **“可审计 n-gram 外部记忆在小型冻结 LLM 上的边界：局部低熵续写的收益与开放生成的代价”**

核心贡献：

1. 提出 PLE Projector：用 backbone hidden state + memory 特征学习每 token 的 scale/bias；
2. 提出 token-level learned policy：在开放生成中控制 PLE 是否启用；
3. 实证：code 局部续写收益、name 短板、开放生成退化；
4. 规模：1k/10k dataset，显示数据规模带来改进；
5. 负结果：PLE 不是通用语义记忆，不能替代 RAG / 参数化 adapter；
6. 可审计：real/control、per-task、可解释路由。

这对 **TMLR / ACL Findings / EMNLP Findings / 低资源 workshop** 是可行的。

---

## 5. 如果目标是更高会议（NeurIPS / ICML / ICLR）

必须补上：

1. 真实公开基准上的正收益；
2. 至少一个任务上 PLE 不可替代；
3. 5–10 seed 显著性；
4. 与 Memory Layers / XMemTransfer / Locas 等直接对比；
5. 理论-实验匹配；
6. 完整消融和规模曲线；
7. 公开可复现 artifact。

否则不建议投顶会，容易被判为“另一个 RAG 变体”或“负结果不完整”。

---

## 6. 推荐优先级

| 优先级 | 任务 | 预计资源 |
|---|---|---|
| P0 | 把 HumanEval 或 NQ 或 TriviaQA 真实子集跑通 | 1–2 周 |
| P0 | 补 kNN-LM / NGM / MemSFT 或 TokenMem 基线 | 1 周 |
| P1 | 5 seed / bootstrap / CI 统计 | 1 周 |
| P1 | 联合系统表：base / +RAG / +PLE / +MoRA / +All | 1 周 |
| P2 | 10k 数据 projector 训练 | 1 周 |
| P2 | 开放生成质量指标 + LLM judge | 1 周 |
| P3 | CPU 量化 + 延迟 | 2–4 周 |
| P3 | 公开 artifact + 容器 | 1 周 |

---

## 7. 结论

> 我们现在距离“完整论文”主要缺的不是更多数学，也不是更多内部实验脚本。
>
> 缺的是：
> 1. **真实公开基准**；
> 2. **直接可比的外部记忆基准**；
> 3. **统一联合系统评测**；
> 4. **统计显著性和多 seed**；
> 5. **生成质量而非仅 logprob**；
> 6. **坦诚的边界/负结果定位**；
> 7. **可公开复现的 artifact**。
>
> 基于当前证据，最稳的论文定位是：
> **“可审计 n-gram 外部记忆在低资源小模型上的能力与边界”**，而不是“我们造了一个更好的通用记忆模型”。

---

## 8. 关键参考文献

- [DeepSeek Engram: Conditional Memory via Scalable Lookup](https://github.com/deepseek-ai/Engram)
- [Memory Grafting: Scaling LM Pre-training via Offline Conditional Memory](https://papers.cool/arxiv/2605.20948)
- [XMemTransfer](https://github.com/OLAResearch/XMemTransfer)
- [Memory Layers at Scale](https://mlanthology.org/icml/2025/berges2025icml-memory/)
- [kNN-LM Does Not Improve Open-ended Text Generation](https://aclanthology.org/2023.emnlp-main.929/)
- [RAGRouter-Bench](https://ar5iv.labs.arxiv.org/html/2602.00296)
- [MemTrapBench](https://www.semanticscholar.org/paper/MemTrapBench%3A-Benchmarking-Cognitive-Traps-in-LLM-Wang-Luo/736d61825a5afed4c85b227951a9880d01e2299f)
- [When Not to Trust Language Models](https://arxiv.org/abs/2212.10511v1)
- [TF-Engram](https://arxiv-org.ezproxy.obspm.fr/html/2607.07388v1)
- [Locas](https://ar5iv.labs.arxiv.org/html/2602.05085)
