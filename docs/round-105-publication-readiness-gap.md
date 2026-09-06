# Round 105：如果要把这项工作投向 AI 顶刊/顶会，还缺什么？

> 日期：2026-09-06  
> 方法：20+ 轮 web search，围绕顶会评审标准、RAG/记忆增强论文评估、统计严谨性、可复现性、理论验证和部署指标。  
> 结论：当前工作已经具备“小模型 + PLE + RAG + Purified OPSD”的完整故事线，但距离顶刊还缺少 **正式基准、强基线、统计显著性、深层消融、理论验证、误差分析、效率分析和可复现发布**。

---

## 1. 当前已有什么

- 0.8B 小模型；
- PLE/可寻址 n-gram 外部记忆；
- 任务 router + per-task 校准；
- RAG；
- MoRA/QLoRA/LoRA；
- Purified OPSD；
- 真实局部任务证据：code/name 正，number 待改进；
- 合成正式风格基准：GSM8K-like / MATH-like / HumanEval-like / MBPP-like；
- 形式化数学推导：CMI、对数意见池、支撑集校准、Blackwell、Hedge、Purified OPSD。

---

## 2. 顶刊/顶会视角最缺的实验

### 2.1 正式基准，而不是 “like”

必须换成真实公开基准：

| 任务 | 需要的数据 |
|---|---|
| GSM8K | 官方测试集 |
| MATH | 官方子集 |
| HumanEval | 官方 164 题 |
| MBPP | 官方子集 |
| 知识 QA | NaturalQuestions / TriviaQA / HotpotQA |
| 长上下文记忆 | LongBench / LongMemEval / MemGym / AMA-Bench |
| 开放生成 | 至少一个可控生成评测 |

如果无法使用官方数据，必须在论文中明确标注“synthetic proxy”，并说明局限性。

### 2.2 强基线

需要对比：

- 原始 base；
- 纯 RAG；
- 纯 LoRA / QLoRA / MoRA；
- RAG + MoRA；
- Purified OPSD + MoRA；
- 全参微调（或至少 LoRA 大 rank）；
- kNN-LM / NGM / 类似非参数记忆；
- MemSFT / TokenMem / Memory Grafting 等外部记忆方法（如果可实现）；
- 我们的完整系统 + PLE。

没有这些基线，审稿人无法判断“PLE 是否真的贡献了不可替代的价值”。

### 2.3 统计严谨性

- 3-seed 不够，顶刊一般希望：
  - 5–10 seeds；
  - 或 bootstrap / paired test；
  - 置信区间；
  - effect size；
- 需要报告：
  - mean ± std；
  - 95% CI；
  - 配对 t-test / Wilcoxon；
  - 至少一个主要任务上的显著性检验。

### 2.4 组件级消融

需要系统消融：

| 消融维度 | 具体实验 |
|---|---|
| 是否使用 PLE | base vs +PLE |
| PLE real vs control | 证明不是噪声 |
| Task router | 有 router vs 无 router |
| Gate | 无 gate vs KL gate vs 真实 Δ gate |
| Calibration | 全局 vs per-task |
| 支持集校准 | 使用 / 不使用 Theorem 3 |
| Memory bank 域 | 同域 vs 跨域 |
| n-gram order | 2 / 3 / 4 / 5 |
| Memory size | 100 / 1k / 10k docs |
| Context length | 8 / 32 / 128 / 512 |
| Router online | 固定权重 vs Hedge |
| Purified OPSD | filter vs no filter |

### 2.5 理论验证

我们有数学定理，但还需要实验验证假设：

- 验证受限 log-linear 族是否真的包含/接近真实条件分布；
- 估计 Blackwell 序是否成立（而非直接假设）；
- 验证支撑集质量定理：
  - 绘制 \(q_\beta(S)\) vs \(p_t(S)\)；
  - 展示 β 校准带来的 NLL 变化；
- 验证 rate-distortion / 记忆规模权衡；
- 验证 Hedge router 的后悔界。

### 2.6 误差分析与定性案例

顶刊喜欢：

- 成功案例：PLE 为什么在 code/name 上帮助；
- 失败案例：number 为什么仍失败；
- 质疑案例：PLE 何时有害，router 如何避免；
- 记忆命中 vs 未命中对比；
- 真实生成文本案例，而不是只看 logprob。

### 2.7 人类评估 / LLM-as-judge

- 至少一个子集做人工评估：
  - 答案正确性；
  - 忠实性；
  - 可读性；
- 如果用 LLM-as-judge：
  - 需要报告与人类一致性；
  - 需要讨论 bias；
  - 不能只靠自动指标。

### 2.8 效率 / 部署指标

顶刊需要“低资源”证明：

- CPU tokens/sec；
- 内存占用；
- 延迟；
- 检索/记忆存储成本；
- 与纯 base 的 overhead；
- 量化后精度损失；
- 是否达到 100 tok/s。

### 2.9 可复现性

- release code；
- release model/adapter weights 或至少 adapter；
- release data/benchmark generation scripts；
- 固定 seeds / random states；
- 提供 manifest；
- 提供 artifact evaluation / reproducibility checklist；
- 记录硬件、软件版本、运行时间。

---

## 3. 论文故事线需要补强的分析

### 3.1 “PLE 是什么，不是什么”要更精确

目前定位是：

> 可审计、可寻址、局部低熵外部记忆。

但顶刊会问：

- 与 n-gram LM 有什么区别？
- 与 kNN-LM 有什么区别？
- 与 Engram 原生 PLE 有什么关系？
- 为什么它不是另一个 RAG 系统？

需要一张“方法定位图”，列出：

- parameterized memory;
- retrieval memory;
- exact n-gram address memory;
- latent memory;
- in-context memory.

### 3.2 PLE 的不可替代性

核心审稿风险：

> “你的 PLE 效果可以被 BM25/ngram/kNN 替代。”

因此必须证明至少一个任务上：

- PLE > kNN-LM；
- PLE > 纯 BM25；
- PLE > 纯 dense；
- PLE > 纯 n-gram LM；
- PLE > 无记忆。

如果没有不可替代性，论文应转成“负结果/边界研究”或系统论文。

### 3.3 与 Purified OPSD 的协同

需要证明：

- 单纯 Purified OPSD；
- 单纯 PLE；
- Purified OPSD + PLE；
- 三者关系是互补还是冗余。

用信息论可以计算：

\[
I(Y;M|H),\quad I(Y;A|H),\quad I(Y;M,A|H)
\]

看是否存在冗余/互补。

---

## 4. 多轮搜索中值得引用的相关重要工作

- kNN-LM “When to rely on retrieval”；
- kNN-LM does not improve open-ended generation；
- Why do Nearest Neighbor Language Models Work?
- NGM；
- MemSFT / TokenMem；
- Memory Grafting；
- RAG-as-noisy-ICL；
- LaRA / LongBench / MemGym / AMA-Bench；
- RAGRouter / L-RAG；
- Log Opinion Pool / Bayesian fusion；
- Hedge / prediction with expert advice；
- Rate-distortion memory；
- Contamination detection literature；
- Statistical evaluation tools (evalci/evalstats)；
- Edge/CPU inference works.

---

## 5. 优先级建议

### 如果目标是 NeurIPS/ICML/ICLR

最高优先级：

1. 真实基准 + 强基线；
2. 统计显著性和多 seed；
3. 组件消融 + 理论验证；
4. 可复现 artifact。

### 如果目标是 ACL/EMNLP 等 NLP 顶会

最高优先级：

1. 真实 NLP 任务 + 人工/LLM-judge 评估；
2. 更强的语言生成实验；
3. 误差分析 and case study；
4. 和现有 memory/RAG 方法的直接对比。

### 如果目标是期刊（如 TMLR/JMLR/AI Journal）

最高优先级：

1. 更完整的理论证明；
2. 更多实验规模和泛化；
3. 长期可复现性；
4. 系统/效率评估。

---

## 6. 结论

> 当前工作不是“缺一个实验”，而是缺一套 **顶刊级证据链**：  
> 真实基准、强基线、统计显著、组件消融、理论-实验对应、误差分析、效率数据和可复现发布。  
> 其中最关键的是：  
> **证明 PLE 在至少一个真实任务上不可替代，且证明 Purified OPSD + PLE + RAG 的协同优于任意单一组件。**
