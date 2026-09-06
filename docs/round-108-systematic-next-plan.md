# Round 108：系统性下一步计划（基于多轮调研 + 现有证据）

> 日期：2026-09-06  
> 状态：研究驱动路线图  
> 时间预算：无限（但按价值排序）  
> 原则：**学习 > 手工规则；局部低熵记忆 > 通用语义记忆；可审计系统 > 黑盒记忆。**

---

## 1. 核心判断

我们已经通过大量实验确认：

- PLE 作为 **通用语义/开放 QA 记忆**：不可替代性弱，BM25/RAG 更强；
- PLE 作为 **code / 局部低熵续写记忆**：有一定价值，尤其是 BM25+PLE 混合；
- 手写 query 规则不可持续，违背 Bitter Lesson；
- 数据驱动的 token/ngram-level policy 已经有初步正信号：
  - 460 样本，accuracy 0.720，AUC 0.809；
  - 在开放生成保护上可以替代关键词规则。

因此下一步不是继续堆关键词，而是：

> **把 PLE 从“手工融合的外部 n-gram”升级为“可学习的记忆头 / PLE Projector”，并配一个可学习的 token/ngram-level router。**

---

## 2. 调研得到的可借鉴结论

### 2.1 外部记忆 / Engram 路线

- **DeepSeek Engram / Qwen PLE 原生结构**
  - 不是简单 n-gram 查表，而是：
    - key/value projection；
    - context-aware gating；
    - ShortConv；
    - 残余注入 hidden states。
  - 说明“记忆必须经过可学习投影和门控，才能进入主干语义空间”。
  - 参考：[DeepSeek Engram System Overview](https://deepwiki.com/deepseek-ai/Engram/3.1-system-overview)、[engram-peft paper alignment](https://github.com/QingGo/engram-peft/blob/master/docs/paper_alignment.md)

- **Memory Grafting**
  - 离线冻结条件记忆 + 轻量 projection/gating；
  - 支持“冻结大表、训练小适配器”的路线。
  - 参考：[Memory Grafting: Scaling LM Pre-training via Offline Conditional Memory](https://papers.cool/arxiv/2605.20948)、[Semantic Scholar](https://www.semanticscholar.org/paper/Memory-Grafting%3A-Scaling-Language-Model-via-Offline-Cheng-Guan/2f4a698decd659e5e6e8d4348cc77403e0a23093)

- **XMemTransfer**
  - 强调 target-side reader adaptation；
  - 关键结论：**5M tokens 才开始有竞争力，20M 基本饱和**。
  - 这解释了我们早期 MLP reader 失败的主因：**训练量差 100 倍以上**。
  - 参考：[XMemTransfer GitHub](https://github.com/OLAResearch/XMemTransfer)、[Qwen3.5-4B 20M adapter](https://huggingface.co/OLAResearchX/xmemtransfer-qwen35-4b-from-pythia160m-20m)

### 2.2 多模态对齐的启示

- 多模态不是“手工调一个 scale”，而是：
  - 冻结 visual encoder；
  - 训练 **projector / Q-Former / Perceiver**；
  - 用 image-text pairs 做对齐；
  - 再通过 instruction tuning 接入 LLM。
- 参考：
  - [Multimodal Alignment and Fusion: A Survey](https://ar5iv.labs.arxiv.org/html/2411.17040v1)
  - [Deciphering Cross-Modal Alignment in LVLMs](https://ar5iv.labs.arxiv.org/html/2410.07167v1)
  - [CapRecover alignment module discussion](https://ar5iv.labs.arxiv.org/html/2507.22828)

对应到 PLE：

```text
外部记忆（n-gram / PLE table）
    ↓
PLE Projector（类似 image projector）
    ↓
Backbone hidden / logit 空间
    ↓
和主干一起生成
```

### 2.3 Learned Router / Adaptive RAG

- **RAGRouter-Bench** 已经把“是否/如何检索”作为一个可学习路由问题；
- **X-Router** 把 knowledge vs reasoning 路由用于成本控制；
- **TokenMem** 面向冻结 LLM 的忠实知识注入；
- 这些都支持我们的方向：
  - 不是“所有任务都开 PLE”；
  - 而是 **让模型/数据学习什么时候用、用多少**。
- 参考：
  - [RAGRouter-Bench](https://ar5iv.labs.arxiv.org/html/2602.00296)
  - [X-Router](https://aclanthology.org/2026.findings-acl.994/)
  - [TokenMem](https://arxiv-org.ezproxy.obspm.fr/html/2607.22625v1)

### 2.4 Bitter Lesson

- 手工规则不可扩展；
- 应该用 **搜索、学习、通用方法** 替代人工特征。
- 参考：[Bitter Lesson Explained](https://www.taskade.com/blog/bitter-lesson-explained)

---

## 3. 我们现在的资产

| 资产 | 状态 |
|---|---|
| P0 局部任务证据 | ✅ code/name/number |
| BM25+PLE 混合 | ✅ 3 seed |
| token/ngram-level learned policy | ✅ 460 样本 |
| learned policy 替代关键词验证 | ✅ |
| PLE 检索通道安全 | ✅ |
| Purified OPSD | ⚠️ 局部 held-out 正，正式基准不稳定 |
| code corpus / serving | ✅ 基础可用 |
| 可复现 manifest | ✅ |
| 真实 HumanEval/GSM8K | ❌ 尚无 |

---

## 4. 下一步技术路线（按价值排序）

### Phase A：PLE Projector（最高优先）

把“是否用 PLE”升级为“如何把 PLE 映射到主干 logit 空间”。

```text
输入：
  - 当前 backbone hidden state
  - 匹配到的 n-gram order
  - memory 分布 / memory entropy
  - density ratio
输出：
  - logit bias 或 scale/bias
目标：
  - next-token cross-entropy
训练策略：
  - 冻结 backbone
  - 训练小 projector / adapter
  - 可选 LoRA 或最后几层解冻
```

参考 XMemTransfer / Memory Grafting / 多模态 projector。

### Phase B：把 learned policy 从“二值 gate”升级为“连续控制”

当前 `TokenPlePolicy` 只决定：

```text
用 PLE 或不用
```

下一步应输出：

```text
scale / bias / 是否应用 / 置信度
```

这样：

- 可以用小 MLP 同时做 router 和 projector；
- 避免硬阈值；
- 更像多模态 connector。

### Phase C：收集“生成质量”标签

当前 policy 标签是：

```text
PLE fusion 是否提升 teacher-forced logprob
```

但这不等于生成质量。

下一步应收集：

- 生成是否退化；
- 是否复读；
- 是否包含正确代码结构；
- 人工/LLM judge 评分；
- 自动 exact-match。

然后把这些标签加入 policy 训练。

### Phase D：扩大本地数据

当前只有 P0 局部任务 ~460 个 token 观测。

应扩展到：

- code corpus 更大；
- wiki / 私有语料；
- 如果可能，构造 1M–5M token 的局部续写数据；
- 用 XMemTransfer 的“5M 才有竞争力”作为目标。

### Phase E：Backbone 策略

不要全冻结：

```text
冻结 backbone
+ PLE Projector
+ 小 LoRA / 最后 N 层解冻
```

参考 Prometheus Mind / XMemTransfer。

### Phase F：开放生成保护与 serving

- PLE 检索通道始终保留；
- PLE logit fusion 由 learned policy/projector 控制；
- 在 code 局部续写上开启；
- 在开放生成上自动关闭或衰减。

### Phase G：评测升级

- 真实 HumanEval / MBPP 子集（如果可以获取）；
- 真实 GSM8K / MATH 子集；
- code 局部续写 paired test；
- 生成质量 LLM judge；
- CPU 延迟/量化；
- 消融：projector vs gate vs fixed calibration vs BM25-only。

---

## 5. 里程碑与 Go/No-Go

### M1：PLE Projector v0
- 在 P0 code/name/number 上：
  - fixed calibration；
  - learned gate；
  - learned projector；
- 3 seed paired；
- 如果 projector > fixed calibration 且不伤害开放生成 → 继续。

### M2：规模放大
- 1M token 局部续写数据；
- projector + LoRA/部分解冻；
- 如果仍无稳定正收益 → 记录边界，转系统/负结果论文。

### M3：真实基准
- HumanEval / MBPP / GSM8K 子集；
- 如果无法获取 → 明确 synthetic proxy 限制。

### M4：产品化
- CPU 量化；
- KV cache；
- PLE retrieval + learned policy serving；
- 100 tok/s 或可接受低资源指标。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 数据量不足 | 扩大本地语料 / 合成/自蒸馏 / 1M+ |
| 过拟合局部任务 | 3+ seed、cross-validation、真实基准 |
| 开放生成退化 | learned policy + 生成质量标签 |
| 手工规则回潮 | 所有策略尽量从数据学习 |
| 通用语义错位 | 不追求全局语义，只做局部低熵 |
| 评测不可信 | per-item、paired test、bootstrap CI |

---

## 7. 结论

> 下一步不是继续调 PLE 的超参，而是：
>
> **把 PLE 做成一个可学习的记忆头 + 可学习的 token/ngram router。**
>
> 借鉴多模态投影器、Engram 原生 gating、XMemTransfer 的 target-side reader、Memory Grafting 的离线冻结记忆，以及 RAGRouter-Bench 的可学习路由思想。
>
> 用数据替代关键词，用局部低熵任务替代通用语义幻想，用可审计检索替代黑盒记忆。
