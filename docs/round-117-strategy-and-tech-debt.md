# Round 117：终极目标、本轮技术债与下一步开发计划

> 日期：2026-09-07
> 目的：在我们已经完成 PLE Projector、真实基准、论文初稿之后，重新校准终极目标和开发顺序。

---

## 1. 终极目标

> **构建一个低资源、可审计、可部署的 0.8B 混合记忆系统：**
> - 以 PLE / n-gram 可寻址外部记忆为“可审计局部记忆”；
> - 与 RAG、参数化 adapter 协同；
> - 严谨证明它的能力边界；
> - 产出可复现论文和 artifact；
> - 最终具备真实部署价值（CPU/低资源推理）。

但这个目标目前已经收敛为一句更诚实的话：

> **证明“可审计 n-gram 外部记忆在低资源小模型上能做什么、不能做什么”，并把它做成一个可复现的系统。**

我们不再追求“PLE 成为通用语义记忆”，因为证据不支持。

---

## 2. 本轮 session 发现了什么

### 2.1 论文层面

- 论文排版已经接近 ICML 风格；
- 但内容厚度仍不足：
  - 缺系统架构图；
  - 缺算法/伪代码；
  - 缺 case study；
  - 缺 ablation / sensitivity；
  - 引用只有 25 条，参考论文 40+；
  - 当前只有 7 页，参考论文 21 页。

### 2.2 实验层面

- HumanEval 只有 20 题；
- TriviaQA exact match = 0；
- pass@k 只有 3×2 小样本；
- LLM-as-judge 只跑了 20 条 TriviaQA；
- 10k 只有 3 seed；
- 没有 memory size / n-gram order / data size 敏感性曲线；
- 没有真实联合系统在真实 HumanEval/TriviaQA 上的完整表。

### 2.3 工程/可复现层面

- 本地环境无法加载 Qwen3.5，远程环境不稳定；
- 没有统一容器化训练/评测环境；
- 没有公开模型权重；
- 没有 CPU/量化效率数据；
- CI 只检查 lint/test，不检查 PDF 或论文构建；
- 论文源码已整理到 `paper/`，但还没有自动化文档构建。

---

## 3. 技术债清单

| 类别 | 技术债 |
|---|---|
| 证据 | HumanEval 20 / TriviaQA 0 / pass@k 小 |
| 证据 | 缺少 10k 5 seed |
| 证据 | 缺少真实联合系统表 |
| 分析 | 缺少 case study / error analysis |
| 分析 | 缺少 ablation / sensitivity |
| 方法 | 无算法伪代码 / 无系统架构图 |
| 方法 | 无 memory scale / n-gram order 实验 |
| 方法 | 无在线 router 完整验证 |
| 工程 | 无容器内可复现训练 |
| 工程 | 无公开权重 |
| 工程 | 无 CPU 100 tok/s |
| 工程 | 无 PDF 构建 CI |
| 论文 | 引用不足 |
| 论文 | 章节内容偏薄 |

---

## 4. 开发计划（按“更稳、更快接近目标”排序）

### Phase 1：论文证据补强（1–2 周）

1. HumanEval 扩到 50；
2. TriviaQA 跑 200 条；
3. 补 10k 5 seed；
4. 补 pass@k 更大样本；
5. 补全 LLM judge 100/200 条；
6. 增加 memory size / data size / n-gram order 曲线。

### Phase 2：论文内容补强（1–2 周）

1. 增加系统架构图；
2. 增加 PLE Projector / router 流程图；
3. 增加算法伪代码；
4. 增加 HumanEval / TriviaQA / open-generation case study；
5. 增加 Threats to Validity；
6. 增加 Broader Impact；
7. 引用扩充到 40+；
8. 每个实验补充 setup / baseline / metric 说明。

### Phase 3：系统与可复现（2 周）

1. 统一 Dockerfile / 环境锁；
2. 增加 PDF 构建 CI；
3. 发布 projector 权重；
4. 发布 adapter 权重；
5. 增加 artifact checksum / release note；
6. CPU 量化与延迟；
7. 训练/评测一键脚本。

### Phase 4：论文投稿准备

1. 最终格式检查；
2. 统计显著性；
3. human/LLM judge 一致性；
4. 附录完整结果；
5. 投稿到 TMLR / ACL Findings / 低资源 workshop。

---

## 5. 可以从类似项目借鉴什么（且不冲突）

| 项目 | 借鉴什么 | 不拿什么 | 如何不冲突 |
|---|---|---|---|
| DeepSeek Engram / Qwen PLE | 可扩展 n-gram lookup、gating、disk-backed memory | 不假设大模型 | 作为底层记忆表 |
| Memory Grafting | offline memory + target-side reader | 不追求全量预训练 | 作为“记忆注入”参考 |
| XMemTransfer | 跨模型 target-side reader、5M–20M scaling | 不照搬大模型资源 | 作为 reader 设计参考 |
| NGM | training-free n-gram memory hook | 不认为它是最终方案 | 作为强基线 |
| MemSFT | 冻结 backbone + token router | 不引入大参数化记忆 | 作为控制层参考 |
| TokenMem | 独立 memory channel + conflict gate | 不把长上下文全塞进 cross-attention | 作为 memory channel 参考 |
| kNN-LM 可靠性工作 | 何时依赖检索/记忆 | 不迷信 kNN | 作为 gate 设计参考 |
| RAG | 文档级检索 | 不替代 token 级记忆 | 作为检索通道 |
| LoRA / QLoRA / MoRA | 参数化能力增强 | 不把 adapter 当记忆 | 作为能力轴 |
| TF-Engram | SSD-backed memory / prefetch | 不依赖大集群 | 作为存储优化 |
| Locas | 局部支持记忆初始化 | 不照搬其训练流程 | 作为记忆理论 |
| Graph of States | 论文结构、图表、case study、严谨性 | 不复制其任务 | 作为写作/实验模板 |
| ReproEvalCard / TMLR | 可复现、评测卡、artifact | 不为了格式牺牲内容 | 作为发布标准 |

---

## 6. 战略原则

1. **证据优先**：先补真实数据和消融，再增加写作。
2. **边界优先**：把“PLE 不能做什么”写清楚，比吹嘘更有价值。
3. **可复现优先**：环境、脚本、权重、评测卡必须完整。
4. **低资源优先**：每一步都要在单卡/CPU 上可完成。
5. **少而深**：与其增加很多弱实验，不如把 3–5 个实验做深。
6. **不冲突架构**：外部记忆、RAG、adapter 不是替代关系，而是三层互补。

---

## 7. 结论

> 我们离目标还差的主要不是“更多方法”，而是：
> - 更完整的证据；
> - 更深入的分析；
> - 更可复现的 artifact；
> - 更厚实的论文内容。

下一步：

```text
先补证据（Phase 1/2）
再补系统（Phase 3）
最后投稿（Phase 4）
```
