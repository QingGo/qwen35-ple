# Round 72：系统性复盘 v2——重新定位 PLE 后的目标与路线

> 日期：2026-09-04
> 状态：在 PLE 主创新定位下重新规划
> 目的：既保留 PLE 的创新价值，又以可复现、低资源、可产品化的方式提升 0.8B 端到端能力。

---

## 1. 终极目标

### 1.1 北极星

> **以 PLE/外部稀疏记忆为核心创新，构建一个低资源、可复现、可审计、可部署的 0.8B 混合智能系统，并尽可能提升其端到端表现。**

PLE 不再是“备选”，而是“主创新”；但主创新不等于“当主要预测器”。

### 1.2 四层目标

| 层次 | 目标 | 验收 |
|---|---|---|
| 科学 | 说明 Engram/PLE 到底能提供哪类信息 | \(I(Y;E|H)\)、\(I(Y;C|E_{\text{ngram}})\)、real/control |
| 创新 | 构造“可寻址残差记忆 / 长尾外部知识库” | 非参数残差记忆 + 多源 router |
| 工程 | 可复现混合系统 | 评测、3-seed、污染审计、CI |
| 产品 | 0.8B + PLE/RAG 可部署变体 | CPU 100 tok/s + bundle |

### 1.3 当前核心认知

- PLE 不适合当作“语义知识预测器”；
- 普通 LLM 智能来自组合计算，不是查表；
- PLE 更适合：
  - 长尾/局部/低熵
  - 可寻址外部知识
  - 非参数残差
  - 边缘/本地稀疏记忆
- 因此：
  - PLE 作为主创新，但角色是“参数模型的互补外部记忆”。

---

## 2. 本轮 session 发现的技术债

### 2.1 科学/证据债

| # | 技术债 |
|---|---|
| S1 | 尚未验证 n-gram/PLE 在低熵/代码/专名任务上 real>control |
| S2 | 尚未度量 \(I(Y;C\mid E_{\text{ngram}})\) |
| S3 | 尚未评估“PLE 只 gate 长尾/低熵”是否保住通用能力 |
| S4 | 尚未测量多源融合的消融 |
| S5 | 多任务评测仍缺真实 GSM8K/MATH/HumanEval/MBPP |
| S6 | 3-seed 覆盖不足 |

### 2.2 方法/架构债

| # | 技术债 |
|---|---|
| M1 | 已实现 NgramLM，但未做 logit 融合实验 |
| M2 | 未实现“非参数残差记忆” |
| M3 | 未实现语义可寻址 PLE（value=文档/知识） |
| M4 | 未实现多源凸 router |
| M5 | 未实现 RAG self-distillation |
| M6 | 未实现真实 teacher logits 蒸馏 |
| M7 | 未做联合小规模预训练/PLE from scratch |

### 2.3 工程债

| # | 技术债 |
|---|---|
| E1 | 混合检索 dense 仍是弱 embedding |
| E2 | 未接入生产 serving |
| E3 | 无 CPU 100 tok/s 闭环 |
| E4 | 数据 provenance 不完整 |
| E5 | 新实验缺少统一评测协议 |

---

## 3. 借鉴矩阵：哪些可以借，且不冲突

### 核心原则

> PLE 作为“外部记忆/残差/长尾知识”，与 RAG、teacher、参数化模型天然互补，不冲突。

| 项目 | 借什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| Memory Grafting / XMemTransfer | 可寻址外部记忆、target-side reader | 不把 PLE 当语义预测器 | 可作为残差记忆接口 |
| Ordo-M / Prometheus Mind | 冻结模型 + 外部稀疏记忆 | 不复制其全部架构 | 支撑“可寻址记忆”主创新 |
| NGM / 经典 n-gram LM | 训练无关 logit 插值 | 不把 n-gram 当最终智能 | 作为局部专家 |
| DeepSeek Engram / Qwen PLE | n-gram 查表、容量卸载 | 不把原论文收益照搬 | 重新定义为互补记忆 |
| RAG / ReAugKD | 输入通道、teacher logits | 不把项目变成纯 RAG | PLE 可做词法 key/长尾旁路 |
| OPD / Purified OPSD | 学生轨迹+teacher 分布 | 不直接照搬长 CoT | 能力迁移，与 PLE 并行 |
| QLoRA / LoRA / MoRA | 低资源 adaptation | 不要求全参数 | 用于蒸馏/后训练 |
| PERK | test-time 适应 | 不作为主线 | 长上下文补充 |
| Hierarchical Memory | 长尾知识与推理分离 | 不重训大表 | 正是 PLE 定位 |
| 信息论/统计决策 | CMI、Bayes 最优、Blackwell 序 | 不替代实验 | 用于设计门禁 |

---

## 4. 后续开发计划

### Phase PLE-1：证明 PLE 的真正价值

目标：

- 确定 PLE/n-gram 在哪些任务上 real>control。

任务：

1. 建低熵/代码/专名/数字评测集；
2. 用 `NgramLM` 跑 real vs control；
3. 估计：
   \[
   \lambda^* = \frac{\mathrm{Cov}(L_t-L_b,\;L_n-L_b)}{\mathrm{Var}(L_n-L_b)}
   \]
4. 测量：
   \[
   I(Y;C\mid E_{\text{ngram}})
   \]
5. 门禁：
   - 若 PLE 在低熵/代码/专名上 real>control → 进入 PLE-2；
   - 若没有 → 仍做“语义可寻址/边缘”等非智能提升路径。

### Phase PLE-2：PLE 主创新架构

目标：

- 实现“非参数残差记忆 / 可寻址外部知识库”。

任务：

1. 非参数残差记忆：
   \[
   \hat Y=P_{\text{base}}(Y|H)+\text{router}\times \text{retrieved residual}
   \]
2. 语义可寻址 PLE：
   - key：n-gram / 词法；
   - value：文档片段 / 段落向量 / 知识条目；
3. 长尾 gate：
   - 只在低熵/长尾/未知 token 激活；
4. 多源凸 router：
   - base + RAG + PLE + teacher。

门禁：

- PLE 在长尾/局部任务上 real>control；
- RAG/通用任务不退化。

### Phase CAP-1：能力提升主线

目标：

- 先提升 0.8B 的推理/代码/格式。

任务：

1. RAG self-distillation；
2. 高质量数据筛选 + QLoRA/MoRA；
3. 若有高 RAM：
   - Qwen3.8 离线 teacher logits；
   - OPD / Purified OPSD。

门禁：

- 多任务上蒸馏后 > 蒸馏前；
- 通用能力不退化。

### Phase CAP-2：混合系统集成

目标：

- 把 PLE、RAG、teacher、base 融合成完整系统。

任务：

1. 多源 router；
2. 消融：base / +RAG / +PLE / +teacher；
3. 3-seed；
4. 污染审计。

### Phase PROD：产品化

目标：

- 0.8B + PLE/RAG 可部署。

任务：

1. 量化/GGUF/ExecuTorch；
2. CPU 100 tok/s；
3. bundle + manifest；
4. e2e。

---

## 5. 如何更稳地前进

1. **先证 PLE 角色，再谈主创新**
   - 不因为“创新”而放弃实验门禁。
2. **先小后大**
   - 每次小规模验证，再放大。
3. **保留负面结果**
   - PLE 若在词法任务也失败，仍有“可寻址外部记忆”的创新价值可做。
4. **能力与创新分开评估**
   - 能力：RAG/蒸馏；
   - 创新：PLE 架构。
5. **可复现**
   - 脚本、配置、manifest、文档、CI。

---

## 6. 优先级一览

| 优先级 | 事项 |
|---|---|
| P0 | PLE 低熵/代码/专名 real vs control |
| P0 | N-gram λ* 与 \(I(Y;C|E_{\text{ngram}})\) |
| P1 | RAG self-distillation |
| P1 | 数据筛选 + QLoRA/MoRA |
| P2 | 非参数残差记忆 |
| P2 | 语义可寻址 PLE |
| P2 | 多源 router |
| P3 | Teacher logits / OPD |
| P3 | CPU serving / 量化 |

---

## 7. 一句话

> PLE 的主创新价值不在于“替代 LLM 的智能”，而在于“作为可寻址、非参数、长尾外部记忆，和参数化模型形成互补”。
>
> 接下来的路线是：先证明 PLE 真正擅长什么，再把它做成主架构，同时用 RAG/蒸馏提升 0.8B 的实际能力。
