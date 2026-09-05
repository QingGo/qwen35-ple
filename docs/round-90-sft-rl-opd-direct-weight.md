# Round 90：SFT / RL / OPD / OPSD 机制与“能否不用 RL 提升能力”多轮调研

> 日期：2026-09-05  
> 状态：完成  
> 方法：30+ 轮 web search，逐步调整关键词；结合数学推导回答以下问题：
> 1. SFT / RL / OPD / OPSD 到底是什么；
> 2. 是否存在“直接调整权重”的方法；
> 3. “SFT 增宽分布、RL 缩窄分布”是否合理；
> 4. 不用 RL 能否实际提升模型能力。

---

## 1. 四种训练方式的核心区别

### 1.1 SFT / Supervised Fine-Tuning

**机制**：使用外部语料 \((x,y)\)，teacher-forcing，最小化

\[
\mathcal L_{\text{SFT}}(\theta)=\mathbb E_{(x,y)\sim D_{\text{data}}}
\Big[-\log p_\theta(y\mid x)\Big].
\]

**特点**：
- 输入分布来自外部数据，不是模型自己的 rollout；
- 目标是“把模型分布向数据分布靠拢”；
- 可以引入新格式、新能力，也可能造成分布偏移/遗忘（alignment tax）。

### 1.2 RL / Reinforcement Learning（特别是 RLVR）

**机制**：模型自己采样轨迹 \(y\sim\pi_\theta(\cdot|x)\)，用奖励/验证器评分，最大化期望奖励，并加 KL 约束防止跑偏：

\[
\max_\theta \mathbb E_{x\sim D,y\sim\pi_\theta}
\big[R(y\mid x)\big]
-\beta\,\mathrm{KL}\big(\pi_\theta\|\pi_{\text{ref}}\big).
\]

**特点**：
- 输入分布是当前策略自己产生的，on-policy；
- 直接优化“对/错、好/坏”等结果信号；
- 适合数学、代码等可验证任务；
- 计算和采样成本高。

### 1.3 OPD / On-Policy Distillation

**机制**：用当前学生策略采样输入/轨迹，再把这些轨迹上的 teacher 分布作为监督：

\[
\mathcal L_{\text{OPD}}(\theta)
=
\mathbb E_{x\sim D,\;y\sim\pi_\theta}
\Big[
\mathrm{KL}\big(\pi_{\text{teacher}}(\cdot\mid x)\;\|\;\pi_\theta(\cdot\mid x)\big)
\Big].
\]

**特点**：
- 数据来自学生自己的 rollout → 减少 off-policy 分布失配；
- 监督信号来自 teacher logits / teacher 文本，而不是奖励；
- 比 RL 便宜；
- 可以保留较多探索/多样性；
- 但若 teacher 本身不好，或学生只会模仿自己的低质量轨迹，会退化。

### 1.4 OPSD / On-Policy Self-Distillation

**机制**：teacher 和学生是同一个模型（或同族模型）的变体，仍然用学生 rollout 构造输入分布，然后向“更优自我版本/验证后的版本”蒸馏。

### 1.5 Purified OPSD

**关键**：On-Policy Self-Distillation 的“提纯”版本，通过验证/过滤/清洗学生 rollout，避免“开卷老师教学生死记硬背”或“模仿低质量 CoT”，从而保住长链思考能力。

代表论文：
- *Purified OPSD: On-Policy Self-Distillation Without Losing How to Think*
- *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*
- *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*

---

## 2. 是否存在“直接调整权重”的方法？

**有，但大多面向局部/事实/行为，不能替代一般能力训练。**

### 2.1 知识编辑类
- ROME：基于因果追踪的 rank-one 直接权重更新；
- MEMIT：多事实、多层的直接权重编辑；
- MEND / AlphaEdit：学习一个 hypernetwork 或 null-space 约束来生成权重更新；
- 适合“改一个事实”，不适合“提升推理能力”。

### 2.2 模型合并 / Task Vector
- Task arithmetic、model merging、MagMax 等；
- 不需要反向传播，直接对权重做加减/插值；
- 适合组合已有能力，不能凭空产生新推理能力。

### 2.3 推理时干预
- Activation steering；
- Representation editing；
- Steer2Edit；
- 不改权重，只改推理时的隐状态/激活方向；
- 可作为低成本“方向性控制”，但不是真正训练。

### 2.4 零梯度外部记忆
- INLAY：frozen model + external addressable memory，约 5ms 零梯度写入、精确删除；
- 与我们 PLE/RAG 方向高度一致；
- 适合外挂知识/事实，不改变模型本身推理。

### 2.5 黑盒/无梯度权重优化
- Evolution Strategies；
- EA4LLM；
- 通过扰动权重/搜索直接优化，不依赖反向传播；
- 近期有 “Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning”；
- 但计算代价仍不低，且主要用于替代 RL/微调，不是魔法。

### 结论
> “直接调整权重”存在，但大多只适合：
> - 单点事实编辑；
> - 已有能力组合；
> - 外挂记忆；
> - 方向性行为控制。
>
> 要凭空提升通用推理/代码/数学能力，仍然需要“信息进入权重”的过程：SFT、RL、OPD 或等效的蒸馏/优化。

---

## 3. “SFT 增宽、RL 缩窄”是否合理？

**部分合理，但不绝对。**

### 3.1 数学表达

设：
- \(P_{\text{pre}}\)：预训练分布；
- \(P_{\text{SFT}}\)：SFT 数据分布；
- \(\pi_{\text{ref}}\)：RL 参考策略；
- \(R\)：奖励。

SFT 最小化：

\[
\mathrm{KL}(P_{\text{SFT}}\,\|\,\pi_\theta)
\]

所以 SFT 会把模型推向外语料分布；若 \(P_{\text{SFT}}\) 比当前模型在某些维度更宽，则增宽；若更窄，则缩窄。因此不能说 SFT 一定是“增宽”。

RL 带 KL 约束的最优解是：

\[
\pi^*(y\mid x)
\propto
\pi_{\text{ref}}(y\mid x)
\exp\!\big(R(y\mid x)/\beta\big).
\]

- 当 \(\beta\to0\)：\(\pi^*\) 收敛到最高奖励轨迹，**分布缩窄/尖峰化**；
- 当 \(\beta\to\infty\)：回到参考分布，不缩窄；
- 中间状态：通常向高奖励区域集中，但也可能因熵奖励保持宽度。

所以：
> SFT 不必然“增宽”，RL 不必然“缩窄”；但 RL 在高奖励任务上确实更容易让分布变尖，SFT 则容易让分布跟着数据走。

### 3.2 相关证据
- Alignment tax / SFT 分布偏移问题；
- “RL preserves prior knowledge better than SFT” 系列工作；
- “Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better”；
- “Reinforcement Fine-Tuning Naturally Mitigates Forgetting”；
- “RL’s Razor: Why Online Reinforcement Learning Forgets Less”；
- “Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT?”。

这些说明：
- RL 的 on-policy 采样减少了与自身分布的不匹配；
- SFT 的外部数据分布容易让模型遗忘/偏移；
- 所以用“增宽/缩窄”描述不够完整，更准确是“数据分布 vs 模型自身分布 + 奖赏浓度”共同决定。

---

## 4. 不用 RL，能否实际提升模型能力？

**能，而且已有大量证据。**

### 4.1 可行路径与论文支撑

| 路径 | 论文/工作 | 要点 |
|---|---|---|
| 强 teacher 蒸馏/SFT | Self-Distilled Reasoner、Purified OPSD、Critique-Guided Distillation、Long CoT synthetic SFT | 用更好模型或验证后的轨迹做监督，不需要 RL |
| On-policy distillation | OPD、Self-Distilled Reasoner、Lite-OPD | 用模型自己的轨迹减少分布失配，比 RL 便宜 |
| Semantic Soft Bootstrapping | “Long Context Reasoning without RL” | 单模型自蒸馏，无 RL，长上下文提升 |
| RAG self-distillation | ReAugKD、我们的 CAP-1 | 检索增强蒸馏，成本低 |
| 高秩 PEFT | MoRA/QLoRA/LoRA/DoRA | 参数高效 post-training，不需要 RL |
| Evolution Strategies | ES at Scale、EA4LLM | 无梯度/无 RL 也能优化模型 |
| 外部记忆 | NGM、TF-Engram、INLAY | 不调权重也能提供知识/记忆 |

### 4.2 限制

- **纯 self-improvement 有天花板**：
  - Self-Improvement Paradox：没有外部脚手架，模型难以真正 bootstrap 新推理能力；
  - 需要 teacher、验证器、新数据或外部知识打破自举循环。
- **RL 的优势**：
  - 对可验证任务（数学/代码）能直接优化正确率；
  - 2025 年后大型模型普遍引入大规模 RL（R1、GRPO、RLVR、Ring-Zero 等）；
  - 但 RL 的采样效率低，资源要求高。
- **我们的结论**：
  - 0.8B 低资源场景**不必先做 RL**；
  - 更实际路线是：
    1. 高质量 RAG self-distill / OPD 式蒸馏；
    2. 验证/过滤轨迹（Purified OPSD 思路）；
    3. MoRA/QLoRA 参数高效适配；
    4. 外挂 PLE/RAG 记忆补足长尾知识；
    5. 若后续有 verifier，再做小规模 RLVR/GRPO。

---

## 5. 对本项目的直接指导

1. **CAP-1 应该升级为 OPD/OPSD 路线**：
   - 当前 RAG self-distill 本质是“外部 teacher 文本 + 学生训练”；
   - 下一步在自采样轨迹上做验证过滤，再蒸馏，就是 Purified OPSD 的本地化实现。
2. **不要只做纯 SFT**：
   - 纯 SFT 容易让知识类能力略降；
   - 应混入 retain 数据、RAG 上下文、验证后的自生成轨迹。
3. **MoRA / QLoRA 保留**：
   - 适合在 8GB GPU 上做 post-training；
   - 作为 CAP-1 的参数化能力提升通道。
4. **PLE/RAG 作为外部记忆**：
   - 不是替代训练，而是把“不需要写进权重”的知识放到可寻址外部；
   - 这与 INLAY、TF-Engram、NGM 的思路一致。
5. **RL 放在后期**：
   - 只有当我们有可验证数学/代码评测和足够采样资源时，才启动小规模 RLVR；
   - 否则用 OPD + 过滤 + 蒸馏更划算。

---

## 6. 关键论文/资源

- *Purified OPSD: On-Policy Self-Distillation Without Losing How to Think*
- *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*
- *Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe*
- *On the Geometry of On-Policy Distillation*
- *Dense Supervision, Sparse Updates: On-Policy Distillation*
- *Semantic Soft Bootstrapping: Long Context Reasoning without RL*
- *Is Human-Written Data Enough? Teaching Reasoning without RL or Distillation*
- *ReAugKD*
- *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*
- *ROME / MEMIT / MEND / AlphaEdit*（直接权重编辑）
- *INLAY*（零梯度外部记忆编辑）
- *TF-Engram / NGM / Memory Grafting*（外部非参数记忆）
- *RL’s Razor / Retaining by Doing / Why RLFT Preserves Prior Knowledge*（RL 遗忘更少）
- *Ring-Zero / DeepSeek-R1 / GRPO*（大规模 RL 路线）

---

## 7. 一句话总结

> SFT 是“用外部数据把分布拉过去”，RL 是“用自己的轨迹、按奖励把分布压到高收益区域”，OPD/OPSD 是“用自己的轨迹、按 teacher 分布做蒸馏”；存在直接权重编辑/模型合并/外挂记忆等方式，但它们适合局部知识和组合，不适合凭空提升通用推理。当前低资源 0.8B 项目应优先走 **RAG self-distill + 验证过滤 + OPD/OPSD + MoRA/QLoRA + PLE/RAG 外挂记忆**，RL 可作为后期可验证任务的可选增强。
