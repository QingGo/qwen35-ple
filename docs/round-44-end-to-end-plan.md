# Round 44：最大化 Qwen3.5-0.8B + PLE 端到端智能水平的实施路线

> 日期：2026-09-03
> 状态：工程 + 研究路线
> 目标：不只是证明 PLE 嫁接是否可行，而是尽量提升最终模型的实际智能水平。

---

## 1. 当前判断

我们已有证据：

- PLE 对 next-token 有正但很小的增量信息；
- 稀有 token 上信号更强；
- 但只训练 target-side reader，real/control 几乎无差异；
- 可训练 h_to_e 会退化；
- differential value 信号太弱，被随机噪声淹没。

因此要提升端到端智能，不能只靠“冻结 backbone + 小 reader + 通用 LM loss”。

---

## 2. 端到端智能提升的总体路线

```text
P0: 问题定义与评测
P1: 记忆读取器增强
P2: backbone 适配（LoRA / 部分解冻）
P3: 后训练 SFT/RL
P4: 混合记忆 / RAG / 蒸馏
P5: 推理效率与 serving
P6: 多次迭代评估
```

---

## 3. P0：先建“端到端智能”评测集

目标不是只看 next-token 或 val loss，而是测真正智能：

- 知识回忆：稀有实体/长尾 QA；
- 推理：多步推理、算术、常识；
- 长上下文：长文档记忆；
- 指令遵循：复杂格式；
- 抗干扰：BoolQ 不失真；
- 真实性：避免编造。

关键对照组：

```text
no-reader
real PLE
control PLE
random PLE
zero PLE
RAG baseline（如有）
```

---

## 4. P1：记忆读取器增强

### 4.1 只注入 E 特有信息

\[
v_{\text{diff}}=
\mathrm{MLP}(H,E_\perp)
-
\mathrm{MLP}(H,0)
\]

### 4.2 条件门控

\[
g(h,e)=
\sigma\!\left(
w_r\cdot \text{rarity}(x)
+
w_u\cdot \text{uncertainty}(h)
+
\text{bilinear}(h,e)
\right)
\]

只在稀有/高不确定性位置放行。

### 4.3 稳定对比训练

用 InfoNCE 或 triplet 训练 value，让 real 与 control/random 产生可区分输出：

\[
L=
-\log\frac{
e^{\mathrm{sim}(v_{\text{real}},R)/\tau}
}{
e^{\mathrm{sim}(v_{\text{real}},R)/\tau}
+
\sum_{c}
e^{\mathrm{sim}(v_c,R)/\tau}
}
\]

### 4.4 防退化

- 固定 ridge 投影；
- 或 FiLM 分离：
  \[
  v=\mathrm{MLP}(E_\perp)\odot\gamma(H)+\beta(H)
  \]

---

## 5. P2：Backbone 适配（很关键）

当前冻结 backbone 可能限制了模型“使用记忆”的能力。

建议尝试：

- **LoRA on selected layers**；
- 或部分解冻后几层；
- 或加入 memory-consuming adapter；
- 训练数据使用“需要记忆才能答对”的任务。

这是从“reader 能注入”到“模型真的会用”的关键步骤。

---

## 6. P3：后训练 SFT/RL

在 reader + backbone 适配之后，做：

- **SFT**：用记忆增强数据，教模型在需要时引用记忆；
- **RL / DPO / GRPO**：奖励正确使用记忆，惩罚 BoolQ 退化；
- 门禁：
  - real > control；
  - real > no-reader；
  - BoolQ 不显著退化；
  - 稀有任务有真实提升。

---

## 7. P4：混合记忆与知识蒸馏

如果 PLE 本身的知识信号不足，可以：

- **RAG / 外部文档记忆**：把 PLE 当作检索索引，或与检索器组合；
- **ReAugKD**：用大模型教师做检索增强知识蒸馏，让 0.8B 学习外部知识；
- **分层记忆**：常见知识由 backbone 负责，长尾由外部记忆负责；
- **MemFlow / MemPO**：参考面向小模型记忆智能体的方法；
- **NVM / MLP Memory**：用更语义化的记忆表示替代 raw PLE。

---

## 8. P5：推理效率与 CPU 100 tok/s

- Store-P / access-order 懒加载；
- 缓存常用 PLE 向量；
- 预计算/量化 e_t；
- vLLM/SGLang/CompileForge 集成；
- 如果 PLE 收益小，考虑：
  - 只在必要 token 上读取；
  - 或把 PLE 表作为推理期外部检索，而不是逐 token 注入。

---

## 9. P6：迭代评估

每个版本都做：

- 全评测集；
- 三线/五线对照；
- 稀有任务分层；
- 真实任务 vs 格式效应；
- 成本/延迟。

---

## 10. 外部研究参考

- Memory Grafting / XMemTransfer
- Pretraining with Hierarchical Memories: separating long-tail and common knowledge
- ReAugKD: Retrieval-Augmented Knowledge Distillation
- MLP Memory / Retriever-Pretrained Memory
- NGM: Training-Free Memory Module
- MemFlow / MemPO for small-model memory agents
- Storage–Retrieval Gap in Parametric Knowledge Graph Memory
- Selective Memory Access / SR-TTT

---

## 11. 优先执行顺序

1. **建立稀有知识评测**：证明 PLE 在真正知识任务上是否 > control；
2. **Fix E_perp + differential value + rare gate**：只注入 E 特有信息；
3. **加入 LoRA / backbone adaptation**：让模型学会用记忆；
4. **SFT/RL 门禁**：real > control 且 BoolQ 不退化；
5. **混合记忆/蒸馏**：如果 PLE 不足，用 RAG/教师蒸馏补足；
6. **CPU/ serving 优化**：最后做产品化。
