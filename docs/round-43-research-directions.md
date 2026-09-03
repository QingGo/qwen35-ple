# Round 43：基于当前证据与外部研究的下一步研究方向

> 日期：2026-09-03
> 状态：研究规划
> 目的：结合已有实证和外部文献，列出后续最值得做的研究方向。

---

## 1. 当前证据总结

| 发现 | 含义 |
|---|---|
| 线性 ΔR²(E∥H) 约 +0.006 | 记忆增量小，但为正 |
| MLP ΔR² 约 +0.02 | 非线性可提取更多，但绝对量仍小 |
| PLS r=64 超过全维度 | 信号可被监督低秩方向捕获 |
| 稀有 token ΔR² 约为常见 token 2 倍 | 记忆更偏向长尾 |
| 高梯度 token ΔR² 为负 | 对“最难”token 不一定有帮助 |
| 可训练 h_to_e 退化 | 会把 H 走私进 E_perp |
| MLP(E_perp) 无法预测 R | E_perp 必须与 H 联合 |
| Differential 注入 real/control 差异被随机噪声淹没 | 当前 E 特有信号太弱 |
| Contrastive hinge 训练发散 | 需要更稳定的对比目标 |

---

## 2. 外部研究启示

### 2.1 XMemTransfer / Memory Grafting

- 需要 **5M–20M token 量级** target-side reader 训练；
- 我们目前只有 1024 token 拟合 MLP，可能远不足以形成可用的记忆读取。

### 2.2 NGM / 训练无关记忆模块

- 或许可以绕过训练不稳定性；
- 适合作为“不需要训练”的对照，或与 trained reader 组合。

### 2.3 MLP Memory / Retriever-Pretrained Memory

- 用检索器预训练记忆编码；
- 可能比直接使用 PLE n-gram hash 特征更语义化。

### 2.4 Storage–Retrieval Gap

- 指出 adapter 可能学到“输出条件化/格式效应”，而不是真正检索存储内容；
- **这非常像我们观察到的 real/control/random 几乎无差异**；
- 应专门设计指标区分“记忆检索”和“通用输出调节”。

### 2.5 When Not to Trust Language Models

- 需要选择“参数记忆真正失败、非参数记忆真正有用”的任务；
- 我们当前用通用 next-token 梯度，可能混入太多局部语言模式。

### 2.6 Selective Memory Access / Gating

- 外部工作强调选择性存储和读取；
- 我们的稀有 token 结果支持在“稀有/高不确定性”位置做条件 gate。

### 2.7 SR-TTT / Surprisal-Aware

- 对高 surprise token 做残差调整；
- 可能比我们的“高梯度”更接近“真正需要外部记忆”的条件。

---

## 3. 建议研究方向（按优先级）

### 方向 A：稀有 token 知识评测

**目标**：不再用通用 next-token 梯度，而是构造“稀有 token 补全 / 长尾 QA / 实体事实回忆”任务。

**实验**：
- 从语料中提取低频实体；
- 构造 cloze / QA；
- 对比 no-reader / real / control / random。

**判读**：
- 如果 real 在稀有任务上显著 > control，则说明当前 PLE 有知识价值；
- 如果仍然差不多，则记录为负面证据。

### 方向 B：Rare-token Conditioned Gate

**目标**：让 gate 只在高稀有度 / 高不确定性 token 上放行。

**实现**：
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

**理论依据**：
- 稀有 token ΔR² 更高；
- 简单 token 上注入容易退化。

### 方向 C：稳定的 Contrastive Value 训练

**目标**：让 value 对 real/control 产生可区分响应，但避免发散。

**替代方案**：
- InfoNCE：
  \[
  L=
  -\log\frac{\exp(\mathrm{sim}(v_{\text{real}},R)/\tau)}
  {\exp(\mathrm{sim}(v_{\text{real}},R)/\tau)+
  \sum_{c\in\{\text{control,random}\}}\exp(\mathrm{sim}(v_c,R)/\tau)}
  \]
- Triplet loss：
  \[
  \max(0,\|v_{\text{real}}-R\|^2-\|v_{\text{control}}-R\|^2+\alpha)
  \]
  但需要梯度截断/谱归一化。

### 方向 D：固定/约束 E_perp

**目标**：防止 h_to_e 把 H 走私进 E_perp。

**方案**：
- 使用固定 ridge 投影；
- 或对 h_to_e 加正交约束；
- 或使用 FiLM/Hypernetwork 分离 H 与 E 路径：
  \[
  v=\mathrm{MLP}(E_\perp)\odot\gamma(H)+\beta(H)
  \]

### 方向 E：大规模训练 + 稀有 token 重采样

**目标**：把 MLP value 的训练量从 1024 token 提升到 100k–1M。

**采样**：
- 过采样稀有 token 上下文；
- 使用梯度残差损失 + LM loss 联合。

### 方向 F：记忆表示学习

**目标**：不再直接用 PLE hash 特征，而是学习语义化记忆编码。

**候选**：
- PLS 监督低秩编码；
- Autoencoder / VQ；
- Retriever-pretrained memory；
- NGM 训练无关模块。

---

## 4. 最优先的 3 个实验

1. **Rare-token FAQ / cloze 评测**：先确定“在真正需要知识的任务上，real 是否 > control”；
2. **Rare-token gate + differential value**：只在稀有 token 上注入，重复 BoolQ 和稀有任务；
3. **稳定 contrastive value loss + 更大训练集**：用 InfoNCE/triplet 替代发散 hinge。

如果这 3 个实验仍不能使 real 稳定优于 control，则建议：
- 将当前 PLE 表定位为“局部语言模式增强”而非“知识记忆”；
- 或换用更语义化的记忆表示（MLP Memory / NGM / 检索器）。

---

## 5. 相关文献

- XMemTransfer / Memory Grafting
- NGM: Plug-and-Play Training-Free Memory Module
- MLP Memory: Retriever-Pretrained Memory
- Storage–Retrieval Gap in Parametric Knowledge Graph Memory
- When Not to Trust Language Models
- Training Language Models with Memory Augmentation
- Selective Memory Access / Memory for LLMs
- SR-TTT: Surprisal-Aware Residual Test-Time Training
- TokenMem: Faithful Knowledge Injection for Frozen LLMs
