# Round 57：相关工作调研与最优记忆注入方法的数学推导

> 日期：2026-09-04
> 状态：完成多轮检索和形式化推导
> 目标：从已有工作出发，推导“在冻结 backbone 下，什么样的记忆注入是最优的”，并给出可实现的构造。

---

## 1. 多轮调研结果

### 1.1 可直接借鉴的工作

| 工作 | 核心贡献 | 对本项目的启示 |
|---|---|---|
| [XMemTransfer / Memory Grafting](https://github.com/OLAResearch/XMemTransfer) | 冻结源记忆表，训练 target-side reader；exact/longest-match + projection + gate + ShortConv | 说明“记忆读取器”需要单独训练，不能只靠几何对齐 |
| [TokenMem](https://arxiv-org.ezproxy.obspm.fr/html/2607.22625v1) | 独立 cross-attention 记忆通道 + conflict-aware gate，避免与 backbone self-attention 竞争 | 支持“独立记忆通道”，但对容量和 gate 有更高要求 |
| [MemSFT](https://arxiv-org.ezproxy.obspm.fr/html/2607.25614v1) | 冻结 backbone + 冻结 memory LM，只训练 token-level router；以 distribution-level memory 融合 | 最优融合应在“分布/logit 层”，而不是 hidden 层 |
| [DeepSeek Engram](https://github.com/deepseek-ai/Engram) | 条件记忆 via scalable lookup；早期层容量卸载，n-gram 局部记忆 | PLE 本质上提供局部 n-gram 先验，不等于语义知识检索 |
| [ReAugKD](https://www.semanticscholar.org/paper/ReAugKD%3A-Retrieval-Augmented-Knowledge-Distillation-Zhang-Muhamed/191bd8f008f80883bba1fa38908e2c836a5f7bbe) | 用检索增强教师蒸馏学生 | 检索上下文提供了比参数记忆更大的 \(I(Y;D|H)\) |
| [OPD / Purified OPSD](https://arxiv-org.ezproxy.obspm.fr/html/2607.02234v1) | on-policy 学生轨迹 + teacher 分布；去除 reference shortcut | 如果要做蒸馏，Purified OPSD 比朴素 OPSD 更稳健 |
| [MoRA](https://huggingface.co/papers/2405.12130) | 同参数下高秩更新 | 适合作为后续 backbone adaptation 的参数化，但需要先证明记忆通道有信息 |

### 1.2 共同结论

多轮调研的共同结论是：

1. 记忆/检索系统的价值取决于 **它能给 base model 多少任务相关的条件信息**；
2. 如果只在 hidden 层注入，还受 backbone Jacobian 可见子空间限制；
3. **直接在 logits/distribution 层融合**可以绕过 hidden 注入的不可见瓶颈；
4. 最优融合在数学上等价于 **在 base logits 上加上条件对数似然比**。

---

## 2. 形式化框架

设：

- \(H\)：backbone 看到的上下文表示；
- \(M\)：外部记忆/检索结果/PLE 特征；
- \(Y\)：目标 token 或任务标签；
- \(q(y|h)\)：当前 frozen backbone 的输出分布；
- \(P(y|h,m)\)：给定额外记忆后的真实条件分布。

### 2.1 允许的函数族

考虑 logit-level 注入族：

\[
\mathcal Q=
\left\{
p(y|h,m)\propto q(y|h)\exp(\delta(y,h,m))
\right\}
\]

其中 \(\delta\) 是任意可测函数。  
这个族覆盖了所有与 \(q\) 同支撑的分布。

### 2.2 最优 logit 修正

**定理 1（Bayes 最优 logit 修正）**

在 \(\mathcal Q\) 中最小化期望交叉熵：

\[
\min_{p\in\mathcal Q}
\mathbb E_{H,M,Y\sim P}
\left[-\log p(Y|H,M)\right]
\]

的唯一最优解是：

\[
\boxed{
p^*(y|h,m)=P(y|h,m)
}
\]

对应的 logit 修正为：

\[
\boxed{
\delta^*(y,h,m)
=
\log\frac{P(y|h,m)}{q(y|h)}
}
\]

**证明：**

对于任意 \(p\in\mathcal Q\)：

\[
\mathbb E[-\log p(Y|H,M)]
=
H(Y|H,M)
+
\mathbb E_{H,M}\Big[
\mathrm{KL}\big(P(Y|H,M)\|p(Y|H,M)\big)
\Big]
\]

KL 非负，且仅在 \(p=P(Y|H,M)\) 时为零，因此最优。

令：

\[
p(y|h,m)=\frac{q(y|h)\exp(\delta(y,h,m))}{Z(h,m)}
\]

与 \(P(y|h,m)\) 相等时：

\[
\delta(y,h,m)
=
\log P(y|h,m)-\log q(y|h)+\log Z(h,m)
\]

由于 \(\sum_y P(y|h,m)=1\)，取 \(Z=1\) 时：

\[
\delta^*(y,h,m)
=
\log\frac{P(y|h,m)}{q(y|h)}
\]

∎

### 2.3 最优方法能达到的精确收益

**定理 2（若 base 已校准，最大 log-loss 改善 = 条件互信息）**

如果 base 分布恰好是真实边际条件分布：

\[
q(y|h)=P(y|h)
\]

则用定理 1 的 \(\delta^*\) 后，相对 base 的期望 log-loss 改善为：

\[
\Delta L^*
=
\mathbb E_{H,M}\Big[
\mathrm{KL}\big(P(Y|H,M)\|P(Y|H)\big)
\Big]
=
I(Y;M|H)
\]

**证明：**

\[
\Delta L^*
=
\mathbb E_{H,M,Y\sim P}
\left[
\log q(Y|H)-\log P(Y|H,M)
\right]
\]

\[
=
\mathbb E_{H,M}
\left[
\int P(y|H,M)\log\frac{P(y|H,M)}{P(y|H)}dy
\right]
\]

\[
=
\mathbb E_{H,M}\mathrm{KL}\big(P(Y|H,M)\|P(Y|H)\big)
=
I(Y;M|H)
\]

∎

**含义：**

- 这就是我们之前的硬上界；
- 但定理 1 告诉我们该上界是可实现的，**只要在 logit 层直接学习条件对数似然比**；
- hidden 注入并不是实现该上界的必要方式，反而是次优通道。

---

## 3. Hidden 注入与 logit 注入的差异

### 3.1 线性化

设 frozen backbone 从 hidden \(h\) 到 logits \(\ell\) 的 Jacobian：

\[
J=\frac{\partial \ell}{\partial h}
\]

我们希望实现的 logit 修正是 \(\delta\ell\)。

### 3.2 最优 hidden 注入

**定理 3（给定 logit 修正下的最优 hidden 注入）**

在最小二乘意义下，最优 hidden 注入是：

\[
\boxed{
\Delta h^*
=
J^\top (J J^\top)^+ \delta\ell
}
\]

实际到达 logits 的修正是：

\[
J\Delta h^*
=
P_{\mathrm{col}(J)}\delta\ell
\]

其中 \(P_{\mathrm{col}(J)}\) 是到 \(J\) 列空间的正交投影。

**证明：**

对任意 \(\Delta h\)：

\[
\|J\Delta h-\delta\ell\|^2
\]

这是关于 \(\Delta h\) 的凸二次函数。对 \(\Delta h\) 求导：

\[
J^\top(J\Delta h-\delta\ell)=0
\]

得：

\[
\Delta h=(J^\top J)^+ J^\top \delta\ell
=
J^\top(J J^\top)^+ \delta\ell
\]

进一步：

\[
J\Delta h
=
J J^\top(J J^\top)^+ \delta\ell
=
P_{\mathrm{col}(J)}\delta\ell
\]

∎

### 3.3 推论：可见信息上界

如果 logit 修正来自记忆 \(M\)，hidden 注入最多只能实现：

\[
P_{\mathrm{col}(J)}\delta^*
\]

因此：

\[
I(Y;H+\Delta(H,M)\mid H)
\le
I\big(Y;P_{\mathrm{col}(J)}\delta^*\mid H\big)
\]

这比 \(I(Y;M|H)\) 更紧，因为它扣除了被 frozen backbone 抹掉的部分。

---

## 4. Router / 多专家融合的最优性

记忆和 backbone 可以看成两个“专家”：

\[
\ell_{\text{fused}}=(1-\alpha)\ell_{\text{base}}+\alpha\ell_{\text{mem}}
\]

**定理 4（逐 token 最优 router）**

对每个 \((h,m)\)，最优 router 是：

\[
\alpha^*(h,m)
=
\arg\min_{\alpha\in[0,1]}
\mathbb E_{Y\sim P(\cdot|h,m)}
\left[
-\log \mathrm{softmax}\big((1-\alpha)\ell_{\text{base}}+\alpha\ell_{\text{mem}}\big)(Y)
\right]
\]

该问题是凸的，因为 softmax 交叉熵关于 logits 是凸函数，而 logits 关于 \(\alpha\) 是仿射的。

**实现建议：**

- 不要手动固定 \(\alpha\)；
- 用一个小 MLP 从 \(h,m\) 预测 \(\alpha\)；
- 用真实任务损失训练；
- 这正是 MemSFT 的“只训练 router”做法。

---

## 5. 由此推导出的最优方法

### 5.1 最优记忆注入管线

```text
外部记忆源 M
  │
  ├─ PLE E        条件信息小
  ├─ 检索文档 D   条件信息大（已验证）
  └─ 教师分布 T   蒸馏目标
  │
  ▼
学习条件对数似然比：
  δ*(y) ≈ log P(y|h,m) − log q(y|h)
  │
  ▼
logit-level 融合：
  ℓ_fused = ℓ_base + router × δ*(y)
  │
  ▼
训练目标：
  CE(ℓ_fused, Y)
```

### 5.2 为什么这比 hidden injection 更优

- Hidden injection 受 \(J\) 的列空间限制，丢失 \(P_{\mathrm{col}(J)}^\perp\) 方向的信息；
- Logit injection 直接修改输出分布，可以实现定理 1 的最优 \(\delta^*\)；
- Router 可以逐 token 决定信任哪个专家；
- 若记忆源本身信息不足（如 PLE），logit 注入也只能得到很小的收益，但这已经是该源的理论最优。

### 5.3 对当前项目的决策

我们已实测：

- RAG 文档 \(D\)：rare Δ ≈ +0.851 logprob；
- PLE \(E\)：rare real−control ≈ +0.00013，不显著。

根据定理 2：

\[
I(Y;D|H)\gg I(Y;E|H)
\]

因此：

- 继续把 PLE 作为主记忆源不划算；
- 应优先实现 **检索文档/教师分布作为 M** 的 logit-level 融合；
- 如果以后仍想用 PLE，应实现 \(\delta^*(y)=\log P(y|H,E)/P(y|H)\) 的估计，而不是继续依赖 hidden reader。

---

## 6. 实现路线

### 6.1 立即可做

1. 用当前 RAG baseline 作为“外部记忆教师”；
2. 训练一个学生/记忆 head，学习：

   \[
   \delta^*(y)\approx\log\frac{P_{\text{teacher}}(Y|H,D)}{P_{\text{base}}(Y|H)}
   \]

3. 在 logit 层相加，并用 learned router 融合；
4. 这等价于 ReAugKD + MemSFT router 的结合。

### 6.2 如果继续 PLE

1. 用更大/更多样的 PLE 源构建 bank；
2. 计算 \(I(Y;E|H)\) 和 \(I(Y;E_{\text{control}}|H)\)；
3. 若仍接近 0，按定理 2 可知即使最优 logit 注入也无法带来显著收益；
4. 停止 PLE 主线，保留为可选局部 n-gram 先验。

### 6.3 理论/实验闭环

建议实现的“最优方法”包含三个可验证声明：

- 声明 1：logit-level 最优修正 = 条件对数似然比；
- 声明 2：logit-level 效果 ≥ hidden-level 效果，因为 hidden 会丢失 \(P_{\mathrm{col}(J)}^\perp\)；
- 声明 3：记忆源选择应依据 \(I(Y;M|H)\)，RAG/教师 > PLE。

这三个声明都可以用同口径 real/control + 离散指标验证。

---

## 7. 引用

- [XMemTransfer](https://github.com/OLAResearch/XMemTransfer)
- [TokenMem](https://arxiv-org.ezproxy.obspm.fr/html/2607.22625v1)
- [MemSFT](https://arxiv-org.ezproxy.obspm.fr/html/2607.25614v1)
- [DeepSeek Engram](https://github.com/deepseek-ai/Engram)
- [ReAugKD](https://www.semanticscholar.org/paper/ReAugKD%3A-Retrieval-Augmented-Knowledge-Distillation-Zhang-Muhamed/191bd8f008f80883bba1fa38908e2c836a5f7bbe)
- [Purified OPSD](https://arxiv-org.ezproxy.obspm.fr/html/2607.02234v1)
- [MoRA](https://huggingface.co/papers/2405.12130)
