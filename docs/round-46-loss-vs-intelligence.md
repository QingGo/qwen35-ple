# Round 46：为什么 Scaling Law 的 Loss 代理有效，而我们这里 Loss 下降不代表智能提升

> 日期：2026-09-03
> 目标：从信息论、统计决策论、分布偏移、流形/任务分解等角度严格解释这一问题。

---

## 1. 问题重述

在标准 Scaling Law 中：

\[
L_{\text{val}}\downarrow \quad \Longrightarrow \quad \text{模型能力通常}\uparrow
\]

但在我们的 PLE 嫁接中：

\[
L_{\text{real}}<L_{\text{control}}<L_{\text{no-reader}}
\]

但：

\[
\text{QA EM: } \text{no-reader} \ge \text{control} \ge \text{real}
\]

为什么 Loss 代理失效？

---

## 2. Loss 到底衡量什么

语言模型 loss 是训练分布上的平均负对数似然：

\[
L(\theta)
=
\mathbb{E}_{x\sim \mathcal D_{\text{LM}}}
\left[
-\frac1T\sum_t \log P_\theta(x_t\mid x_{<t})
\right]
\]

它可以写成：

\[
L(\theta)
=
H_{\mathcal D_{\text{LM}}}(X)
+
\mathbb{E}_{x\sim \mathcal D_{\text{LM}}}
\left[
\mathrm{KL}\big(P^*(x_t\mid x_{<t})
\;\|\;
P_\theta(x_t\mid x_{<t})\big)
\right]
\]

其中 \(H_{\mathcal D_{\text{LM}}}(X)\) 是语料本身不可约熵。

因此：

> **Loss 衡量的是：模型在“训练/验证语料分布”上预测下一个 token 的平均校准程度。**

它不是“通用智能”。

---

## 3. Scaling Law 为什么通常有效

在标准 Scaling Law 中，通常满足以下隐含假设：

1. **训练语料分布与下游任务分布高度重叠**；
2. **下游任务可以表示为 next-token 预测的某种函数**；
3. **模型容量/数据增加时，KL 项整体下降，所有相关条件分布都变好**；
4. **评测指标对概率校准单调**或至少近似单调；
5. **没有明显的 shortcut/分布偏差**。

在这些条件下：

\[
\mathrm{KL}(P^*\|P_\theta)\downarrow
\]

会同时改善很多下游任务的预测分布，因此 Loss 可以作为代理。

---

## 4. 为什么代理在 PLE 场景失效

### 4.1 任务分布不同

QA 的评测分布：

\[
\mathcal D_{\text{task}}
\]

通常与：

\[
\mathcal D_{\text{LM}}
\]

不是同一个分布。

Loss 改善只保证：

\[
\mathrm{KL}_{\mathcal D_{\text{LM}}}
\]

下降，不保证：

\[
\mathrm{KL}_{\mathcal D_{\text{task}}}
\]

下降。

### 4.2 评测指标不是 Log-Loss

即使：

\[
\mathrm{KL}_{\mathcal D_{\text{task}}}
\]

下降，EM 也可能不变：

- EM 对概率校准不敏感；
- 只需要 top-1 正确；
- 很多 loss 改进来自次优 token 的概率重排，不影响 top-1；
- 甚至可能让模型更“自信”但更错。

### 4.3 Loss 下降可能来自无关维度

把文本条件分布分解为：

\[
P^*(x_t\mid x_{<t})
=
P^*_{\text{content}}(c_t\mid \cdot)
\cdot
P^*_{\text{style}}(s_t\mid c_t,\cdot)
\]

Loss 下降可能主要来自：

\[
P_{\text{style}}
\]

例如：

- 更常见的格式；
- 更常见的 n-gram；
- 更长的常见文本；
- 更低的整体熵。

而任务智能更多依赖：

\[
P_{\text{content}}
\]

### 4.4 PLE 的本质：n-gram 局部记忆

PLE 是 n-gram 记忆表，天然更容易改善：

- 局部共现；
- 常见短语；
- 格式；
- 低频但局部可预测的 token。

而不一定改善：

- 需要复杂推理；
- 需要跨句实体关系；
- 需要世界知识问答。

这解释了为什么 real 和 control 都能降低 Loss，但任务能力没有提升。

---

## 5. 数学分解：Loss 下降不等于“记忆信息”增加

设：

\[
\Delta L
=
L_{\text{no-reader}}-L_{\text{real}}
\]

它可以分解为：

\[
\Delta L
=
\Delta L_{\text{style}}
+
\Delta L_{\text{content}}
\]

其中：

\[
\Delta L_{\text{style}}
=
\mathbb E_{D_{\text{LM}}}
\left[
\mathrm{KL}(\text{style}_P^*\|\text{style}_\theta)
\right]
\]

\[
\Delta L_{\text{content}}
=
\mathbb E_{D_{\text{LM}}}
\left[
\mathrm{KL}(\text{content}_P^*\|\text{content}_\theta)
\right]
\]

如果：

\[
\Delta L_{\text{style}}\gg \Delta L_{\text{content}}
\]

那么：

\[
\Delta L>0
\]

但任务能力可能不变。

### control 的解释

control 的 PLE 是打乱顺序的同一组向量：

\[
E_{\text{control}}=E_{\text{real}}[\pi]
\]

二者具有相同的：

- 边际分布；
- 范数；
- 局部统计量（在 marginal 意义上）。

因此：

\[
\Delta L_{\text{control}}\approx \Delta L_{\text{real}}
\]

但如果真实语义顺序才是任务所需信息：

\[
I(Y;E_{\text{real}}\mid H)>I(Y;E_{\text{control}}\mid H)
\]

那么 Loss 差异小，任务差异也应小。

---

## 6. 信息论证明：Loss 下降上界不等于任务提升

### 6.1 任务提升的信息量上界

设任务标签 \(Y\)，模型输出 \(\hat Y\)。

\[
I(Y;\hat Y)\le I(Y; \text{all model internals})
\]

若 PLE 提供的新信息为：

\[
\Delta I=I(Y;E\mid H)
\]

则任何端到端任务提升上界为：

\[
\Delta I
\]

### 6.2 Loss 下降与 \(\Delta I\) 的关系

Loss 下降：

\[
\Delta L
\]

主要衡量：

\[
\mathrm{KL}\big(P^*_{\text{LM}}\|P_\theta\big)
\]

而任务提升需要：

\[
\Delta I(Y;E\mid H)
\]

这两者没有必然不等式关系。

可以构造反例：

- \(P_\theta\) 降低了常见文本的 surprisal；
- 但对 \(Y\) 的条件信息没有任何增加；
- 此时：
  \[
  \Delta L>0,\qquad \Delta I=0
  \]

因此：

\[
\boxed{
\Delta L>0 \not\Rightarrow \Delta I(Y;E\mid H)>0
}
\]

---

## 7. Scaling Law 中 Loss 代理的局限

外部研究也指出：

- 下游指标与 perplexity 排名并不总是一致；
- 存在“perplexity 是 surprise 的代理，不是 truth 的代理”；
- Emergent ability 往往在 loss 跨过某个阈值后才出现；
- 小 loss 下降可能落在“平坦区”。

因此 Scaling Law 的 Loss 代理是“经验上有条件成立”，不是“数学上必然成立”。

---

## 8. 对 PLE 场景的结论

我们场景中 Loss 下降但智能未提升，理论解释是：

1. **Loss 下降主要来自 style/局部 n-gram/格式效应**；
2. **control 也能获得同样的 Loss 收益**，说明 Loss 收益主要来自“PLE 向量整体分布”，而不是“真实 token 顺序”；
3. **任务级智能需要的是 \(\Delta I(Y;E|H)\)**，而我们实测很小（线性 0.006，MLP 0.02）；
4. **EM 等离散指标对概率校准不敏感**，小的信息增益不足以改变 top-1 输出；
5. **任务分布与 LM 训练分布不同**，因此 val loss 不具备跨任务代理性。

---

## 9. 数学建议

后续应主要依赖：

\[
\Delta R^2(Y;E\mid H),
\quad
I(Y_{\text{task}};E\mid H),
\quad
\text{real}-\text{control on task accuracy}
\]

而不是：

\[
\text{val loss}
\]

只有当我们能证明：

\[
\Delta I_{\text{task}}>0
\]

并且：

\[
\Delta I_{\text{real}}\gg \Delta I_{\text{control}}
\]

时，Loss 才能重新作为“可靠代理”。
