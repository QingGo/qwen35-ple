# Round 55：P1/RAG 结果与数学推导的一致性检查

> 日期：2026-09-04
> 状态：理论需要从“必要上界”扩展为“可实现的通道/容量/任务相关性”理论
> 结论：现有实验没有推翻核心信息论上界，但证明该上界是必要不充分条件；需要补充实现通道、任务相关信息分解和评测协议。

---

## 1. 总体判断

我们此前最重要的数学结论是：

\[
I(Y;H+\Delta(H,E)\mid H)\le I(Y;E\mid H)
\]

以及：

\[
\Delta R^2(Y;E\mid H)
=
\frac{\|P_{E_\perp}Y\|^2}{\|Y\|^2}
\]

P1 和 RAG 的结果与这两个结论 **不矛盾**：

- Phase A 已经测得真实 PLE 的 \(\Delta R^2\sim 10^{-4}\sim10^{-3}\)，非常小；
- P1 实测 rare real−control ≈ \(+0.00013\)，与“条件信息量极小”一致；
- RAG 的收益远超 PLE，也与“外部文档的条件信息远大于 PLE n-gram 信息”一致。

所以问题不是“数学错了”，而是：

> **我们此前把必要条件误当成了充分条件：以为只要 \(I(Y;E|H)>0\)，或者 reader 能提升 \(R^2\)，就应有端到端收益。**

---

## 2. 实验结果与已有理论一致的部分

### 2.1 信息上界

- 理论：任何 reader 增益受限于 \(I(Y;E|H)\)。
- 实测：纯 PLE 的 task-level \(\Delta R^2\) 只有约 \(10^{-4}\sim10^{-3}\)。
- 结论：P1 没有显著 real−control 是理论上可预期的。

### 2.2 冻结 backbone + 小 reader 可能无效

- Round 45 已指出：注入 hidden 后还要经过冻结层 \(J_F\)，如果 \(\Delta\) 落在 \(J_F\) 的近零空间，则不改变输出。
- P1 正是“冻结 backbone + 小型 memory module”，实证结果支持这一风险。

### 2.3 Loss/logprob 提升不等于任务提升

- Round 46 已指出：loss 主要衡量 LM 分布，可能主要来自 style/局部 n-gram。
- P1 中 memory module 对 real 和 control 都大幅提升 answer-logprob，但 real−control 几乎为零。
- 这直接验证了“通用分布效应/格式效应”，而不是 PLE 内容特有的因果增益。

### 2.4 EM/离散指标对小概率变化不敏感

- Round 46 已指出：小的 logprob 变化可能不改变 top-1。
- P1 first-token hit 在所有条件下相同，验证了这一点。

### 2.5 RAG/蒸馏可能是更直接路线

- Round 45 已预测：外部检索文档可能提供
  \[
  I(Y;D\mid H)\gg I(Y;E_{\text{PLE}}\mid H)
  \]
- RAG baseline 实测：
  - rare Δ = +0.851，152/182 wins；
  - common Δ = +2.070，77/88 wins。
- 这支持“问题不在小模型，而在 PLE 通道本身提供的信息量/类型”。

---

## 3. 需要修改的理论部分

### 3.1 从“条件信息上界”到“可实现信息上界”

原理论只给出：

\[
\text{端到端增益} \le I(Y;E\mid H)
\]

应改为：

\[
\text{端到端增益}
\le
\min\Big(
I(Y;E\mid H),\;
\underbrace{\sup_{\Delta\in\mathcal{C}} I(Y;H+\Delta\mid H)}_{\text{可实现通道容量}}
\Big)
\]

其中 \(\mathcal{C}\) 是当前 reader 类别 + backbone 是否可训练 + 信息注入位置决定的实现类。

如果：

- reader 输出主要落在 \(J_F\) 的近零空间；
- 或 reader 只能表达低秩/common 校正；
- 或 frozen backbone 无法把 hidden 注入转化为输出变化；

那么即使 \(I(Y;E|H)>0\)，端到端收益仍可接近 0。

### 3.2 区分“输入通道”和“hidden 注入通道”

RAG 和 PLE 的信息进入方式不同：

| 通道 | 机制 | 对 frozen backbone 的依赖 |
|---|---|---|
| 输入上下文（RAG） | 直接把文档 token 放入输入，所有层可见 | 低；主流 attention 自然可用 |
| hidden 注入（PLE reader） | 在某一层向 hidden 加 \(\Delta\) | 高；取决于 \(J_F\) 是否保留该方向 |

因此：

> “需要 backbone adaptation”这一结论应限定为 **hidden-state 记忆注入**；不适用于输入上下文增强（RAG/蒸馏）。

这是本轮最重要的理论修正之一。

### 3.3 增加“任务相关信息分解”

PLE 可能包含信息，但这些信息可能不是任务所需的信息：

\[
I(Y;E\mid H)
=
I(Y;E_{\text{task}}\mid H)
+
I(Y;E_{\text{local/style}}\mid H)
\]

- PLE 擅长局部 n-gram、短语、格式，因此能降 loss、能提升 real/control 的绝对 logprob；
- 但在 rare 知识任务上，真正需要的是实体/事实/跨句关系，PLE 的 \(I(Y_{\text{task}};E|H)\) 很小。

因此理论门禁应改为：

\[
\boxed{
I(Y_{\text{task}};E_{\text{real}}\mid H)
-
I(Y_{\text{task}};E_{\text{control}}\mid H)
}
\]

而不是只看：

\[
I(Y_{\text{LM}};E\mid H)
\]

### 3.4 明确“正 CMI”不是充分条件

原证明中的“存在性定理”说的是存在理论最优表示 \((H,E)\)，但并没有说“我们的小 reader + frozen backbone 能实现它”。

需要增加阶梯：

```text
I(Y;E|H) > 0
    ↓ 必要
reader 能表示 Δ* ≈ P_{E⊥}R
    ↓ 必要
backbone/head 不被该方向的零空间抹掉
    ↓ 必要
训练/优化能逼近该表示
    ↓ 必要
任务级 real−control（不是绝对 logprob）显著
```

任何一步不满足，即使 \(I>0\) 也可能没有端到端收益。

### 3.5 修改评测证据标准

不应以以下任一作为“PLE 有效”的证据：

- val loss 下降；
- real 相对 no-reader 的绝对 logprob 提升；
- memory module 本身能改善 QA。

应改为：

- 同口径 **real vs control** 的 paired 差异；
- 任务级 **first-token/EM/accuracy**，而不仅是平均 logprob；
- 3-seed 或至少 paired t-test；
- 控制 bank/控制上下文不能包含测试答案泄漏。

---

## 4. 对后续实验的影响

| 原计划 | 修改后 |
|---|---|
| 只有 \(\Delta R^2>0\) 就推进 backbone adaptation | 必须先有任务级 real−control，或证明 hidden 注入通道的 \(J_F\Delta\neq0\) |
| 把 PLE 作为主记忆路径 | 暂时降级为局部语言先验；主路径转 RAG/蒸馏 |
| 只看 logprob 提升 | 必须看 real−control + 离散指标 |
| 认为冻结 backbone 不够就应解冻 | 仅对 hidden 注入成立；RAG 输入通道不需要解冻也有收益 |

---

## 5. 一句话总结

我们的数学没有错，但之前把它说得太强了：

> **条件互信息是端到端记忆收益的必要上界，不是充分条件。**

P1 的负结果和 RAG 的正结果共同说明：真正需要补的理论是“信息如何通过可实现通道到达输出”，以及“任务需要的信息是否真的在 PLE 中”。
