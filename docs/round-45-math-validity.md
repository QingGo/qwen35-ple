# Round 45：用数学与数据验证“端到端提升路线”的合理性与边界

> 日期：2026-09-03
> 目标：对 round 44 的端到端路线做数学/信息论论证，指出哪些步骤有理论依据，哪些需要谨慎。

---

## 1. 信息论上界

对任意任务 \(Y\) 和任意 reader \(\Delta(H,E)\)：

\[
I(Y;H+\Delta(H,E)\mid H)\le I(Y;E\mid H)
\]

因此：

\[
\boxed{
\text{任何记忆注入的端到端收益都受限于}
I(Y;E\mid H)
}
\]

### 1.1 用线性/MLP ΔR² 估计上界

在 Gaussian 近似下：

\[
I(Y;E\mid H)
\approx
-\frac12\log(1-\Delta R^2)
\]

我们测到：

- 线性 ΔR² ≈ 0.006；
- MLP ΔR² ≈ 0.02。

对应：

\[
I_{\text{lin}}\approx 0.003\ \text{nats},\quad
I_{\text{mlp}}\approx 0.010\ \text{nats}
\]

这些是**每个 token 的条件互信息上界**，绝对量很小。

### 1.2 说明

- 这不等于“PLE 对 QA 一定没用”，因为 QA 的 \(Y\) 不是 next-token；
- 但说明：如果只用 next-token 代理，端到端提升上界非常有限；
- 因此必须建立 **真正知识任务的 \(Y\)**，不能只看 next-token。

---

## 2. 为什么“只训 reader + 冻结 backbone”可能不足

设注入发生在第 \(l\) 层：

\[
h_l'=h_l+\Delta
\]

后续冻结层 \(F\) 将 hidden 映射到 logits。对损失的一阶近似：

\[
\delta\ell
\approx
\nabla_z\ell^\top J_F\Delta
\]

其中：

\[
J_F=\frac{\partial z}{\partial h_l'}
\]

### 2.1 可影响维度

如果 \(\Delta\) 落在 \(J_F\) 的近似零空间中：

\[
J_F\Delta\approx 0
\]

那么即使：

\[
I(Y;E\mid H)>0
\]

注入也不会改变输出。

### 2.2 推论

- 只训练 reader 无法改变 \(J_F\)；
- 因此为了让记忆真正影响端到端输出，**必须适配 backbone**：
  - 部分解冻；
  - 或 LoRA；
  - 或 memory-consuming adapter。

这就是 P2 的数学依据。

---

## 3. 为什么“只注入 E 特有部分”更合理

把 value 分解：

\[
v(H,E)=v_H(H)+v_E(H,E)
\]

其中：

\[
v_H(H)=\mathbb{E}[v(H,E)\mid H]
\]

\(v_H\) 是 H 可预测的部分，不是记忆新增信息。
\(v_E\) 是真正由 E 引起的部分。

### 3.1 数据验证

我们测到 concat MLP reader 的：

\[
\cos(v_{\text{real}}, v_{\text{zero}})\approx 0.98
\]

因此：

\[
\frac{\|v_E\|^2}{\|v\|^2}
=
1-\cos^2(v_{\text{real}},v_{\text{zero}})
\approx 3\%
\]

也就是说：

> 当前 reader 的输出只有约 3% 来自 E 特有内容，其余约 97% 是 H/common 校正。

### 3.2 结论

如果不做 differential / E-specific injection，即使 MLP R² 很高，注入的也主要是 H-correction，而不是 PLE 内容。

因此：

\[
v_{\text{diff}}=v(H,E_\perp)-v(H,0)
\]

是更合理的注入形式。

---

## 4. 为什么 rare-token gate 有依据

令稀有度为 \(R\)。条件增量：

\[
I(Y;E\mid H,R=\text{rare})
\]

我们的数据：

- 稀有 token 线性 ΔR² ≈ 0.015；
- 常见 token 线性 ΔR² ≈ 0.0079。

在 Gaussian 近似下：

\[
I_{\text{rare}}\approx 0.0075,\quad
I_{\text{common}}\approx 0.0040
\]

所以：

\[
I_{\text{rare}}>I_{\text{common}}
\]

因此最优 gate 应依赖稀有度：

\[
g^*(h,e)=\mathbb{1}[\text{rare}(x)\ \text{或}\ \text{uncertainty}(h)]
\]

---

## 5. 为什么高梯度 ≠ 需要记忆

我们观察到：

- 高梯度 token：ΔR² = -0.0048；
- 低梯度 token：ΔR² = +0.0149。

说明：

\[
\text{高梯度} \not\equiv \text{记忆有用}
\]

可能原因：

- 高梯度 token 包含更多不可预测噪声；
- 外部记忆不一定能降低这些梯度；
- 稀有 token 和“难 token”不是同一集合。

因此：

\[
\text{rarity}(x)
\]

比 \(\|\nabla_h\ell\|\) 更适合作为 gate 信号。

---

## 6. 为什么 SFT/RL 可能有用但必须有前提

信息论只给出上界：

\[
I(Y;\hat Y)\le I(Y;E|H)
\]

实际模型可能远低于上界，因为：

- backbone 不会自动使用记忆；
- reader 可能只学到格式效应；
- 模型可能把记忆和普通上下文混淆。

因此：

- 如果上界小（接近 0），SFT/RL 无法创造信息；
- 如果上界为正，SFT/RL 可以缩小“实现差距”。

所以：

\[
\boxed{
\text{SFT/RL 合理的必要条件是}
I(Y;E\mid H)>0
}
\]

当前 next-token 代理下上界为正但很小，因此不能直接断言 SFT/RL 一定有效，必须先在真实知识任务上测 \(I(Y;E|H)\)。

---

## 7. 为什么混合记忆 / RAG / 蒸馏是合理的备选

外部检索文档 \(D\) 可以提供：

\[
I(Y;D\mid H)\gg I(Y;E_{\text{PLE}}\mid H)
\]

如果真实知识任务上 PLE 的 CMI 远小于检索器的 CMI，那么：

- 继续优化 PLE reader 的收益有限；
- 混合 RAG 或教师蒸馏可能更直接提升端到端智能。

这也解释了为什么要把 RAG baseline 纳入评测。

---

## 8. 总结：哪些结论合理，哪些需要谨慎

### 合理且有数学依据

1. 记忆增益有信息论上界；
2. 只训 reader 不够，需要 backbone 适配；
3. differential / E-specific 注入更合理；
4. rare-token gate 有数据支持；
5. SFT/RL 只有在 \(I(Y;E|H)>0\) 时才有意义；
6. 混合记忆/RAG/蒸馏是合理备选。

### 需要谨慎/尚未证明

1. PLE 在真正知识任务上的 \(I(Y;E|H)\) 仍未直接测量；
2. 当前 next-token 上界很小，不能直接外推到 QA；
3. LoRA / backbone adaptation 是否能真正提升，需要实验；
4. “更大训练量一定能提升”尚未证明；
5. RAG 是否优于 PLE，需要同口径对比。

---

## 9. 下一步应当优先测的数学量

1. 在稀有知识任务上测：
   \[
   \Delta I_{\text{task}}=I(Y_{\text{task}};E|H)
   \]
2. 测：
   \[
   \frac{\|v_E\|^2}{\|v\|^2}
   \]
   是否随训练提升；
3. 测 LoRA/backbone adaptation 后：
   \[
   J_F\Delta
   \]
   是否显著非零；
4. 测 RAG baseline 与 PLE 的条件互信息差。
