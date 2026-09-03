# Round 29：从数学角度推导“最有效对齐”的目标与实验指引

> 日期：2026-09-03
> 状态：调研 + 数学推导草稿
> 目标：把当前实证结论上升为可检验的数学量，指导后续 reader / loss / 训练实验。

---

## 1. 调研结论

已有工作的核心共识：

- **Memory Grafting / XMemTransfer**：冻结源记忆表，训练 target-side reader，让目标模型学会读取记忆；训练量通常需要数百万 token，不只是“几何对齐”。
- **Contrastive Learning**：InfoNCE / 对比损失本质上是在对齐正负样本分布，可被看成一种分布对齐问题。
- **Representational Alignment**：CKA、Procrustes、CCA、Gromov-Wasserstein 等工具衡量表示之间的相似性/可映射性，但它们并不直接回答“记忆是否能带来任务增益”。

我们当前的实证显示：

- PLE e_t 与 Qwen hidden 的 CKA 低、Procrustes 残差大、kNN overlap 接近随机；
- real 与 control 都显著提高 next-token entropy，random/zero 不明显；
- real 相对 control 的优势依赖注入强度：1.0 附近最大，2.0 时 control 反超。

因此需要重新定义“对齐”：**不应该追求 e_t 与 h 的全局相似，而应该追求 e_t 对“基座无法预测的任务残差”的条件可解码性。**

---

## 2. 符号与问题

设：

- \(H \in \mathbb{R}^m\)：目标模型某层 hidden state；
- \(E \in \mathbb{R}^d\)：PLE 记忆向量（\(d=2560\)）；
- \(Y\)：下一个 token 或下游任务标签；
- \(W\)：语言模型输出头（固定或与训练无关）；
- \(R = Y - \mathbb{E}[Y \mid H]\)：基座已经无法预测的“任务残差”。

记忆注入的数学目标是：

\[
\min_{\text{reader}} \mathbb{E}\big[\ell(Y; W(H + \Delta(H,E)))\big].
\]

在局部线性化下，最优注入量可理解为：

\[
\Delta^*(H,E) \approx \mathbb{E}\big[R \mid H,E\big].
\]

也就是说，**reader 应该逼近给定 (H,E) 后任务残差的条件期望**，而不是让 E 和 H 长得像。

---

## 3. 核心数学量：条件增量可解释性

任意 reader 能带来的最大信息增益受数据不等式约束：

\[
I(Y; \Delta(H,E) \mid H) \le I(Y; E \mid H).
\]

因此第一个要测的量不是 CKA，而是：

\[
\Delta R^2(Y; E \mid H)
= R^2(Y; H,E) - R^2(Y; H).
\]

在线性近似下，它是“在控制 H 后，E 对 Y 的增量线性可解释方差”。

### 关键命题

**命题 1（冗余无害的边界）**
如果 \(E = L H + \varepsilon\)，且 \(\varepsilon \perp H\)，那么：

\[
\Delta R^2(Y; E \mid H) = R^2(Y; \varepsilon \mid H).
\]

若 \(E\) 被 H 线性决定（\(\varepsilon \approx 0\)），则：

\[
\Delta R^2(Y; E \mid H) \approx 0.
\]

因此 **CKA / Procrustes 越高，不一定越好；当它反映的是 E 被 H 线性预测时，它恰恰说明记忆是冗余的。**

**命题 2（最优线性 reader）**
令 \(\tilde E = E - \Pi_H E\) 为去掉 H 线性可预测部分后的记忆残差。若 reader 采用线性注入：

\[
\Delta = A \tilde E,
\]

则最优 \(A\) 满足：

\[
A^* = (W^\top W)^{-1} W^\top \Sigma_{R\tilde E}\Sigma_{\tilde E}^{-1},
\]

其中 \(\Sigma_{R\tilde E} = \mathbb{E}[R \tilde E^\top]\)。

对应的增量增益为：

\[
\eta^2
= \frac{\operatorname{Tr}(\Sigma_{R\tilde E}\Sigma_{\tilde E}^{-1}\Sigma_{\tilde E R})}
{\operatorname{Tr}(\Sigma_{R R})}.
\]

**如果 \(\eta^2\) 接近 0，那么任何线性 reader 都无法带来真实记忆收益。**  
这个量比 CKA 更能指导我们是否应继续投入 5M–20M / RL。

---

## 4. Gate 的数学角色

当前官方 reader 的 gate 本质是双线性形式：

\[
g(h,e) = \sigma\!\left(
\frac{\langle W_q h,\; W_k e\rangle}{\sqrt{m'}}
\right).
\]

它可以学习一个低秩交叉协方差矩阵：

\[
B = W_q^\top W_k \in \mathbb{R}^{m \times d}.
\]

因此：

- **Key/value 应当分工**：
  - Key 路径 \(W_k E\) 负责“门控相关性”：应该与 \(W_q H\) 有一定局部/线性对齐，使模型能判断“现在该不该读记忆”；
  - Value 路径 \(W_v E\) 负责“内容注入”：应该尽量包含 \(H\) 无法提供的新信息，而不是重复 \(H\) 的预测。

- 如果 Value 路径没有做 H-正交化，它会把基座已经能预测的信息再注入一次，造成：
  - 对已正确 BoolQ 的扰动；
  - real 和 control 都提高 entropy；
  - 高 scale 时 control 反超。

**建议架构方向**：
\[
E_{\parallel} = \Pi_H E,\qquad
E_{\perp} = E - \Pi_H E.
\]

- Gate 使用 \(E_{\parallel}\) / 或从 \(E\) 学到的“与 H 对齐”的 key 表示；
- Value 使用 \(E_{\perp}\) / 或从 \(E\) 学到的“与 H 去相关”的 value 表示。

这可以在数学上同时满足：

1. 信息新异性：\(E_{\perp}\) 与 H 不相关；
2. 门控可行性：key 与 query 可对齐。

---

## 5. 可推导的实验指导

### 5.1 先做“增量可解释性”诊断（最重要）

在线性探针意义下计算：

- \(R^2(Y; H)\)
- \(R^2(Y; H, E)\)
- \(R^2(Y; H, E_{\perp})\)
- \(\Delta R^2(Y; E \mid H)\)
- \(\Delta R^2(Y; E_{\perp} \mid H)\)

如果 \(\Delta R^2\) 很小，说明当前 PLE 记忆对目标任务的“新信息”很少，问题不在对齐方法，而在记忆表或任务选择。  
如果 \(\Delta R^2\) 主要来自 \(E_{\perp}\)，说明正交化 value 路径有理论依据。

### 5.2 建立“门控可对齐性”指标

- 对 Key 表示 \(K=W_k E\) 和 Query 表示 \(Q=W_q H\)，计算：
  - CCA / CKA；
  - kNN overlap；
  - 线性可分性 / contrastive accuracy。
- 目标不是让 \(E\) 与 \(H\) 全局相似，而是让 **门控所用的表示** 局部可比。

### 5.3 修改 reader / loss

建议优先做以下实验：

```text
L = L_LM
  + λ1 * L_alignment(K(E), Q(H))        # 门控可对齐
  + λ2 * L_task(V(E), R)                # 内容可解码
  + λ3 * ||Corr(V(E), H)||^2            # 内容去冗余
  + λ4 * H(gate) / gate sparsity        # 控制注入强度，防止破坏 BoolQ
  + λ5 * KL(reader_on || reader_off)    # 防止扰动基座
```

其中 \(L_task(V(E), R)\) 可以直接用 next-token 损失对 hidden 的梯度作为回归目标，或用线性 probe 的增量 R² 作为代理。

### 5.4 Scale 结论的数学解读

我们的 scale sweep 显示：

| scale | real−control | real entropy |
|---:|---:|---:|
| 0.25 | +0.14 | 0.89 |
| 0.5 | +0.38 | 1.20 |
| 1.0 | +0.59 | 2.23 |
| 2.0 | -0.27 | 3.95 |

- 过强的线性放大不会增加“条件互信息”，只会放大已有信号和噪声；
- 2.0 时 control 反超，说明噪声项被放大到超过真实信号；
- 因此最优不是无限加大 scale，而是提高 **信噪比**：即提高 \(\Delta R^2\)，同时压低 gate 非选择性注入。

---

## 6. 总结

从数学上看，最有效的“对齐”不是让 PLE e_t 与 Qwen hidden 在 CKA/Procrustes 意义上相似，而是：

1. **让记忆向量对“基座无法预测的任务残差”具有高增量可解释性**；
2. **让 gate 所使用的 key/query 表示具备局部可对齐性**；
3. **让 value 注入与 H 去相关**，避免重复已有信息；
4. **控制 gate 选择性与注入强度**，减少对简单任务（BoolQ）的扰动。

因此后续实验应该：

- 先测 \(\Delta R^2(Y; E \mid H)\) 与 \(\Delta R^2(Y; E_{\perp} \mid H)\)；
- 再据此决定是改门控、改 loss，还是承认当前 PLE 表/任务不提供足够新信息；
- 在没有看到明确 \(\Delta R^2\) 正信号之前，不进入 5M–20M 和 RL。
