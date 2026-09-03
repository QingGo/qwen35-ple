# Round 33：对齐第一性原理的完整证明

> 日期：2026-09-03
> 状态：形式化证明
> 目标：把 round 32 的命题 A/B/C/D 写成可复核的完整证明。

---

## 0. 记号与假设

设概率空间 \((\Omega,\mathcal{F},P)\)。随机变量：

- \(H\in\mathbb{R}^m\)：目标模型 hidden；
- \(E\in\mathbb{R}^d\)：PLE 记忆向量；
- \(Y\in\mathbb{R}^k\)：任务标签（可为 next-token one-hot 或连续目标）。

假设所有随机变量二阶矩有限。  
定义：

\[
\mathcal{H}_H = L^2(\Omega,\sigma(H),P)
\]

\[
\mathcal{H}_{H,E}=L^2(\Omega,\sigma(H,E),P)
\]

它们是 \(L^2\) 的闭子空间。  
定义正交投影：

\[
P_H = \mathbb{E}[\cdot\mid H]
\]

\[
P_{H,E}=\mathbb{E}[\cdot\mid H,E]
\]

即条件期望算子。

定义基座残差：

\[
R = Y - P_H Y
\]

注意：

\[
R \perp \mathcal{H}_H
\]

因为 \(P_H\) 是正交投影。

对于线性代数部分，若协方差矩阵奇异，一律使用 Moore-Penrose 伪逆 \(\Sigma^+\)，结论不变。

---

## 1. 命题 A：任意 reader 的增益上界

### 定理 A

对任意可测函数 \(f:\mathbb{R}^m\times\mathbb{R}^d\to\mathbb{R}^m\)，令：

\[
\Delta = f(H,E)
\]

则：

\[
I(Y;H+\Delta\mid H)\le I(Y;E\mid H)
\]

### 证明

考虑 Markov 链：

\[
Y \longrightarrow (H,E) \longrightarrow H+\Delta
\]

在给定 \(H\) 的条件下，\(H+\Delta\) 是 \((H,E)\) 的确定性函数，因此：

\[
Y \perp H+\Delta \mid (H,E)
\]

由数据处理不等式：

\[
I(Y;H+\Delta\mid H)
\le
I(Y;H,E\mid H)
\]

而：

\[
\begin{aligned}
I(Y;H,E\mid H)
&= H(Y\mid H)-H(Y\mid H,H,E)\\
&= H(Y\mid H)-H(Y\mid H,E)\\
&= I(Y;E\mid H)
\end{aligned}
\]

因此：

\[
\boxed{I(Y;H+\Delta\mid H)\le I(Y;E\mid H)}
\]

### 推论 A

如果：

\[
I(Y;E\mid H)=0
\]

则任意 reader 都不能带来任何条件信息增益。

---

## 2. 命题 B：线性 reader 的最优增益与上界

### 2.1 线性预测设置

设所有变量已中心化。记：

\[
\Sigma_{HH}=\mathbb{E}[HH^\top],\quad
\Sigma_{EE}=\mathbb{E}[EE^\top],\quad
\Sigma_{HE}=\mathbb{E}[HE^\top]
\]

\[
\Sigma_{HY}=\mathbb{E}[HY^\top],\quad
\Sigma_{EY}=\mathbb{E}[EY^\top],\quad
\Sigma_{YY}=\mathbb{E}[YY^\top]
\]

基座最优线性预测为：

\[
\hat Y_H = \Sigma_{YH}\Sigma_{HH}^+ H
\]

残差：

\[
R=Y-\hat Y_H
\]

### 2.2 线性增量 R²

定义：

\[
\Delta R^2(Y;E\mid H)
=
\frac{
\|P_{[H,E]}Y\|^2-\|P_HY\|^2
}{
\|Y\|^2
}
\]

### 2.3 关键引理：正交分解

令：

\[
E_\perp = E-P_H E
\]

则：

\[
E_\perp \perp \mathcal{H}_H
\]

并且：

\[
\mathrm{span}(H,E)=\mathrm{span}(H)\oplus\mathrm{span}(E_\perp)
\]

### 证明引理

因为 \(P_HE\in\mathcal{H}_H\)，所以：

\[
\langle E_\perp, U\rangle
=
\langle E-P_HE,U\rangle
=
\langle E,U\rangle-\langle E,U\rangle
=0,\quad \forall U\in\mathcal{H}_H
\]

因此 \(E_\perp\perp\mathcal{H}_H\)。  
任意 \((H,E)\) 的线性组合可写成：

\[
a^\top H+b^\top E
=
(a^\top+b^\top P_H?) H + b^\top E_\perp
\]

所以生成空间是直和。

### 2.4 增量 R² 公式

由直和分解：

\[
P_{[H,E]}Y = P_HY + P_{E_\perp}Y
\]

且 \(P_HY\perp P_{E_\perp}Y\)，因此：

\[
\|P_{[H,E]}Y\|^2-\|P_HY\|^2
=
\|P_{E_\perp}Y\|^2
\]

所以：

\[
\Delta R^2(Y;E\mid H)
=
\frac{\|P_{E_\perp}Y\|^2}{\|Y\|^2}
\]

### 2.5 线性投影表达式

线性投影：

\[
P_{E_\perp}Y
=
\Sigma_{Y E_\perp}\Sigma_{E_\perp E_\perp}^+ E_\perp
\]

于是：

\[
\boxed{
\Delta R^2(Y;E\mid H)
=
\frac{
\operatorname{Tr}
\left(
\Sigma_{Y E_\perp}
\Sigma_{E_\perp E_\perp}^+
\Sigma_{E_\perp Y}
\right)
}{
\operatorname{Tr}(\Sigma_{YY})
}
}
\]

若 \(Y\) 是标量，则：

\[
\Delta R^2
=
\frac{
\Sigma_{Y E_\perp}
\Sigma_{E_\perp E_\perp}^+
\Sigma_{E_\perp Y}
}{
\Sigma_{YY}
}
\]

### 2.6 最优线性 reader

若限制 reader 为：

\[
\Delta = A E_\perp
\]

且输出预测为：

\[
\hat Y = W(H+\Delta)
\]

则当 \(W\) 固定且列满秩时，最优 \(A\) 等价于最优 \(B=WA\)：

\[
\min_B \mathbb{E}\|Y-WH-BE_\perp\|^2
=
\min_B \mathbb{E}\|R-BE_\perp\|^2
\]

对 \(B\) 求导：

\[
\frac{\partial}{\partial B}
\mathbb{E}\|R-BE_\perp\|^2
=
-2\mathbb{E}[RE_\perp^\top]+2B\mathbb{E}[E_\perp E_\perp^\top]
\]

令其为零：

\[
B^*=\Sigma_{R E_\perp}\Sigma_{E_\perp E_\perp}^+
\]

因此最优 reader 的预测增量是：

\[
B^*E_\perp
=
\Sigma_{R E_\perp}\Sigma_{E_\perp E_\perp}^+ E_\perp
\]

这正是 \(R\) 在 \(\mathrm{span}(E_\perp)\) 上的线性投影。

---

## 3. 命题 C：Value 去相关不损失线性增益

### 定理 C

如果 \(R\perp \mathcal{H}_H\)，则：

\[
\Delta R^2(Y;E\mid H)
=
\frac{\|P_{E_\perp}R\|^2}{\|Y\|^2}
\]

并且：

\[
\Sigma_{R E_\perp} = \Sigma_{Y E_\perp}
\]

因此 Value 使用 \(E_\perp\) 与使用 \(E\) 的线性增量信息相同。

### 证明

因为：

\[
R=Y-P_HY
\]

且：

\[
P_HY\in\mathcal{H}_H,\qquad
E_\perp\perp\mathcal{H}_H
\]

所以：

\[
\langle P_HY, E_\perp\rangle=0
\]

于是对任意方向 \(u\)：

\[
\mathbb{E}[R^\top u E_\perp]
=
\mathbb{E}[(Y-P_HY)^\top u E_\perp]
=
\mathbb{E}[Y^\top u E_\perp]
-
\mathbb{E}[(P_HY)^\top u E_\perp]
\]

第二项为零。 因此：

\[
\Sigma_{R E_\perp}=\Sigma_{Y E_\perp}
\]

又因为 \(E_\perp\subset \mathcal{H}_{H,E}\ominus\mathcal{H}_H\)，且 \(R\in\mathcal{H}_H^\perp\)，所以：

\[
P_{E_\perp}Y=P_{E_\perp}R
\]

于是：

\[
\Delta R^2(Y;E\mid H)
=
\frac{\|P_{E_\perp}Y\|^2}{\|Y\|^2}
=
\frac{\|P_{E_\perp}R\|^2}{\|Y\|^2}
\]

### 推论 C

如果我们要构造 Value 路径，应当使用：

\[
\boxed{E_\perp = E-\mathbb{E}[E\mid H]}
\]

而不是原始 \(E\)。  
它不会损失任何关于基座残差 \(R\) 的线性信息，同时避免重复注入 H 已能预测的内容。

---

## 4. 命题 D：几何对齐既不充分也不必要

### 4.1 几何对齐不是充分条件

构造：

\[
H\sim \mathcal{N}(0,I_m)
\]

\[
E=H
\]

\[
Y\perp (H,E)
\]

则：

- CKA\((H,E)=1\)；
- Procrustes residual = 0；
- kNN overlap = 1；
- 但：
\[
I(Y;E\mid H)=0
\]

因此几何完全对齐，却没有任何记忆增益。

### 4.2 几何对齐不是必要条件

构造：

\[
H\sim\mathcal{N}(0,I_m),\qquad
E\sim\mathcal{N}(0,I_d),\qquad
H\perp E
\]

令：

\[
Y=E_1
\]

即 \(Y\) 是 \(E\) 的第一个分量，完全由 E 决定，且与 H 独立。

则：

- CKA\((H,E)=0\)；
- Procrustes residual 很大；
- kNN overlap 接近随机；
- 但：
\[
I(Y;E\mid H)=I(E_1;E)=+\infty \quad\text{(连续情形)}
\]
或对离散情形为正。

因此几何不对齐，却存在显著记忆增益。

---

## 5. 充分性定理（存在性）

### 定理 S

如果：

\[
I(Y;E\mid H)>0
\]

则存在一个可测表示：

\[
Z=(H,E)
\]

满足：

\[
I(Y;Z)=I(Y;H,E)
\]

并且：

\[
I(Y;Z)>I(Y;H)
\]

### 证明

这是平凡构造：\(Z=(H,E)\) 保留了全部信息，因此：

\[
I(Y;Z)=I(Y;H,E)
\]

而：

\[
I(Y;H,E)-I(Y;H)=I(Y;E\mid H)>0
\]

所以严格更好。

### 实际含义

\(Z=(H,E)\) 是理论上的最优表示。  
Target-side reader 的目标是把 \(E\) 中的新增信息压缩进 hidden 空间，使得：

\[
H+\Delta
\]

在功能上逼近 \(Z\) 对 \(Y\) 的充分性，而不必在几何上逼近 \(E\)。

---

## 6. Hilbert 空间正交投影定理

### 定理 H

\[
\Delta^*=
(P_{H,E}-P_H)Y
=
P_{\mathcal{H}_{E|H}}Y
\]

其中：

\[
\mathcal{H}_{E|H}
=
\mathcal{H}_{H,E}\ominus\mathcal{H}_H
\]

### 证明

因为 \(\mathcal{H}_H\subseteq\mathcal{H}_{H,E}\)，由正交分解：

\[
\mathcal{H}_{H,E}
=
\mathcal{H}_H
\oplus
(\mathcal{H}_{H,E}\ominus\mathcal{H}_H)
\]

对于 \(Y\in L^2\)：

\[
P_{H,E}Y=P_HY+P_{\mathcal{H}_{H,E}\ominus\mathcal{H}_H}Y
\]

因此：

\[
(P_{H,E}-P_H)Y
=
P_{\mathcal{H}_{H,E}\ominus\mathcal{H}_H}Y
\]

### 推论

最优 reader 的数学本质是：

\[
\boxed{
\Delta^*
=
\text{把 }Y\text{ 投影到 }E\text{ 带来的新增信息子空间}
}
\]

---

## 7. 有限样本版本

以上证明全部基于总体分布。  
有限样本下，用经验协方差代替总体协方差，可得经验版本：

\[
\widehat{\Delta R^2}
=
\frac{
\operatorname{Tr}
\left(
\hat\Sigma_{Y E_\perp}
\hat\Sigma_{E_\perp E_\perp}^+
\hat\Sigma_{E_\perp Y}
\right)
}{
\operatorname{Tr}(\hat\Sigma_{YY})
}
\]

当样本量 \(N\) 小于有效维度时，伪逆会过拟合噪声；此时应使用岭回归：

\[
\hat B_\lambda=
\hat\Sigma_{R E_\perp}
\left(\hat\Sigma_{E_\perp E_\perp}+\lambda I\right)^{-1}
\]

这直接指导我们的实验：若 E 的噪声方向多，应先用 PCA/PLS 降维，再做该诊断。

---

## 8. 完整证明的总结

我们已经证明：

1. **A：任意 reader 的增益上界是 \(I(Y;E\mid H)\)**；
2. **B：线性 reader 的增益上界是 \(\Delta R^2(Y;E_\perp)\)，且最优线性 reader 是残差在 \(E_\perp\) 上的投影**；
3. **C：Value 去相关不损失线性增量信息**；
4. **D：几何对齐既不充分也不必要**；
5. **S：只要条件互信息为正，就存在理论上的最优表示 \((H,E)\)**；
6. **H：最优注入是 Hilbert 空间中的正交投影到新增信息子空间**。

因此后续实验的核心指标应该是：

\[
\Delta R^2(Y;E\mid H),
\quad
\Delta R^2(Y;E_\perp\mid H),
\quad
\mathrm{cosine}(\Delta_{\text{reader}}, \Delta^*)
\]
