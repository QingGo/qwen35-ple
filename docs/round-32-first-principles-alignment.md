# Round 32：第一性原理——对齐的本质不是几何相似，而是条件充分性

> 日期：2026-09-03
> 状态：理论框架 + 证明草图 + 实验设计
> 目标：从第一性原理重新定义“对齐”，并推导什么才是最有效的对齐方式。

---

## 1. 第一性问题

在 PLE 嫁接中，“对齐”到底指什么？

我们之前默认它是：

> PLE e_t 空间与 Qwen hidden 空间在某种度量下相似。

但这个定义有几个问题：

1. CKA / Procrustes / kNN 高，不代表记忆有用；
2. CKA / Procrustes / kNN 低，不代表记忆无用；
3. 不同任务可能要求不同的“对齐”。

因此需要从更基本的问题出发：

> 一个表示 \(Z\) 的本质是什么？
> 两个表示“对齐”到底要保留什么？

---

## 2. 表示的本质：充分统计量

令：

- 输入 \(X\)
- 表示 \(Z=f(X)\)
- 任务标签 \(Y\)

表示的本质是**对输入的有损压缩**。  
用信息论语言，一个表示 \(Z\) 对任务 \(Y\) 的“好”程度由：

\[
I(Y;Z)
\]

衡量。

### 2.1 充分统计量

如果：

\[
I(Y;Z)=I(Y;X)
\]

则 \(Z\) 是关于 \(Y\) 的充分统计量。  
它保留了任务所需的全部信息，尽管可能丢失其他信息。

### 2.2 对齐的定义

对两个表示 \(Z_1,Z_2\)，我们可以定义“关于任务族 \(\mathcal{T}\) 对齐”：

\[
Z_1 \equiv_{\mathcal{T}} Z_2
\iff
\forall T \in \mathcal{T}:\quad
I(Y_T;Z_1)=I(Y_T;Z_2)
\]

也就是：

> 两个表示如果对某个任务族保留完全相同的信息，它们就是关于这个任务族对齐的。

这完全不需要几何相似。

---

## 3. 几何相似既不是必要条件，也不是充分条件

### 反例 1：几何相似但无用

设：

\[
E = H + \varepsilon,\qquad \varepsilon \perp Y
\]

则：

- CKA 很高；
- Procrustes 残差很低；
- 但：
\[
I(Y;E\mid H)=0
\]

所以几何对齐很高，但记忆完全无用。

### 反例 2：几何不相似但有用

设：

\[
E = g(H) + \text{noise}
\]

其中 \(g\) 是非线性函数，且 noise 含有稀有 token 信息。  
则：

- CKA 可能低；
- Procrustes 可能差；
- 但：
\[
I(Y;E\mid H)>0
\]

所以几何不对齐，但记忆有用。

### 结论

> 几何相似性是“足够但不必要也不充分”的性质。  
> 真正必要且充分的条件是：**E 对某个任务族具有条件增量信息**。

---

## 4. 第一性原理定理

### 4.1 必要性

对任意 reader \(\Delta(H,E)\)：

\[
I(Y;H+\Delta(H,E)\mid H)
\le
I(Y;E\mid H)
\]

因此如果：

\[
I(Y;E\mid H)=0
\]

则不存在任何 reader 能带来记忆增益。

### 4.2 充分性

如果：

\[
I(Y;E\mid H)>0
\]

并且 reader 的函数类包含或可以逼近：

\[
\Delta^*(H,E)
=
\mathbb{E}[Y\mid H,E]
-
\mathbb{E}[Y\mid H]
\]

那么从原则上，记忆增益可以达到。

### 4.3 推论

对齐问题本质上不是“把 E 映射到 H 的空间”，而是：

\[
\boxed{
\text{学习条件期望差}
\ \Delta^* =
\mathbb{E}[Y\mid H,E]
-
\mathbb{E}[Y\mid H]
}
\]

所以最优 reader 是条件期望差的一个函数逼近器。

---

## 5. 从 Hilbert 空间看：这就是一个正交投影

令：

- \(\mathcal{H}_H = L^2(\sigma(H))\)
- \(\mathcal{H}_{H,E}=L^2(\sigma(H,E))\)
- 目标函数空间 \(L^2(Y)\)

则：

\[
\mathbb{E}[Y\mid H]
\]
是 \(Y\) 在 \(\mathcal{H}_H\) 上的正交投影：

\[
P_H Y
\]

而：

\[
\mathbb{E}[Y\mid H,E]
\]
是 \(Y\) 在 \(\mathcal{H}_{H,E}\) 上的正交投影：

\[
P_{H,E}Y
\]

因此：

\[
\Delta^* = (P_{H,E}-P_H)Y
\]

它是 \(Y\) 在“新增信息子空间”上的投影：

\[
\mathcal{H}_{E|H} = \mathcal{H}_{H,E} \ominus \mathcal{H}_H
\]

### 5.1 这给我们的几何直觉

真正需要对齐的是：

- 记忆信息 \(E\) 的**新增子空间**；
- 它与任务 \(Y\) 的关系；
- 而不是 \(E\) 与 \(H\) 的整体几何关系。

---

## 6. 不同数学分支重新推导同一结论

### 6.1 信息论

\[
\Delta I = I(Y;E\mid H)
\]

这是对齐的“信息量”。

### 6.2 线性代数

\[
\Delta R^2 = R^2(Y;H,E)-R^2(Y;H)
\]

这是线性情况下的“信息量”。

### 6.3 贝叶斯 / GP

\[
\Delta^* = \mathbb{E}[Y\mid H,E]-\mathbb{E}[Y\mid H]
\]

是后验均值差。

### 6.4 微分几何

\[
\Delta^* \in T_h\mathcal{M}_H
\]

好的实现应把新增信息投影到目标流形切空间。

### 6.5 最优控制

\[
\Delta^*_l = \text{该层能影响最终任务的最优控制量}
\]

不同层对应不同可控制性。

### 6.6 范畴 / 函子

如果任务具有对称性，\(\Delta^*\) 应满足交换图：

\[
\eta \circ F(f) = G(f)\circ \eta
\]

即 reader 应与任务对称性可交换，而不是任意投影。

---

## 7. 最有效的对齐策略

### 7.1 先定义任务族

必须明确我们想让记忆解决什么任务：

- 稀有实体 next-token；
- 长尾 QA；
- BoolQ passage 判断；
- 通用语言建模。

不同任务族的 \(\Delta I\) 可能完全不同。

### 7.2 估计条件增量信息

\[
\Delta I(Y;E\mid H)
\]

或其线性代理：

\[
\Delta R^2(Y;E\mid H)
\]

### 7.3 学习低维充分统计量

不是直接用原始 E，而是寻找：

\[
Z = P_{\text{mem}}(E)
\]

使得：

\[
I(Y;Z\mid H)\approx I(Y;E\mid H)
\]

且 \(\dim Z\) 尽量小。  
这可以用信息瓶颈、PLS、CCA、kernel mean embedding 实现。

### 7.4 把 \(Z\) 映射到 H 的新增子空间

\[
\Delta = g(h,z)\cdot W_v z_\perp
\]

其中：

\[
z_\perp = z-\Pi_{\text{span}(H)}z
\]

### 7.5 让 gate 学会按需注入

gate 应近似：

\[
g^*(h,z) = \mathbb{1}[\Delta^*(h,z)\neq 0]
\]

而不是对所有 token 均匀开启。

---

## 8. 关键实验设计

### 8.1 任务族定义

构造“记忆任务集”：

\[
\mathcal{T}_{\text{mem}} =
\{
\text{稀有 token next-token},
\ \text{长尾 QA},
\ \text{BoolQ passage},
\ \text{普通 next-token}
\}
\]

### 8.2 Oracle reader 上界

先不做轻量 reader，先训练一个表达能力强的 oracle：

\[
\Delta_{\text{oracle}} = \text{MLP}(H,E)
\]

看它能把 \(\Delta R^2\) 做到多少。  
如果 oracle 都不能显著提升，说明问题不在 reader，而在 E 或任务本身。

### 8.3 核 / 非线性诊断

用核岭回归估计：

\[
\hat{\Delta}^* = \mathbb{E}[Y\mid H,E]-\mathbb{E}[Y\mid H]
\]

比较：

- 线性 \(\Delta R^2\)
- 核 \(\Delta R^2\)

如果核显著更好，说明 reader 必须是非线性的。

### 8.4 正交化诊断

计算：

\[
E_\perp = E-\Pi_H E
\]

测：

\[
\Delta R^2(Y;E_\perp\mid H)
\]

如果接近 \(\Delta R^2(Y;E\mid H)\)，说明 Value 去相关是安全且必要的。

### 8.5 切空间诊断

测：

\[
\rho = \frac{\|\Pi_{T_h}\Delta^*\|}{\|\Delta^*\|}
\]

如果 \(\rho\) 低，说明即使 oracle 也把信息注入流形外，当前架构可能根本用不上。

---

## 9. 具体证明草图

### 命题 A：记忆增益的必要条件

假设 reader 任意，则：

\[
I(Y;H+\Delta(H,E)\mid H)
\le I(Y;E\mid H)
\]

证明：  
\(H+\Delta(H,E)\) 是 \((H,E)\) 的函数，由数据处理不等式：

\[
I(Y;H+\Delta\mid H)
\le I(Y;H,E\mid H)
= I(Y;E\mid H)
\]

### 命题 B：线性 reader 的增益上界

若 \(\Delta=A E\)，则最优可达到的增量 R² 为：

\[
\eta^2_\text{lin}
=
\frac{
\operatorname{Tr}(\Sigma_{R E}\Sigma_E^{-1}\Sigma_{E R})
}{
\operatorname{Tr}(\Sigma_{RR})
}
\]

### 命题 C：正交化不损失线性增益

令：

\[
E_\perp = E-\Pi_H E
\]

如果 \(R\perp H\)（即任务是基座残差），则：

\[
\Sigma_{R E} = \Sigma_{R E_\perp}
\]

因此：

\[
\eta^2(Y;E\mid H)
=
\eta^2(Y;E_\perp)
\]

也就是说，在线性任务残差意义下，**Value 去相关不损失任何增量信息**。

### 命题 D：几何对齐不是充分条件

存在构造使 CKA=1 但 \(\Delta I=0\)，故几何对齐不能作为充分条件。  
同样也存在 \(\Delta I>0\) 但 CKA 很低的构造，故几何对齐也不是必要条件。

---

## 10. 最终结论

从第一性原理看：

1. **对齐的本质是条件充分性，不是几何相似性**；
2. 最有效的对齐是：
   \[
   \Delta^*(H,E)=
   \mathbb{E}[Y\mid H,E]-
   \mathbb{E}[Y\mid H]
   \]
3. 后续实验应先测：
   \[
   \Delta R^2(Y;E\mid H),\qquad
   \Delta R^2(Y;E_\perp\mid H)
   \]
4. 然后实现 oracle reader，确定可以达到的上界；
5. 再设计低维、去相关、可门控的 reader。

如果 oracle reader 都无法显著提高 \(\Delta R^2\)，那么问题不在“对齐方法”，而在于记忆表或任务本身。
