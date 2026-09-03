# Round 40：为什么 MLP Value Reader 需要非线性——理论推导与设计指导

> 日期：2026-09-03
> 状态：理论解释 Oracle MLP 结果
> 目标：用数学解释：
> 1. 为什么当前线性目标侧适配器容量不足；
> 2. 为什么 MLP Value Reader 能提升 3–4 倍；
> 3. 为什么 E⊥ 更好；
> 4. 为什么 PLS 对 MLP 帮助有限；
> 5. 理论如何指导下一步 reader 设计。

---

## 1. 形式化模型

设：

\[
R = Y - \mathbb{E}[Y\mid H]
\]

是基座无法预测的任务残差。
最优记忆注入在函数空间中为：

\[
\Delta^*(H,E)
=
\mathbb{E}[R\mid H,E]
-
\mathbb{E}[R\mid H]
\]

为简化，固定 H，只关注 E 到 R 的非线性映射：

\[
R = g(E) + \varepsilon,\qquad \varepsilon\perp E,\quad \varepsilon\perp H
\]

其中 \(g\) 是记忆相关内容，\(\varepsilon\) 是噪声。

---

## 2. 线性 reader 只能捕获 g 的线性部分

定义线性投影：

\[
g_{\text{lin}}(E)
=
\Sigma_{RE}\Sigma_{EE}^{-1}E
\]

这是 \(g(E)\) 在 \(\mathrm{span}(E)\) 上的 \(L^2\) 投影。定义非线性残余：

\[
g_{\perp}(E)=g(E)-g_{\text{lin}}(E)
\]

显然：

\[
g_{\perp}\perp \mathrm{span}(E)
\]

### 线性可解释增量

\[
\Delta R^2_{\text{lin}}
=
\frac{\|g_{\text{lin}}\|^2}{\|R\|^2}
\]

### 非线性 MLP 可达到的增量

由万能逼近定理，足够宽/深的 MLP 可以任意逼近连续函数 \(g\)，因此：

\[
\Delta R^2_{\text{mlp}}
\approx
\frac{\|g\|^2}{\|R\|^2}
\]

### 比率

\[
\frac{\Delta R^2_{\text{mlp}}}{\Delta R^2_{\text{lin}}}
\approx
\frac{\|g\|^2}{\|g_{\text{lin}}\|^2}
\]

---

## 3. 实验结果的数学解释

实验：

\[
\text{linear: } 0.0058,\qquad
\text{MLP: } 0.0206
\]

因此：

\[
\frac{\|g\|^2}{\|g_{\text{lin}}\|^2}
\approx
\frac{0.0206}{0.0058}
\approx 3.55
\]

也就是说：

> 当前 PLE 到梯度残差的条件期望主要是非线性的；
> 线性部分只解释了约 \(1/3.55\approx 28\%\) 的可提取信号；
> 约 **72% 的信息藏在非线性映射中**。

这从数学上证明了：

> 当前 reader 使用线性 out_proj，是容量瓶颈，不是数据/记忆无信息。

---

## 4. 为什么 E⊥ 在 MLP 下更好

### 正交分解

\[
E = E_\parallel + E_\perp
\]

其中：

\[
E_\parallel=\mathbb{E}[E\mid H]
\]

即 H 可线性预测的部分。

### 定理：E∥ 对 R 没有增量信息

因为：

\[
R\perp \mathcal H_H
\]

所以对任意 \(U\in\mathcal H_H\)：

\[
\mathbb{E}[R^\top U]=0
\]

而 \(E_\parallel\in\mathcal H_H\)，因此：

\[
\mathrm{Cov}(R,E_\parallel)=0
\]

在线性意义下，关于 R 的信息全部来自：

\[
E_\perp=E-E_\parallel
\]

### 非线性情形

即使最优映射 \(g(E)\) 是非线性的，如果满足条件独立性：

\[
R \perp E_\parallel \mid E_\perp
\]

那么 E∥ 对 R 没有额外信息。
即使不完全成立，E∥ 也会占用 MLP 容量并引入与 H 相关的冗余，导致过拟合。

### 实验支持

\[
\text{MLP H+E: } 0.0206,\qquad
\text{MLP H+E}_\perp: 0.0228
\]

E⊥ 略优于 E，说明：

> 去相关不仅不损失信息，还通过降低冗余/噪声帮助泛化。

---

## 5. 为什么 PLS 对 MLP 帮助有限

### PLS 的优化目标

PLS 方向求解：

\[
\max_w
\frac{\mathrm{Cov}^2(Ew,R)}{\mathrm{Var}(Ew)}
\]

这是**线性相关性最大化**。

### 非线性最优压缩不同

如果最优映射 \(g(E)\) 是非线性的，那么最优低维表示 \(z=P(E)\) 应由：

\[
\max_P I(R;P(E)\mid H)
\]

决定，而不是线性协方差最大化。

因此：

- 对线性 reader，PLS 是很好的压缩；
- 对 MLP reader，PLS 可能丢掉非线性可用的方向。

### 实验支持

\[
\text{MLP H+PLS64: } 0.0122
\]

低于：

\[
\text{MLP H+E: } 0.0206
\]

说明 PLS64 对非线性模型反而是次优压缩。

---

## 6. 对下一步 Reader 设计的数学指导

### 6.1 架构

根据以上推导，建议：

\[
\boxed{
\Delta
=
g(h,e)\cdot \mathrm{MLP}\big(E_\perp\big)
}
\]

其中：

- \(E_\perp=E-\mathbb{E}[E\mid H]\)；
- \(\mathrm{MLP}\) 至少 2 层、带激活；
- \(g(h,e)\) 可用现有 gate，但应强调稀有 token / 高不确定性选择。

### 6.2 为什么

- MLP 可以逼近非线性 \(g(E)\)；
- E⊥ 去除 H 冗余；
- gate 防止对简单 token 扰动。

### 6.3 训练目标

除了 LM loss，建议加入显式残差监督：

\[
L_{\text{aux}}
=
\mathbb{E}\left\|
\mathrm{MLP}(E_\perp)-\widehat{R}
\right\|^2
\]

其中：

\[
\widehat{R}=-\frac{\partial L_{\text{LM}}}{\partial h}
\]

这是对“最优 reader 应逼近条件期望差”的直接实现。

### 6.4 信息瓶颈版本

更理论化的目标是：

\[
\max_{P}
I(R;P(E)\mid H)
-\beta I(E;P(E)\mid H)
\]

这可以指导：不用固定 PLS，而是学习一个“对 R 信息量最大、对 E 压缩量最大”的低维编码。

---

## 7. 结论

数学上：

1. **Oracle MLP 的 3–4 倍提升**表明当前线性适配器只捕获了约 28% 的可提取信号；
2. **E⊥ 更好**是因为 \(R\perp \mathcal H_H\)，E∥ 对 R 没有增量信息；
3. **PLS 对 MLP 帮助有限**是因为 PLS 优化线性相关，不是非线性信息瓶颈；
4. 因此下一步应该实现：

\[
\Delta=g(h,e)\cdot \mathrm{MLP}(E_\perp)
\]

并加上残差监督 loss 和稀有 token 门控。
