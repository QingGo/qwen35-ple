# Round 31：更深的数学分支推导——对齐的几何、因果、贝叶斯、控制与拓扑视角

> 日期：2026-09-03
> 状态：理论扩展，尚未实证
> 目标：在 round 29/30 基础上，用更多数学分支给出新的可检验猜想和实验设计。

---

## 1. 为什么还需要更多数学视角

Round 30 已经给出核心框架：

- 最优 reader 近似 \(\mathbb{E}[R|H,E]\)；
- 应测 \(\Delta R^2(Y;E|H)\)；
- Key/Gate 与 Value 应分工；
- 高维 E 会稀释信号。

但还有一些问题没有回答：

1. 记忆到底应该注入到哪一层？
2. 是否应该只在部分 token 上注入？
3. 如何判断注入是“因果有效”还是“相关偏移”？
4. 如何从数学上解释 BoolQ 退化？
5. 是否存在比线性投影更合适的几何结构？

下面用更多数学分支来尝试回答。

---

## 2. 统计决策论视角：不只是平均损失

### 2.1 最小最大风险形式

当前我们主要看平均 preplexity / QA EM。但记忆注入的真正问题可能是：

\[
\mathcal{R}_{\max}(\Delta) = \max_{c \in \mathcal{C}} \mathbb{E}[\ell(Y; H+\Delta) \mid c]
\]

其中 \(\mathcal{C}\) 是任务类别：Trivia / NQ / BoolQ / 简单语言建模。

记忆可能降低长尾知识风险，但增加 BoolQ 风险。因此最优 reader 应满足：

\[
\Delta^* = \arg\min_\Delta \max_c \ \mathbb{E}[\ell_c]
\]

这就解释了为什么单纯降低平均 val loss 或平均 answer loss 不够。

### 2.2 可检验结论

- 把所有 QA 按任务和“是否需要外部记忆”分组；
- 对每组分别计算 real / control / no-reader 的风险；
- 如果 real 在某类任务风险上升，说明 gate 没有做到“按需注入”。

---

## 3. 因果推断视角：real 是否真的因果有效

### 3.1 干预 vs 观察

我们目前测的是：

\[
\Delta_{\text{obs}} = \mathbb{E}[\ell(Y;H+\Delta^{\text{real}})] - \mathbb{E}[\ell(Y;H+\Delta^{\text{control}})]
\]

但这仍是观察性比较。真正的因果问题是：

\[
\tau = \mathbb{E}[\ell(Y \mid do(\Delta^{\text{real}}))] - \mathbb{E}[\ell(Y \mid do(\Delta^{\text{control}}))]
\]

### 3.2 为什么 control 不够

control 只是对 token 顺序做 permutation，它并不构成所有可能的“反事实记忆”。  
更严格的干预集应包括：

- real e_t；
- shuffled control；
- random same-norm noise；
- zero；
- top-k 截断；
- 用另一个记忆表的相似向量替换。

如果 real 相对所有这些干预都没有显著优势，那么“PLE 内容”本身就不是有效因果变量。

### 3.3 可检验结论

设计一个 factoral patching 矩阵：

\[
\text{condition} \times \text{inject\_layer} \times \text{inject\_scale} \times \text{gate\_mode}
\]

然后估计 real 的平均因果效应（ATE）和异质性（CATE）：

\[
\tau(x) = \mathbb{E}[\ell^{\text{real}}-\ell^{\text{control}} \mid X=x]
\]

其中 \(x\) 可以是任务类别、问题长度、是否包含稀有词。

如果 \(\tau(x)\) 只在某些子群为正，说明应做 **conditional / gated injection**，而不是全局注入。

---

## 4. 贝叶斯 / 高斯过程视角：记忆是后验更新

### 4.1 后验均值形式

把记忆注入看作对残差 \(R\) 的高斯过程回归：

\[
R(h,e) \sim \mathcal{GP}\big(0, k((h,e),(h',e'))\big)
\]

给定观测数据集后，最优注入是后验均值：

\[
\Delta^*(h,e) = k(h,e; \mathcal{D})^\top
\big(K+\sigma^2 I\big)^{-1} R_{\mathcal{D}}
\]

### 4.2 核结构的猜想

如果记忆有用，一个自然的核分解是：

\[
k((h,e),(h',e')) = k_H(h,h')\cdot k_E(e,e')
\]

在这种乘积核下，gate 可以自然分解为：

\[
g(h,e) \propto k_H(h,\text{query})\cdot k_E(e,\text{key})
\]

这正是当前双线性 gate 的贝叶斯解释。

### 4.3 有效维度与样本复杂度

核的有效维度：

\[
d_{\text{eff}}(k) = \frac{\big(\sum \lambda_i\big)^2}{\sum \lambda_i^2}
\]

如果 \(d_{\text{eff}}\) 很大，需要更多样本来估计后验均值。  
这可以解释为什么当前只有 1M token 时，real−control 很小。

### 4.4 可检验结论

- 用核岭回归（RBF / 线性核）估计非线性 \(\Delta R^2\)；
- 比较线性 \(\Delta R^2\) 与核 \(\Delta R^2\)；
- 如果核方法显著更好，说明需要一个非线性 value 路径。

---

## 5. 微分几何视角：注入应该落在目标切空间上

### 5.1 切空间

假设 Qwen hidden 生活在低维流形 \(\mathcal{M}_H\) 上。  
每个点 \(h\) 有一个切空间：

\[
T_h\mathcal{M}_H \subset \mathbb{R}^m
\]

好的记忆注入应该主要落在 \(T_h\mathcal{M}_H\) 内，才能被后续层利用；  
如果注入方向与切空间正交，则相当于向“流形外”增加噪声，容易造成退化。

### 5.2 度量对齐

设 PLE 空间有度量 \(g_E\)，Qwen 空间有度量 \(g_H\)。  
理想的 reader 是拉回映射：

\[
\phi^*g_H \approx g_E
\]

即“记忆空间的局部度量经过 reader 后，应近似 Qwen 空间的局部度量”。

### 5.3 可检验结论

- 对每个 token，用邻域 PCA 估计 \(T_h\mathcal{M}_H\)；
- 计算 reader 输出 \(\Delta\) 在切空间上的投影占比：
  \[
  \rho = \frac{\|\Pi_{T_h}\Delta\|}{\|\Delta\|}
  \]
- 如果 \(\rho\) 低，说明注入大量进入流形外方向，可能是 BoolQ 退化的几何原因；
- 如果 \(\rho\) 高但 real−control 仍小，说明问题不在几何，而在信息量。

---

## 6. 最优控制视角：确定最佳注入层

### 6.1 分层控制问题

把 Transformer 看作动态系统：

\[
h_{l+1} = f_l(h_l + \Delta_l),\qquad l=1,\dots,L
\]

记忆注入是最优控制：

\[
\min_{\{\Delta_l\}} \sum_t \ell_t(y_t; h_L)
\]

每个层的“可控制性”可近似用损失对该层 hidden 的梯度范数衡量：

\[
c_l = \mathbb{E}\big[\|\nabla_{h_l}\ell\|^2\big]
\]

### 6.2 猜想

- 如果某一层 \(c_l\) 很大，说明该层对最终输出影响大，可能是好的注入层；
- 但注入太深可能已经错过了需要修改的表示；
- 最优层应该满足“可控性大 + 表示尚未完全确定”的平衡。

### 6.3 可检验结论

- 对 layer 1、4、8、12、16、20、23 分别计算 \(\|\nabla_{h_l}\ell\|\)；
- 同时测该层 reader 注入后 real−control 差异；
- 如果某层梯度范数大但 real−control 小，说明该层不是记忆信息能发挥作用的层；
- 如果梯度范数小但 real−control 大，说明深层已经在使用记忆。

---

## 7. 拓扑 / 持久同调视角：记忆是否引入新拓扑结构

### 7.1 简单想法

Qwen hidden 的局部拓扑结构可能缺少某些“记忆簇”。  
PLE 可以在目标空间中引入新的拓扑连接。

### 7.2 可检验结论

- 对 hidden 和 reader 注入后的 hidden 分别估计 kNN 图的连通分量数量、Betti 数、持久同调；
- 如果注入后拓扑结构发生显著变化，说明记忆确实在改变表示空间；
- 如果变化主要来自 control 也产生，说明变化是“扰动型”而非“语义型”。

---

## 8. 范畴 / 函子视角：局部平移不变性猜想

### 8.1 为什么 n-gram 记忆应该有局部结构

PLE 是 n-gram 记忆表，因此记忆读取应满足局部平移不变性：

\[
\Delta(h_t, e_t) \approx \Delta(h_{t+1}, e_{t+1})
\quad\text{当局部上下文相似时}
\]

这启发我们：

- Key/Value 路径应包含 **局部卷积 / ShortConv**（当前已有）；
- Gate 不应只依赖单点 \(h_t\)，而应依赖局部窗口 \(h_{t-k:t+k}\)。

### 8.2 可检验结论

- 比较“单点 gate”和“局部窗口 gate”；
- 如果窗口 gate 能提高 real−control 且降低 BoolQ 退化，说明局部结构很重要；
- 如果无差异，说明当前 ShortConv 没有真正建模记忆的局部性。

---

## 9. 信息几何视角：自然梯度与 Fisher 信息

### 9.1 最佳更新方向

在统计流形上，最优参数更新不是欧氏梯度，而是自然梯度：

\[
\Delta \theta^* \propto \mathcal{I}(\theta)^{-1} \nabla_\theta \ell
\]

对 reader 来说，这等价于：

\[
\Delta^* \approx \text{投影 of score function onto } \mathrm{span}(E)
\]

### 9.2 Fisher 信息与记忆有用性

如果 E 在 Fisher 信息意义下与 Y 的 score 正交，则 memory 无法提供梯度方向。  
因此可定义：

\[
\mathcal{F}(E;Y|H) =
\mathbb{E}\left[
\frac{\partial \log p(Y|H)}{\partial H}^\top
\Sigma_E
\frac{\partial \log p(Y|H)}{\partial H}
\right]
\]

这可以作为“记忆方向与任务梯度方向匹配度”的度量。

### 9.3 可检验结论

- 用 next-token loss 对 hidden 的梯度作为 score 近似；
- 计算 \(\mathrm{cosine}(E, \nabla_h \ell)\)；
- 如果很低，说明 reader 虽然训练了，但注入方向与任务需要的方向不一致。

---

## 10. 综合推导：一个带门控的最优 reader 形式

综合上述，我们猜想最优 reader 应具有如下结构：

\[
z = P_{\text{mem}}(E) \in \mathbb{R}^r
\]

\[
g(h,z) = \sigma\!\left(
\frac{\langle W_q h, W_k z\rangle}{\sqrt{r}}
\right)
\]

\[
v(h,z) = W_v \big(z_\perp\big),
\qquad
z_\perp = z - \Pi_{\text{span}(H)} z
\]

\[
\Delta = g(h,z)\cdot v(h,z)
\]

\[
H' = H + \Delta
\]

其中：

- \(P_{\text{mem}}\) 由 PLS / 信息瓶颈学出；
- \(z_\perp\) 去相关；
- \(g\) 由局部窗口 + 谱对齐正则；
- \(v\) 与任务残差 \(R\) 回归。

这可以从多个数学分支同时推出：

- 信息论：\(z\) 是 \(E\) 关于 \(Y|H\) 的瓶颈表示；
- 线性代数：\(z_\perp\) 去除 H 冗余；
- 微分几何：\(v\) 尽量落在 \(T_h\mathcal{M}_H\)；
- 最优控制：层选择由梯度范数决定；
- 贝叶斯：\(v\) 是后验均值近似。

---

## 11. 建议新增实验

### 11.1 快速诊断（1–2 天）

1. 计算线性 \(\Delta R^2(Y;E|H)\)；
2. 计算核 \(\Delta R^2\)；
3. 计算 \(E_\perp\) 的 \(\Delta R^2\)；
4. 计算 reader 输出与 backprop 残差的 cosine。

### 11.2 几何诊断（2–3 天）

5. 估计 hidden 流形切空间；
6. 计算注入向量在切空间的投影比例 \(\rho\)；
7. 按层计算梯度范数 \(c_l\)。

### 11.3 因果诊断（3–5 天）

8. 扩展 patching 矩阵到 random / zero / top-k / 不同 layer；
9. 估计 real−control 的 CATE，按任务和 token 稀有度分层；
10. 检查 BoolQ 风险是否被 gate 抑制。

### 11.4 新 reader 实验（1–2 周）

11. 实现 \(P_{\text{mem}}\) 低秩信息瓶颈；
12. Key/Gate 使用局部窗口 + SpecAlign loss；
13. Value 使用 \(z_\perp\) 与任务残差回归；
14. 加入 gate entropy / KL 约束；
15. 比较 real−control、BoolQ、entropy、\(\Delta R^2\)。

---

## 12. 总结

从更多数学分支看，“最有效对齐”不是一个几何相似度问题，而是一个 **带条件的、分层的、因果的、低维的统计决策问题**：

\[
\boxed{
\text{有效对齐}
=
\text{条件信息}
+
\text{门控可对齐性}
+
\text{切空间兼容性}
+
\text{去冗余}
+
\text{低维压缩}
}
\]

我们后续实验应围绕这五项分别建立可测量指标，而不是继续只看 val loss / QA EM。
