# Round 30：多视角数学推导——如何最有效实现 PLE 与目标模型的对齐

> 日期：2026-09-03
> 状态：理论推导 + 可检验假设
> 目标：用不同数学分支给出可操作的实验设计，指导下一阶段不是继续堆 scale，而是找到真正的“记忆有用性”条件。

---

## 0. 统一问题

我们有一个冻结记忆向量 \(E \in \mathbb{R}^d\)（PLE，\(d=2560\)），目标 hidden \(H \in \mathbb{R}^m\)（\(m=1024\)），以及任务标签 \(Y\)（如 next token / BoolQ 答案）。

当前 reader 实际在做：

\[
H' = H + \Delta(H,E)
\]

数学目标是：

\[
\min_{\Delta} \mathbb{E}[\ell(Y; W(H+\Delta))]
\]

这里 \(W\) 是输出头。我们下面用不同数学分支分别刻画“最优 \(\Delta\)”应满足什么条件。

---

## 1. 信息论视角：条件充分统计量与信息瓶颈

### 1.1 最优注入是条件期望

局部线性化后：

\[
\Delta^*(H,E) \approx \mathbb{E}[R \mid H,E],\qquad
R = Y - \mathbb{E}[Y \mid H]
\]

任何 reader 的增益上界：

\[
I(Y;\Delta(H,E)\mid H) \le I(Y;E\mid H)
\]

因此“记忆是否有用”的数学本质是：

\[
\boxed{\Delta I = I(Y;E\mid H)}
\]

### 1.2 信息瓶颈形式

如果 reader 输出只能是低维向量 \(Z\)，则最优 reader 是如下信息瓶颈的解：

\[
\max_{p(z|h,e)} I(Y;Z\mid H) - \beta I(H,E;Z)
\]

在 Gaussian 线性情形下，最优 \(Z\) 是 \(E\) 对 \(Y\) 给定 \(H\) 后的 CCA / 线性回归方向，而不是 \(E\) 对 \(H\) 的 CKA 方向。

### 1.3 可检验结论

- 实验应测：\(I(Y;E\mid H)\) 的线性代理 \(\Delta R^2\)。
- 如果 \(\Delta R^2\) 很小，说明当前记忆表/任务选择本身不提供新信息，继续调 reader 无效。
- 如果 \(\Delta R^2\) 大但 reader 没用好，则问题在容量、正则、或 gate 设计。

---

## 2. 线性代数 / 谱视角：子空间角与互补性

### 2.1 主角度

设：

\[
S_H = \mathrm{span}(H),\qquad S_E = \mathrm{span}(E)
\]

两者之间的主角度 \(\theta_1,\dots,\theta_{\min(m,d)}\) 描述子空间结构：

- 主角度小：两个空间有大量共享方向；
- 主角度接近 \(\pi/2\)：两个空间几乎互补。

我们的 CKA 低、kNN overlap 低，说明 **E 和 H 的共享方向少，互补方向多**。  
这其实对“记忆提供新信息”是好的，但不利于“用 H 作为 query 去 gate”。

### 2.2 关键分解

定义投影：

\[
E_\parallel = \Pi_H E,\qquad
E_\perp = E - \Pi_H E
\]

- \(E_\parallel\)：与 H 共享的方向，适合做 **gate/key**；
- \(E_\perp\)：H 无法预测的方向，适合做 **value/注入内容**。

### 2.3 最优 Value 子空间

给定任务残差 \(R\)，最优 value 子空间是：

\[
S_{\text{value}} =
\mathrm{range}\big(\Sigma_{E_\perp R}\big)
=
\mathrm{range}\big(\mathbb{E}[E_\perp R^\top]\big)
\]

它的主方向可由 SVD 得到：

\[
\Sigma_{E_\perp R} = U \Lambda V^\top
\]

### 2.4 可检验结论

- 计算 \(S_H\) 与 \(S_E\) 的 principal angles；
- 计算 \(S_{\text{value}}\) 与 \(S_H\) 的夹角；
- 如果 \(S_{\text{value}}\) 和 \(S_H\) 很接近，说明 value 路径在重复 H 信息；
- 如果 \(S_{\text{value}}\) 很大但大多数奇异值很小，说明有用信号稀疏，需要低秩正则或 top-k selection。

---

## 3. 随机矩阵 / 高维统计视角：信噪比与过拟合

### 3.1 高维回归的噪声问题

真实数据中：

\[
E = S + N
\]

其中 \(S\) 是任务相关信号，\(N\) 是高维噪声（含 hash collision、无关 n-gram 方向）。

线性回归的最优解是岭回归：

\[
\hat A = \Sigma_{ER}\big(\Sigma_E + \lambda I\big)^{-1}
\]

高维下，可实现的增量 R² 大约为：

\[
\eta^2 \approx
\frac{\|\Sigma_{R S}\|^2_{\text{signal}}}
{\|\Sigma_R\|^2 \cdot \big(1 + \text{noise}/\text{sample}\big)}
\]

当前实验只有几十万到 1M token，但 E 的有效维数约 766，实际可能有很多噪声方向。  
因此**当前低 real−control 差异可能主要不是“没对齐”，而是“信噪比太低/过拟合噪声维度”。**

### 3.2 可检验结论

- 对 E 做 PCA/PLS 降维到 \(r \in \{16,32,64,128,256\}\)；
- 用 ridge regression 测 \(\Delta R^2\)；
- 如果降维后 \(\Delta R^2\) 上升，说明高维噪声在稀释真实信号；
- 如果降维后 \(\Delta R^2\) 下降，说明有用信息分散在高维，需要非线性读法。

---

## 4. 最优传输 / 度量几何视角：部分对齐而非全局对齐

### 4.1 Gromov-Wasserstein 对齐

全局 GW 问题：

\[
GW^2 = \min_{\pi}
\int\int |d_H(h,h') - d_E(e,e')|^2\,d\pi\,d\pi'
\]

我们的 kNN overlap 显示当前 E 和 H 的局部几何几乎不一致。  
但这不一定是否定结论，因为**记忆只需要在“该用记忆的时刻”对齐**。

### 4.2 加权对齐

定义相关性权重：

\[
w_i = \|R_i\| \quad\text{或}\quad w_i = H(Y_i \mid H_i)
\]

加权 Procrustes / GW：

\[
\min_{\phi} \sum_i w_i \|\phi(e_i) - h_i\|^2
\]

这比全局无权重对齐更接近真实任务：  
我们应该只在**基座不确定、需要外部记忆**的 token 上要求 E 与 H 可映射。

### 4.3 可检验结论

- 按 next-token entropy / 任务残差大小对 token 分桶；
- 只在高残差桶里计算 CKA / Procrustes / kNN overlap；
- 如果高残差桶的对齐显著更好，说明当前 PLE 其实有局部记忆价值，只是被大量普通 token 稀释。

---

## 5. RKHS / 核方法视角：条件 HSIC 与核对齐

### 5.1 定义

\[
\mathrm{HSIC}(X,Y) = \| \mathbb{E}[K_X \otimes K_Y] \|^2_{HS}
\]

CKA 是归一化 HSIC。  
但我们真正需要的是：

\[
\mathrm{HSIC}\big(V(E),\; R \mid H\big)
\]

即“在给定 H 后，V(E) 与任务残差的条件依赖”。

### 5.2 可检验结论

- 用核岭回归 / 条件 HSIC 估计非线性增量；
- 如果线性 \(\Delta R^2\) 低但条件 HSIC 高，说明需要非线性 reader（MLP / deeper gate）；
- 如果两者都低，说明记忆信息本身不足。

---

## 6. 图 / 谱聚类视角：门控需要局部图对齐

### 6.1 当前 gate 的数学形式

\[
g(h,e) = \sigma\!\left(
\frac{\langle W_q h, W_k e\rangle}{\sqrt{m'}}
\right)
\]

这要求 \(W_k E\) 与 \(W_q H\) 在局部邻域上可比。

### 6.2 谱图对齐指标

构建：

- H 的 kNN 图 \(G_H\)
- E（或 \(W_k E\)）的 kNN 图 \(G_E\)

比较 Laplacian 前 \(k\) 个特征向量的余弦相似度：

\[
\text{SpecAlign} = \frac{1}{k}\sum_{j=1}^k
|\langle u_j^H, u_j^E\rangle|
\]

如果 SpecAlign 低，gate 很难学到“该用哪段记忆”。

### 6.3 可检验结论

- 对原始 E 和经过 key projection 的 \(W_k E\) 分别测 SpecAlign；
- 如果 projection 后的 SpecAlign 提升但 real−control 没提升，说明 gate 不是瓶颈；
- 如果 projection 后 SpecAlign 仍低，说明需要显式加 key 对齐 loss，而不是只靠 LM loss。

---

## 7. 优化动力学视角：reader 是否真的学到了任务残差

### 7.1 理想梯度

固定 backbone 时，reader 的梯度方向等价于：

\[
\frac{\partial L}{\partial \Delta} \propto -\mathbb{E}[R \mid H,E]
\]

因此训练好的 reader 输出应该和高维残差 \(R\) 正相关。

### 7.2 可检验结论

- 计算实际 reader 输出 \(\Delta\) 与 backprop 残差 \(R\) 的 cosine / CKA；
- 如果很低，说明当前 reader 主要学到的是“分布偏移/格式/语言风格”，而不是“记忆内容”；
- 这能直接解释为什么 control 也提高 logprob：control 也学到了同一类分布偏移。

---

## 8. 流形假设视角：先降维到记忆流形

### 8.1 当前问题

- PLE raw intrinsic dimension ≈ 766；
- Qwen hidden intrinsic dimension ≈ 40–80；
- 两者维度差异过大，直接线性映射会把大量噪声维度带入目标空间。

### 8.2 猜想

存在一个低维“记忆语义流形”：

\[
E \xrightarrow{P} z \in \mathbb{R}^r,\qquad r \ll 2560
\]

使得：

\[
I(Y; z \mid H) \approx I(Y; E \mid H)
\]

即大部分任务相关信息集中在一个低维子空间里。

### 8.3 可检验结论

- 用 PCA / autoencoder / PLS 把 E 压到 \(r=32,64,128\)；
- 测压缩后 CKA、kNN、\(\Delta R^2\)；
- 如果压缩后信息不损失且对齐指标显著上升，说明当前 reader 的瓶颈是“高维噪声稀释”，而不是“表没有信息”。

---

## 9. 综合数学框架

基于以上所有视角，我们提出一个统一的 reader 设计：

```text
E
├── Key 路径:
│   E → W_k → K(E)
│   目标: 与 Q(H) = W_q H 做局部谱/核对齐, 用于 gate
│
└── Value 路径:
    E → 去相关/降维 →  E⊥ 或 z
    → W_v → V(E)
    目标: 与任务残差 R 高相关, 与 H 低相关
```

对应损失：

\[
\begin{aligned}
L =\ & L_{\text{LM}} \\
&+ \lambda_1 \cdot \text{SpecAlign / CKA / InfoNCE}\big(K(E), Q(H)\big) \\
&+ \lambda_2 \cdot \text{Regression / HSIC}\big(V(E), R \mid H\big) \\
&+ \lambda_3 \cdot \|\mathrm{Corr}(V(E), H)\|^2 \\
&+ \lambda_4 \cdot \text{Gate Entropy / Sparsity} \\
&+ \lambda_5 \cdot KL(\text{reader\_on} \| \text{reader\_off})
\end{aligned}
\]

其中：

- \(\lambda_1\) 提高 gate 可对齐性；
- \(\lambda_2\) 提高 value 对任务残差的可解码性；
- \(\lambda_3\) 防止 value 重复 H 已有信息；
- \(\lambda_4\) 防止对所有 token 无差别注入；
- \(\lambda_5\) 防止破坏基座能力。

---

## 10. 给后续实验的具体排序

1. **必须最先做**：增量 R² 诊断
   \[
   R^2(Y;H),\quad R^2(Y;H,E),\quad R^2(Y;H,E_\perp)
   \]
   以及 \(\Delta R^2(Y;E\mid H)\)。

2. **第二个做**：高维噪声诊断
   - PCA/PLS 将 E 降到 \(r=32,64,128\)；
   - 测压缩前后 \(\Delta R^2\) 与 CKA/kNN。

3. **第三个做**：条件对齐诊断
   - 在高残差 token 子集上重新计算 CKA / Procrustes / kNN；
   - 测 Key projection 后的 SpecAlign。

4. **第四个做**：gate/value 分工商
   - 按第 9 节的架构修改 reader；
   - 增加对应 loss；
   - 比较 real−control、BoolQ 退化、entropy。

5. **第五个做**：如果以上都显示正信号，再进入 5M–20M 训练和 RL；如果都没有，记录为负面机制证据。

---

## 11. 核心结论

最有效的“对齐”不是让 PLE 和 Qwen hidden 全局相似，而是：

1. **在任务残差方向上有增量可解释性**；
2. **在 gate 所需局部邻域上可对齐**；
3. **Value 与 H 去相关，避免重复注入**；
4. **先压缩掉 E 的高维噪声，再对齐**；
5. **用加权/条件方式只对齐真正需要记忆的 token**。

数学上，我们建议把“CKA 对齐”替换为“条件增量可解释性 + 门控谱对齐 + 去相关 value”三件套。
