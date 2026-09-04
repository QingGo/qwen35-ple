# Round 70：用数学推导“什么是最有效的路径”

> 日期：2026-09-04
> 状态：多视角数学推导
> 目标：给出可指导后续实验的最优路径，并说明为什么。

---

## 1. 问题形式化

设：

- \(H\)：0.8B backbone 能看到的上下文；
- \(Y\)：目标任务 token / 标签；
- \(p_\theta(y|h)\)：当前 base 输出分布；
- \(\mathcal{M}\)：可用记忆源集合：
  - \(D\)：RAG 文档；
  - \(E\)：PLE / n-gram 记忆；
  - \(T\)：teacher 输出。
- 注入通道 \(\mathcal{C}\)：
  - 输入上下文；
  - logit 修正；
  - hidden 注入。

我们的目标：

\[
\min_{\text{source, channel, training}}
\mathbb E[\ell(Y,\hat Y)]
\]

受限于本地资源：

- 8GB GPU；
- 约 15GB RAM；
- 可选的高 RAM/云 teacher 推理。

---

## 2. 定理 1：通道有效性排序

对任意记忆源 \(M\)：

\[
I(Y;\text{input}(M)\mid H)
\ge
I(Y;\text{logit}(M)\mid H)
\ge
I(Y;\text{hidden}(M)\mid H)
\]

**证明思路：**

- 输入通道：记忆直接进入上下文，所有层都能使用；
- logit 通道：可以实现最优修正：
  \[
  \delta^*(y)=\log\frac{P(y|H,M)}{P(y|H)}
  \]
  其 loss 改善精确等于 \(I(Y;M|H)\)；
- hidden 通道：只能实现：
  \[
  J\Delta h = P_{\mathrm{col}(J)}\delta\ell
  \]
  所以信息量一般更小。

因此：

> **能走 logit/input 就不要走 hidden。**

---

## 3. 定理 2：源的 Blackwell 信息序

如果记忆源 \(M_1\) 对任务 \(Y\) 在给定 \(H\) 下是 Blackwell 更充分的，则对任意决策问题，使用 \(M_1\) 的最优决策风险不大于使用 \(M_2\)。

\[
M_1 \succeq_{\text{Blackwell}} M_2
\Rightarrow
\forall \text{decision rules},\quad
R(M_1)\le R(M_2)
\]

**直觉：**

- RAG 文档提供了“任务相关语义信息”；
- PLE e_t 主要提供“局部 n-gram 先验”；
- 对知识问答，RAG 的信息序应高于 PLE；
- 对代码/低熵 token，PLE/n-gram 可能更充分。

因此：

> **不存在一个“永远最优”的记忆源；最优路径是任务相关的信息序选择。**

---

## 4. 定理 3：多源最优融合

如果同时有 base、RAG、n-gram、teacher，且在 logit 层做线性融合：

\[
\ell_{\text{fused}}=
\ell_{\text{base}}
+
\sum_i \lambda_i \log\frac{P(y|H,M_i)}{P(y|H)}
\]

则最优 \(\{\lambda_i\}\) 是凸优化：

\[
\min_{\lambda}
\mathbb E\left[
-\log \mathrm{softmax}(\ell_{\text{fused}})(Y)
\right]
\]

**证明要点：**

- softmax CE 关于 logits 凸；
- logits 关于 \(\lambda\) 线性；
- 所以整体凸，存在全局最优。

这意味着：

> RAG、n-gram、teacher 不是非此即彼，可以用一个可学习 router/权重统一融合。

---

## 5. 定理 4：资源约束下的最优选择

设记忆源 \(M_i\) 的单位成本为 \(c_i\)，条件信息量为：

\[
I_i=I(Y;M_i\mid H)
\]

在资源有限时，优先选择“信息/成本比”高的源：

\[
\text{choose } i^*=\arg\max_i \frac{I_i}{c_i}
\]

在当前资源下估算：

| 源 | 信息 | 本地成本 | 性价比 |
|---|---|---|---|
| RAG | 高（语义） | 低（只需检索） | 极高 |
| Teacher 离线 | 高（推理/代码） | 高（需高 RAM/云） | 中高 |
| N-gram/PLE | 低（语义）/中（词法） | 极低 | 词法任务高 |
| Hidden PLE reader | 低 | 中 | 低 |

因此：

> **先做高性价比的 RAG/teacher logit 融合；PLE 只在词法/低熵任务上作为低成本专家。**

---

## 6. 定理 5：N-gram 与神经网络最优插值

对局部低熵任务，设：

- \(L_b = \log p_\theta(y|h)\)
- \(L_n = \log p_{\text{ngram}}(y|\text{context})\)
- \(L_t = \log P(y|h,\text{true})\)

考虑融合：

\[
L = L_b + \lambda L_n
\]

最优 \(\lambda\) 近似：

\[
\lambda^*
=
\frac{\mathbb E[(L_t-L_b)(L_n-L_b)]}
{\mathbb E[(L_n-L_b)^2]}
\]

**推导：**

这是最小二乘投影：

\[
\min_\lambda \mathbb E[(L_t - L_b - \lambda L_n)^2]
\]

求导：

\[
\lambda
=
\frac{\mathrm{Cov}(L_t-L_b,\;L_n-L_b)}
{\mathrm{Var}(L_n-L_b)}
\]

因此：

- 如果 n-gram 的方向与“base 的残差”相关，\(\lambda>0\)；
- 如果不相关，\(\lambda\approx0\)；
- 如果相关但符号相反，\(\lambda<0\)。

这给出了一个**无需大规模训练**的判断方法：

> 在低熵/代码/专名任务上，先估计 \(\lambda^*\)，如果显著为正，PLE/n-gram 就可以作为 logit prior 使用。

---

## 7. 定理 6：自蒸馏的收益上界

设 teacher 伪标签噪声为 \(\epsilon\)，学生从 teacher 学到的有效信号为：

\[
S = I(Y; \hat Y_T \mid H)
\]

学生最终收益：

\[
\Delta R
\le
\frac{S}{S+\text{noise}}
\]

**含义：**

- teacher 越准，收益越大；
- 伪标签噪声越大，收益越小；
- 因此自蒸馏/self-training 前必须做：
  - 数学答案验证
  - 代码运行验证
  - 检索一致性验证
  - 自洽性过滤

---

## 8. 猜想：当前资源下的最有效路径

综合以上数学：

> **最优路径 = “输入/ logit 通道 + 高信息性价比源 + 多源凸融合”。**

具体为：

1. **RAG 语义知识**：
   - 直接输入上下文；
   - 或训练 RAG-augmented self-distillation 到 student；
2. **Teacher 离线蒸馏**（等到高 RAM）：
   - 导出 teacher logits/text；
   - 在 logit 层做 KL；
3. **PLE / N-gram 词法专家**：
   - 仅用于低熵/代码/专名；
   - 用定理 5 估计 \(\lambda^*\)；
4. **统一 router**：
   - 对 RAG、teacher、n-gram、base 做 log-linear 凸融合；
5. **最后**：
   - 量化 + CPU serving。

---

## 9. 下一步实验指导

| 实验 | 验证哪一个数学结论 |
|---|---|
| RAG vs PLE rare QA | 信息序 / CMI |
| N-gram \(\lambda^*\) on code/names | 定理 5 |
| RAG self-distillation | 定理 1/4 |
| Multi-source router | 定理 3 |
| Teacher distillation | 定理 6 |
| PLE local real vs control | 判断 PLE 是否值得进 router |

---

## 10. 结论

> 最有效的路径不是“把 PLE 做大”，而是：
>
> 1. 用 RAG 获取高信息性价比；
> 2. 用 logit 层做最优融合；
> 3. 用 teacher 蒸馏提升推理/代码；
> 4. 用 n-gram/PLE 作为低成本局部专家；
> 5. 全部通过一个凸 router 统一。
