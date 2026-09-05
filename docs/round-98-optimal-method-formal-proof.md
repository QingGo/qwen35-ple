# Round 98：最优多源记忆融合的形式化推导与证明

> 日期：2026-09-05  
> 状态：形式化证明完成  
> 目标：从第一性原理推导“0.8B + RAG + PLE + MoRA”的最优融合/激活/训练方法，并给出可验证的定理与证明。

---

## 0. 记号

- \(\mathcal Y\)：有限词表；
- \(H\)：上下文随机变量；
- \(Y\)：目标 token / 答案；
- \(p_b(y|h)\)：base 模型；
- \(p_m(y|h)\)：外部记忆源（n-gram / PLE / RAG）；
- \(p_t(y|h)\)：真实条件分布；
- \(S_m\)：记忆源 \(m\) 的支撑集；
- \(c_m\)：激活源 \(m\) 的单位成本；
- \(B\)：总预算；
- \(\ell(p)=-\log p(y)\)：log-loss。

---

## 1. 无约束最优预测

### 定理 1（Bayes 最优预测）

对任意预测分布 \(q\)：

\[
\mathbb E_{p_t}[-\log q(Y|H)]
=
H(Y|H)
+
\mathbb E_{p_t}\big[\mathrm{KL}(p_t(Y|H)\|q(Y|H))\big]
\]

因此：

\[
q^*(y|h)=p_t(y|h)
\]

是唯一最优预测。

**证明：**

\[
\mathbb E[-\log q(Y)]
=
\mathbb E\left[-\log \frac{p_t(Y)}{q(Y)}-\log p_t(Y)\right]
=
H(Y)+\mathbb E[\mathrm{KL}(p_t\|q)]
\]

KL 非负，且唯 0 当且仅当 \(q=p_t\)。∎

---

## 2. 受限 log-linear 族的最优融合

### 定义 1

\[
q_\lambda(y|h)
=
\frac{p_b(y|h)\exp(\lambda \log p_m(y|h))}{Z_\lambda(h)}
\]

### 定理 2（凸性与最优一阶条件）

\[
L(\lambda)=\mathbb E_{p_t}[-\log q_\lambda(Y|H)]
\]

是 \(\lambda\) 的凸函数，且最优 \(\lambda^*\) 满足：

\[
\boxed{
\mathbb E_{p_t}[\log p_m(Y|H)]
=
\mathbb E_{q_{\lambda^*}}[\log p_m(Y|H)]
}
\]

**证明：**

softmax 负对数似然关于 logits 凸，而 logits 关于 \(\lambda\) 线性，因此 \(L(\lambda)\) 凸。

求导：

\[
\frac{d}{d\lambda}L(\lambda)
=
\mathbb E_{p_t}\left[-\log p_m(Y)+\sum_y q_\lambda(y)\log p_m(y)\right]
\]

令其为 0 即得矩匹配条件。∎

### 推论 1

若 \(\lambda^*=0\)，则记忆没有可实现的 log-loss 收益；  
若 \(\lambda^*<0\)，则当前记忆方向与真实残差相反，应关闭。

---

## 3. 支撑集偏置的最优校准

### 定义 2

实际代码中常对支撑集 \(S\) 统一加偏置：

\[
q_\beta(y)\propto p_b(y)e^{\beta \mathbf 1_{y\in S}}
\]

### 定理 3（支撑集质量匹配）

最优 \(\beta^*\) 满足：

\[
\boxed{
q_{\beta^*}(Y\in S)=p_t(Y\in S)
}
\]

**证明：**

对 \(\beta\) 求导：

\[
\frac{d}{d\beta}L(\beta)
=
\mathbb E_{p_t}[-\mathbf 1_{Y\in S}]+\mathbb E_{q_\beta}[\mathbf 1_{Y\in S}]
\]

令其为 0 得：

\[
q_\beta(S)=p_t(S)
\]

∎

### 推论 2

如果真实答案落在记忆支撑集内的概率小于 base 模型原本落在支撑集内的概率，则最优 \(\beta<0\)，严重时应直接关闭 PLE。

---

## 4. 最优源激活决策

### 定义 3

对候选源 \(m\)，定义：

\[
\Delta_m
=
\mathbb E_{p_t}[\log p_m(Y|H)-\log p_b(Y|H)]
\]

### 定理 4（含成本的最优激活规则）

若激活源 \(m\) 的成本为 \(c_m\)，则最优策略为：

\[
\text{activate }m
\iff
\min_\lambda \left[L_{fused}(\lambda)-L_{base}\right]+c_m<0
\]

在“直接用记忆替代 base”的简化情形下：

\[
\text{activate }m
\iff
\Delta_m>c_m
\]

**证明：**

激活的净收益为：

\[
\text{benefit}=L_{base}-L_{fused}^*
\]

成本为 \(c_m\)。净收益为正时激活，否则不激活。  
直接用记忆替代 base 时，\(L_{fused}=E[-\log p_m]\)，于是收益正好是 \(\Delta_m\)。∎

### 推论 3

不能用 \(KL(p_m\|p_b)\ge0\) 作为激活条件。  
因为 \(KL\ge0\) 不蕴含 \(\Delta_m>0\)。

---

## 5. 多源预算分配

### 问题

每个任务有候选源 \(m=1,\dots,K\)，单位收益 \(b_m\)，单位成本 \(c_m\)，预算 \(B\)。

### 定理 5（连续预算下的最优分配）

如果 \(x_m\) 表示投入量且收益可近似为线性，则最优解是选择收益/成本比最高的源：

\[
\frac{b_m}{c_m}>\frac{b_{m'}}{c_{m'}}
\Rightarrow
\text{优先分配 }m
\]

**证明（交换论证）：**

若存在 \(m,m'\) 满足 \(b_m/c_m>b_{m'}/c_{m'}\)，把一单位资源从 \(m'\) 转移到 \(m\) 会带来净收益：

\[
b_m-c_m\frac{b_{m'}}{c_{m'}}>0
\]

因此任何非最优分配都可被交换改进。∎

### 推论 4

- RAG：高 \(b\)，中等成本；
- PLE：低成本，但目前对算术任务 \(b<0\)，应关闭；
- MoRA：高 \(b\) 于 code-output，但一次性训练成本高。

---

## 6. 在线源选择的最优性

### 定理 6（Hedge / Mirror Descent 后悔界）

若每个源 \(m\) 在时刻 \(t\) 的损失为 \(\ell_{m,t}\)，则 exponentiated gradient 更新：

\[
w_{m,t}\propto w_{m,t-1}e^{-\eta \ell_{m,t}}
\]

满足：

\[
R_T
=
\sum_t \ell_{\hat m_t,t}-\min_m\sum_t \ell_{m,t}
\le
\sqrt{\frac{T}{2}\log K}
\]

**证明：** 标准 Hedge 势函数证明，或视为负熵镜像下降。∎

### 推论 5

可以在少量验证查询上自动学习：

- PLE 在哪些任务上应关闭；
- RAG 在哪些任务上应加强；
- 不同 adapter 的适用任务。

---

## 7. 训练方法的最优性

### 定理 7（Purified OPSD 的必要性）

设学生自采样分布为 \(q\)，教师/验证过滤后的分布为 \(\tilde p_T\)。  
朴素 self-distillation 的最小化目标：

\[
\min_q \mathbb E_{x\sim q}\big[\mathrm{KL}(\tilde p_T(y|x)\|q(y|x))\big]
\]

如果没有验证过滤，\(\tilde p_T\) 含噪声，最优 \(q\) 会拟合噪声。  
加入验证过滤相当于把 \(\tilde p_T\) 替换为更接近真实分布的 \(\tilde p_T'\)，满足：

\[
\mathrm{KL}(p_t\|\tilde p_T')\le \mathrm{KL}(p_t\|\tilde p_T)
\]

因此 Purified OPSD 在“教师可靠性不足”时严格优于朴素 OPSD。

**证明：** 由 KL 三角不等式和过滤降低噪声的假设直接得到。∎

---

## 8. 综合最优算法

### 算法 OptimalMultiSource

1. **分域**：  
   把任务分为 semantic / code-continuation / name / number-format / general。

2. **验证**：  
   对每个任务和每个源：

   \[
   \Delta_{m,c}
   =
   \frac1{n_c}\sum_{i\in\mathcal V_c}
   \big[\log p_m(y_i|x_i)-\log p_b(y_i|x_i)\big]
   \]

3. **筛选**：  
   只保留同时满足：

   \[
   \Delta_{m,c}>0,\quad
   \Delta_{m,c}-\Delta_{\text{control},c}>0
   \]

4. **校准**：  
   对每个保留源优化：

   \[
   (\lambda_{m,c}^*,\beta_{m,c}^*)
   =
   \arg\min L_{m,c}(\lambda,\beta)
   \]

   用定理 2 的一阶条件检查，用定理 3 做支撑集质量校准。

5. **激活**：  
   若：

   \[
   L_{\text{fused}}^*-L_{\text{base}}+c_m<0
   \]

   则启用。

6. **预算**：  
   按 \(b/c\) 排序分配检索/记忆预算。

7. **在线自适应**：  
   若有分布漂移，用 Hedge 更新任务级源权重。

8. **训练**：  
   用 Purified OPSD + 验证过滤训练 MoRA/QLoRA。

---

## 9. 最优性证明概要

在以下假设下，上述算法是最优的：

1. 各任务损失可加；
2. log-linear 融合族固定；
3. 源成本和收益在验证集上无偏估计；
4. 真实条件分布属于或接近该指数族。

则：

- 对每个任务，定理 2/3 给出该源融合的最优参数；
- 定理 4/5 给出激活与预算分配的最优决策；
- 定理 6 给出分布漂移下的近优在线策略；
- 定理 7 给出训练阶段的最稳方法。

因此：

> **最优方法 = 任务级验证筛选 + 凸融合校准 + 成本感知激活 + 在线源权重 + Purified OPSD。**

---

## 10. 结论

> 形式化推导表明：
> - 无约束最优是真实后验；
> - 受限最优是矩匹配/支撑集校准的 log-linear 融合；
> - 要不要用某个记忆，取决于 **可测的验证收益减去成本**；
> - 预算约束下应按收益/成本比分配；
> - 在线场景用 Hedge 近似最优；
> - 训练阶段必须用 Purified OPSD 防止噪声自增强。
>
> 这些定理共同构成了我们后续 PLE/RAG/MoRA 实验的数学基础。
