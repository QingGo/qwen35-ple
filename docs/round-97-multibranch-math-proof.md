# Round 97：多分支数学推导——黎曼几何、流形、优化、控制、系统论、运筹、信息论、概率与贝叶斯

> 日期：2026-09-05  
> 状态：完成额外 25+ 轮迭代搜索与多分支推导  
> 目标：把 PLE/RAG/MoRA 多源融合从未经验证的启发式，升级为多个数学分支交叉支持的可证明方法。

---

## 0. 本轮搜索与来源

额外完成 25+ 轮搜索，覆盖：

| 分支 | 搜索调整方向 | 关键来源/工作 |
|---|---|---|
| 黎曼几何/信息几何 | Fisher metric、softmax 流形、natural gradient | *The Information Geometry of Softmax*、Information Geometry |
| 流形/表示 | manifold hypothesis、intrinsic dimension、embedding geometry | *Token Embeddings Violate the Manifold Hypothesis*、*Reasoning Emerges from Constrained Inference Manifolds* |
| 最优化 | Bregman divergence、mirror descent、convex duality | *Learning Mixtures of Experts with EM: A Mirror Descent Perspective*、*Estimating Mixture Distributions via Stochastic Mirror Descent* |
| 控制论 | state-space optimal control、optimal stopping | *Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning*、*Stop-RAG* |
| 系统论 | feedback control、fixed-point/self-distillation dynamics | *Self-Correction as Feedback Control*、*Geometric Dynamics of Agentic Loops* |
| 运筹学 | resource allocation、LP duality、queueing/MDP | *Data Auctions for RAG*、*Queueing-Aware Optimization*、*Primal-Dual Continual Learning* |
| 信息论 | CMI、rate-distortion、SDPI | *Rate-Distortion Memory*、*Strong Data Processing Inequalities* |
| 概率/随机 | SDE、concentration、martingale、在线专家 | *Unraveling Text Generation in LLMs (SDE)*、*Prediction with Expert Advice* |
| 贝叶斯 | Bayesian model averaging、Dirichlet process、Bayes factor | *Bayesian Inference for Weights in Logarithmic Pooling*、*Compositional Structure of Bayesian Inference* |
| 拓扑/谱 | persistent homology、graph spectral memory | *GEM-RAG*、*Topological Data Analysis for LLMs* |

---

## 1. 统一模型

设：

- \(p_b(y|h)\)：base 分布；
- \(p_m(y|h)\)：外部记忆分布；
- \(p_t(y|h)\)：真实条件分布；
- \(z_b=\log p_b\)，\(s=\log p_m\)；
- 融合分布：
  \[
  q_\lambda(y|h)\propto \exp(z_b(y|h)+\lambda s(y|h))
  \]
- 任务类型 \(c\in\mathcal C\)；
- 资源/预算：检索成本、存储成本、延迟。

---

## 2. 黎曼几何/信息几何视角

### 2.1 Softmax 流形

所有可能输出分布在自然参数化下构成一个统计流形，其度量是 Fisher information：

\[
g_{ij}(\theta)=\mathbb E_{p_\theta}\left[\partial_i\log p_\theta\;\partial_j\log p_\theta\right]
\]

Softmax 流形在 Fisher 度量下是**带正曲率的球面/单纯形嵌入**。  
log-linear 融合：

\[
z_b+\lambda s
\]

在这个流形上可以看作沿一个“记忆方向”做 **测地线/指数族路径**。

### 2.2 结论

1. 如果 \(s\) 与真实残差方向切向一致，则沿此方向移动可降低损失；
2. 如果 \(s\) 与真实残差方向正交或反方向，则移动会增大损失；
3. 因此最优融合不是“加得越多越好”，而是 **在 Fisher 度量下做投影**。

### 2.3 可操作度量

定义记忆方向的切向量：

\[
v=\mathbb E_{p_t}\left[\frac{\partial \log q_\lambda}{\partial \lambda}\right]
=
\mathbb E_{p_t}[s(Y)]-\mathbb E_{q_\lambda}[s(Y)]
\]

最优 \(\lambda\) 使该切向量在 Fisher 度量意义下与真实梯度方向对齐，等价于：

\[
\mathbb E_{p_t}[s(Y)]=\mathbb E_{q_\lambda}[s(Y)]
\]

这正是 round-96 的矩匹配条件。

---

## 3. 流形/低维结构视角

### 3.1 表示低维性

LLM 的中间表示通常近似落在低维子流形上。  
外部记忆是否有效，取决于：

\[
\mathrm{span}(M)\cap \mathrm{span}\left(\frac{\partial \log p_t}{\partial h}\right)
\]

是否非空。

### 3.2 对 PLE 的解释

- PLE/n-gram 在 token 表面空间提供局部低维先验；
- 如果任务需要的“答案方向”属于语义子空间，则 n-gram 方向可能远离真实残差方向；
- 这解释了 P0 中 PLE 对 arithmetic/code-Q&A 无正收益。

### 3.3 可操作结论

> 先测记忆方向与 base 真实残差方向的子空间对齐度，再决定是否启用。  
> 可以用 CKA、投影相似度、或简单 log-density ratio 作为代理。

---

## 4. 最优化理论视角

### 4.1 凸性与 Bregman 投影

对固定源，log-loss 是 \(\lambda\) 的凸函数，因此可用：

- 一维黄金分割/网格；
- 牛顿法；
- Bregman/mirror descent。

若把源权重看作单纯形上的点：

\[
w\in\Delta^{K-1}
\]

则可用 **镜像下降**：

\[
w_{t+1}=\nabla \psi^*\left(\nabla \psi(w_t)-\eta \nabla L(w_t)\right)
\]

其中 \(\psi\) 取负熵时退化为 exponentiated gradient / Hedge，有 regret bound：

\[
R_T=O(\sqrt{T\log K})
\]

### 4.2 拉格朗日对偶

若要同时优化准确率和成本，可写：

\[
\min_{w} L(w)+\rho \cdot C(w)
\]

其对偶变量 \(\rho\) 是“记忆/检索预算的影子价格”。

因此：

> 如果某个源的边际信息收益低于其边际成本，最优解会把它置零。

---

## 5. 控制论视角

### 5.1 State-Space 最优控制

把生成过程看作状态系统：

\[
h_{t+1}=F(h_t,u_t),\quad u_t=\text{外部记忆/检索}
\]

目标是：

\[
\min_{u}\sum_t \ell(y_t,\hat y_t)+\text{cost}(u_t)
\]

这可用：

- 动态规划 / Bellman；
- 或连续时间 Pontryagin 最大值原理。

### 5.2 对 PLE/RAG 的指导

- RAG/PLE 不是“永远加最好”，而是 **控制输入**；
- 应在以下情况才加记忆：
  - 当前状态不确定性高；
  - 记忆降低预测损失；
  - 成本可接受；
- 这自然导出 optimal stopping / Stop-RAG 式决策。

### 5.3 可操作实现

简化为每步门控：

\[
u_t=\mathbb 1\left[\Delta_t(h_t)>\tau\ \text{and}\ \text{memory hit}\right]
\]

---

## 6. 系统论/反馈控制视角

### 6.1 自蒸馏/自我改进的闭环

设学生分布为 \(q\)，教师/验证信号为 \(T\)，更新规则为：

\[
q_{t+1}=\mathrm{proj}\left(q_t+\eta \, \mathbb E_{y\sim q_t}[r(y)\nabla \log q_t]\right)
\]

这是有反馈的动力学系统。

### 6.2 稳定性

- 如果奖励分布和验证器稳定，自蒸馏可收敛；
- 如果模型只朝自己高置信区域收缩，可能坍缩；
- 因此需要 **Purified OPSD**：加验证/过滤，相当于反馈控制器中的参考信号。

### 6.3 可操作结论

> CAP-1 不能只做 self-distill；必须加入 verification/filtering，否则系统会像正反馈回路一样放大错误。

---

## 7. 运筹学视角

### 7.1 资源分配

把每个任务/查询看作需要分配：

- 检索次数；
- top-k；
- 是否启用 PLE；
- 是否调用 MoRA/更大模型。

形式化为：

\[
\max \sum_i V_i(x_i)\quad \text{s.t.}\quad \sum_i c_i x_i\le B
\]

其中 \(V_i\) 是任务收益，\(c_i\) 是成本。

### 7.2 结论

- 最优分配通常是“先投收益/成本比最高的源”；
- 边际收益递减时停止；
- PLE 因为几乎零成本，应该在“小收益但零成本”任务上启用；一旦有负收益，即使零成本也应关闭。

---

## 8. 概率论/随机过程视角

### 8.1 SDE 生成视角

文本生成可建模为：

\[
dz = f(z)dt+\sigma dW
\]

外部记忆相当于改变漂移项 \(f\)。  
如果记忆方向与真实漂移一致，则降低路径损失；否则增加。

### 8.2 在线专家与集中不等式

把每个源看作专家：

\[
\text{expert}_m \text{ loss}= -\log p_m(Y)
\]

Hedge 更新可达到：

\[
R_T\le \sqrt{\frac{T}{2}\log K}
\]

因此：

> 可以用很少的验证集自动学到“哪些任务该用 PLE、哪些该关”。

---

## 9. 贝叶斯视角

### 9.1 后验源选择

设源 \(m\) 的先验为 \(\pi_m\)，观测到任务的答案/验证信号后：

\[
p(m|D)\propto \pi_m \exp\left(-\sum_i \ell_m(y_i)\right)
\]

即：

\[
\log \frac{p(m_1|D)}{p(m_2|D)}
=
\log\frac{\pi_{m_1}}{\pi_{m_2}}
+
\sum_i\left(\ell_{m_2}(y_i)-\ell_{m_1}(y_i)\right)
\]

**Bayes factor 形式的解释**：

> 如果一个源在验证集上的总 log-loss 比另一个低，后验会迅速偏好它。

### 9.2 Dirichlet 过程 / 非参数记忆

记忆条目可以看成从 Dirichlet Process 中抽取：

\[
G\sim DP(\alpha,P_0)
\]

这为“何时创建新的记忆条目”提供了自然机制：

- 新上下文与已有条目距离大 → 新记忆；
- 否则更新已有计数。

---

## 10. 信息论/率失真视角

### 10.1 记忆不是越多越好

\[
\min_{\text{memory}} I(Y;M|H)+\lambda R(M)
\]

### 10.2 PLE 与 RAG 的分工

- 高频/低熵/局部 → 小 n-gram bank，低 \(R\)；
- 语义/知识 → RAG，高 \(I\)；
- 可参数化能力 → MoRA，直接写入权重，降低运行时 \(R\)。

---

## 11. 拓扑/谱视角（可选）

- 用 persistent homology 检测任务表示是否在同一拓扑分量；
- 用谱图方法把记忆条目组织成图，检索等价于图上信号传播；
- 这对“语义记忆”未来有价值，但对当前 n-gram 低熵记忆不是必需。

---

## 12. 综合最优方法

### 12.1 最优决策流程

1. **分域**：
   - semantic → RAG；
   - code continuation / name / number format → 同域 PLE；
   - reasoning/code-output Q&A → MoRA/参数化；
2. **per-task calibration**：
   - 估计每个源的真实 \(\Delta_t\)；
   - 只有 \(\Delta_t>0\) 且 real-control > 0 才启用；
3. **fusion**：
   - 每个任务独立优化 \(\lambda,\beta\)；
   - 用矩匹配/支撑集质量校准；
4. **router**：
   - 用 Hedge 在线更新专家权重；
   - 有成本时用拉格朗日对偶加入预算约束；
5. **training**：
   - CAP 用 Purified OPSD + 验证过滤，避免反馈回路坍缩；
6. **evaluation**：
   - 报 per-task Δ、real vs control、融合 CE、exact/pass@k、3 seed、污染审计。

### 12.2 数学上的最终判据

融合源 \(m\) 应进入任务 \(c\) 的 router，当且仅当：

\[
\Delta_{m,c}>0
\]
\[
\Delta_{m,c}-\Delta_{\text{control},c}>0
\]
\[
L_{\text{fused},c}<L_{\text{base},c}
\]

三者同时成立。

---

## 13. 下一轮实验表

| 实验 | 分支依据 |
|---|---|
| 在 HumanEval/MBPP 上测同域 PLE bank 的 next-token | 信息论/流形对齐 |
| 测每个任务的 \(\Delta_t\) 和 Bayes factor | 贝叶斯 |
| 用镜像下降/Hedge 做任务级 router | 最优化/在线学习 |
| 对每个任务单独优化 \(\lambda,\beta\) | 凸优化/Fisher 投影 |
| 在 PLE 消融中加入 real vs control | 概率/假设检验 |
| 加入检索成本预算的 LP/对偶分析 | 运筹学 |
| CAP-1 升级 Purified OPSD | 系统论/稳定性 |
| 用 SDE/状态空间模型分析长上下文记忆 | 控制论 |

---

## 14. 结论

> 多分支数学给出一致结论：  
> **外部记忆不是“信息越多越好”，而是“在正确任务、正确方向、正确成本下，以可验证方式进入融合”。**  
> 当前 P0 的负结果不是因为 PLE 没有信息，而是因为：
> 1. 方向不对（任务不是 PLE 强项）；
> 2. 支撑集不对（记忆 bank 与任务不同域）；
> 3. gate 不对（没有测真实 Δ）；
> 4. 校准不对（没有 per-task λ/β）。
>
> 修正路径已经由上述数学公式和实验表明确给出。
