# Round 96：35 轮迭代搜索 + 最优多源记忆融合的数学推导与证明

> 日期：2026-09-05  
> 状态：多轮搜索与数学推导完成  
> 目标：从多视角、多数学分支推导“当前 0.8B + PLE + RAG + MoRA 的最优融合方法”，并给出可执行的后续实验协议。

---

## 0. 摘要

本轮进行了 **35 轮以上** web search，每轮根据上一轮结果调整关键词。  
检索到的关键相关工作包括：

- kNN-LM 的“何时依赖检索”理论；
- NGM 免训练记忆模块；
- MemSFT / TokenMem / Memory Grafting；
- RAG as Noisy In-Context Learning；
- Local Sufficiency / Mixture Identifiability；
- Blackwell 信息序；
- Logarithmic Opinion Pool；
- Calibration / Temperature Scaling；
- Selective Prediction / Entropy 不足；
- Rate-Distortion 记忆压缩；
- 在线专家选择 / 预测与专家建议。

核心结论：

> 之前“\(I(Y;M|H)>0\) 所以 PLE 可用”的数学推导是不完整的。  
> 正确的最优判据是：**在真实任务分布下，可实现的融合分布必须真正降低 log-loss；并且该条件要在 held-out 上测量，而不是用非负的 KL 代理。**

---

## 1. 问题形式化

设：

- \(H\)：模型已见上下文；
- \(Y\)：下一个 token 或任务答案；
- \(p_b(y|h)\)：当前 base 模型；
- \(p_m(y|h)\)：外部记忆/检索源给出的分布，例如 n-gram；
- \(p_t(y|h,m)\)：真实未知条件分布；
- \(S\)：记忆支撑集，即 n-gram 赋了非零概率的 token 集合。

我们考虑 log-linear 融合族：

\[
q_{\lambda,\beta}(y|h) \propto p_b(y|h)\exp\left(\lambda \log p_m(y|h) + \beta \mathbf{1}_{y\in S}\right)
\]

多源时推广为：

\[
q_{\lambda}(y|h) \propto p_b(y|h)\exp\left(\sum_i \lambda_i \log p_{m_i}(y|h)\right)
\]

---

## 2. 定理 1：CMI 是必要不充分条件

### 2.1 上界

任意从 \((H,M)\) 构造的注入 \(Z=f(H,M)\)，由数据处理不等式：

\[
I(Y;Z|H) \le I(Y;M|H)
\]

因此记忆的“潜在信息”不可能超过条件互信息。

### 2.2 为什么不足

最优 log-loss 可达到的下降是：

\[
\min_{q}\mathbb E[-\log q(Y|H,M)] - H(Y|H,M)
=
\min_{q}\mathbb E_{p_t}\big[\mathrm{KL}(p_t(Y|H,M)\|q(Y|H,M))\big]
\]

如果限定 \(q\) 只能取上面的 log-linear 族，那么：

\[
\Delta_{\text{achievable}}
\le I(Y;M|H)
\]

且等号成立 **当且仅当真实 \(p_t\) 本身就属于这个指数族**。

证明要点：

1. 对任意 \(q\)，log-loss 可分解为条件熵加 KL；
2. KL 非负，最小化 KL 得到可达到的损失；
3. 当 \(p_t\) 不在受限族内时，即使 \(I(Y;M|H)>0\)，受限融合也可能无法利用该信息。

**这直接解释了 P0 结果**：

> PLE-1/PLE-2 证明 n-gram 有真实信息，但那只是 \(I>0\)。  
> 我们的 \(q=\alpha\log p_m+\beta\) 是一个很窄的指数族，不一定包含真实条件分布，因此端到端可能无收益甚至负收益。

---

## 3. 定理 2：真正可操作的启用条件

### 3.1 正确判据

在真实分布 \(p_t\) 下，单独使用记忆分布 \(p_m\) 比 base \(p_b\) 更好的充要条件是：

\[
\Delta_t
=
\mathbb E_{p_t}\big[\log p_m(Y|H)-\log p_b(Y|H)\big]
> 0
\]

但这只是“直接用记忆替代 base”的条件。  
对于融合分布 \(q\)，真正要检验的是：

\[
\Delta_q
=
\mathbb E_{p_t}\big[-\log q(Y|H)\big]
-
\mathbb E_{p_t}\big[-\log p_b(Y|H)\big]
< 0
\]

即 **held-out 上融合 CE 必须严格小于 base CE**。

### 3.2 非负 KL 不能作为 gate

我们此前的 gate 使用：

\[
\mathrm{KL}(p_m\|p_b)
=
\sum_y p_m(y)\log\frac{p_m(y)}{p_b(y)}
\]

它恒非负，但 **不能推出 \(\Delta_t>0\)**。

反例：

设真实答案 \(Y=1\)，base \(p_b(1)=0.99\)，memory \(p_m(1)=0.5\)，  
memory 把剩余 0.5 分配到大量 base 认为极不可能的 token。

则：

\[
\mathrm{KL}(p_m\|p_b)>0
\]

但：

\[
\log p_m(1)-\log p_b(1)=\log 0.5-\log 0.99<0
\]

所以 \(\Delta_t<0\)，记忆有害。

**结论**：

> 运行时 gate 不能使用“KL 非负”或“KL > 阈值”。  
> 必须使用有标签 held-out 上的经验 \(\Delta_t\) 或直接测融合 CE 差。

---

## 4. 定理 3：最优 log-linear 融合权重

### 4.1 凸性

固定 \(p_b,p_m\)，函数：

\[
L(\lambda)=\mathbb E_{p_t}[-\log \mathrm{softmax}(z_b+\lambda s)(Y)]
\]

其中 \(s(y)=\log p_m(y)\)。

因为 softmax 负对数似然关于 logits 凸，且 logits 关于 \(\lambda\) 线性，所以 \(L(\lambda)\) 是凸函数。  
因此可以用梯度/一维搜索找到全局最优。

### 4.2 最优一阶条件

\[
\nabla_\lambda L(\lambda)
=
\mathbb E_{p_t}\big[\mathbb E_{q_\lambda}[\log p_m(Y)]-\log p_m(Y)\big]
=
0
\]

即：

\[
\boxed{
\mathbb E_{p_t}[\log p_m(Y)]
=
\mathbb E_{q_\lambda}[\log p_m(Y)]
}
\]

含义：

> 最优融合应使“模型当前预测分布下的记忆平均 log 概率”等于“真实目标分布下的记忆平均 log 概率”。  
> 这不是简单的相关性，而是指数族下的矩匹配条件。

### 4.3 近似最小二乘形式

如果近似认为真实答案的对数似然由 base 加噪声产生：

\[
L_t \approx L_b + \epsilon
\]

则单源最优权重近似：

\[
\lambda^*
\approx
\frac{\mathrm{Cov}(L_t-L_b,\;L_n-L_b)}
{\mathrm{Var}(L_n-L_b)}
\]

这正是 round-70 的形式。  
但它的适用前提是“误差近似线性”；更可靠的方法仍是直接优化 CE，而不是只算协方差。

---

## 5. 定理 4：偏置 \(\beta\) 的精确最优含义

我们目前的融合是对记忆支撑集 \(S\) 中所有 token 增加同一个 \(\beta\)。

考虑：

\[
q_\beta(y)\propto p_b(y)e^{\beta \mathbf{1}_{y\in S}}
\]

对 \(\beta\) 求导并令为零：

\[
\mathbb E_{p_t}[\mathbf{1}_{Y\in S}]
=
\mathbb E_{q_\beta}[\mathbf{1}_{Y\in S}]
\]

所以：

\[
\boxed{
q_\beta(Y\in S)=p_t(Y\in S)
}
\]

**含义**：

> 最优偏置 \(\beta\) 的作用是让融合模型在“记忆支撑集”上的总概率质量，等于真实分布在该支撑集上的质量。  
> 如果真实答案经常落在记忆支撑集内，\(\beta\) 应偏正；反之应偏负甚至关闭。

**这可以直接用来初始化/校准 PLE**：

- 在 held-out 上统计：
  \[
  \hat p_t(S)=\frac{\text{样本中真 token 落在记忆支撑集的比例}}{1}
  \]
- 调整 \(\beta\)，使融合模型在 \(S\) 上的概率质量接近 \(\hat p_t(S)\)；
- 如果 \(\hat p_t(S)\) 低于 base 模型在 \(S\) 上的质量，则说明该记忆支撑集没有提供额外真实支持，应关闭 PLE。

当前 P0 中：

- 对算术/code 题，记忆支撑集可能包含很多无关 token；
- 固定 \(\beta=-1\) 没有按任务估计 \(\hat p_t(S)\)；
- 所以可能把融合质量推向错误方向。

---

## 6. 定理 5：Blackwell 信息序与任务级源选择

### 6.1 定理

若记忆源 \(M_1\) 对 \(M_2\) 是 Blackwell 充分的，即存在随机变换把 \(M_1\) 映射到 \(M_2\)，则对任意决策规则：

\[
R(M_1)\le R(M_2)
\]

### 6.2 对受限融合的修正

Blackwell 充分性保证的是 **存在某个最优决策规则** 下 \(M_1\) 不差于 \(M_2\)，而不是保证我们的 log-linear 融合一定不差。

因此：

> 用 Blackwell 决定“哪一个源在任务级更值得作为主信息源”，  
> 但还要用“可实现融合 CE”决定“我们当前的接口是否真的能利用它”。

---

## 7. 定理 6：Logarithmic Opinion Pool 是校准专家的自然聚合

相关工作（Logarithmic Opinion Pool / Bayesian Inference for Weights in Log Pooling）指出：

- 对数意见池是校准专家条件下唯一满足某些外部贝叶斯一致性公理的聚合方式；
- 在我们的设定中，`base logits + λ·log p_m` 正是指数族下的对数意见池。

**关键限制**：

> 对数意见池假设各专家是“校准的”。  
> 如果 n-gram 专家未按任务校准，则融合不会自动变好，只会放大错误专家的置信度。

因此：

1. 先用 Theorem 2/4 判断专家是否在该任务上“比 base 更接近真实”；
2. 再决定是否允许进入对数意见池。

---

## 8. 定理 7：Entropy 单独不足以作为 gate

相关研究（*Entropy Alone is Insufficient for Safe Selective Prediction in LLMs*）和 L-RAG 都说明：

- 高熵可能来自“真实多答案”，不是信息不足；
- 低熵也可能来自“自信但错误”；
- 不确定性趋势比单点熵更有用。

因此对 round-89 的低熵 gate 做修正：

> 不能仅用 \(H(Y|H)<\tau\) 激活 PLE。  
> 必须同时满足任务级 \(\Delta_t>0\)，最好再满足：
> \[
> \Delta_{\text{real}}-\Delta_{\text{control}}>0
> \]
> 即真实记忆相对打乱 control 有可测优势。

---

## 9. 定理 8：Rate-Distortion 视角下的记忆容量分配

相关研究（*What to Keep, What to Forget* / *Rate-Distortion Framework for Agent Memory*）给出：

\[
\min_{\text{memory}} I(Y; M|H) + \lambda \cdot R(M)
\]

- \(R(M)\)：记忆存储/检索成本；
- n-gram/PLE：低 \(R\)，适合高重复、低熵、局部模式；
- RAG：高 \(I\)，适合语义/知识；
- 参数化（MoRA/LoRA）：高 \(C_p\)，适合将高频模式压进权重。

因此任务级最优分配为：

\[
\text{prefer}\ \arg\max_i \frac{I_i(Y;M_i|H)-\text{redundancy}}{c_i}
\]

这与 round-70 的“性价比”结论一致，但我们现在知道还必须扣除与 base 的冗余。

---

## 10. 定理 9：在线路由与专家建议的后悔界

如果把每个任务看成“从若干记忆源中选择专家”，并且每步能观测到每个源的 log-loss，那么 multiplicative weights / Hedge 有：

\[
R_T \le O(\sqrt{T\log K})
\]

其中 \(K\) 是源数量，\(T\) 是查询数。

这为后续提供一个轻量级自适应 router：

- 不需要训练大模型；
- 每个任务/域维护专家权重；
- 根据每步真实答案的 log-loss 更新；
- 可自动发现“PLE 在哪些任务上应关闭”。

---

## 11. 综合：最优方法

### 11.1 构造

1. **分域记忆**：
   - 语义/知识 → RAG / dense；
   - 代码补全 → 同域代码 n-gram bank；
   - 专名拼写 → 实体 n-gram bank；
   - 数字/日期格式 → 格式文本 n-gram bank；
   - 不要用单一大而杂的语料。

2. **per-task calibration**：
   - 在 held-out 上估计：
     \[
     \Delta_t = E[\log p_m(Y)-\log p_b(Y)]
     \]
   - 只保留 \(\Delta_t>0\) 且 real-control 差 > 0 的源。

3. **per-task fusion**：
   - 对每个任务分别优化 \((\lambda,\beta)\)；
   - 用定理 4 初始化/校准 \(\beta\)；
   - 直接最小化 held-out CE，因为凸性保证全局最优。

4. **router**：
   - 先做规则任务路由；
   - 再用在线专家权重自适应；
   - 语义任务强制关闭 PLE。

5. **评估**：
   - 使用真正的 code continuation、name/date/number 任务；
   - 报告：
     - per-task \(\Delta_t\);
     - real vs control;
     - 融合 CE 差；
     - exact match / pass@k;
     - 3 seed。

### 11.2 停止条件

> 如果某个真实局部任务上：
> \[
> \Delta_{\text{real}}>0,\quad
> \Delta_{\text{real}}-\Delta_{\text{control}}>0,\quad
> \Delta_{\text{fused}}<0
> \]
> 则 PLE 应进入该任务的 router。  
> 如果任意一项不满足，则关闭该任务的 PLE，直到改进记忆或校准。

---

## 12. 对下一轮实验的具体指导

| 实验 | 验证的数学结论 |
|---|---|
| 用同域代码语料构建 PLE bank，在 HumanEval/MBPP next-token 上评测 | 定理 1/2：CMI 与可实现性 |
| 对每个任务估计 \(\Delta_t\) | 定理 2：正确的 gate |
| 对每个任务单独优化 \((\lambda,\beta)\) | 定理 3/4：凸性与支撑集质量 |
| 在 PLE 消融中加入 real vs control | 定理 7：真实优势 vs 噪声 |
| 用在线专家权重做 router | 定理 9：轻量自适应 |
| 比较固定 \(\lambda\)、per-task \((\lambda,\beta)\)、support-mass 校准 | 定理 4/6 |
| 加 base entropy + memory top1 advantage 的 gate | 定理 7：熵不足 |
| 用 RAG as noisy ICL / local sufficiency 检查语义任务为何 PLE 无效 | 定理 5：Blackwell 信息序 |

---

## 13. 主要参考来源

- [You can’t pick your neighbors, or can you? When and how to rely on retrieval in kNN-LM](https://aclanthology.org/2022.findings-emnlp.218/)
- [NGM: A Plug-and-Play Training-Free Memory Module for LLMs](https://arxiv.org/html/2605.16893)
- [MemSFT: Mitigating Alignment Tax with an External Parametric Memory](https://arxiv.org/html/2607.25614)
- [TokenMem: Faithful Knowledge Injection for Frozen LLMs](https://arxiv.org/html/2607.22625)
- [Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory](https://ar5iv.labs.arxiv.org/html/2605.20948)
- [When Is Next-Token Prediction Useful? Local Sufficiency, RAG, Tools, and Programming](https://arxiv.org/html/2605.23278)
- [A Statistical Framework for Data-dependent Retrieval-Augmented Models](https://arxiv.org/html/2408.15399)
- [L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading](https://export.arxiv.org/pdf/2601.06551)
- [RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models](https://www.semanticscholar.org/paper/RAGRouter%3A-Learning-to-Route-Queries-to-Multiple-Zhang-Liu/bbc2dcac1d4e52e607dc17104414b6b0cee5fb44)
- [Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://www.semanticscholar.org/paper/Entropy-Alone-is-Insufficient-for-Safe-Selective-in-Phillips-Gustafsson/11019d48271fe2faaaf13a928af061ba900571d2)
- [What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction](https://arxiv.org/html/2607.08032)
- [Bayesian Inference for the Weights in Logarithmic Pooling](https://projecteuclid.org/journals/bayesian-analysis/volume-18/issue-1/Bayesian-Inference-for-the-Weights-in-Logarithmic-Pooling/10.1214/22-BA1311.full)
- [No-Regret Learning with Unbounded Losses: The Case of Logarithmic Pooling](https://papers.neurips.cc/paper_files/paper/2023/file/44ecfb60950e868a13172b935b7964a9-Paper-Conference.pdf)
- [A simple proof of Blackwell’s theorem on the comparison of experiments](https://www.sciencedirect.com/science/article/pii/S016517652400630X)
