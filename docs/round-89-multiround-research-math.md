# Round 89：多轮搜索调研 + 多视角数学推导

> 日期：2026-09-05  
> 状态：完成  
> 方法：33 轮 web search，每轮根据前一轮结果调整关键词；随后从信息论、统计决策、凸优化、谱方法、最优控制/预算视角、因果/分布偏移、PAC-Bayes 等角度做数学推导。  
> 用途：为 PLE-2 非参数可寻址记忆、CAP-1 参数微调、多源 router 提供可指导实验的结论。

---

## 1. 搜索轮次与关键发现

### 1.1 PLE / Engram / 非参数 n-gram 记忆

| 轮次 | 关键词方向 | 关键发现 |
|---|---|---|
| 1-4 | DeepSeek Engram / Qwen PLE / n-gram injector | Engram 是“条件记忆 + 可扩展查表”的稀疏轴；Qwen3.8-Flash-Next 原生 2/3-gram；4-gram 需要外部 bank |
| 4 | hot-tier / collision-free Engram | 高频 key 碰撞可能是 Engram 瓶颈；免碰撞 hot-tier 扩展可改善训练动态 |
| 5-6 | NGM / kNN-LM | NGM 是训练无关、即插即用记忆；kNN-LM 是经典非参数 logit 插值 |
| 7 | Lngram | n-gram 条件记忆可以进入 latent space，不只是 token 表面 |
| 10 | Memory Grafting / TF-Engram | 离线条件记忆可用于预训练扩展；TF-Engram 用 SSD + 预测预取实现 train-free 大容量记忆 |
| 16 | X-gram / long-tail | 超越固定 n-gram，数据感知 X-gram 更适合长尾 embedding 扩展 |

### 1.2 RAG / 检索 / 多源融合

| 轮次 | 关键词方向 | 关键发现 |
|---|---|---|
| 9-12 | RAG logit fusion / retrieval law | log-form retrieval law 比 power law 拟合更好；检索规模增长有边际递减 |
| 11-12 | memorize vs retrieve / inference scaling law | 参数记忆与检索存在最优分配；长上下文 RAG 推理有 scaling law |
| 17-18 | RAGRouter / query routing | 学习式 query routing 可显著改善多 RAG 模型选择 |
| 29-30 | entropy lazy loading / uncertainty trends | 基于熵/不确定性趋势决定何时检索，是低熵 gate 的天然相关工作 |
| 24-25 | kNN-LM negative / long-tail crisis | 非参数记忆不一定改善开放式生成；长尾是 kNN-LM 的危机 |
| 25 | When Not to Trust LM | 参数化/非参数记忆各有适用条件，需要门控 |

### 1.3 微调 / 高秩 PEFT / 蒸馏

| 轮次 | 关键词方向 | 关键发现 |
|---|---|---|
| 13 | ReAugKD / OPD | 检索增强蒸馏是低资源能力提升的有效路径 |
| 14 | MemSFT / TokMem / RETRO | 外部参数化记忆可缓解 alignment tax；TokMem 是 token 级记忆通道 |
| 21-22 | MoRA / DoRA / spectral | MoRA 用多个低秩/旋转/压缩实现高秩更新；DoRA 分解方向与幅度 |
| 32-33 | PAC-Bayes / RAG vs FT math | 存在 RAG 与 fine-tuning 的严格数学比较；记忆和参数化各有 generalization 边界 |

### 1.4 其他数学/工程相关

| 轮次 | 关键词方向 | 关键发现 |
|---|---|---|
| 19 | Blackwell / comparison of experiments | Blackwell 序是选择信息源的理论基础 |
| 20 | temperature scaling / logit fusion | 温度缩放是 logit 融合校准的默认工具 |
| 23 | causal / distribution shift RAG | 分布偏移下需要因果不变性/鲁棒检索 |
| 27-28 | sparse memory / gated memory | 稀疏门控记忆可用于长上下文和线性 RNN 状态扩展 |

---

## 2. 可借鉴相关工作矩阵

| 工作 | 可借鉴点 | 不借鉴点 | 与我们的关系 |
|---|---|---|---|
| DeepSeek Engram | 稀疏查表条件记忆、规模轴 | 不把 n-gram 当语义预测器 | PLE 的核心来源 |
| NGM | 训练无关、即插即用、logit 修正 | 不解决语义知识 | 与 NgramLM 同类 |
| Lngram | latent-space n-gram memory | 需要额外训练 latent 表示 | PLE-2 可升级方向 |
| Memory Grafting | 离线条件记忆 + target reader | 需要大规模预训练 | 我们当前冻结 backbone 的 reader 对应 |
| TF-Engram | 大容量 SSD + 预测预取 | 工程复杂，本地暂不需要 | 未来产品化容量扩展 |
| kNN-LM 系列 | 非参数 logit 插值经典 | 开放式生成可能退化 | 我们已用 real/control 门禁 |
| ReAugKD | RAG 增强蒸馏 | 不把 RAG 当唯一主线 | CAP-1 核心 |
| RAGRouter | 学习式 query routing | 需要较多训练数据 | 多源 router 参考 |
| L-RAG / uncertainty trends | 熵/不确定性触发检索 | 依赖在线动态信号 | 低熵 gate 参考 |
| MoRA/DoRA | 高秩/结构化参数更新 | 不直接解决记忆 | CAP-1 高秩微调 |
| RAG scaling laws | 检索规模/log-form 规律 | 不能直接套到 n-gram | 决定记忆容量优先级 |
| Blackwell order | 信息源比较 | 需可测条件 | 源选择理论基础 |
| Temperature calibration | logit 尺度校准 | 不能替代信息选择 | 融合必要组件 |

---

## 3. 数学框架

### 3.1 记号

- \(H\)：backbone 已看到的上下文；
- \(Y\)：下一个 token / 任务标签；
- \(p_\theta(y|h)\)：base 分布；
- \(M\)：外部记忆/检索源；
- \(D\)：RAG 文档；
- \(E\)：PLE/n-gram 记忆；
- \(T\)：teacher 输出；
- \(\ell_f = \log p_\theta(y|h) + \sum_i \lambda_i \log p_{S_i}(y|h)\)：log-linear logit 融合。

### 3.2 定理 1：记忆信息的通道上界

**定理**：对任意记忆源 \(M\) 和任意可计算注入 \(Z=f(M,H)\)，有

\[
I(Y;Z|H)\le I(Y;M|H).
\]

进一步，在 log-loss 下，最优可达到的损失下降为

\[
\Delta \ell^* = I(Y;M|H).
\]

**证明思路**：

1. \(Y-H-M\) 是条件 Markov 链；
2. \(Z\) 是 \((H,M)\) 的函数，因此 data processing inequality 给出 \(I(Y;Z|H)\le I(Y;M|H)\)；
3. 令最优预测为 \(p^*(y|h,m)=\Pr(Y=y|H=h,M=m)\)，则 log-loss 从 base 到最优的下降正好是
   \[
   \mathbb E_{h,m}\big[\mathrm{KL}(p^*(\cdot|h,m)\|p_\theta(\cdot|h))\big]=I(Y;M|H).
   \]

**指导**：
- 不要追逐“看起来像记忆”的特征，先测 \(I(Y;M|H)\) 的上界；
- PLE/n-gram 在语义知识上 \(I\) 很小，在代码/专名/低熵上相对更高，实测与此一致。

### 3.3 定理 2：最优 logit 修正是条件对数似然比

**定理**：若允许对 base logits 做任意加性修正 \(\Delta(y|h,m)\)，则 log-loss 最优修正是

\[
\Delta^*(y|h,m)=\log\frac{p^*(y|h,m)}{p_\theta(y|h)}.
\]

**证明思路**：

对任意预测分布 \(q\)，log-loss 的泛函导数为零时得到后验分布；因 softmax 的完备性，任意分布都可以表示为 base logits + 一个加性修正。

**指导**：
- 我们应当让记忆模块输出“条件似然比”或等效的对数先验，而不是直接输出 hidden 向量；
- 这正是我们选择 logit 通道的原因。

### 3.4 定理 3：log-linear 融合是凸优化

**定理**：固定记忆源 \(S_1,\dots,S_k\)，对

\[
\ell_f(\lambda)=\ell_b+\sum_i\lambda_i\log p_{S_i}
\]

来说，期望 log-loss

\[
L(\lambda)=\mathbb E[-\log \mathrm{softmax}(\ell_f(\lambda))(Y)]
\]

是 \(\lambda\) 的凸函数。

**证明思路**：

1. \(\mathrm{softmax}(\ell)\) 的负对数似然关于 \(\ell\) 凸；
2. \(\ell_f(\lambda)\) 关于 \(\lambda\) 线性；
3. 仿射变换保凸；所以整体凸，存在全局最优，且可用梯度/网格搜索稳定求解。

**指导**：
- 不需要用复杂神经网络 router 来融合固定 logit 源；
- 但若 router 权重依赖 \(H\)，则变成函数学习问题，需要正则化和足够数据。

### 3.5 定理 4：使用外部记忆的判别条件

**定理**：若 base 分布为 \(p_b\)，记忆分布为 \(p_m\)，则在真实分布 \(p_t\) 下，使用记忆的期望 log-loss 低于不使用记忆，当且仅当

\[
\mathbb E_{p_t}\big[\log p_m(Y|H)-\log p_b(Y|H)\big]>0.
\]

**证明思路**：

比较两个预测分布的 log-loss 差：

\[
\mathbb E_{p_t}[-\log p_m]- \mathbb E_{p_t}[-\log p_b]
=
\mathbb E_{p_t}[\log p_b-\log p_m].
\]

因此记忆更优当且仅当上述期望为负，即 \(\mathbb E[\log(p_m/p_b)]>0\)。

**指导**：
- 这是最直接、最可操作的 gate：在 held-out 上测平均 log-density ratio；
- 我们 real/control 的 paired logprob 差正是此量的经验估计；
- 如果为负，应关闭记忆或校准后使用。

### 3.6 定理 5：Blackwell 信息序与源选择

**定理**：如果记忆源 \(M_1\) 相对 \(M_2\) 是 Blackwell 更充分的（即 \(M_1\) 是 \(M_2\) 的“加噪”逆操作），则对任意决策规则和任意损失，使用 \(M_1\) 的最优风险不大于使用 \(M_2\)。

**证明思路**：

Blackwell 充分性等价于存在一个随机变换把 \(M_1\) 的分布映到 \(M_2\) 的分布，因此任何基于 \(M_2\) 的决策都可以通过 \(M_1\) 模拟。

**指导**：
- 在真实系统中，RAG/teacher 在语义任务上往往比 n-gram 更充分；
- 在低熵局部任务上，n-gram 可能更充分；
- 因此按任务选择信息源，而不是追求统一“最强记忆”。

### 3.7 定理 6：记忆/参数化容量分配

**定理**：设任务真实条件熵为 \(H(Y|H)\)，参数化模型可稳定记忆 \(C_p\) bits，外部记忆可提供 \(C_m\) bits，且二者冗余为 \(I_{\text{red}}\)，则有效容量

\[
C_{\text{eff}}=\min\big(C_p+C_m-I_{\text{red}},\;H(Y|H)\big).
\]

**证明思路**：

这是信息容量的并集/交叠基本不等式。

**指导**：
- 高频、重复、低熵模式应尽量参数化；
- 长尾、稀有、局部模式放入外部记忆；
- 检索/记忆与参数化之间冗余较大时，收益递减；这也解释了 RAG scaling law 的对数/幂律形式。

### 3.8 定理 7：低熵 gate 与长尾任务分解

**定理**：对 token 级任务，若

\[
H(Y|H)<\tau,\quad \text{且}\quad \Pr(\hat Y_{\text{ngram}}=Y)>\Pr(\hat Y_{\text{base}}=Y),
\]

则 n-gram 记忆应被激活。

**指导**：
- 使用 base 模型熵作为 gate 信号；
- 在低熵/代码/专名/数字上才启用 PLE，语义任务交给 RAG/Dense；
- 这与 L-RAG、uncertainty trends 等工作的思想一致。

### 3.9 定理 8：校准参数的最优性

**定理**：当使用 \(l_f=l_b+\alpha\log p_m+\beta\) 时，最优 \((\alpha,\beta)\) 可由最小化 held-out log-loss 得到；固定 \(\alpha\) 时最优 \(\beta\) 满足“使记忆候选 token 族的平均 logit 与 base 对齐”。

**证明思路**：

对 \(\beta\) 求导，得到 \(\beta\) 只影响记忆候选集合的整体偏置；对 \(\alpha\) 求导得到 log-density ratio 与 logit 尺度匹配条件。

**指导**：
- 不能用单一 \(\lambda\)；应使用 scale + bias（+ temperature）；
- 我们已在真实 base logits 上验证 scale+bias 比单 λ 更有效。

### 3.10 定理 9：检索规模递减与 log-form law

**定理**：如果每个新增检索文档与已有文档的信息重叠率为 \(\rho\)，则前 \(k\) 个文档的累积信息近似为

\[
I_k \approx I_\infty\big(1-\rho^k\big),
\]

当 \(I_\infty\) 为所有相关文档总信息、\(\rho<1\) 时，信息随 \(k\) 指数饱和；在对数尺度上表现为 log-form scaling。

**指导**：
- 不是“检索越多越好”；
- 应优先提高检索精度和去重，而不是无限增加 top-k。

---

## 4. 对我们后续实验的指导

### 4.1 PLE-2：非参数可寻址记忆

1. **以 logit 层似然比为主通道**，不要再走 hidden 注入；
2. **使用 log-density ratio 门控**：只有 \(E[\log(p_m/p_b)]>0\) 才激活；
3. **按任务分域**：
   - code / name / number / low-entropy → PLE/n-gram；
   - semantic QA → Dense/RAG；
4. **校准参数**：scale + bias + temperature，不要用单一 λ；
5. **value 质料**：函数块/段落/实体条目已覆盖；实体知识应交给 RAG，不要指望 n-gram 精确记忆。

### 4.2 CAP-1：RAG self-distillation + LoRA/QLoRA/MoRA

1. RAG self-distill 对 code/arithmetic 有效，但对 knowledge 可能略降；
2. 应做 **任务条件 adapter 或 router**，而不是单 adapter 全局替换；
3. MoRA 已 vendor 并可训练，应增加多 seed 和更大规模对比；
4. 训练数据应排除 self-retrieval 泄漏（已做 `--exclude-source`）；
5. 与 RAG 联合时，保留外部知识，让 adapter 专注“格式/推理/局部模式”。

### 4.3 多源 router

1. 固定 logit 融合是凸优化，可直接求最优权重；
2. 若 router 依赖输入，按任务/熵/不确定性做 gating；
3. 可借鉴 RAGRouter 的 query routing，但本地先做规则/轻量模型；
4. 最终评估应同时看：
   - held-out log-loss；
   - exact match / pass@k；
   - real vs control；
   - contamination audit。

---

## 5. 结论

> 最有效的路径不是“让某一记忆替代 LLM 智能”，而是：
>
> 1. 用信息论上界筛选记忆源；
> 2. 用 Blackwell/CMI 决定任务级源选择；
> 3. 用 log-density ratio 做记忆 gate；
> 4. 用凸 log-linear logit 融合做多源整合；
> 5. 用 scale+bias+temperature 做校准；
> 6. 用 RAG self-distill + MoRA/QLoRA 提升 base 的格式/推理/局部能力；
> 7. 用任务条件 router 把 PLE、RAG、参数化能力组合成最终系统。

---

## 6. 已保存/可复现内容

- 本文档：`docs/round-89-multiround-research-math.md`
- 之前的实现与实验均在仓库中：
  - `src/qwen35_ple/addressable_memory.py`
  - `src/qwen35_ple/fusion.py`
  - `src/qwen35_ple/router.py`
  - `scripts/run_ple1_ngram_eval.py`
  - `scripts/run_rag_channel_ablation.py`
  - `scripts/run_fusion_calibration.py`
  - `scripts/run_ple2_semantic_values_3seed.py`
  - `scripts/run_ple2_entity_memory_eval.py`
  - `scripts/run_cap1_eval_lora.py`
  - `scripts/run_lora_distill.py`（支持 LoRA/QLoRA/MoRA）
  - `vendor/peft-mora/src/peft`
