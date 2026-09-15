# Round 170 调研:稀疏记忆"让它发挥作用"的五个独立诊断

**目的**:为论文 B("纯 PLE 嫁接 + 少量训练 → 提升 0.8B 通用性能")建立文献基线。
**方法**:先按第一性原理拆出"价值只能从哪来",再对每一轴做迭代检索,最后收敛。

---

## 0. 第一性原理:一个 token-keyed memory 的价值只能从三处来

由界 `I(Y;e|h_t) ≤ I(Y;w_t|h_t)`,可动的量只有三个:

```text
① 通道能携带多少   I(Y; w_t | h_t)      ← 由窗口与表决定
② 骨干没记住多少   (界的条件项 h_t)      ← 决定①里有多少是"新增信息"
③ 读出送出去多少   rung ④ 实测:有效维度 1.001

⇒ 任何"让 PLE 更起作用"的思路,必须落在其中一处。否则它只是在换说法。
```

五条独立研究线**在同一个诊断上收敛** —— 而那个诊断不在①,而在**每个条目的证据量**。

---

## 1. 五个独立诊断:瓶颈是"每个条目的证据量",不是容量/寻址/优化器

| 工作 | 诊断 | 数字 |
|---|---|---|
| **本项目**(B1) | 三元组计数分布极度倾斜 ⇒ 行只拿到 1–4 步梯度 | **87.5% 的三元组是单例**;667,404 行里只有 5,063 行计数 ≥10 |
| **SCONE** [2502.01637](https://ar5iv.labs.arxiv.org/html/2502.01637) | 词表变大 ⇒ 尾部条目更新次数崩塌 ⇒ 表示质量下降 | 100M token 训练:32K 词表 **97.6%** 的 token 拿到 >100 次更新;2M 词表只剩 **7.3%** |
| **Memory Grafting** [2605.20948](https://ar5iv.labs.arxiv.org/html/2605.20948) | 记忆从零学 ⇒ 记忆一扩容就要重训 | **记忆占总参数 >70% 后训练效率开始下降** |
| **Engram** [2601.07372](https://ar5iv.labs.arxiv.org/html/2601.07372) | 表从零学,扩容成本高 | 引 Memory Grafting 的同一诊断 |
| **本项目**(更早) | 更长的 key 不赚 | SCONE 独立复现:**f-gram 长度 K 从 2 涨到 4,困惑度反而上升**,之后才走平 |

**三条线用三种完全不同的语言说出了同一件事:**

> **一张稀疏表的每一个条目都需要足够的更新次数;当条目数增长快过语料时,
> 直接对表做梯度下降就失效了。**

我们的"行选择"({count ≥ 10} 才训)只是这个诊断的**最弱回应** —— 它承认了问题,
但只是**不去碰**那些条目,没有**修**它们。

---

## 2. 两个已发表的解法(我们都没试过)

### 解法 A —— SCONE:不要把嵌入当**参数**,把它当成一个**函数的输出**

```text
直接反传                             SCONE
─────────────                       ────────────────────────────────
表: V_f-gram → ℝ^d                   小 transformer A_f-gram: n-gram → ℝ^d
每个条目独立、只被自己的计数更新        相似 n-gram **共享参数**
⇒ 单例条目 = 单样本估计               ⇒ 单例条目也能拿到好嵌入
推理:查表                            推理:预计算 → 缓存成查找表(可卸载到内存/SSD)
```

**论文原话**:*"The direct approach fails to exploit dependencies between n-grams,
leading to fewer updates per embedding... Scone addresses this by parameterizing
embeddings with an f-gram transformer, avoiding the sparse update problem."*

**实测**:419M 主模型 + 170M f-gram 模型 → **ppl 23.4**,胜过 **589M 基线(24.7)**;
1B 加速器参数 + 1B f-gram 嵌入 → 胜过 1.9B 基线,而推理 FLOPs/显存只有一半。
f-gram 表从 512K 涨到 100M,困惑度**持续下降**。

**其他设计点**:f-gram 用 BPE 式发现、**最低频次阈值 5**、按频次取 top-N、
**最长匹配优先**、K=5。

### 解法 B —— Memory Grafting:表不学,直接从**更强的模型**那里抄

```text
离线跑一个 grafting model(teacher)
对每个频繁 n-gram,取它**最后一个 token 的隐状态** → 存成记忆值(冻结)
recipient 用**精确最长后缀匹配**检索,轻量 projection + gate 注入
未命中的上下文回退到 hash-based Engram
```

**实测**(2.8B 可训、100B token、匹配架构与预算):

| | MoE | vanilla Engram | **Memory Grafting** |
|---|---|---|---|
| 平均基准分 | 51.95 | 52.43 | **53.86** |

0.92B 规模下所有 grafting 变体都优于基线,**teacher 越强增益越大**
(Qwen3.5-35B-A3B 最强)。

**它的核心主张**:*"A sufficiently pre-trained model has already learned strong
representations for these patterns. If these encodings can be used as conditional
memory, the memory capacity can scale without increasing the recipient's training
cost."*

---

## 3. Engram 自己消融出的四条工程杠杆(受控:3B MoE,100B token,1.6B 记忆)

来源:Engram §6.2 / §6.3 / 图 5。参考配置 = **{2,3}-gram 注入在 layer 2 和 layer 6**。

| 杠杆 | 结论 | **我们的现状** |
|---|---|---|
| **层位置** | 单模块扫描 1→12:**layer 2 最优**,越深越差 | 我们就是**单层 layer 2** ✓ 对,但见下 |
| **层数** | **同样的预算分成两个模块放在 (2, 6) 比单模块更好** —— "把早期干预与晚期丰富上下文门控结合起来" | ❌ **我们只有一层,是次优配置** |
| **门控** | 组件消融里回退最大的三个之一 | ❌ 我们的读出击塌成 **1.001 维** ⇒ 没有逐位置门控能力 |
| **tokenizer 压缩** | 三个最重要组件之一;NFKC + 小写把语义等价的 token 折叠(**128K 词表缩 23%**) | ❌ 我们用**原始 token id** |
| **阶数** | {2,3}-gram;4-gram 在固定预算下**略差**("稀释了更高频 2/3-gram 的容量") | ❌ 我们只有**单一 3-gram** |
| **多分支** | 三个最重要组件之一 | ❌ 单流 |

**§6.3 的敏感性实验**(推理时完全抑制记忆输出):**事实知识掉到原性能的 29–44%,
阅读理解保留 81–93%。** 即:在他们那个共训模型里,记忆确实是**参数知识的主要仓库**。

**门控可视化**:门控在**多 token 命名实体与公式化短语**上持续激活 —— 与我们的
"高 surprisal 尾部"完全一致。

**注意**:Engram 用 **Muon** 训练,但训练的是 **MoE 骨干的稠密权重**,不是稀疏嵌入行。
这与我们"Muon 不适用于稀疏按行的嵌入表"的结论**不矛盾** —— 适用对象不同。

---

## 4. 另一条独立线索:读出必须被"按住"

| 工作 | 发现 | 与我们的关系 |
|---|---|---|
| **Ordo-M** [2608](https://www.alphaxiv.org/abs/2608.ordo-m-sparse-memory-frozen-model) | 冻结模型 + 稀疏记忆,**parameter-free positional gate 把 text damage 降低 96–97%**;记忆会"shouting over"基座 | 我们实测 **46.3% 的位置受害**、top 1% 承载 65.6% ⇒ 同一个病 |
| **Ordo-M** | 知识必须是**"答案形状"**:原始文本几乎无效,短陈述句达 69.4% | 与项目 1.5e"去掉答案长度监督后结论改变"同一现象 |
| **Engram** §6.2 | 门控是回退最大的组件之一 | 我们塌成 1 维 ⇒ 门控名存实亡 |
| **本项目** rung ④ | 注入向量有效维度 **1.001**,h_t 是 1.480 | 一个近常数向量**无法**逐位置门控 |

**⇒ 我们的 +0.176 nats 是下界。** oracle sign 门控给 **+0.355**(2×),
oracle surprisal 门控给 **+0.207**。

---

## 5. 把文献映射到我们的约束上

**我们的设定**:0.8B **冻结**骨干 + 官方 FP8 行表 + **少量训练**。
**Engram 的设定**:从头共训 MoE + 记忆,262B token。

| 文献杠杆 | 在"冻骨干"下可行? | 成本 |
|---|---|---|
| **tokenizer 压缩后再做 key** | ✅ 纯离线,不动模型 | **极低** |
| **多阶 {2,3}-gram + 最长匹配** | ✅ 需要改寻址 | 低 |
| **(2,6) 双层注入** | ✅ 需要改注入点 | 低 |
| **门控/非塌陷读出** | ✅ 训一个小 reader | 低 |
| **SCONE 式 f-gram 生成器**(表 = 函数的输出) | ✅ **不需要骨干** | 中 |
| **Memory Grafting 式 teacher 表**(用 4B 隐状态造表) | ✅ 项目已有 4B 模型 | 中 |
| **共训骨干**(Engram 的"有效深度") | ⚠️ 改命题,单卡 4090 成本高 | 高 |

**注意一条重要的负面约束**:Engram 的"有效深度"收益(§6.1/6.2 LogitLens + CKA 证明
浅层表征上移到 MoE 的第 12 层)**结构上要求骨干参与**。冻骨干拿不到这一项 ——
我们实测的 +0.176 是纯内容效应。

---

## 6. 仍未回答的问题(检索未找到)

1. **有没有人在"冻结的小骨干"上做过 SCONE 式 f-gram 生成器?** 未找到。
2. **tokenizer 压缩对 n-gram **表**的覆盖率增益有多大?** Engram 只报了词表缩 23%,
   没报压缩后 n-gram 计数分布的变化 —— 而这正是我们最需要的数字
   (它直接决定有多少单例会变成非单例)。
3. **读出塌陷与门控失效是否同一件事的可测形式?** 未找到直接测量。
4. **在冻结骨干上,多阶/双层/tokenizer 压缩三者的增益能否叠加?** 未找到。

---

## 7. 出处

- Engram: *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs*,
  arXiv [2601.07372](https://ar5iv.labs.arxiv.org/html/2601.07372),
  代码 [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)
- SCONE: *Scaling Embedding Layers in Language Models*, arXiv
  [2502.01637](https://ar5iv.labs.arxiv.org/html/2502.01637)(NeurIPS 2025)
- Memory Grafting: arXiv [2605.20948](https://ar5iv.labs.arxiv.org/html/2605.20948)
- Ordo-M: [alphaxiv 2608](https://www.alphaxiv.org/abs/2608.ordo-m-sparse-memory-frozen-model)
- Product-Key Memory: arXiv [1907.05242](https://www.alphaxiv.org/abs/1907.05242)
- XMemTransfer: arXiv [2608.17050](https://huggingface.co/papers/2608.17050)
- PSGD: [lixilinx/psgd_torch](https://deepwiki.com/lixilinx/psgd_torch/2.3-mathematical-foundations)

---

# 第二轮(16 轮检索):换掉框架之后才看得见的东西

第一轮我一直在找"更好的 PLE 变体"。**第一性原理下我们的装置不是 PLE,而是:**

```text
冻结 LM + 外部存储 + 在某个地址上把存储内容注入残差流
```

一旦这样说,该读的文献就变成**半参数/非参数记忆的整个分支**,而不是 PLE 的亲戚。
换框之后,最重要的东西出现了。

## 8. kNN-LM:冻结模型 + 零训练,+2.9 困惑度

Khandelwal et al., *Generalization through Memorization*,
[arXiv 1911.00172](https://ar5iv.labs.arxiv.org/html/1911.00172)

```text
datastore:(f(c_i), w_i)   f = 末层 FFN 输入(过 layernorm)← 整个前缀的表示
推理:p(y|x) = λ·p_kNN(y|x) + (1−λ)·p_LM(y|x)      ← 在【logit 层】插值
```

| | 困惑度 |
|---|---|
| 基线 LM(Baevski & Auli) | 18.65 |
| **+ kNN-LM(零训练)** | **16.12** |
| + kNN-LM + continuous cache | **15.79** |

**换算成 nats**:`ln(18.65) − ln(16.12)` = **+0.146 nats**;含 continuous cache = **+0.166 nats**。

> **⇒ 我们实测的 +0.17593 nats 与 kNN-LM 的 +0.146 nats 是同一量级。**
> **我们的数字不小。小的是我们把增益锁进了一个 3-token 窗口。**

其他关键读数:

- **"retrieving nearest neighbors from the corpus outperforms training on it"**:
  在 100M 上训练 + 从 3B 建 datastore → **13.73**,胜过**在 3B 上训练**(15.17)。
- datastore 从 512K 涨到 3B,**单调改善且未饱和**;且 **最优 λ 随 datastore 变大而上升**
  (模型越依赖非参数部分)。
- **帮助的正是长尾**:"particularly helpful in predicting rare patterns, such as factual
  knowledge"、人名、近重复句。
- 键的选择很讲究:**末层 FFN 输入、过 layernorm** 最好(17.96 → 16.06)。
- `k=1024`,`λ=0.25`(域内)/ `0.65`(域适应)。

### 8.1 为什么这不违反我们的界 —— 而这正是重点

| | 键是什么 | 受我们的界约束? |
|---|---|---|
| **我们的嫁接** | ≤3 个 token 的哈希 ⇒ `I(Y;e\|h_t) ≤ I(Y;w_t\|h_t)` | ✅ 受 |
| **kNN-LM** | 末层 FFN 输入 = **整个前缀**经注意力的函数 | ❌ **不受** |

论文自己的结论已经写了:*"Two mechanisms escape the bound, and only two:
**retrieval, which addresses by query rather than by a fixed window**, and parametric adapters."*

**⇒ kNN-LM 就是那条"检索逃逸路线",它在冻结模型上零训练即可工作,
而且它擅长的地方恰好是我们实测增益所在的地方(高 surprisal 长尾)。**

## 9. 尾部的"第三种修法":我们只找到过两种

| 修法 | 做什么 | 出处 |
|---|---|---|
| 表 = **函数的输出**(相似 n-gram 共享参数) | SCONE | [2502.01637](https://ar5iv.labs.arxiv.org/html/2502.01637) |
| 表 = **teacher 的隐状态** | Memory Grafting | [2605.20948](https://ar5iv.labs.arxiv.org/html/2605.20948) |
| **表 = 按频率聚类的低秩分解**(+ adaptive softmax) | **Adaptive Input Representations** | Baevski & Auli 2019,[1809.10853](https://ar5iv.labs.arxiv.org/html/1809.10853) |
| **按频率缩放每个参数的学习率** | **Frequency-Aware SGD**(**有可证收益**) | ICLR 2022,[链接](https://mlanthology.org/iclr/2022/li2022iclr-frequencyaware/) |

**注意**:kNN-LM 的基线模型(Baevski & Auli)本身就是**罕见词嵌入的经典解法** ——
按频率把词表切成簇,每簇一个低秩投影。**这是 2019 年就有的标准答案,而我们在 2026 年重新发现了问题。**

**Frequency-Aware SGD** 更直接:**它把我上一轮从第一性原理推出来的"步长应随证据收缩"
做成了有定理的算法。** 我们的"行选择"是它的一个粗糙特例。

## 10. "把它按住"这件事,文献里出现了**三次**

| 出处 | 说法 |
|---|---|
| **Ordo-M** | 记忆中会 shouting over 基座;**parameter-free positional gate 把 text damage 降 96–97%** |
| **Engram §6.2** | 组件消融中**回退最大**的三个之一就是上下文感知门控 |
| **TRAMS** | *Training-free Memory Selection for Long-range Language Modeling*([2023.findings-emnlp.331](https://aclanthology.org/2023.findings-emnlp.331/)) —— **免训练的记忆选择** |

**加上我们自己的实测:46.3% 的位置受害、top 1% 承载 65.6%。**
**四个独立来源说同一件事:读出不按住,记忆是净噪声源。**

## 11. 被我们完全漏掉的一整类:从上下文复制

| 工作 | 机制 |
|---|---|
| **Pointer Sentinel Mixture Models**([1609.07843](https://ar5iv.labs.arxiv.org/html/1609.07843)) | 指针网络 + 混合,专治**罕见词** |
| **Continuous cache**(Grave et al. 2017) | 从**测试文档内部**检索,与 kNN-LM **可叠加** |
| **infini-gram**([2401.17377](https://arxiv.org/html/2401.17377v1)) | 后缀数组上的**无界 n-gram**,5T token,无训练 |
| **RETRO** / **GPT vs RETRO**([EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1081/)) | 检索 + PEFT 的交叉 |

**infini-gram 对我们特别相关**:它证明了**不做任何训练**、纯计数、无界阶数,
就能和神经 LM 插值并改善。**它是"表"这一侧的极限形态。**

## 12. 判据:我们缺的那一块,有现成的

**LongTail-Swap**(Algayres et al., EMNLP 2025 Findings,
[aclanthology](https://aclanthology.org/2025.findings-emnlp.601/)):

- **只测分布的尾部** —— 模型用极少曝光学会新词的能力,像婴儿一样
- 形式:**可接受 / 不可接受句对**,零样本,取两句平均 log 概率
- 已有 10M / 100M 词 BabyLM 两个版本,评了 16 个模型
- 结论:**LM 在罕见词上表现很差;而且架构差异在长尾上比在头部显著得多**
- **代码公开,可以为任意英文语料生成** ⇒ **我们可以为自己的语料生成一份**

> 这是"能力判据"的直接候选,而且**它的成功判据正好落在我们测到增益的位置上**。

其他判据/严谨性来源:
- **When Not to Trust Language Models**([ACL 2023](https://aclanthology.org/2023.acl-long.546/)):
  参数记忆 vs 非参数记忆的边界
- **Quantifying Variance in Evaluation Benchmarks** / [evalstats](https://github.com/ianarawjo/evalstats) /
  seed-variance reporting:小样本评测的统计功效

## 13. "冻结基座 + 小预算"的其它成熟路线(不是记忆)

| 路线 | 代表 | 为什么值得看 |
|---|---|---|
| **激活引导** | ITI / control vectors / [Householder 伪旋转](https://ar5iv.labs.arxiv.org/html/2409.10053) | 冻结模型上的推理期干预;**方向-幅度视角**与我们的塌陷直接相关 |
| **侧网络** | Side-tuning / [Symbiotic Tuning](https://ieeexplore.ieee.org/document/11227100) | 冻结骨干 + 小侧网络,是我们嫁接的替代架构 |
| **权重空间编辑** | task arithmetic / [ROME / MEMIT](https://levelup.gitconnected.com/rome-vs-memit-the-evolution-of-mass-editing-transformer-memory-e3e4af2ca206) | 不改前向的"记忆" |
| **数据质量** | Phi / "Textbooks Are All You Need" | **小模型的提升常常来自数据而非架构** |
| **同策略蒸馏** | [CADENCE](https://huggingface.co/papers/2607.16955) / OPSD | 用 teacher 换小模型能力,与我们已有的 4B 契合 |
| **子词正则** | BPE dropout / [2605.13436](https://arxiv-org.ezproxy.obspm.fr/html/2605.13436v1) | 直接改尾部表征 |
| **联想记忆理论** | Modern Hopfield / [fast weights](https://ar5iv.labs.arxiv.org/html/2510.27258) | 给"注入一个向量"提供理论框架 |
| **检索缩放律** | [log-form retrieval law, 2604.00715](https://huggingface.co/buckets/huggingchat/papers-content/tree/2604/2604.00715.md) | 检索收益的**幂律/对数律拟合** |

## 14. 换框之后的结论

```text
① 我们的数字(0.176 nats)与 kNN-LM(0.146 nats)同量级 —— 不是小效应
② 差别在【地址】:我们锁在 3-token 窗口里,kNN-LM 用整个前缀
③ 论文自己说检索是两条逃逸路线之一 —— 而它零训练
④ 尾部问题文献里有四种修法,我们一种没试
⑤ "按住读出"四个独立来源,我们实测 46.3% 受害
⑥ 判据有现成的:LongTail-Swap,可为任意语料生成
```

**⇒ 论文 B 的问题因此变了。** 原命题"纯 PLE 嫁接 + 少量训练 → 提升通用性能"
在**冻结骨干 + 3-token 寻址**下,结构上被自己的界限制住了。
而**同一套装置换成检索式寻址**(仍然冻结模型、仍然零/极少训练),
文献说它能把 18.65 做到 15.79,并且**检索胜过训练**。

**这不是放弃 PLE,而是把它放回它真正的位置:一个 token 键的先验通道,
与一个查询键的检索通道互补 —— 而后者才是能力增量的来源。**
