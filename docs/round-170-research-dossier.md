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
