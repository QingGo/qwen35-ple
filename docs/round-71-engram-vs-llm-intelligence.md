# Round 71：重新反思：为什么普通 LLM 能产生智能，而嫁接 Engram 不能

> 日期：2026-09-04
> 状态：深层反思 + 数学化
> 核心：Engram/PLE 不是“少了多少智能”，而是它的机制本质上属于“非参数局部记忆”；智能来自参数化组合计算，不能仅靠查表注入。

---

## 1. Engram/PLE 到底是什么

Engram/PLE 的关键机制：

- 对 token 序列做 n-gram 哈希；
- 查到一个固定宽度的 embedding 向量 \(e_t\)；
- 在早期层注入到 hidden；
- 本质是一个：

> **非参数、离散键、固定阶、表面形式记忆。**

它确实能大幅降低语言建模 loss，因为：

- 局部 n-gram 是语言中非常可预测的部分；
- 查表比让网络学习这些规律更直接；
- 所以 loss 下降是合理的。

---

## 2. 普通 LLM 有什么，Engram 嫁接缺了什么

### 2.1 组合性

Transformer 不是查表，而是：

\[
h' = \text{Attention}(h) + \text{MLP}(h)
\]

- 每个 token 的表示由整个上下文动态计算；
- 可以组合任意距离的信息；
- 可以形成“实体-关系-属性”等高阶结构。

Engram \(e_t\) 只是对最近 2/3 个 token 的固定哈希向量：

- 没有 long-range interaction；
- 没有 query-dependent routing；
- 没有“组合出新的未见 n-gram”的能力。

### 2.2 归纳头 / 上下文学习

普通 LLM 中会出现 induction heads：

\[
A,B,\dots,A \to B
\]

它能在**当前上下文**中发现模式并外推。

Engram 只能看“训练语料中见过的 n-gram”。

### 2.3 深度抽象

Transformer 的每一层都在做非线性变换：

\[
\text{token} \to \text{syntax} \to \text{semantics} \to \text{world model} \to \text{reasoning}
\]

Engram 注入的是“一个固定语义平面的查表向量”，不经过同样的深度演化。

### 2.4 压缩压力

普通 LM 必须把海量数据压缩到参数中，因此被迫学习：

- 通用语法规则
- 语义类别
- 推理模板
- 世界知识

而如果有外部 n-gram 表直接给出答案，模型对“学习这些规则”的压力就降低了：

> **查表把最容易预测的部分拿走了，但正是那些“不容易查表”的部分，才逼迫网络形成智能。**

---

## 3. 数学化解释

### 3.1 信息分解

设 \(C\) 为完整上下文，\(E_{\text{ngram}}\) 为 n-gram 记忆能提供的预测信息：

\[
I(Y;C)
=
I(Y;E_{\text{ngram}})
+
I(Y;C\mid E_{\text{ngram}})
\]

- 第一项：n-gram 表能记住的局部统计；
- 第二项：需要**参数化组合计算**才能得到的部分。

普通 LLM 必须同时学习两项。  
嫁接 Engram 后，第一项可以靠外部表解决，但**第二项仍然必须由网络学习**。

### 3.2 为什么 loss 下降不带来智能

\[
\Delta L_{\text{engram}}
\approx
H(Y|C)-H(Y|C,E_{\text{ngram}})
\]

\[
=
I(Y;E_{\text{ngram}}\mid C)
\]

这主要来自**第一项**。

而智能可能更多依赖：

\[
I(Y;C\mid E_{\text{ngram}})
\]

也就是“n-gram 表无法解释、必须靠组合计算才能预测”的部分。

所以：

> **Engram 降低的 loss 可能主要是 local memorization loss；它没有自动增加 compositional intelligence。**

### 3.3 模型类包含关系

n-gram 模型本质是有限阶 Markov 模型：

\[
P(y_t\mid y_{<t})
=
P(y_t\mid y_{t-n+1:t})
\]

Transformer 可以表示：

\[
P(y_t\mid y_{<t})
\]

对任意长上下文、任意高阶组合。

因此：

\[
\text{n-gram models}
\subsetneq
\text{Transformer expressivity}
\]

嫁接 n-gram 不会增加表达力，只是给网络一个“局部 shortcut”。

### 3.4 为什么 word2vec 单 token embedding 能复用

- word2vec embedding 编码了**词级分布语义**；
- 所以可以被下游分类、NER、情感分析复用；
- 但它也是**静态、无上下文**的；
- 后来的 LLM 之所以更强，正是因为不再只用静态 embedding，而是通过 Attention + MLP 把它动态组合。

Engram 的 n-gram embedding 比 word2vec 多了局部上下文，但仍然是：

> **静态键 → 固定向量 → 查表。**

它没有自动获得“上下文动态组合”的能力。

---

## 4. 为什么原论文 Engram 可能有效

原论文不是简单“嫁接后看效果”，而是：

1. 大规模联合预训练；
2. 早期层静态重建卸载；
3. 把注意力/深度容量释放给推理；
4. 模型从头学习如何与 Engram 协作。

所以原论文的收益可能来自：

> **容量重分配 + 联合适应，而不是 Engram 本身承载了智能。**

---

## 5. 如果要让 PLE 成为主创新，应该怎么做

### 5.1 不要把它当“主要预测器”

如果 PLE 用于预测大部分 token，它只是非参数记忆，会把模型训练成“局部复制器”。

应该把它设计成：

> **非参数残差记忆 / 稀疏语义地址 / 长尾知识旁路。**

### 5.2 新架构提案：PLE 作为“可寻址残差记忆”

不是让 PLE 直接提供 \(e_t\) 作为特征，而是让它存储：

\[
r_t = E[Y\mid \text{context}] - E[Y\mid \text{base model}]
\]

也就是“base 模型预测不了的残差”。

推理时：

\[
\hat Y =
P_{\text{base}}(Y|H)
+
\text{router}\times \text{retrieved residual}
\]

这样：

- 不需要 PLE 取代网络；
- 只补足“参数模型没学会的长尾局部规律”；
- 保留网络对通用/组合部分的建模压力；
- PLE 成为主创新而不是备选。

### 5.3 让 PLE 只负责长尾 / 低熵

这是现有 n-gram 最擅长的：

```text
常见/通用/组合任务 → 参数化 LLM
长尾/局部/低熵任务 → PLE/n-gram
```

这样不会让 PLE 变成整体能力的“天花板”。

### 5.4 把 PLE 升级为语义可寻址记忆

现在的 PLE 键是离散 n-gram。  
可以进一步：

- 把 n-gram 作为“地址”；
- 但 value 不是 e_t，而是：
  - 文档片段
  - 段落向量
  - 知识三元组
  - 代码片段
- 这样 PLE 从“词法记忆”变成“可寻址外部知识库”。

### 5.5 联合训练而非冻结注入

如果希望 PLE 成为主创新，应该：

- 在预训练/CPT 阶段就加入 PLE；
- 让 backbone 从一开始学习如何使用它；
- 而不是训练完后从外面插一个 reader。

---

## 6. 可以做的验证实验

| 实验 | 目的 |
|---|---|
| 测量 \(I(Y;C\mid E_{\text{ngram}})\) | 看 PLE 之后还剩多少“必须组合计算”的信息 |
| PLE 只 gate 到低熵/长尾 | 防止 shortcut，看通用能力是否保住 |
| 非参数残差记忆 | 验证“只补残差”是否比“直接注入 e_t”更好 |
| 联合小规模预训练 | 看 PLE 从训练开始参与是否带来不同结果 |
| 语义可寻址 PLE | 把 n-gram 键映射到文档/知识，验证知识能力 |

---

## 7. 结论

PLE 不是“没有价值”，而是被用错了：

> **Engram 的价值不在于“替代 LLM 做智能”，而在于作为一个非参数、可寻址、低成本的外部记忆，去补足参数模型的长尾/局部/低熵部分。**

普通 LLM 产生智能的原因是：

- 注意力
- 深度组合
- 归纳头
- 压缩压力
- 世界模型

这些都不能靠查 n-gram 表获得。

所以要让 PLE 成为主创新，正确方向是：

> **把 PLE 改造成“可寻址残差记忆 / 长尾外部知识库”，与参数化模型形成互补；而不是把它当成一个额外的预测器去替代参数化智能。**
