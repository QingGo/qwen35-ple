# Round 68：把 PLE 用起来的新思路——训练无关的 n-gram 词法记忆

> 日期：2026-09-04
> 状态：提出新定位并实现原型
> 核心观点：PLE 不应继续作为“学习式语义记忆”，而应作为一个 **训练无关的稀疏 n-gram 词法记忆 / 局部低熵先验**。

---

## 1. 为什么还值得做 PLE

用户的价值判断：

- 创新：PLE/Engram 是外部冻结 n-gram 记忆，本身有独特架构价值；
- 影响范围：可以作为“小模型 + 外部稀疏记忆”这一方向的开源参考；
- 效果：当前在 rare QA 上不足，但需要换任务、换用法。

因此我们不应放弃 PLE，而应：

> 把 PLE 从“失败的语义知识记忆”重新定义为“成功的稀疏词法记忆”。

---

## 2. 之前为什么效果差

之前测试的用法：

- 把 PLE e_t 作为特征；
- 学习 MLP/logit head；
- 看 rare QA real>control。

结果：

- PLE e_t 的 task-level 信息极低；
- 连 logit-space 也放大不了。

但这不是 n-gram 记忆的唯一用法。

---

## 3. 新用法：作为训练无关的 n-gram LM

### 3.1 核心思想

不学习 e_t 的映射，而是直接从同一个 token 语料里统计：

\[
P(\text{next} \mid \text{last } n \text{ tokens})
\]

- 2-gram / 3-gram / 4-gram 精确匹配；
- 未命中时 backoff；
- 与 base model 在 logit 层融合：

\[
\ell_{\text{final}}=\ell_{\text{base}}+\lambda \log P_{\text{ngram}}
\]

这本质是一个：

> **可审计、可版本化、训练无关的稀疏 n-gram 记忆模块。**

### 3.2 为什么可能有效

- 它正是 PLE 擅长的事情：局部低熵 token、代码、名字、格式；
- 不像 e_t 需要学习投影；
- 可以直接和 RAG 形成：
  - RAG = 语义/文档级知识；
  - N-gram memory = 词法/局部先验；
  - Base model = 推理/格式。

### 3.3 已实现原型

```text
src/qwen35_ple/ngram_lm.py
```

功能：

- `NgramLM.from_tokens`
- `distribution(context)`
- `logprob(token, context)`
- `topk(context)`
- `interpolate_logits`

测试：

```text
tests/test_ngram_lm.py —— 3 passed
```

---

## 4. 如何证明这个新用法有效

需要新的 real/control 门禁，但“real”改为：

| 条件 | 含义 |
|---|---|
| real | 用真实语料统计的 n-gram 记忆 |
| control | 用打乱 token 顺序的语料统计的 n-gram 记忆 |
| no-memory | 只用 base model |

关键原因是：这样能分离“n-gram 顺序信息”与“边际 token 分布”。

### 4.1 新评测任务

| 任务 | 为什么适合 n-gram |
|---|---|
| 低熵 token 预测 | 前文高度决定 next |
| 代码补全 | 括号/关键字/API 顺序 |
| 专名/实体拼写 | 名字的后缀强烈依赖前文 |
| 数字/日期 | 格式模式 |
| 长尾短语 | 4-gram 精确命中 |

### 4.2 预期指标

- real vs control 的 next-token logprob；
- top-1 accuracy；
- RAG + n-gram vs 单独 base/RAG。

---

## 5. 与 RAG 的融合架构

```text
输入
 ├─ RAG 文档（语义知识）
 │
 ▼
Base 0.8B
 ├─ base logits
 └─ NgramLM log-prior（local lexical）
      │
      ▼
log-linear fusion / router
      │
      ▼
输出
```

如果 NgramLM 在低熵任务上 real>control，它就成为一个有独立创新价值的组件。

---

## 6. 下一步

1. 用 `data/ple-books-160k/tokens.npy` 训练 NgramLM；
2. 建立低熵/代码/专名评测集；
3. 跑 real / control / no-memory；
4. 再测试 RAG + NgramLM 组合；
5. 如果通过，把 NgramLM 集成到 RAG serving adapter。

---

## 7. 相关参考

- [NGM: Plug-and-Play Training-Free Memory Module for LLMs](https://huggingface.co/papers/2605.16893)
- [DeepSeek Engram](https://github.com/deepseek-ai/Engram)
- [flash-next-8gb](https://github.com/lna-lab/flash-next-8gb)
