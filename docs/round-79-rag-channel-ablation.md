# Round 79：真实 RAG 三通道检索消融

> 日期：2026-09-05  
> 状态：完成  
> 结论：BM25 在“原文段落检索”上最强；Dense 在“QA 答案包含检索”上最强；N-gram/PLE 对语义 QA 无效，但对局部词法检索是独立通道。

---

## 1. 实验设置

- 新增 `scripts/run_rag_channel_ablation.py`
- 语料：200 篇 Wiki 文档
- 评测 1：文档检索
  - 每篇文档用首句作为 query，目标为该文档自身
  - 200 queries
- 评测 2：QA answer-containment
  - 使用 `data/qa-expanded-150.json`
  - 只有答案字符串实际出现在索引语料中的 34 题被保留
  - 指标：top-k 是否包含至少一个含有答案的文档
- 通道：
  - BM25
  - Dense（Qwen token embedding mean-pool）
  - N-gram 精确寻址（PLE）
  - Hybrid（BM25 weight=4, Dense=1, Ngram=1, RRF）

---

## 2. 文档检索结果

| Channel | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| bm25 | 0.9850 | 0.9950 | 0.9950 | 0.9892 |
| dense | 0.8100 | 0.8750 | 0.9000 | 0.8463 |
| ngram | 0.3000 | 0.3550 | 0.3650 | 0.3289 |
| hybrid | 0.8150 | 0.9850 | 0.9900 | 0.8938 |

解读：

- BM25 因为 query 是文档首句，词面重叠极高，因此接近满分；
- N-gram 也能单独召回部分文档，但远弱于 BM25/Dense；
- 加权 Hybrid 在 Recall@3/5 接近 BM25，Recall@1 仍被 BM25 的强 lex 信号拉低。

---

## 3. QA answer-containment 检索结果

| Channel | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| bm25 | 0.2941 | 0.3529 | 0.4706 | 0.3451 |
| dense | 0.3235 | 0.5588 | 0.5882 | 0.4387 |
| ngram | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | 0.2941 | 0.4412 | 0.5294 | 0.3770 |

解读：

- **Dense 在 QA 答案包含检索上最好**，MRR 0.439；
- **N-gram/PLE 在语义 QA 上完全无效**（0%）——这再次验证 PLE 不是语义知识检索器；
- Hybrid 介于 BM25 与 Dense 之间，说明需要按任务 router，而不是固定融合。

---

## 4. 对路线的影响

| 任务类型 | 应使用通道 |
|---|---|
| 原文/词面/代码/专名局部检索 | BM25 + N-gram/PLE |
| 语义知识问答 | Dense（或后续更强 dense/rerank）首推 |
| 混合系统 | 按任务置信度做 router，而不是固定 RRF |

### 这支持 PLE-2 的定位

- PLE/n-gram 不应作为通用语义检索；
- 它应作为：
  - 代码/低熵/专名的局部寻址通道；
  - 与 Dense/BM25 互补的非参数外部记忆；
  - 只有通过长尾 gate 或任务 router 才激活。

---

## 5. 下一步

1. 用校准后的 n-gram logit fusion 做代码/专名任务，而不是语义 QA；
2. 实现任务条件 router：
   - 语义/常识 → Dense/RAG；
   - 局部/低熵/代码 → N-gram/PLE；
3. 继续提升 dense embedding（当前只是 token mean-pool，不是 sota）；
4. 跑真实 QA 生成的 RAG 效果。
