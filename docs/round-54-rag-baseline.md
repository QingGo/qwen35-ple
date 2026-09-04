# Round 54：RAG 同口径 baseline 与转向决策

> 日期：2026-09-04
> 状态：已完成同口径 RAG baseline，结果显著优于 PLE memory interface
> 结论：外部检索能带来 large real-world 可观测收益；PLE exact bank + TokenMem 在当前冻结 backbone 下不能。

---

## 1. 为什么做这个 baseline

P1 门禁未通过后，按 round-50 停止条件需要比较：

1. PLE memory interface（已完成：rare real−control ≈ 0.0001，不显著）；
2. 外部检索 RAG（本报告）；
3. 后续 OPD / Purified OPSD。

RAG baseline 的目的是建立“能力提升是否必须依赖 PLE”的对照：如果简单外部检索已经显著提升 rare/common 知识问答，而 PLE 记忆接口不能，则说明当前瓶颈不在“有没有 PLE”，而在“小模型能否直接利用外部知识/更强监督”。

## 2. 设置

| 项 | 值 |
|---|---|
| Backbone | Qwen3.5-0.8B（frozen） |
| Retriever | 自研轻量 BM25（无外部依赖） |
| Corpus | `data/sources/wikitext.jsonl`，23,767 docs |
| Top-k | 3 |
| QA | `data/rare-kb-v1.json`，270 题（rare 182 / common 88） |
| Metric | 与 P1 相同：answer-token 平均 logprob |
| 条件 | no-context vs RAG（context + question + answer） |

## 3. 结果

| 分组 | n | no-context logprob | RAG logprob | Δ | wins |
|---|---:|---:|---:|---:|---:|
| all | 270 | -7.186 | -5.938 | **+1.248** | 229/270 (84.8%) |
| rare | 182 | -4.566 | -3.716 | **+0.851** | 152/182 (83.5%) |
| common | 88 | -12.604 | -10.534 | **+2.070** | 77/88 (87.5%) |

## 4. 与 P1 对比

| 方法 | rare Δ (real/control 或 RAG/no-context) | wins | 判定 |
|---|---:|---:|---|
| P1 exact bank + TokenMem + router | +0.000131 | 60/182 | 不显著 |
| RAG BM25 top-3 | +0.851 | 152/182 | 显著且大 |

结论：

- 冻结 0.8B 本身可以利用外部检索上下文显著提高知识问答；
- 当前 PLE 记忆接口的贡献远小于简单 BM25 RAG；
- 继续投入大规模 PLE backbone adaptation 的优先级应下调；
- 更合理的路线是：**RAG/蒸馏为主，PLE 作为可选局部语言先验**。

## 5. 产物

- `scripts/run_rag_baseline.py`
- WSL 输出：
  - `outputs/rag-baseline-200.json`
  - `outputs/rag-baseline-tail.json`
  - `outputs/rag-baseline-100.json`（smoke）

## 6. 下一步

1. 跑 OPD / Purified OPSD 蒸馏（不依赖 PLE）；
2. 将 RAG 与蒸馏 student 做同口径对比；
3. 如果蒸馏 student 能接近或超过 RAG，则进一步验证“小模型能力提升来自更强监督，而不是 PLE 表”；
4. PLE 暂时保留为低优先级实验/可选增强，不进入大规模 RL。
