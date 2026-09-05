# Round 94：P0 多源 Router 消融——base / RAG / PLE / MoRA / all（3-seed）

> 日期：2026-09-05  
> 状态：P0 完成（含 3-seed 多源消融）  
> 结论：RAG 对知识任务有明确正提升，MoRA 对 code-output 有明确正提升；当前 PLE 在多源消融中尚未带来正收益，算术任务上为负。该结果作为可审计证据保留。

---

## 1. 实验设置

- 脚本：`scripts/run_multisource_ablation.py`
- 模型：`Qwen3.5-0.8B`
- Adapter：`outputs/cap1-mora-160`（CAP-1 MoRA-160）
- RAG：BM25，top-3，语料 `data/sources/wikitext.jsonl`
- PLE：可寻址 n-gram 外部记忆，语料 `data/cap1-rag-distill-160.jsonl`
- PLE 融合：`TaskConditionedNgramLogitProcessor` + `configs/ngram-fusion-router.json`
  - `knowledge → semantic`（关闭 PLE）
  - `arithmetic → number`（允许 PLE）
  - `code-output → code`（允许 PLE）
- 种子：0 / 1 / 2
- 每 seed 评测：20 knowledge + 10 arithmetic + 10 code-output
- 指标：answer-token 平均 log-probability（教师强制），first-token hit

---

## 2. 聚合结果（3-seed 均值）

### Mean answer logprob（越低越好）

| Source | Knowledge | Arithmetic | Code-output |
|---|---:|---:|---:|
| base | -7.569 | -6.801 | -14.250 |
| +RAG | **-6.376** | -6.801 | -14.250 |
| +PLE | -7.569 | -7.272 | -14.250 |
| +MoRA | -7.756 | -6.869 | **-13.317** |
| +RAG+MoRA | -7.022 | -6.869 | **-13.317** |
| +PLE+MoRA | -7.756 | -7.340 | **-13.317** |
| +all | -7.022 | -7.340 | **-13.317** |

### Delta vs base（正值 = 提升）

| Source | Knowledge | Arithmetic | Code-output |
|---|---:|---:|---:|
| +RAG | **+1.193** | 0.000 | 0.000 |
| +PLE | 0.000 | -0.471 | 0.000 |
| +MoRA | -0.187 | -0.068 | **+0.933** |
| +RAG+MoRA | +0.547 | -0.068 | **+0.933** |
| +PLE+MoRA | -0.187 | -0.539 | **+0.933** |
| +all | +0.547 | -0.539 | **+0.933** |

---

## 3. 解读

1. **RAG 是知识任务的最有效来源**  
   - Knowledge：+1.19 nats，稳定正提升；
   - 对算术/code 无影响（当前 RAG 只在 knowledge 上检索）。

2. **MoRA 是 code-output 的最有效来源**  
   - Code-output：+0.93 nats；
   - 与之前 CAP-1 多任务结果一致；
   - Knowledge/arithmetic 略有下降，体现单一 adapter 的 tradeoff。

3. **当前 PLE 没有正收益**  
   - Knowledge 为 0（任务 router 主动关闭，符合定位）；
   - Code-output 为 0（当前 n-gram 记忆没有命中这些简单表达式）；
   - Arithmetic 为 **-0.47 nats**（负收益）。

4. **+all 的收益来自 RAG + MoRA，而非 PLE**  
   - all 与 +RAG+MoRA 在 knowledge/code 上几乎一致；
   - arithmetic 上 all 与 +PLE+MoRA 几乎一致，说明 PLE 在算术上是净负担。

---

## 4. 对 P0/PLE 的判定

### 已完成

- [x] 任务分类 Router：`TaskClassifier / TaskRouter`
- [x] Log-density gate：`LogDensityRatioGate`
- [x] 任务条件生成处理器：`TaskConditionedNgramLogitProcessor`
- [x] 校准参数持久化：`configs/ngram-fusion-router.json`
- [x] Serving 自动加载：`RAGServingAdapter`
- [x] 多源消融：base / +RAG / +PLE / +MoRA / +all
- [x] 3-seed

### 证据门禁

> PLE 在真实多源消融中目前没有产生正贡献，并且在 arith 上为负。
> 因此不能把 PLE 当成“已经可用的外部记忆增益”写进产品；
> 应继续做 PLE 的窄口径改进，或至少在算术任务上关闭 PLE。

---

## 5. 下一步（P1 前置）

1. 改进 PLE 记忆：
   - 用**代码/数字专用语料**或更匹配的 n-gram bank；
   - 尝试更高阶 n-gram、top-k 约束、更严格 density gate；
2. 把 PLE 的 `task_scale` 对 arithmetic 暂时调低/关掉；
3. 在真实 HumanEval/MBPP/代码补全任务上单独评测 PLE，而不是只依赖合成 arithmetic/code；
4. 扩大 PLE 真实校准样本，替换当前规则阈值；
5. 跑正式评测集与 Purified OPSD。
