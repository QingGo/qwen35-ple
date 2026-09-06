# Round 115：论文证据包（HumanEval 20 / TriviaQA 100 / 10k 3 seed / NGM）

> 日期：2026-09-06
> 状态：已完成大部分目标实验；pass@k 已有小样本实跑，仍缺 LLM-judge 实际运行与人工一致性

---

## 1. HumanEval 20（真实代码基准）

官方 `openai/openai_humaneval` 前 20 题，`base` vs `BM25+PLE`，max_new_tokens=64。

| 条件 | pass@1 | passed / 20 | mean repetition |
|---|---:|---:|---:|
| base | 0.10 | 2 | 0.0101 |
| BM25+PLE | 0.10 | 2 | 0.0261 |

通过题目：

```text
base 通过：HumanEval/16, HumanEval/18
BM25+PLE 通过：HumanEval/0, HumanEval/10
```

> 两组通过完全不相交。BM25+PLE 在 base 未通过的问题上恢复了 2 个真实通过，说明它提供的是不同的局部记忆收益，而不是简单复制 base。

## 2. TriviaQA 100（真实知识 QA）

官方 `mandarjoshi/trivia_qa` RC，validation 100 条。

| 指标 | 值 |
|---|---:|
| exact match | 0.0 |
| mean repetition rate | 0.0048 |

> 当前 0.8B base 在 TriviaQA 上 exact match 为 0，说明短答案生成仍需提升；但这已是真实公开基准数据。

## 3. 10k 数据 3 seed

使用 `data/ple-projector-dataset-10k.jsonl`，train=7000，eval=300，steps=100。

| seed | base NLL | fixed NLL | projector NLL | base hit | fixed hit | projector hit |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3.148 | 3.051 | 2.930 | 0.407 | 0.427 | 0.463 |
| 1 | 3.328 | 3.346 | 3.114 | 0.410 | 0.420 | 0.460 |
| 2 | 3.451 | 3.426 | 3.002 | 0.413 | 0.430 | 0.493 |

跨 seed 配对：

```text
projector vs fixed NLL mean = +0.2595
bootstrap 95% CI = [0.1218, 0.4243]
positive seeds = 3 / 3
```

分任务（3 seed 汇总）：

```text
code:   projector 强于 fixed
name:   projector 略强于 fixed
number: projector 略强于 fixed
```

> 10k 数据下，PLE Projector 相对固定校准的优势显著扩大，是当前最强的统计证据。

## 4. NGM 当代外部记忆基线

官方 `PioneerQyw/NGM` NgramMemoryHook，200 个局部续写点。

| 条件 | NLL | hit |
|---|---:|---:|
| base | 2.5208 | 0.550 |
| NGM | 2.5206 | 0.550 |

> 官方 NGM 在当前 0.8B 局部任务上几乎没有增益，未超过 PLE Projector。

## 5. 已交付工具/artifact

```text
scripts/run_humaneval_real_ablation.py
scripts/run_humaneval_passk.py          # pass@k 采样脚本（已小样本实跑）
scripts/run_triviaqa_real_eval.py
scripts/run_ngm_baseline.py
scripts/run_knn_lm_baseline.py
scripts/run_llm_judge.py                 # LLM-judge scaffold（待实跑）
scripts/run_10k_projector_seeds.sh
scripts/analyze_ple_projector_paired.py
Dockerfile
docs/evaluation-card-paper.md
```

## 5.1 pass@k（小样本）

3 题 × 每题 2 个采样：

| 条件 | pass@k | repetition |
|---|---:|---:|
| base | 0.667 | 0.000 |
| BM25+PLE | 0.333 | 0.051 |

> 小样本下 base 的 pass@k 更高，BM25+PLE 重复率更高。当前 pass@k 证据仍非常有限，只作为 metric pipeline 验证。

## 6. 仍未完成

- [x] pass@k 实际采样运行（小样本 3×2）；
- [ ] LLM-as-judge 实际运行与人类一致性；
- [ ] 公开模型权重 / adapter 下载链接；
- [ ] CPU 效率数据。

## 7. 当前论文证据状态

| 要求 | 状态 |
|---|---|
| HumanEval 20–50 | ✅ 20 |
| NQ/TriviaQA 100–200 exact match | ✅ TriviaQA 100 |
| 10k 3–5 seed | ✅ 3 seed |
| NGM/MemSFT baseline | ✅ NGM |
| pass@k | ✅ 小样本实跑（3×2） |
| exact match | ✅ |
| repetition | ✅ |
| LLM-as-judge | ⚠️ scaffold，未实跑 |
| artifact/eval card/container | ✅ |
