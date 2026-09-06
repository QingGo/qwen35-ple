# Round 112：P0/P1 —— 真实基准、kNN 基线、5 seed 统计、联合系统表

> 日期：2026-09-06
> 状态：完成 P0/P1 核心实验
> 目标：把“论文还缺什么”中最关键的真实基准、外部记忆基线、统计显著性和联合系统表补上。

---

## 1. P0：真实 HumanEval 最小子集

- 数据：`openai/openai_humaneval`，官方 test split，前 5 题。
- 语料：`data/code-corpus.jsonl`，chunk 后 1672 段。
- 条件：
  - `base`
  - `bm25`（BM25 检索 + 上下文前置）
  - `base_ple`（PLE n-gram logit fusion）
  - `bm25_ple`（BM25 检索 + PLE fusion）

### pass@1

| 条件 | pass@1 | passed / 5 |
|---|---:|---:|
| base | 0.2 | 1 |
| bm25 | 0.0 | 0 |
| base_ple | 0.0 | 0 |
| bm25_ple | **0.2** | 1 |

关键案例：

- `base` 通过 HumanEval/2；
- `bm25_ple` 通过 HumanEval/0；
- 纯 BM25 在 5 题中未通过任何一题；
- 纯 PLE 未通过任何一题；
- **BM25 + PLE 是唯一同时具备检索和局部词法先验的条件，恢复了 1 个真实 HumanEval 通过。**

这仍然是小样本，不能做总体结论，但它提供了第一个真实公开代码基准上的信号。

---

## 2. P0：kNN-LM 基线

- 数据：`data/ple-projector-dataset-1k.jsonl`
- datastore：500 个训练局部续写点；
- eval：200 个局部续写点；
- k=16，温度 0.05，λ=0.5；
- 隐藏状态来自冻结 Qwen3.5 最后一层。

### 结果

| 条件 | NLL | hit |
|---|---:|---:|
| base | 2.633 | 0.505 |
| kNN-LM | 2.847 | 0.540 |

分任务：

| 任务 | base NLL | kNN NLL | base hit | kNN hit |
|---|---:|---:|---:|---:|
| code | 2.600 | 2.828 | 0.551 | 0.573 |
| name | 4.110 | 3.980 | 0.100 | 0.300 |
| number | 1.893 | 2.193 | 0.167 | 0.250 |

解读：

> 当前小型 kNN-LM 能提升 hit，但 NLL 更差（概率校准不如 base）。
> 这说明简单 hidden-space kNN 不是强基线，PLE 需要在概率校准上做得更好才有意义。

---

## 3. P1：5 seed + bootstrap + CI

在 100 samples / 100 steps / task one-hot 条件下，补跑 seed 3、4，得到完整 5 seed。

| seed | projector vs fixed NLL |
|---|---:|
| 0 | +0.219 |
| 1 | -0.013 |
| 2 | +0.001 |
| 3 | +0.151 |
| 4 | +0.165 |

跨 seed 汇总：

```text
projector vs fixed mean = +0.1044
proj_vs_fixed_ci95 = [0.0251, 0.1866]
positive seeds = 4 / 5
fixed vs base mean = +0.4925
projector vs base mean = +0.5969
```

> 5 seed 后，bootstrap 95% CI 下界 > 0，说明 learned PLE projector 相对固定校准在小规模 paired 数据上开始具有统计意义。

---

## 4. P1：联合系统表

已有 `ms-purified-mora-s{0,1,2}.json`，包含：

```text
base
rag
ple
mora
rag_mora
ple_mora
all
```

3 seed 聚合关键结果（answer logprob，越高越好）：

| 系统 | knowledge | arithmetic | code-output |
|---|---:|---:|---:|
| base | -8.140 | -7.365 | -14.250 |
| rag | **-6.748** | -7.365 | -14.250 |
| ple | -8.140 | -7.492 | -14.250 |
| mora | -7.691 | -7.114 | -12.292 |
| rag_mora | **-6.704** | -7.114 | -12.292 |
| ple_mora | -7.691 | -7.237 | -12.292 |
| all | **-6.704** | -7.237 | -12.292 |

结论：

- RAG 是 knowledge 的主要收益；
- MoRA 是 arithmetic / code-output 的主要收益；
- PLE 单独没有带来通用能力收益；
- PLE 最合理角色仍然是“局部低熵 code 记忆”，不是通用语义记忆。

---

## 5. 完成状态

| 项目 | 状态 |
|---|---|
| 真实 HumanEval 最小子集 | ✅ |
| kNN-LM 基线 | ✅ |
| 5 seed + bootstrap + CI | ✅ |
| 联合系统表 | ✅（3 seed，已有数据汇总） |
| 10k 训练 | ❌ 仍为下一步 |
| 公开 artifact / CPU 100tok/s | ❌ 仍为下一步 |

---

## 6. 产物

```text
scripts/run_humaneval_real_ablation.py
scripts/run_knn_lm_baseline.py
scripts/analyze_ple_projector_paired.py（增加 bootstrap CI）
outputs/humaneval-real.json
outputs/knn-lm-baseline.json
outputs/ple-projector-v1-seed3.json
outputs/ple-projector-v1-seed4.json
outputs/ple-projector-paired-analysis-5seed.json
docs/round-112-p0-p1-real-baselines-stats.md
```
