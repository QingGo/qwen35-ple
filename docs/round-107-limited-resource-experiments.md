# Round 107：有限资源实验执行——Purified OPSD 多 seed 与 PLE 基线消融

> 日期：2026-09-06  
> 状态：已完成第一批低成本高信息实验，含 Purified OPSD 多 seed 与 PLE 基线消融  
> 设备：单张 GTX1070 8GB；远程 WSL

---

## 1. 本轮完成的事情

1. **Purified OPSD MoRA-80 多 seed 训练**
   - Seed 0：已有 `outputs/cap1-purified-mora-80`
   - Seed 1：新增 `outputs/cap1-purified-mora-80-s1`
   - Seed 2：新增 `outputs/cap1-purified-mora-80-s2`
   - 数据：`data/purified-opsd-train.jsonl`（138 条验证保留）
   - 每 seed 80 steps，MoRA type 1，r=8，alpha=16

2. **Per-item 评测与统计**
   - 新增 `scripts/run_cap1_formal_eval.py`，输出 per-item 数据
   - 新增 `scripts/analyze_seed_peritem.py`，计算 paired delta、bootstrap 95% CI、p 值
   - 覆盖：
     - CAP-1 held-out 39 条
     - 正式风格基准：GSM8K-like / MATH-like / HumanEval-like / MBPP-like

3. **PLE 不可替代性基线脚本**
   - 新增 `scripts/run_ple_baseline_ablation.py`
   - 对比：base / BM25 context / n-gram retrieval context / PLE logit fusion / raw n-gram
   - 同域 code / name / number 局部任务
   - 修复第一次运行时 BM25 长文档导致 OOM：检索上下文截断到 `--max-rag-tokens`（默认 256）

4. **Purified OPSD 下 LoRA / QLoRA / MoRA 对照**
   - 新增 `outputs/cap1-purified-lora-80`
   - 新增 `outputs/cap1-purified-qlora-80`
   - 与 MoRA seed 0/1/2 统一在 CAP-1 + 正式风格基准上做 per-item paired 分析

---

## 2. Purified OPSD 多 seed 结果

### 2.1 CAP-1 held-out（39 条，answer logprob，越高越好）

| Run | Mean logprob | First-hit | Δ vs base | 95% CI | p | wins/losses |
|---|---:|---:|---:|---:|---:|---:|
| base | -1.3370 | 0.0000 | — | — | — | — |
| pur-mora-s0 | -1.2522 | 0.0000 | +0.0848 | [0.058, 0.111] | <0.0001 | 34/5 |
| pur-mora-s1 | -1.2378 | 0.0000 | +0.0992 | [0.072, 0.127] | <0.0001 | 35/4 |
| pur-mora-s2 | -1.2400 | 0.0000 | +0.0971 | [0.063, 0.132] | <0.0001 | 34/5 |

**结论**：在 CAP-1 自蒸馏 held-out 上，Purified MoRA-80 三个 seed 均一致正提升，且 paired 检验显著。说明“Purified OPSD 数据被模型吸收了”这个局部结论是稳健的。

### 2.2 正式风格基准（20 条/family，answer logprob）

| Family | base | s0 | s1 | s2 | s0 Δ | s1 Δ | s2 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K-like | -8.781 | -8.492 | -9.213 | -9.007 | +0.289 ** | -0.431 ** | -0.226 ** |
| MATH-like | -4.590 | -4.638 | -5.237 | -5.041 | -0.048 ns | -0.647 ** | -0.450 ** |
| HumanEval-like | -1.560 | -1.464 | -1.600 | -1.543 | +0.096 ** | -0.040 ** | +0.017 ns |
| MBPP-like | -1.497 | -1.373 | -1.559 | -1.436 | +0.124 ** | -0.063 ** | +0.061 ** |

**关键发现**：

- Seed 0 在 GSM8K-like、HumanEval-like、MBPP-like 上正；MATH-like 基本持平。
- Seed 1、2 在 GSM8K-like 和 MATH-like 上显著为负，代码类正负混杂。
- 因此 **Purified OPSD 的能力提升目前只在 CAP-1 held-out 上稳健，尚未在正式风格基准上达到跨 seed 稳定正收益**。
- 这直接说明：不能把 Purified OPSD 单独作为论文的“能力提升”主卖点；需要继续做数据/过滤/训练稳定性改进，或把它作为“局部自蒸馏闭环”证据，而不是通用能力提升。

### 2.3 LoRA / QLoRA / MoRA 对照（Purified OPSD，seed 0）

| Adapter | CAP-1 Δ | GSM8K-like Δ | MATH-like Δ | HumanEval-like Δ | MBPP-like Δ |
|---|---:|---:|---:|---:|---:|
| Purified LoRA-80 | +0.111 ** | -0.497 ** | +0.031 ns | -0.208 ** | -0.031 ** |
| Purified QLoRA-80 | +0.100 ** | -0.810 ** | -0.016 ns | -0.250 ** | -0.059 ** |
| Purified MoRA-80 s0 | +0.085 ** | +0.289 ** | -0.048 ns | +0.096 ** | +0.124 ** |
| Purified MoRA-80 s1 | +0.099 ** | -0.431 ** | -0.647 ** | -0.040 ** | -0.063 ** |
| Purified MoRA-80 s2 | +0.097 ** | -0.226 ** | -0.450 ** | +0.017 ns | +0.061 ** |

**观察**：

- 三种 PEFT 方法在 CAP-1 held-out 上都能一致提升自蒸馏答案 logprob；
- 但在正式风格基准上，只有 MoRA seed 0 出现“3/4 正”，其余多数为负；
- LoRA/QLoRA 在 GSM/HumanEval/MBPP 上均明显回退；
- 这意味着 **当前 Purified OPSD + 80 步小训练不足以带来通用能力正迁移，更像是对特定自蒸馏分布的过拟合/局部适配**。

---

## 3. PLE 不可替代性基线结果

- 脚本：`scripts/run_ple_baseline_ablation.py`
- 配置：seed 0，code/wiki 同域 train/test split，context_len=32，top-k=3，max-rag-tokens=256
- 指标：answer-token mean logprob（越高越好），top-1 hit

### 3.1 Code（代码续写，n=80）

| 条件 | Mean logprob | Top-1 hit | Δ vs base |
|---|---:|---:|---:|
| base | -2.1897 | 0.600 | — |
| BM25 context | -1.9001 | 0.575 | +0.290 |
| n-gram retrieval context | -1.9286 | 0.600 | +0.261 |
| **PLE logit fusion** | **-1.6618** | **0.675** | **+0.528** |

- PLE vs BM25：**+0.238**
- PLE vs n-gram retrieval：**+0.267**
- raw n-gram 单模型 logprob：-4.762（很弱）

**代码任务上 PLE 确实优于两个便宜基线，且 top-1 hit 也最高。**

### 3.2 Name（人名/大写词续写，n=63）

| 条件 | Mean logprob | Top-1 hit | Δ vs base |
|---|---:|---:|---:|
| base | -4.6021 | 0.270 | — |
| BM25 context | **-3.8815** | **0.381** | **+0.721** |
| n-gram retrieval context | -4.5173 | 0.302 | +0.085 |
| PLE logit fusion | -4.5359 | 0.270 | +0.066 |

- PLE vs BM25：**-0.654**
- PLE vs n-gram retrieval：-0.019

**名字任务上 PLE 几乎不优于 base，而 BM25 检索上下文收益巨大；PLE 被 BM25 明显替代。**

### 3.3 Number（数字续写，n=50）

| 条件 | Mean logprob | Top-1 hit | Δ vs base |
|---|---:|---:|---:|
| base | -1.0696 | 0.680 | — |
| BM25 context | **-0.9158** | **0.700** | **+0.154** |
| n-gram retrieval context | -1.4071 | 0.520 | -0.338 |
| PLE logit fusion | -1.0610 | 0.640 | +0.009 |

- PLE vs BM25：**-0.145**
- PLE vs n-gram retrieval：+0.346

**数字任务上 PLE 只比 base 略好，且不如 BM25；但显著优于 n-gram retrieval。**

### 3.4 基线结论

> PLE 不是在所有局部任务上都不可替代。
>
> - **Code**：PLE 是价值来源，强于 BM25 和 n-gram retrieval；
> - **Name**：BM25 明显更有效，PLE 不可替代性不成立；
> - **Number**：PLE 弱于 BM25，但强于 n-gram retrieval。
>
> 结合 Purified OPSD 多 seed 在正式基准上的不稳定性，当前证据不支持把 PLE 作为唯一的、普适的主创新卖点。更合适的定位是：
>
> **“可审计的局部/低熵外部记忆，在特定 code 类任务上有优势，但在通用知识/名字类任务上不能替代传统 RAG。”**

---

## 4. 本轮新增/修改文件

- `scripts/run_cap1_formal_eval.py`：per-item CAP-1 + formal 评测
- `scripts/analyze_seed_peritem.py`：paired delta + bootstrap CI + p 值
- `scripts/run_ple_baseline_ablation.py`：PLE vs BM25 / n-gram retrieval / base
- `scripts/run_purified_seed_evals.sh`：多 seed per-item 评测运行器
- `.github/workflows/ci.yml`：将新增 Python 脚本加入 ruff lint 列表

---

## 5. 下一步判断

1. **PLE 不可替代性只在 code 类任务部分成立**：
   - Code：PLE > BM25 > n-gram retrieval；
   - Name/Number：BM25 更优或至少不差。
   - 因此 **不建议把 PLE 作为普适主创新**。

2. **Purified OPSD 作为能力提升也不够稳健**：
   - CAP-1 held-out 三 seed 一致正；
   - 正式风格基准只有 seed 0 正，seed 1/2 明显回退。
   - 因此不能把它单独写成“通用能力提升”。

3. **建议转向的论文/产品定位**：
   - 主故事：**低资源可审计混合记忆系统**；
   - PLE 作为“code/局部低熵任务的专用通道”；
   - RAG/BM25 作为通用语义/名字任务主通道；
   - Purified OPSD 作为“局部自蒸馏闭环”和部署数据工程证据，而不是通用能力来源。

4. **仍需补齐的低成本实验**：
   - 增加 LoRA / QLoRA 对照（尤其是 Purified OPSD 下）；
   - 在 code 类任务上做 PLE vs BM25 的多 seed + paired 检验；
   - 增加真实公开基准（如 HumanEval/GSM8K 子集，若可获取）；
   - 增加失败案例：name 为什么 BM25 强、number 为什么 PLE 弱；
   - CPU 100 tok/s 与量化部署证据。
