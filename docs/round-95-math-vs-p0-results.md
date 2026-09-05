# Round 95：数学推导与 P0 多源实测的差异、原因与修正

> 日期：2026-09-05  
> 状态：分析完成  
> 结论：P0 结果并不否定“PLE/n-gram 在局部词法任务上有增量信息”，但否定了“当前这种融合方式在这些评测任务上已经可用”。主要差异来自任务口径、记忆域、校准和 gate 的实现，而非数学本身错误。

---

## 1. 数学推导原本预期什么

来自 `docs/round-89-multiround-research-math.md` 和 `docs/round-70-most-effective-path-math.md` 的核心预期：

1. **信息上界**：记忆是否值得用，先看  
   \[
   I(Y;M|H)
   \]
   大于 0。PLE/n-gram 在 code/name/number 上应较大，在语义知识上很小。

2. **最优 logit 修正**：  
   \[
   \Delta^*=\log\frac{p^*(y|h,m)}{p_\theta(y|h)}
   \]
   因此 logit 层融合是正确通道。

3. **判别条件**：真实任务分布下，  
   \[
   \mathbb E_{p_t}[\log p_m(Y|H)-\log p_b(Y|H)]>0
   \]
   时，记忆能降低 log-loss。

4. **源选择**：Blackwell 序决定任务级源选择：  
   - 语义/knowledge → RAG/Dense；  
   - 低熵/code/name/number → PLE/n-gram 可能更充分。

5. **校准**：需要 scale + bias + temperature，且最优参数要在 held-out 上估。

---

## 2. P0 实测结果

从 `docs/round-94-p0-multisource-ablation.md`：

| Source | Knowledge | Arithmetic | Code-output |
|---|---:|---:|---:|
| +RAG | **+1.193** | 0.000 | 0.000 |
| +PLE | 0.000 | -0.471 | 0.000 |
| +MoRA | -0.187 | -0.068 | **+0.933** |
| +all | +0.547 | -0.539 | **+0.933** |

即：

- RAG 在知识上正；
- MoRA 在 code-output 上正；
- **PLE 没有正贡献**，arithmetic 上甚至为负；
- all 的收益基本来自 RAG + MoRA。

---

## 3. 主要差异

| 数学/前期结论 | P0 实测 | 差异类型 |
|---|---|---|
| PLE 在 code/name/number 上 real>control，有增量信息 | P0 的 code-output 无提升，arithmetic 负 | 任务口径不同 |
| 只要 \(E[\log(p_m/p_b)]>0\) 就应激活 | 实际激活后没有正收益 | gate 没有正确估计该量 |
| 校准后的 logit 融合应能提升局部任务 | 当前融合反而伤害 arithmetic | 校准域不匹配 + 非最优修正 |
| PLE 作为低成本局部专家应进入多源 router | PLE 在 all 中无净贡献 | 当前 memory bank 与任务不匹配 |
| 多源凸融合应能取各源之长 | all 只保留 RAG+MoRA，PLE 未加分 | 信息序正确，但 PLE 未提供任务相关信息 |

---

## 4. 为什么会有这些差异

### 4.1 评测任务不是 PLE 的强项任务

PLE-1/PLE-2 的强证据来自：

- 同一代码/文档域内训练 n-gram；
- 预测 **下一个 token**；
- 评测的是局部 continuation / 精确 n-gram 寻址。

P0 多源消融的代码任务是：

> “What does `len([1,2,3])` evaluate to?”

这是 **语义/计算题**，不是代码补全。  
算术任务是：

> “What is 23 + 45?”

这是 **计算题**，不是数字格式/日期/低熵词法任务。

因此：

- PLE 的 \(I(Y;M|H)\) 在这些任务上可能本来就很小；
- 不能把 code-output Q&A 当作 PLE 最擅长的 code continuation。

### 4.2 记忆 bank 与评测任务不匹配

P0 中 PLE memory 来自：

```text
data/cap1-rag-distill-160.jsonl
```

这是数学/代码 **解题文本**，不是：

- 被评测代码本身的语料；
- Python 源码树；
- 数字/日期格式语料；
- 函数/变量名领域语料。

所以当前 n-gram 对 `len([1,2,3])`、`1 + 2 * 3` 这类表达式几乎无有效命中，等于在相关 token 上添加了无关先验。

### 4.3 我们没有真正测量论文中的判别条件

数学条件要求：

\[
\mathbb E_{p_t}[\log p_m(Y|H)-\log p_b(Y|H)]>0
\]

这是一个 **在有标签 true target 下** 才能估计的量。

我们实现的门控用了运行时代理：

\[
KL(p_m\|p_b)=E_{p_m}[\log p_m-\log p_b]
\]

这个量 **非负**，只要 p_m 与 p_b 不同就会通过，不表示 p_m 更接近真实分布。

因此 `expected_kl` 模式在阈值 0 时几乎总是放行，无法阻止有害的 n-gram 先验。

### 4.4 校准参数来自错误的小样本域

当前 `configs/ngram-fusion-router.json` 的：

```json
scale = 1.0
bias = -1.0
temperature = 0.5
```

来自 `outputs/fusion-calibration.json`，而该文件是在：

- 只有 4 个样本；
- wiki 域；
- 32 token 上下文；

上校准的。

没有在：

- arithmetic；
- code-output；
- 真实代码补全；
- 更大样本；

上分别校准。  
所以把 wiki 上的参数直接用于 arithmetic，可能产生负向偏置。

### 4.5 任务分类把“计算题”错误当成“数字局部任务”

当前 `TaskClassifier` 将：

- `arithmetic` → `number`
- 从而在 PLE 任务集合中放行。

但“23 + 45 等于多少”不是低熵词法任务，而是需要推理/计算的任务。  
数学中的低熵 gate 指向的是：

- 代码 token 补全；
- 专名拼写；
- 日期/数字格式；
- 高频局部模板。

不是所有含数字的问题都应启用 PLE。

### 4.6 当前 logit 修正不是最优条件对数似然比

即使某处 \(I(Y;M|H)>0\)，也只有：

\[
\log\frac{p^*(y|h,m)}{p_\theta(y|h)}
\]

是最优修正。

我们实际用的是：

\[
\alpha \log p_m + \beta
\]

这是对最优修正的一个很粗糙的近似，且没有使用检索到的外部 value，也没有学习条件于 \(H\) 的 router。

所以：

- “信息上界为正”不等于“我们的简单融合一定能提升”；
- 这与 round-55 的结论一致：**CMI 是必要不充分，还需可实现通道**。

---

## 5. 需要修正什么

### 5.1 修正评测口径

- 单独构建 **真实代码补全** 集：HumanEval / MBPP / Python 源码 next-token；
- 单独构建 **专名/数字格式** 集；
- 不要把“代码题问答”和“算术计算题”当作 PLE 主战场。
- 对 PLE 使用：
  - next-token logprob / perplexity；
  - real vs control paired 差；
  - 而不是仅看 answer-token logprob。

### 5.2 修正记忆 bank

- PLE memory 必须与评测同域：
  - 代码：用源码文件、函数体、代码片段；
  - 专名：用实体/人名/地名语料；
  - 数字：用日期、号码、格式文本；
- 或者使用检索到的外部 value 进行注入，而不只是 continuation prior。

### 5.3 修正 gate

- 在**有标签 held-out** 上直接估计：
  \[
  \Delta=\mathbb E[\log p_m(Y)-\log p_b(Y)]
  \]
- 只有 real 显著大于 control，且 \(\Delta>0\)，才允许该任务启用 PLE；
- 运行时 gate 可以使用：
  - base entropy 低；
  - memory top1 advantage > 0；
  - pseudo-label ratio > 0；
  - 不能用“KL 非负”作为通过条件。

### 5.4 修正任务 taxonomy

- 把 `arithmetic` 从 `ple_tasks` 中移出，或单独设 `arithmetic_ple_enabled=false`；
- 新增：
  - `code_continuation`
  - `name_entity`
  - `number_format`
  - `low_entropy_local`
- 只有这些任务才允许 PLE。

### 5.5 修正校准

- 按域分别校准：
  - code / name / number / general；
- 扩大样本；
- 保存为 per-task calibration；
- 每个任务单独估计 \(\lambda^*\)（round-70 定理 5）：
  \[
  \lambda^*=
  \frac{\mathrm{Cov}(L_t-L_b,\;L_n-L_b)}
  {\mathrm{Var}(L_n-L_b)}
  \]
- 如果 \(\lambda^*\le0\)，直接关闭该源的 PLE。

### 5.6 修正融合形式

- 从“固定 scale+bias”升级为：
  - 任务条件 scale；
  - 检索 value 注入；
  - 可学习或轻量级 log-density router；
- 在多源消融中增加：
  - PLE real vs PLE control；
  - 无 gate vs 有 gate；
  - 每个任务的 \(\lambda^*\) 正负判断。

---

## 6. 一句话结论

> 数学没有错，错的是我们把“PLE 在局部词法任务上有信息”直接外推成“当前记忆 bank + 当前 gate + 当前评测任务上就能端到端提升”。  
> 需要的修正不是放弃 PLE，而是：**换到真正的 PLE 强项任务、用同域 bank、测量真实 log-density ratio、按任务关闭不适用域、按任务重新校准。**
