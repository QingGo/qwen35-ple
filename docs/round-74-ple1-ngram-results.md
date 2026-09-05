# Round 74：PLE-1 第一阶段结果——N-gram 词法记忆 real vs control 通过

> 日期：2026-09-05  
> 状态：PLE-1 核心证据已建立  
> 结论：真实有序 n-gram 记忆在代码、专名、数字任务上显著优于打乱顺序的控制组；这支持 PLE 作为“可寻址/局部词法外部记忆”的主创新方向。

---

## 1. 本次完成内容

1. 新增可复现实验脚本：
   - `scripts/run_ple1_ngram_eval.py`
2. 建立两个域的训练/评测划分：
   - **wiki**：`data/sources/wikitext.jsonl`，300 篇文档，80/20 切分；
   - **code**：本仓库 + `engram-peft` 的 Python 源码，120 个文件，80/20 切分。
3. 对每个域训练两个 n-gram 记忆：
   - **real**：保持原始 token 顺序；
   - **control**：在每个文档内部打乱 token 顺序。
   - 二者共享相同的 unigram 边际分布，因此差异只来自“有序 n-gram 结构”。
4. 输出：
   - `outputs/ple1-ngram-eval.json`
   - `outputs/ple1-ngram-eval-report.md`
   - `outputs/ple1-ngram-base-fusion.json`

---

## 2. 主结果（每个域 2000 个评测位置）

| Domain | N | real logprob | control logprob | Δ logprob | real top1 | control top1 | Δ top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wiki | 2000 | -7.329 | -8.742 | +1.413 | 0.245 | 0.042 | +0.203 |
| code | 2000 | -3.417 | -6.813 | +3.397 | 0.500 | 0.045 | +0.455 |

- wiki 的 real 困惑度约 1524，control 约 6263；
- code 的 real 困惑度约 30.5，control 约 910；
- 配对 t 检验：
  - wiki：t≈25.7，p≈0；
  - code：t≈48.1，p≈0。

---

## 3. 分类结果

### 3.1 wiki

| Category | N | Δ logprob | real top1 | control top1 |
|---|---:|---:|---:|---:|
| name | 314 | +1.998 | 0.213 | 0.000 |
| number | 190 | +2.489 | 0.416 | 0.016 |
| other | 1496 | +1.154 | 0.230 | 0.053 |

### 3.2 code

| Category | N | Δ logprob | real top1 | control top1 |
|---|---:|---:|---:|---:|
| name | 113 | +4.265 | 0.451 | 0.000 |
| number | 73 | +3.829 | 0.685 | 0.000 |
| other | 1814 | +3.325 | 0.496 | 0.050 |

结论：

> n-gram 顺序信息对专名拼写、数字/日期格式、代码局部结构都有显著可测增益。
>
> 其中 code 和 number 是最强区域，这与“PLE/n-gram 适合低熵、局部、长尾词法记忆”的定位一致。

---

## 4. Base model logit 融合小样本（wiki，n=8）

使用 Qwen3.5-0.8B 本地模型在 CPU 上做了 8 个位置的 logit 融合探索：

| 指标 | 数值 |
|---|---:|
| base NLL | 2.0470 |
| base+real ngram 最优 NLL | 1.7480 |
| base+control ngram 最优 NLL | 2.0438 |
| Δ bits（base → base+real） | 0.4314 |
| Δ bits（base → base+control） | 0.0046 |
| real 相对 control 的 Δ bits | 0.4268 |
| base+real 最优 λ | -1.95（贴边界，仍需校准） |
| base+control 最优 λ | 6.00（贴边界） |

解读：

- 即使只有 8 个位置，真实 n-gram 带来的 NLL 下降也远大于 control；
- 但最优 λ 为负且贴边界，说明 **直接把 `log P_ngram` 加到 raw base logits 并不是最终可用形式**：
  - n-gram 概率的尺度与模型 logits 不匹配；
  - 需要在 PLE-2 中做学习式 router/温度/比例校准；
  - 这也为“非参数残差记忆 + 多源凸 router”提供了直接动机。

---

## 5. 对 PLE 主创新定位的意义

### 5.1 PLE-1 门禁已通过

| 门禁 | 状态 |
|---|---|
| n-gram/PLE 在低熵/代码/专名任务上 real > control | ✅ 通过 |
| 真实 n-gram 信息超过打乱控制 | ✅ 通过（大 Δ logprob、大 Δ top1、p≈0） |
| 代码/数字/专名是最强区域 | ✅ 与预期一致 |
| 直接 log p 插值可直接用于系统 | ⚠️ 未通过，需要 router 校准 |

### 5.2 新定位得到实证支撑

- PLE 不是“语义知识记忆”；
- PLE 是 **有序 n-gram / 局部低熵 / 可寻址外部词法记忆**；
- 它在代码补全、实体拼写、数字/日期格式等任务上确实具备独立于 unigram 的增量信息。

---

## 6. 局限与未完成

1. 当前 n-gram 记忆是训练无关的纯词法记忆，没有和 PLE 真实表、RAG、teacher 结合；
2. 尚未真正测量 \(I(Y;C\mid E_{\text{ngram}})\)，目前仅有 NLL 下降代理；
3. Base fusion 样本只有 8 个位置，且没有对 code 域跑；
4. λ 校准问题未解决；
5. 尚未做 3-seed 统计；
6. 尚未构造真实 HumanEval / MBPP 代码补全评测集；
7. “name” 分类只是按首字母大写的简单启发式，不是完整 NER。

---

## 7. 下一步

### PLE-2a：Router/温度校准

- 用少量 base logits 学一个缩放/门控：
  \[
  \ell_{\text{fused}} = \ell_{\text{base}} + g(H)\cdot \alpha \cdot \log P_{\text{ngram}}
  \]
- 对比：
  - 固定 \(\lambda\)
  - 可学习 \(\alpha\)
  - 按域/置信度 gate

### PLE-2b：非参数残差记忆

- 将 n-gram 作为 key；
- value 可以是：
  - 原文片段；
  - 检索到的文档；
  - 代码/命名实体片段；
- 形成“可寻址外部知识库”。

### PLE-2c：代码/专名/数字正式评测集

- 从 HumanEval/MBPP 格式生成代码补全；
- 从 Wikidata/DBpedia 抽取实体拼写；
- 从日期/数字语料构造格式预测。

### CAP 并行

- RAG self-distillation；
- QLoRA/MoRA；
- 小规模 n-gram + RAG 多源 router 实验。

---

## 8. 一句话

> 本轮把 PLE 从“失败语义记忆”变成了“被实证支持的局部有序词法记忆”：真实 n-gram 在 code/name/number 上显著优于打乱控制，PLE 可以作为第一类外部记忆进入后续混合系统。
