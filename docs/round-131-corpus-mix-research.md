# Round 131：语料配比多轮调研——别人怎么选训练数据

> 日期：2026-09-08
> 问题：1M PLE reader/target-side training 中，“loss 降但任务不涨”，是否是训练语料来源/配比不合适？
> 别人是怎么做的？

---

## 1. 核心结论

调研后最关键的发现：

> **XMemTransfer 不是靠复杂的多域混合比例，而是靠“源记忆/目标 reader 与下游任务领域匹配”的语料选择。**

他们很少把 chat / tool / CoT 按固定比例混在一起，而是：

```text
单一高质量语料
：
- WikiText
- FineWeb-Edu
- Nemotron-CC HQ-DQA（STEM Q&A）
- Nemotron-CC HQ（有机 web）
```

---

## 2. XMemTransfer 的语料设计

### 2.1 三个主要语料

| 语料 | 用途 |
|---|---|
| WikiText | 简单/同 tokenizer 的 source memory 与基础适配实验 |
| FineWeb-Edu | 下游 FW-matched 实验 |
| Nemotron-CC HQ-DQA | 8B STEM Q&A，domain-aligned 实验 |
| Nemotron-CC HQ | 26B 有机 web，corpus-control 实验 |

### 2.2 关键实验：corpus control

XMemTransfer 有一个专门脚本：

```text
run/22_corpus_control.sh
```

它比较：

```text
HQ 26B（large organic web）
vs
HQ-DQA 8B（small STEM Q&A）
```

目的：

```text
判断收益来自“语料规模”
还是“任务领域结构匹配”。
```

这正说明：

> 对外部记忆/reader 而言，**领域匹配可能比单纯语料大小更重要。**

### 2.3 Reader 训练语料与下游匹配

在 `run/20_downstream_fw_matched.sh` 中：

```text
source memory：FineWeb-Edu
target adaptor：FineWeb-Edu
downstream：RTE / BoolQ / OpenBookQA / SciQ / TruthfulQA / RACE
```

在 `run/21_domain_alignment.sh` 中：

```text
source memory：Nemotron-CC HQ-DQA
target adaptor：Nemotron-CC HQ-DQA
downstream：同样知识任务
```

也就是说：

```text
他们用“与评测任务同域的语料”训练 reader。
```

---

## 3. Smol Training Playbook 的启示

Hugging Face 的 SmolLM3 训练手册也强调：

```text
数据混合不是固定不变的；
可以随训练阶段调整。
```

常见做法是：

```text
第一阶段：通用高质量 web/text
中间阶段：加入 code / math / reasoning
后训练阶段：加入 instruction / chat / CoT
```

也就是说：

```text
“把 chat / CoT / tool 和 general text 一次性按比例混合”
不一定是最优做法。
```

更稳的路线是：

```text
按阶段或按任务匹配。
```

---

## 4. Data Mixing Laws 的启示

近年有专门研究：

```text
Scaling Laws for Optimal Data Mixtures
```

核心思想：

```text
最优混合比例取决于：
- 目标任务分布
- 训练算力
- 数据来源质量
- 是否做多阶段训练
```

所以：

```text
没有一个万能配比；
必须针对目标评测任务做消融。
```

---

## 5. 我们自己 M1–M5 的问题

我们已有 M1–M5 配比：

```text
M1: general 50 / chat 20 / wiki 20 / cot 6 / tool 4
M2: general 40 / chat 30 / wiki 20 / cot 6 / tool 4
M3: general 30 / chat 40 / wiki 20 / cot 6 / tool 4
M4: general 30 / chat 30 / wiki 30 / cot 6 / tool 4
M5: general 20 / chat 40 / wiki 20 / cot 10 / tool 10
```

问题：

```text
1. “general”来源是杂乱网页文本；
2. chat 比例过高，可能让模型学会 instruct 格式但削弱 n-gram/local 信号；
3. CoT/tool 比例太低，无法形成任务格式；
4. 缺乏纯 WikiText / FineWeb-Edu 对照；
5. 缺乏纯 STEM QA / 纯代码对照；
6. 没有做“领域匹配 vs 规模”的 corpus-control。
```

---

## 6. 建议的新语料矩阵

### 6.1 一组“单一纯净语料”对照

```text
W  : WikiText
FW : FineWeb-Edu / 高质量教育 web
CODE : Python/代码语料
STEM : Nemotron-CC HQ-DQA 类 STEM Q&A
GENERAL : 高质量 general web
```

### 6.2 一组“任务匹配”对照

```text
M-W ：纯 WikiText（XMemTransfer 基础口径）
M-FW ：纯 FineWeb-Edu
M-STEM ：纯 STEM Q&A
M-CODE ：纯代码
M-GEN ：纯高质量 general
```

### 6.3 一组“小规模混合”

```text
M-FW+CODE ：FineWeb-Edu 70% + Code 30%
M-FW+STEM ：FineWeb-Edu 60% + STEM 40%
M-FW+CODE+STEM+CHAT ：模仿实际应用，但 chat 比例低
```

---

## 7. 评估口径也要改

语料选择之外，还要改评测：

```text
1. 生成长度：至少 64–128 tokens；
2. 答案抽取：从生成文本中提取答案，而不是前 16 token 严格 EM；
3. 归一化：
   - 数字
   - 同义词
   - 冠词
   - 大小写
4. 同时报：
   - PPL
   - QA EM
   - 代码 pass@1 / pass@k
   - 数学 acc
```

否则会重复出现：

```text
real 其实更“会解释”，但被严格 EM 误判为失败。
```

---

## 8. 建议的 1M 三线矩阵

```text
条件：
- real PLE
- control PLE
- no-reader

语料：
- WikiText
- FineWeb-Edu
- STEM Q&A
- Code
- FineWeb-Edu + Code
- FineWeb-Edu + STEM

种子：
- 3 seeds

规模：
- 1M tokens（和已有 M1 可比）
```

选语料标准：

```text
在 3 seeds 下，
real 稳定 > control，
并且至少一个任务上 real > no-reader。
```

---

## 9. 一句话

> 别人（XMemTransfer）的做法不是“复杂多域混合”，而是：
> **source memory / target reader / downstream 三者领域匹配。**
>
> 我们 M1–M5 最大的问题可能是：
> **把 chat/tool/cot 和杂乱 general 混在一起，反而丢失了与目标任务匹配的纯净信号。**
>
> 下一步应先做：
> **WikiText / FineWeb-Edu / STEM / Code 的单一语料三线对照，再考虑混合。**
