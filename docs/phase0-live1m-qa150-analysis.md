# 1M 150 题 QA 三线结果与 Bad Case 分析

> 日期：2026-09-02（WSL）
> 数据：`data/wet-1m-first.npy` 1M tokens
> Reader：`official_source_qwen_v1`，layer 8，500 steps，lr 1e-4
> 评测：`assets/qa-expanded-150.json`，150 题，greedy exact-match，max_new_tokens=16
> 产物：`outputs/reader-real-seed0.pt`、`outputs/reader-control-seed0.pt`、三线 QA JSON

---

## 1. 总览

| 线 | QA EM | TriviaQA | NQ | BoolQ | PPL |
|---|---:|---:|---:|---:|---:|
| no-reader | **53.3%** | 70.0% | 0.0% | **90.0%** | 19.88 |
| real（PLE） | 42.0% | 66.0% | 0.0% | 60.0% | 16.27 |
| control（shuffled PLE） | 30.7% | 62.0% | 2.0% | 28.0% | — |

核心结论：

- **real > control**：说明 PLE 内容本身有信号，不是纯噪声。
- **real < no-reader**：当前 1M + 当前 reader 在 150 QA 上总体没有跑赢无 PLE 基线。
- 差距主要来自 **BoolQ**：no-reader 90%，real 只剩 60%，说明 PLE reader 对“基于 passage 的简单判断”产生了明显干扰。
- NQ 三线都接近 0，说明当前 reader 对开放域 NQ 没有明显帮助。

---

## 2. 新做对 / 新做错（基于逐题对比）

### 2.1 real vs no-reader

| 类别 | 数量 |
|---|---:|
| real 新做对（no-reader 错，real 对） | 9 |
| real 新做错（no-reader 对，real 错） | 26 |

新做对 9 题：

- TriviaQA 5 题：
  - Isaac Newton（牛顿运动定律）
  - William Shakespeare（《哈姆雷特》作者）
  - Saturn（有光环的行星）
  - Rome（意大利首都）
  - Atlantic（美洲与欧洲之间的大洋）
- BoolQ 4 题：均为 passage 内可推理的是/否题。

新做错 26 题：

- 大量是 no-reader 本来能对的 BoolQ，real 反而答错或生成格式异常。
- 也有 TriviaQA 基础题被干扰，例如：
  - Brazil 的官方语言
  - 谁发现青霉素
  - 袋鼠所在国家
  - 最快陆地动物
  - 《奥德赛》作者
  - 六边形边数
  - 1492 年发现美洲者

### 2.2 real vs control

| 类别 | 数量 |
|---|---:|
| real 新做对（control 错，real 对） | 26 |
| real 新做错（control 对，real 错） | 9 |

其中 real 相对 control 的新做对大量集中在 BoolQ（22 题），说明 **真实 PLE e_t 确实能提供比随机排列更有用的语义信息**。

---

## 3. 语料重叠检查

语料：`data/wet-1m-one.txt`（约 1M token，实际是中文/混合短文本流）。

### 3.1 real 相对 no-reader 新做对的 9 题

| 题 | answer | 是否在语料中 |
|---|---|---|
| Newton | Isaac Newton | 否 |
| Hamlet | William Shakespeare | 否 |
| Saturn | Saturn | 否 |
| Rome | Rome | 是＊ |
| Atlantic | Atlantic | 是＊ |
| BoolQ 4 题 | yes/no | 多为常见词，参考意义有限 |

> ＊“Rome / Atlantic” 这类词可能只是文本中偶然出现的英文单词，不一定代表该知识条目已出现在语料中。
> 更严格的方法是检查“问题 + 答案”连续片段或该事实句是否出现。

### 3.2 判断

- 3 个最典型的 trivia 新做对（Newton、Shakespeare、Saturn）**没有出现在 1M 训练语料中**。
- 这符合用户希望的信号：**不是简单记忆语料中的答案，而是 PLE 表提供了外部语义知识，reader 学会了如何把这种知识嫁接到 Qwen3.5 上。**
- 但当前净效果仍被 BoolQ/基础题干扰掩盖，所以还不能作为最终正收益结论。

---

## 4. Bad Case 归类

### 4.1 最严重的 bad case：BoolQ 退化

- no-reader BoolQ EM = 90%
- real BoolQ EM = 60%
- control BoolQ EM = 28%

说明：
- 基座模型本身已经很擅长从 passage 中做简单是非判断；
- PLE reader 注入后，可能改变了 hidden state 的分布/回答格式，导致很多原本正确的 BoolQ 被带偏；
- 这比“记忆新知识”更值得关注，说明当前 reader 的门控/注入强度或训练目标还不适合这类任务。

### 4.2 少量真实知识新做对

这些是当前方法最 positive 的信号：

- Newton
- Shakespeare
- Saturn

它们不在 1M 语料中，却能被 real 做对而 no-reader 做错，说明 PLE 表确实存在可迁移的知识信号。

### 4.3 NQ 全部无效

- no-reader / real / control 的 NQ EM 都接近 0 或极低。
- 当前 reader 对开放域、答案不是短实体/常见词的任务没有帮助。

---

## 5. no-reader 的 good case / bad case 与原因

### 5.1 no-reader 相对 real 的 good case

| 比较 | 数量 |
|---|---:|
| no-reader 对，real 错 | **26** |
| no-reader 错，real 对 | 9 |

no-reader 的 26 个 good case 分布：

- BoolQ：19 题
- TriviaQA：7 题

典型 good case：

- Brazil 的官方语言：no-reader 直接给 **Portuguese**；real 生成长句，16 token 内没出现答案。
- 谁发现青霉素：no-reader 直接列出 **Alexander Fleming**；real 进入“思考/解释”模式。
- 袋鼠在哪个国家：no-reader 直接给 **Australia**；real 开始拆解问题。
- 最快陆地动物：no-reader 给 **cheetah**；real 给成了 **Giant Panda**（真错）。
- 《奥德赛》作者：no-reader 给 **Homer**；real 只说“古希腊文学”，没给作者。
- 六边形边数：no-reader 写 **six**；real 写 **6**，归一化不匹配。
- 大量 BoolQ：no-reader 能从 passage 直接稳定输出 yes/no；real 经常变成“是否适合阅读/是否安全”等异常格式。

### 5.2 no-reader 相对 real 的 bad case

no-reader 错、real 对一共 9 题：

- TriviaQA 5 题：Newton、Shakespeare、Saturn、Rome、Atlantic
- BoolQ 4 题：passage 推理

这些正是 PLE 最正向的信号：real 能在 no-reader 不会/答错的地方给出正确答案。

### 5.3 no-reader 相对 control 的 good case

| 比较 | 数量 |
|---|---:|
| no-reader 对，control 错 | **41** |
| no-reader 错，control 对 | 7 |

no-reader 明显优于 control，说明：

- 基座模型本身已经具备大部分简单 QA 能力；
- control 的“随机记忆注入”会大幅破坏这种能力；
- 真正需要的是 **在不伤害基座能力的前提下加入 PLE 知识**，而不是简单注入。

### 5.4 为什么 no-reader 反而做对更多

从逐题生成文本看，主要原因不是“PLE 知识更少”，而是：

1. **回答格式 / 截断**
   - real 的生成变得更“解释性/思考性”；
   - 答案词经常被推迟到 16 token 之外；
   - exact-match 因此失败，即使模型知识可能是对的。

2. **数字 / 同义词归一化差异**
   - `six` vs `6`；
   - 类似问题导致 no-reader 更容易命中严格 exact-match。

3. **BoolQ 稳定性下降**
   - no-reader 在 passage 简单判断上非常稳定；
   - real 注入后容易跑偏成“判断 passage 是否安全/是否适合阅读”等错误任务格式。

4. **少量真实知识错误**
   - 例如 fastest land animal → Giant Panda；
   - Odyssey 作者只给“古希腊文学”；
   - 这说明当前 reader 在部分实体召回上还不够准。

5. **当前训练量仍不足**
   - real > control 说明 PLE 内容有信号；
   - 但 1M token 下 reader 还没学会“只补知识、不干扰基座”，所以净效果仍被 no-reader 反超。

---

## 6. 下一步建议

1. **优先修 BoolQ 干扰**
   - 降低 reader 注入强度 / 调整 gate 初始化；
   - 或在训练目标中增加“不伤害原有能力”的约束；
   - 或只在特定 layer/任务上启用 PLE。
2. **扩大训练量到 5M–20M**
   - 当前 1M 下 real > control，但 < no-reader；
   - XMemTransfer 经验显示 5M 才开始有竞争力。
3. **用更严格的语料命中检测**
   - 检测“完整事实句”是否在语料中，而不是简单单词匹配；
   - 对 BoolQ 只看 passage 内推理，不看普通 yes/no。
4. **多 seed 复跑**
   - 当前每个 arm 只有 seed 0，需 3 seeds 才能判断稳定性。
