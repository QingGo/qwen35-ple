# Round 26：系统性思考——从“低 loss”回到“语义对齐的证据”

> 日期：2026-09-03
> 范围：M1 结果分析、control 机制、语义对齐分析、技术债、路线图、借鉴矩阵
> 状态：M2–M5 正在后台运行；M1 已完成；下一阶段以“机制分析”为最高优先级

---

## 1. 终极目标（不变）

> **用最小可复现实验证明“Qwen3.5 主干 + 冻结 Flash-Next PLE 记忆表”的嫁接是否成立；**
> 若成立，交付 0.8B 后训练模型 + CPU 100 tok/s 可复现推理闭环；
> 若不成立，留下可审计的负结果。

具体验收轴：

| 轴 | 含义 |
|---|---|
| 科学 | 是否真正发生“语义对齐”，而非“分布拟合/格式改善” |
| 工程 | reader / bundle / serving / 复现资产完整 |
| 产品 | 真实 serving A/B + CPU 100 tok/s |
| 过程 | 每个结论有机制证据，不只靠 val loss 或 QA EM |

---

## 2. 当前坐标

### M1 已完成结果

| 线 | val loss | PPL | QA EM | TriviaQA | NQ | BoolQ |
|---|---:|---:|---:|---:|---:|---:|
| no-reader | 2.4563 | 11.66 | 53.3% | 70% | 0% | 90% |
| real | 2.3949 | 10.97 | 50.7% | 76% | 0% | 76% |
| control | 2.4391 | 11.46 | 52.7% | 84% | 4% | 70% |

核心观察：

1. **val loss：real < control < no-reader**，PLE 对语言建模仍有正信号。
2. **QA EM：no-reader > control > real**，PLE 没有带来任务级净收益。
3. **control 也退化**，说明“注入任何向量 + 训练 reader”本身就会干扰 BoolQ。
4. **control 也有知识型 good case**：
   - control 也能做对 Shakespeare、Newton、Rome、Poseidon；
   - 说明这些不能单独作为“真实 PLE 语义对齐”的证据。
5. **real 独有且不在语料中的增益很弱**：
   - 例如 Leonardo da Vinci；
   - 其余主要是 BoolQ 上的 yes/no 差异，证据力不足。

M2–M5 正在后台运行，但即便跑完，也不能仅凭 QA EM 决定是否放大。

---

## 3. 本轮关键认知升级

### 3.1 val loss 不是“能力”

- 混合语料 val loss 更低，主要是因为验证集也来自同一混合分布；
- 新分布更加规范、英文比例更高、结构更清晰；
- 这只能说明“更容易拟合”，不能说明“模型更强”。

### 3.2 control 不是“原版模型”

- control = Qwen3.5 + 训练后的 reader + 随机打乱的 PLE e_t；
- control 的 good case 来自格式收益 + 基座已有知识；
- control 的 bad case 证明“注入扰动”本身会破坏简单判断题。

### 3.3 需要从“黑盒结果”进入“机制证据”

接下来最重要的不是继续堆 mix，而是回答：

```text
1. reader 参数是否真的发生了变化？
2. reader 输出是否真的携带 PLE 语义？
3. 真实 PLE 和随机 PLE 在表征层的差异是什么？
4. BoolQ 退化发生在哪一层？
5. 哪些 token 的 PLE 真正影响了最终答案？
```

只有这些问题有了答案，才能谈“语义对齐”。

---

## 4. 本轮发现的技术债

| 编号 | 技术债 | 影响 | 处置方向 |
|---|---|---|---|
| V181 | 混合语料 val loss 跨语料不可比 | 容易误判混比有效 | 改用固定外部 LM probe |
| V182 | control 也有知识型 good case | 可能高估 real PLE 信号 | 必须做 real-specific 因果证据 |
| V183 | 没有 reader 参数有效性分析 | 不知道训练是否真的生效 | 记录参数变化/输出 norm/gate |
| V184 | 没有表征层对齐分析 | 无法证明语义对齐 | CKA + probe + activation patch |
| V185 | BoolQ 退化机制未知 | 不知道是门控/层/强度/格式问题 | 做 layer/scale/门控扫描 + logit lens |
| V186 | 固定外部评测缺失 | 无法科学选 mix | 建立 fixed eval suite |
| V187 | 仅单 seed / 单次生成 | 结论不稳定 | 最优 mix 跑 3 seeds |
| V188 | 没有 PLE 检索忠实度分析 | 无法判断记忆是否真的被使用 | 记录每条命中的 n-gram 与语义 |
| V189 | 没有严格区分“记忆新知识”和“基座已有知识” | 会污染 PLE 结论 | real/control/no-reader 三线 + 外部 knowledge probe |
| V190 | 尚无 RL 决策门禁 | 可能过早进入高成本 RL | 设置明确 go/no-go 条件 |

---

## 5. 后续开发计划

### Phase A：机制与可解释性分析（最高优先）

目标：拿到“语义对齐是否存在”的直接证据。

#### A1. Reader 参数有效性

- 记录训练前后 reader 参数 L2 变化；
- 记录 reader 输出 norm；
- 记录 gate 激活值 / ShortConv 输出；
- 对比 random-init reader、真实 reader、control reader。

#### A2. 表征层对齐

- 计算：
  - `CKA(Qwen hidden, PLE e_t)`
  - `CKA(Qwen hidden, reader output)`
- 比较 real / control / no-reader。

#### A3. Activation Patching

- 对关键 token：
  - 将 real e_t 替换为 random / control / 零向量；
  - 观察最终答案是否改变；
  - 如果只有 real 能恢复正确输出，即因果证据。

#### A4. Logit Lens / 答案方向

- 从中间层投影到词表；
- 计算正确答案 token 与错误答案 token 的概率差；
- 比较 real / control / no-reader。

#### A5. BoolQ 专项诊断

- 记录每道 BoolQ：
  - 正确答案；
  - real/control/no-reader 的输出；
  - 是否反转极性；
  - 是否出现重复/格式退化；
  - PLE 命中的 n-gram。

#### A6. 语义对齐证据报告

- 对 150 QA 输出一个“证据表”：
  - real 独有做对且不在语料中的题目；
  - control 也能做对的题目；
  - logit 方向改善的题目；
  - activation patch 能翻转的题目。

### Phase B：固定外部评测与科学选 mix

#### B1. 固定外部 LM Probe

- 用同一个 held-out 语料（如 WikiText test / 固定原始 wet test）测所有 mix 的 no-reader/real/control PPL；
- 不再用各 mix 自己切出的 val。

#### B2. 固定任务评测

- 150 QA + 扩展 BoolQ/Trivia/NQ + 后续 CoT/tool；
- 所有 mix 用同一份题目。

#### B3. Mix 选择判据

- 只看：
  - `real - no-reader`
  - `real - control`
  - 以及“real 独有且不在语料中的题目数量”
- 不看各自语料 val loss。

#### B4. 3 Seeds

- 最优 mix 跑 3 seeds；
- 报告 mean ± std。

### Phase C：训练与对齐修正

在确定有真实 PLE 信号后：

- 增加 BoolQ/QA 格式数据；
- 调整 reader 注入层 / scale / gate；
- 增加“不伤害基座能力”的训练约束；
- 考虑是否进入 5M–20M。

### Phase D：RL 决策门禁

不提前做 RL。

只有当以下条件同时满足：

```text
1. real > no-reader
2. real > control
3. 存在 real 独有且不在语料中的新做对
4. BoolQ 不再显著退化
5. 3 seeds 稳定
```

才考虑小规模 RL，例如：
- DPO / GRPO；
- 目标是“正确调用 PLE 记忆”；
- 而不是简单刷 BoolQ。

### Phase E：产品化

- 真实 vLLM / SGLang serving A/B；
- CPU 100 tok/s 闭环；
- 与 LLM-CompileForge 对接。

---

## 6. 借鉴矩阵

| 来源 | 借什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| **Scaling Law 解构（苏剑林 11833）** | 将 loss 分解为 data/opt/arch 误差；数据混比是独立变量；不要用同分布 val loss 选 mix | 不照搬公式，不改变训练框架 | 帮助我们科学判断“loss 降低”的本质 |
| **XMemTransfer** | target-side reader 训练量级 5M–20M；先验证再放大 | 不照搬模型结构 | 提供训练量级与路径 |
| **Memory Grafting** | 冻结外部记忆 + 轻量 projection/gating；记忆与训练语料解耦 | 不复制记忆提取 | 支撑“语义对齐而非背题” |
| **DeepSeek Engram / engram-peft** | 条件记忆、稀疏检索、ShortConv、gate、训练基础设施 | 不引入第二套存储 | 直接复用现有 PLE 引擎 |
| **EngramDB** | manifest、证据库、位级一致、Store/View/SlotIndex | 不改存储核心 | 作为底层事实源与可重建资产 |
| **RAG / Memory-Augmented LLM** | 检索忠实度、query-memory alignment、passage grounding | 不做外部检索系统 | 用来评估 PLE 是否真的被正确使用 |
| **Mechanistic Interpretability** | CKA、probing、activation patching、logit lens、knowledge neuron | 不替代实验 | 提供语义对齐的底层证据 |
| **Data Mixing / CMR / Aioli / DUET** | 用下游任务反馈优化混比，不用训练损失 | 不盲目套比例 | 指导 mix 选择 |
| **Benchmark Contamination** | held-out、n-gram 审计、provenance | 不因污染而放弃知识评测 | 保证结论可信 |
| **SFT/RL 实践** | SFT 先、RL 后；先修格式和门控，再强化 | 不提前 RL | 避免掩盖机制问题 |
| **vLLM / SGLang / LLM-CompileForge** | serving 薄适配、CPU 性能闭环 | 不复制推理引擎 | 产品化阶段使用 |

---

## 7. 当前纪律

1. **不再用各 mix 自己的 val loss 作为 mix 选择依据。**
2. **不再用“答案不在语料中”单独证明 PLE 有效，必须加 real-specific 因果证据。**
3. **在机制分析完成前，不进入 5M–20M，也不进入 RL。**
4. **所有关键结论至少 3 seeds。**
5. **所有实验保留 manifest、污染审计、命令、日志。**
6. **如果机制分析显示 reader 参数没有真正学习，直接记录负结果，停止放大。**
