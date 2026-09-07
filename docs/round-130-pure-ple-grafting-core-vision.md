# Round 130：恢复核心愿景——纯 PLE 嫁接小模型

> 日期：2026-09-08
> 核心愿景（不改）：
>
> **把 Qwen3.8-Flash-Next 的 PLE 记忆表，通过纯 PLE 嫁接（不是 RAG、不是蒸馏、不是通用 adapter），
> 用少量训练显著提升 Qwen3.5-0.8B 的通用性能。**
>
> RAG / 蒸馏 / 通用 adapter 都只是“如果必要才用的实现手段”，不是研究核心。

---

## 1. 之前为什么看起来“不可能”

我们之前的结论需要修正：

```text
我们并没有真正测试过“纯 PLE 嫁接”。
```

我们真正测过的是：

- 冻结 0.8B；
- 只训练一个很小的 logit-level PLE Projector；
- 用人工 memory features + hidden → scale/bias；
- 数据量只有 10k；
- 没有把 PLE 当作完整外部条件层插入 0.8B 的较早层；
- 没有让 0.8B 的后续层真正学习利用 PLE 行。

所以：

```text
之前的结果证明的是“窄 logit 通道 + 10k 样本不够”，
不是“纯 PLE 嫁接不可能”。
```

---

## 2. 我们其实有真实 PLE 表资产

远程 WSL 上已经有：

```text
/home/zeng/qwen38-rows
≈ 48 GB 真实 PLE 行表
```

格式：

```text
16 头 × 160 维 FP8
PLE_QWEN_V1
```

同时 engram-peft 已支持：

```text
engine = "qwen_ple"
table_source = "engramdb:store"
install_disk_multi_head_embedding(store)
```

因此：

```text
我们实际上具备做“真表 PLE 层注入”实验的核心资产。
```

---

## 2.1 已有尝试与新计划的关键区别

| 已有尝试 | 是否 A0/A1 | 新计划是否重复 |
|---|---|---|
| CPU/合成小素数 PLE 表，A0/A1 各 10 步 | 不完整 A0/A1 | **不重复** |
| 1M tokens / 真实 qwen38-rows / official reader + 2层MLP / layer 8 / 500 steps / 3 seeds | **已完成 A1 级 1M** | **不重复** |
| logit-level PLE Projector，10k 样本 | 不是 PLE 层注入 | **不重复** |
| hidden-state reader / MLP reader | 窄通道 | **不重复** |
| RAG + BM25 + PLE 混合 | 系统层 | **不做核心** |
| Purified OPSD / MoRA | 能力手段 | 仅作为后备手段 |
| 蒸馏 teacher-text | 能力手段 | 仅作为后备手段 |

已经完成的 1M 纯 reader 实验：

```text
模型：Qwen3.5-0.8B
表：qwen38-rows（真实 48GB）
Reader：OfficialSourceQwenReader + 2 层 MLP bridge/out_proj
注入层：layer 8
训练：1M tokens, 500 steps, 3 seeds
三线：no-reader / real / control
结果：PPL real > control > no-reader
QA 9题：real 51.85% > control 48.15% > no-reader 44.44%
```

所以 1M 已经做过，**不要重复**。

新计划真正未做的：

```text
5M / 20M tokens
+ 更标准的 QA 集（≥50题/任务）
+ 可选：layer 2 vs layer 8 注入位置对比
+ 可选：更多 reader 变体
```

---

## 3. 纯 PLE 嫁接的标准实验设计

### 3.1 对比

```text
A0：Qwen3.5-0.8B 原版
A1：Qwen3.5-0.8B + 真实 PLE 层（第 2 层注入）
    - 只训练记忆层内的 key/value proj + gate + conv
    - 表冻结
    - 不加 RAG
    - 不加通用 LoRA/MoRA
A2：A1 + 最小 target-side reader/proj（如果 A1 需要对齐）
    - 这仍属于“嫁接必需接口”，不是 RAG/蒸馏
A3：A1 + 小 LoRA（仅作实现手段，不是核心）
```

### 3.2 数据

纯 PLE 嫁接不应该用 RAG 数据。

建议语料：

```text
- 通用文本：WikiText / FineWeb 类
- 代码：Python / 代码语料
- 数学：可验证的 CoT 数据（可用于训练，但不是核心）
- 长上下文：长文档
```

训练规模阶梯（因为 1M 已做过）：

```text
5M tokens  → 初步判断（1M 已有基线）
20M tokens → 关键判断
```

### 3.3 评测

必须有：

```text
1. 通用 held-out：
   - 知识
   - 代码
   - 数学
   - 长上下文/记忆利用
2. real / control：
   - real PLE 表
   - shuffled/control 表
3. 基础模型回归：
   - 不能出现严重退化
```

核心 Gate：

```text
如果 A1/A2 在“至少一个通用任务上”显著优于 A0，
并且 real > control，
那么“纯 PLE 嫁接提升 0.8B 通用性能”就成立。
```

---

## 4. 训练成本

### 4.1 1070 估算

只训练记忆层（冻结 backbone，forward 为主）：

```text
1M tokens ≈ 1–3 小时
5M tokens ≈ 5–15 小时
20M tokens ≈ 20–60 小时
```

### 4.2 4090 云租用

```text
20M tokens 可能 1–5 小时
成本约 $1–5
```

如果 1070 太慢，建议：

```text
租 4090 做 5M/20M 关键实验。
```

---

## 5. 为什么“少量训练”仍然可能

- 我们不需要训练 50B 参数；
- 只需要训练：
  - 记忆层适配器（几十 MB 参数）；
  - 或最小 target-side reader/proj；
- 冻结表、冻结 backbone；
- 训练量 1M–20M tokens，而不是 20B tokens。

所以：

```text
“少量”是相对预训练/CPT 而言：
20M tokens 是少量，
10k tokens 才是真正太少。
```

---

## 6. 如果失败，才考虑其他手段

顺序应该是：

```text
第一步：纯 PLE 层注入 + 记忆层/reader 训练
第二步：如果不行，加最小 LoRA（作为对齐手段）
第三步：如果还不行，再考虑蒸馏
第四步：RAG 永远放在系统层，不进入核心研究问题
```

而不是：

```text
一开始就把 RAG 当成核心。
```

---

## 7. 可以借鉴但不冲突的项目

| 项目 | 借鉴什么 |
|---|---|
| XMemTransfer | target-side reader 架构与训练规模 |
| Memory Grafting | 条件记忆表 + 预训练扩展方法 |
| DeepSeek Engram / Qwen PLE | 表架构、缩放律、磁盘 offload |
| PEFT（LoRA/MoRA） | 只作为可能需要的对齐手段 |
| OPD / Purified OPSD | 只在“蒸馏成为必要手段”时使用 |
| kNN-LM / NGM | 作为失败模式对照，不替代核心 |
| RAG / Self-RAG | 只作为系统层扩展，不进核心研究定义 |

---

## 8. 下一步行动

### 第一步（本周，基于已有 1M 结果）

1. 确定 5M 正式协议：
   - 3 seeds；
   - real / control / no-reader；
   - PPL；
   - 标准 QA 集（≥50 题/任务）；
   - 代码/数学/长上下文；
2. 直接跑 5M tokens 的 memory-layer-only 训练；
3. 与已有 1M 结果比较收益曲线。

### 第二步（如果 5M 有信号）

```text
20M tokens
```

### 第三步（如果 5M 仍不够）

```text
比较注入层：
- layer 2
- layer 8
比较 reader 变体：
- official reader + MLP
- 更深的 reader
- minimal target-side reader
```

### 第四步（根据结果决定）

```text
如果正：纯 PLE 嫁接愿景成立，继续深化。
如果负：记录严格负结果，再考虑最小 LoRA/蒸馏作为手段。
```

---

## 9. 结论

> 我们之前把“纯 PLE 嫁接”和“RAG/蒸馏系统”混在一起了。
>
> 现在恢复核心：
>
> **核心问题只有一个：真实 PLE 层注入 0.8B，少量训练，能否显著提升通用性能。**
>
> 1M 纯 reader 已经测过（PPL 正，QA 小样本方向性正）；
> 但 5M/20M 和标准 QA 集还没有真正做过。
>
> 我们有 48GB 真实 PLE 表、已有 1M 基线、engram-peft 支持、0.8B 模型。
> 下一步应直接做 5M 正式矩阵，而不是重复 1M。
