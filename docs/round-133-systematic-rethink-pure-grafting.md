# Round 133：系统性复盘（纯 PLE 嫁接版）

> 日期：2026-09-08
> 背景：上一轮复盘把 RAG/蒸馏放到核心，被纠正后，本轮以“纯 PLE 嫁接”为唯一核心重新规划。

---

## 1. 终极目标

**唯一核心目标：**

> 把 Qwen3.8-Flash-Next 的 PLE 记忆表，通过纯 PLE 层注入 + 少量训练，显著提升 Qwen3.5-0.8B 的通用性能。

分解为可验收问题：

```text
Q1：真实 PLE 层注入 0.8B 是否可行？
Q2：只训练记忆层/reader 是否能把 PLE 信息转化为通用能力？
Q3：在训练时未见过的知识库上，real 是否仍显著优于 control？
Q4：5M/20M 后，收益是否随数据规模上升？
Q5：如果 A1 不达标，最小的非核心手段是什么？（LoRA/蒸馏，仅作为手段）
```

RAG 不是核心；蒸馏、adapter 都只是“如果必要”的后备手段。

---

## 2. 本轮 Session 发现了什么

### 2.1 已确认

- 1M 纯 reader（真实 48GB PLE 表）已经做过；
- PPL：real < control < no-reader，正；
- QA：no-reader > control > real 或接近，未形成净收益；
- 1M 只跑了 M1，M2–M5 未系统跑三线；
- 训练语料很可能是关键变量；
- 评测协议（16 token 严格 EM）可能严重惩罚 real 的“解释式输出”。

### 2.2 已调研

- XMemTransfer 使用“领域匹配”语料，而非复杂多域混合；
- 他们明确做 corpus control：large organic web vs small task-matched STEM；
- Smol 训练手册建议分阶段数据混合；
- Data mixing laws 指出最优混合依赖目标任务和算力；
- 记忆能力评测应区分“参数化知识”和“外部记忆读取能力”。

---

## 3. 当前技术债

### 3.1 科学债

| 债 | 说明 |
|---|---|
| 语料选择未系统完成 | 只有 M1，未跑纯 WikiText/FW/STEM/Code |
| 评测协议可能失真 | 16 token EM 惩罚解释式回答 |
| 未做 unseen KB 记忆实验 | 无法证明“读取任意外部记忆” |
| 未做 5M/20M | 1M 不足以判断 |
| 未做 layer2 vs layer8 | 1M 用的是 layer8 |
| 未做 reader 变体 | 只有 official reader + 2层MLP |
| 未做严格 QA split + 污染审计 | 需要知识库/QA 双 split |
| 未做规模曲线 | 不知道 5M/20M 是否继续上升 |

### 3.2 工程债

| 债 | 说明 |
|---|---|
| 1070 跑大规模较慢 | 需要 4090 云租用做关键轮 |
| M1–M5 批量脚本未完整跑 | 只有 M1 |
| 新语料（FineWeb-Edu/STEM）未接入 | 需要数据工程 |
| 评测 answer extraction 未完善 | 需要实现抽取/归一化 |
| contamination audit 未与 KB split 联动 | 需要一体化 |

### 3.3 资产债

- 有真实 48GB PLE 表；
- 有 1M reader 基线；
- 有 M1–M5 语料；
- 缺少：
  - FineWeb-Edu 子集；
  - FineWeb-Edu 类数据 token 量；
  - Nemotron-CC HQ-DQA 类 STEM；
  - unseen KB 构建脚本；
  - 标准 QA split 与审计报告。

---

## 4. 开发计划

### Phase 1：语料与评测协议先定（1 周）

1. 构建语料矩阵：
   - WikiText
   - FineWeb-Edu
   - STEM Q&A
   - Code
   - FineWeb+Code
   - FineWeb+STEM
2. 构建知识库 split：
   - KB.train → PLE/memory
   - KB.eval → 评测知识
   - QA.train / QA.eval
3. 完善评测：
   - 生成长度 64–128
   - answer extraction
   - 数字/同义词/冠词归一化
   - 代码 pass@k
4. 污染审计：双向 + KB split。

### Phase 2：1M 三线语料矩阵（1 周，1070 可跑）

```text
每个语料：
real / control / no-reader
3 seeds
1M tokens
```

选出：

```text
real 稳定 > control
且至少一个通用任务 real > no-reader
```

的语料。

### Phase 3：5M 关键验证（1 周，建议 4090）

用选出的语料，跑：

```text
5M tokens
3 seeds
real / control / no-reader
PPL + QA + Code + Math
unseen KB
```

### Phase 4：20M 或负结果收口（1–2 周）

- 如果 5M 正：跑 20M，验证规模曲线；
- 如果 5M 负：先试：
  - layer2 vs layer8；
  - deeper reader；
  - minimal target reader；
  - 如果仍负 → 记录严格负结果，不硬上 LoRA/蒸馏。

### Phase 5：发布

- 论文：纯 PLE 嫁接边界 / 规模曲线；
- artifact：语料、split、审计、reader、结果；
- 入 HF / GitHub。

---

## 5. 可从类似项目借鉴什么（不冲突）

| 项目 | 借鉴什么 | 不拿什么 |
|---|---|---|
| XMemTransfer | matched corpus、corpus-control、target reader、5M/20M | 不照搬模型 |
| Memory Grafting | 条件记忆 + 预训练扩展 | 不追求大预训练 |
| DeepSeek Engram / Qwen PLE | 表架构、gate、磁盘 offload | 不重训大表 |
| Smol Training Playbook | 分阶段数据混合 | 不照搬 11T 语料 |
| Data Mixing Laws | 任务匹配的混合比例 | 不追求全局最优公式 |
| kNN-LM / NGM | 失败模式、基线 | 不让它们替代核心 |
| PEFT（LoRA/MoRA） | 仅作为A1失败后的最小手段 | 不把 adapter 当核心 |
| OPD / Purified OPSD | 仅作为A1失败后的蒸馏手段 | 不把蒸馏当核心 |
| TMLR / RepoEval | 可复现、评测卡、污染审计 | 不迁就格式 |
| 我们自己的 P0 沉淀 | 跨仓库分工、契约、测试 | 不让实验仓承担核心库职责 |

---

## 6. 核心战略

1. **核心唯一**：纯 PLE 嫁接；
2. **先定语料和评测**，再谈规模；
3. **先做 unseen KB**，再谈通用能力；
4. **先做 1M 矩阵**，再上 5M/20M；
5. **失败后才用 LoRA/蒸馏**，不作为第一步；
6. **每次实验都带 real/control/no-reader**。

---

## 7. 一句话结论

> 我们离目标只差：
> 1. 选对语料（领域匹配）
> 2. 修对评测（不再被 16 token EM 误杀）
> 3. 做 unseen KB 记忆实验
> 4. 跑 5M/20M 规模曲线
>
> 如果这四件事做完，real 仍不能 > control 且无法提升通用任务，
> 那才算是“纯 PLE 嫁接”的严格负结果。
