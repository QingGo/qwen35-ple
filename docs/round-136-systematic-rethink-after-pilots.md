# Round 136：Pilot 后的系统性复盘（纯 PLE 嫁接）

> 日期：2026-09-08
> 背景：Phase 1 工具链完成，Phase 2 六语料 3 步 pilot 与 unseen-KB pilot 都给出 PPL 正向信号，
> 但任务级/多 seed/大训练规模仍未验证。本轮重新收敛问题、技术债、计划与借鉴矩阵。

---

## 1. 终极目标（不变）

> **纯 PLE 嫁接**：用 Qwen3.8-Flash-Next 的冻结 PLE 表，通过少量训练把记忆读取能力嫁接到
> Qwen3.5-0.8B，并在通用任务上带来显著、稳定、可归因于“真实 PLE 内容”的提升。

关键验收问题：

```text
Q1: 真实 PLE 层注入是否可行？
Q2: 只训练 reader/记忆层是否能把 PLE 信息转化为通用任务能力？
Q3: 在训练未见过的 KB 上，real 是否仍稳定 > control？
Q4: 1M -> 5M -> 20M 是否形成规模曲线？
Q5: 如果纯 PLE 失败，是否愿意记录严格负结果，而不靠 RAG/蒸馏救场？
```

---

## 2. 本轮 Session 发现

### 2.1 已补齐

- 6 个 1M 匹配语料（Wiki / FineWeb / STEM / Code / FW+Code / FW+STEM）；
- KB/QA split + 双向污染审计；
- 答案抽取/归一化/contains/extracted EM 评测；
- 六语料 3 步、1 seed、无 QA pilot：
  - **所有语料 real PPL < control PPL < no-reader PPL**；
- unseen-KB pilot（seen 训练 / unseen 评测）：
  - **real 43.296 < control 43.732 < no-reader 44.842**。

### 2.2 仍不确定 / 尚未回答

- 3 步 pilot 只证明“语言建模”有差异，不能证明通用任务提升；
- 只有 1 个 seed，无法判断稳定性；
- 没有 QA / Code / Math 生成式结果；
- 没有 5M / 20M；
- 没有 layer2 / layer8 / reader 变体对照；
- unseen-KB 只用了 Wiki 一个 KB split，且无 QA。

---

## 3. 技术债

### 3.1 科学债（按优先级）

| 优先级 | 债 | 为什么关键 |
|---|---|---|
| P0 | 正式 1M 全量矩阵（500 步 × 3 seeds × 150 QA） | 目前只有 3 步 pilot，不足以进下一阶段 |
| P0 | 任务级评测（QA/Code/Math + extraction） | PPL 正向不代表通用能力；这是历史最大教训 |
| P0 | 多 seed 稳定性 | 单 seed 无法排除随机 |
| P1 | 领域选择标准 | 不能只看 PPL，要建立 real-control 与 real-no-reader 的种子级阈值 |
| P1 | unseen-KB 正式化 | 只跑了一个 split，需要多种 KB + QA |
| P1 | 5M / 20M 规模 | 决定是否值得进入“positive scaling”叙事 |
| P2 | layer2 vs layer8 | 当前一层结论可能只对 layer8 成立 |
| P2 | reader 变体 | 当前只有 official reader |
| P2 | 污染审计门禁 | 目前是报告，还不是硬失败条件 |

### 3.2 工程债

- 本地 CPU 可跑 pilot，但正式 500 步 × 6 语料 × 3 seeds × QA 成本高；
- 尚未在 4090/WSL 统一跑正式矩阵；
- 尚未把 `outputs/` 正式结果作为 artifact 发布；
- unseen-KB runner 暂时无 QA 长生成；
- Store-P/slot index 若能用，可显著加速，但尚未接入本次 pilot；
- 缺少“自动选择语料/自动门禁”脚本。

### 3.3 资产债

- 已有：6 语料、KB split、token 流、reader checkpoint、pilot JSON、评测脚本；
- 缺：正式 500 步结果、3-seed 结果、QA/Code/Math 结果、5M/20M 数据、最终论文图表。

---

## 4. 后续开发计划

### Phase 2A：正式 1M 全量（建议 4090）

```text
6 语料 × 3 seeds × 500 步
real / control / no-reader
150 QA（96 token 生成）+ PPL
```

产出：

```text
outputs/phase2/<corpus>.json
docs/round-1xx-phase2-full-results.md
```

### Phase 2B：选语料与门禁

预注册判定：

```text
若 real 在 ≥5/6 语料、≥2/3 seeds 上满足：
  real-PPL < control-PPL
并且至少一个通用任务 real > no-reader
则进入 5M。
否则先诊断 layer/reader/corpus，不盲目扩规模。
```

### Phase 3：unseen-KB 正式化

```text
多 KB split（Wiki / FW / STEM / Code）
每 split：
  seen 训练 reader
  unseen 评测 real / control / no-reader
  加入 QA，而非只 PPL
```

### Phase 4：5M / 20M

```text
用选出的 1–2 个语料：
  5M × 3 seeds × 500 步
  若继续正 → 20M
  若不再上升 → 记录饱和点
```

### Phase 5：负结果或发布

```text
若任务级持续不达：
  先试 layer2 / deeper reader / minimal target reader
  若仍不达 → 严格负结果报告
  不主动把 LoRA/蒸馏作为“核心创新”包装
```

---

## 5. 借鉴矩阵（更新版）

| 项目 | 借鉴什么 | 不拿什么 | 与本项目如何不冲突 |
|---|---|---|---|
| XMemTransfer | matched corpus、corpus-control、target reader、1M→5M→20M | 不照搬模型/数据规模 | 用于语料选择和实验分层 |
| Memory Grafting | 条件记忆 + 预训练扩展 | 不做大型预训练 | 只作为概念框架 |
| DeepSeek Engram / Qwen PLE | 表架构、gate、磁盘 offload | 不重训 PLE 表 | 表是资产，不是研究对象 |
| Smol Training Playbook | 分阶段数据混合 | 不照搬 11T 语料 | 仅在 scale-up 阶段参考 |
| Data Mixing Laws | 任务匹配的最优混合比例 | 不追求全局最优公式 | 等有任务级信号后再用 |
| kNN-LM / NGM | 失败模式、基线 | 不让它们替代核心 | 用作对照与解释工具 |
| RAG / BM25 | 作为“外部检索”上界 | 不作为核心路线 | 只做 baseline |
| PEFT / LoRA / MoRA | 仅作为失败后的最小手段 | 不把 adapter 当核心 | 不在纯 PLE 路径中提前使用 |
| OPD / Purified OPSD | 蒸馏/后训练手段 | 不把蒸馏当核心 | 仅作后备 |
| TMLR / RepoEval | 可复现、评测卡、污染审计 | 不迁就格式 | 用于论文和 artifact 组织 |
| EngramDB / engram-peft 契约 | 仓库分工、稳定 ABI、测试 | 不让实验仓承担核心库职责 | 保持工程边界 |

---

## 6. 核心纪律（避免漂移）

1. **核心唯一**：纯 PLE 嫁接。
2. **每次实验**带 real / control / no-reader。
3. **数据 split**：KB.train 与 KB.eval、QA.train 与 QA.eval 严格不共享。
4. **评测先于结论**：用长生成、答案抽取、normalized EM，不只 PPL。
5. **规模有门禁**：任务级通过才上 5M/20M。
6. **失败有预案**：先在 layer/reader/corpus 内诊断，再考虑 LoRA/蒸馏。
7. **所有结果可复现**：命令、manifest、hash、审计、checkpoint 都保留。

---

## 7. 一句话

> Pilot 给我们的不是“成功”，而是“值得继续投入”的证据。
> 下一步的关键不是再跑更多 3 步 pilot，而是：
> **在 4090 上把正式 1M 全量矩阵和任务级评测补齐，用多 seed 判断 real 是否真的稳定优于 control 和 no-reader。**
