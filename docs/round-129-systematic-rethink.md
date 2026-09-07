# Round 129：系统性复盘——终极目标、本轮技术债、未来计划与借鉴矩阵

> 日期：2026-09-08
> 状态：Phase A/B 完成、P0 沉淀完成、第一性原理与成本分析完成。
> 本文回答：我们现在到底在追求什么？离目标还差什么？下一步怎么走才更稳？

---

## 1. 终极目标

经历了 Phase A/B 和第一性原理分析后，终极目标应当更诚实、更收敛：

> **用可复现的低资源实验，证明“可审计 n-gram 外部记忆在小模型上的能力边界”，并把它与 RAG、蒸馏、适配器组合成一个真正可部署的低资源系统。**

我们不再声称：

```text
“把 Qwen3.8 PLE 嫁接到 0.8B 就能显著提升通用性能”
```

第一性原理告诉我们：

```text
PLE 表可能有隐式语义，
但要让 0.8B 把它变成可用知识，
需要大规模 target-side reader / 蒸馏 / RAG 式证据通道。
```

所以终极目标分为两个层次：

### 1.1 科学层

证明：

```text
PLE/n-gram 外部记忆 = 可审计的局部低熵记忆通道
```

并明确：

```text
哪些任务能提升
哪些任务不能提升
需要多少训练预算才能改变边界
```

### 1.2 系统层

交付：

```text
0.8B + PLE + RAG + adapter + 验证过滤蒸馏
一体化的低资源可部署系统
```

---

## 2. 本轮 Session 发现了什么

### 2.1 证据已完成

- HumanEval 50；
- TriviaQA 200；
- 10k PLE Projector 5 seeds；
- pass@k 10×3；
- LLM judge 全量；
- sensitivity order / memory size；
- CPU float32 / int8 基线。

### 2.2 关键结论

```text
局部低熵任务：可提升，10k 5 seeds 显著。
通用知识：不可提升，TriviaQA exact = 0.005。
开放生成：必须 token policy。
RAG / MoRA：分别承担知识 / 能力。
```

### 2.3 关键认知修正

- PLE 表不是“没有语义”，而是“隐式语义不能直接被 0.8B 使用”；
- “纯 PLE 嫁接通用提升”基本不可行；
- XMemTransfer 需要 50M source / 20M target-side fitting；
- 我们只有 10k 训练样本，差约 3 个数量级。

### 2.4 已沉淀

- engram-peft：fusion / projector / policy；
- EngramDB：addressable n-gram reference memory；
- LLM-CompileForge：external memory fusion contract。

---

## 3. 当前技术债

### 3.1 科学债

| 债 | 说明 |
|---|---|
| 未做真实大规模 target-side reader | 10k vs 20M，差距巨大 |
| 未做真实 Qwen3.8 teacher logits | 目前只有小规模 teacher-text |
| 未做 OPD vs teacher-text 对照 | 不知道严格 OPD 是否值得 |
| 未做完整 PLE 层注入 + 大规模 CPT | 只做了 logit-level projector |
| 未做 NQ / 更大正式基准 | 只有 HumanEval/TriviaQA |
| 未做 5M/20M 训练曲线 | 不知道收益是否随规模上升 |

### 3.2 工程债

| 债 | 说明 |
|---|---|
| 1070 无法跑大 teacher | 需要云端/API 一次性 teacher 数据 |
| 全量 logits 数据量不可行 | 需要 top-k / logprob 方案 |
| 上游 P1/P2 未做 | release 工具、统一 benchmark harness 未落地 |
| 远程 WSL 无法 push | 需要本地 clone 推送流程，已用临时 clone 解决 |
| CI PDF 构建未实际跑过 | 已添加 workflow，未验证 GitHub Actions 结果 |

### 3.3 数据/资产债

| 债 | 说明 |
|---|---|
| 离线 teacher-text 规模太小 | 2.3k 条，不是 Qwen3.8 真实 logits |
| 无 teacher top-k logits 数据集 | OPD 无法直接开始 |
| 无 4090/云端租用实测 | 只有估算 |
| 无正式 OPD 训练脚本 | 只有 CAP-1 self-distill |

---

## 4. 下一步开发计划

### Phase C：小规模 OPD / teacher-logit 可行性验证（1 周）

目标：

```text
判断严格 OPD 是否值得做
```

步骤：

1. 设计 top-k teacher logits 格式；
2. 本地 1070 生成 100k–500k rollout tokens；
3. 云端 A100/H100 或 API 获取 top-k logprobs；
4. 本地训练 LoRA/MoRA；
5. 对比：
   - teacher-text SFT；
   - OPD top-k；
   - Purified OPSD；
6. 用 real/control 和 held-out 评测。

Gate：

```text
如果 OPD 在 100k–500k 上没有明显优于 teacher-text
→ 不扩大到 20M
→ 转向 RAG + 蒸馏系统论文
```

### Phase D：选择主路线（1 周）

根据 Phase C 结果选择：

| 路线 | 说明 |
|---|---|
| A. 边界论文 | PLE 作为可审计局部记忆，加上 RAG/adapter 联合系统 |
| B. 低资源能力论文 | RAG + Purified OPSD + MoRA，不依赖 PLE 通用能力 |
| C. 混合 | PLE 局部 + RAG 知识 + 蒸馏能力，形成完整系统 |

建议：

```text
优先 C，但以 A/B 为科学底座。
```

### Phase E：工程补全（2 周）

- 实现 OPD / Purified OPSD 训练脚本；
- 实现 teacher top-k logits 生成/消费；
- 统一 benchmark harness；
- 完成 P1 release 工具；
- 再次更新 HF artifact。

### Phase F：论文与发布

- 写入完整 benchmark 结果；
- 写入训练成本/数据量；
- 写入第一性原理边界；
- 形成“低资源可审计记忆系统”的可复现发布。

---

## 5. 可以从类似项目借鉴什么（且不冲突）

| 项目 | 借鉴什么 | 不拿什么 |
|---|---|---|
| XMemTransfer | target-side reader 训练、20M 预算、跨模型解耦 | 不照搬其特定模型 |
| Memory Grafting | 条件记忆 + 预训练扩展方法 | 不追求大规模预训练 |
| Purified OPSD | 验证过滤、防自蒸馏坍缩 | 不盲信自生成 |
| ReAugKD | RAG 增强蒸馏，低成本知识迁移 | 不替代 PLE 局部记忆 |
| DeepSeek Engram / Qwen PLE | 表架构、缩放律、磁盘 offload | 不重训练大表 |
| kNN-LM | 开放生成失败模式 | 不把 kNN 当最终方案 |
| NGM | 训练无关 n-gram 基线 | 不认为它能替代 learned reader |
| Self-RAG / FAIR-RAG | 自适应检索、可靠性判断 | 不放弃 token 级记忆 |
| LoRA / QLoRA / MoRA | 参数化能力增强 | 不把 adapter 当记忆 |
| Lngram / Tensorizing | 记忆表示优化 | 不引入第二套存储 |
| TMLR / RepoEval | 可复现、评测卡、artifact | 不为了格式牺牲内容 |
| Diagram Design | 图表规范 | 不改变科学方法 |
| 我们自己的 P0 沉淀 | 跨仓库分工、契约、测试 | 不让实验仓承担核心库职责 |

---

## 6. 核心战略

1. **证据优先**：先做 100k–500k OPD 小规模验证。
2. **成本优先**：本地 1070 训练学生，云端只做一次性 teacher。
3. **不赌单一方法**：PLE、RAG、蒸馏、adapter 是互补，不是竞争。
4. **边界优先**：写清楚“能做什么、不能做什么”，比追求虚假通用提升更有价值。
5. **可复现优先**：日志、数据、top-k、checksum、评测卡都要留档。
6. **可部署优先**：最终系统必须能在 1070/低资源环境跑通。

---

## 7. 一句话结论

> 原始“纯 PLE 嫁接显著提升 0.8B 通用性能”的愿景，从第一性原理和成本角度基本不可行；
> 但“低资源可审计记忆系统 + 局部 PLE + RAG + 蒸馏 + adapter”是可行的，而且我们已经离它很近。
>
> 下一步最值得做的，不是烧钱做 20M 全量训练，而是：
> **用 100k–500k tokens 做一次严格 OPD vs teacher-text 对照，然后根据结果选择主路线。**
