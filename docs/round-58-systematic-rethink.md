# Round 58：系统性复盘、技术债、借鉴矩阵与后续开发计划

> 日期：2026-09-04
> 状态：P1 负结果、RAG 正结果后，重新校准终极目标和开发顺序
> 目的：不把项目变成“必须让 PLE 成功”的项目，而是以“可复现、低资源、可产品化地提升 0.8B 实际能力”为中心。

---

## 1. 终极目标

### 1.1 北极星

> **在低资源、可复现、可审计和 CPU 可部署的前提下，尽可能提升 Qwen3.5-0.8B 的端到端实际能力。**

注意：这里不再以“必须证明 PLE 有用”作为第一目标。

### 1.2 分层目标

| 层次 | 目标 | 验收方式 |
|---|---|---|
| 科学 | 回答“冻结 PLE 是否提供任务级因果增益” | real vs control + 3 seed + 离散指标 |
| 工程 | 形成可复现的 RAG/蒸馏/记忆实验栈 | CI、golden、配置化、文档 |
| 产品 | 交付一个实际可用的 0.8B 变体 | 评测集 + CPU 100 tok/s e2e |
| 过程 | 每次决策有证据、有门禁、可回滚 | pre-registration + gate + 文档 |

### 1.3 当前主要结论

- PLE 纯特征有微弱条件信息；
- 但当前 frozen backbone + hidden reader/P1 memory interface 不能转化为任务级 real>control；
- RAG BM25 已经带来显著提升；
- 因此：
  - **PLE 从主路径降级为可选局部语言先验**；
  - **主路径切换为 RAG + 教师蒸馏**。

---

## 2. 本轮 session 发现的技术债

### 2.1 科学/证据债

| # | 技术债 | 现状 |
|---|---|---|
| TD-S1 | 没有“真实任务级 real>control” | Phase A P1 均未通过 |
| TD-S2 | 没有 RAG/蒸馏/PLE 三方同口径完整评测 | 只有 rare QA logprob |
| TD-S3 | 没有推理/代码/长上下文评测 | 未建立 |
| TD-S4 | 没有 3-seed 显著性和 paired test | 当前仅单次/双 checkpoint |
| TD-S5 | 没有答案泄漏/语料污染审计 | RAG 和记忆 bank 均需审计 |
| TD-S6 | B2/B3 界未测 | 只有 B0/B1 估算脚本 |

### 2.2 方法/实现债

| # | 技术债 | 现状 |
|---|---|---|
| TD-M1 | 没有实现最优 logit-level memory head | 只有 hidden 注入 + 简易 router |
| TD-M2 | 没有学习“条件对数似然比” | 未实现 |
| TD-M3 | RAG 只有 BM25，无语义检索/rerank | 简单 baseline |
| TD-M4 | 没有 OPD / Purified OPSD 流程 | 未实现 |
| TD-M5 | 没有 MoRA/GaLore/ReLoRA 实验 | 若后续训练需要，尚未选型 |
| TD-M6 | PLE bank 只来自单一书籍语料 | 覆盖不足 |
| TD-M7 | 没有将记忆/检索模块接入 serving | 仅离线脚本 |

### 2.3 工程债

| # | 技术债 | 现状 |
|---|---|---|
| TD-E1 | 新脚本缺少统一 registry/config | 原型脚本偏一次性 |
| TD-E2 | RAG/bounds 脚本没有单元测试 | 需补 |
| TD-E3 | 没有 CPU 100 tok/s serving e2e | 未闭环 |
| TD-E4 | 没有完整 bundle + manifest | 需后续 |
| TD-E5 | 数据 provenance/license 记录不完整 | 需补 |

---

## 3. 借鉴矩阵：如何借而不冲突

### 3.1 核心原则

1. **不把 PLE 当作前提**，只把它当作众多可选的记忆源之一；
2. **吸收“方法”，不吸收“立场”**；
3. **所有路径都可以共存**，因为有统一接口：logit-level 融合 + router。

### 3.2 具体借鉴

| 项目 | 借什么 | 不拿什么 | 为什么与当前目标不冲突 |
|---|---|---|---|
| XMemTransfer | 冻结记忆表 + target-side reader | 不把“记忆必须来自 PLE”当目标 | 可作为任何记忆源的读取协议 |
| TokenMem | 独立通道 + conflict gate | 不认为只有 cross-attention 才能增强 | 如果未来做语义记忆，可复用 |
| MemSFT | distribution-level memory + router | 不要求 memory 是参数化 LLM | 和 RAG/蒸馏都可共存 |
| DeepSeek Engram / Qwen PLE | n-gram 查找、局部先验 | 不追求复现大模型收益 | 只用于理解 PLE 为什么弱 |
| ReAugKD / RAG | 检索增强教师 + 学生蒸馏 | 不把项目变成纯 RAG 产品 | 这是当前主路径 |
| OPD / Purified OPSD | 学生轨迹 + teacher 分布 | 不直接照搬到长 CoT | 用于能力迁移 |
| MoRA/GaLore/ReLoRA+sMuon | 低资源 backbone adaptation | 不把它们当做 PLE 必要配套 | 只用于蒸馏/后训练阶段 |
| PERK / test-time LoRA | 长上下文 test-time adaptation | 不替代主训练 | 低优先级，按需使用 |
| Hierarchical Memory | 长尾知识 + 推理 anchor 分离 | 不重训大表 | 可作为最终产品架构 |
| Probabilistic/Info-theoretic framework | CMI 上界、logit 最优修正、router 最优性 | 不用它代替实验 | 用于设计门禁和预期 |

### 3.3 融合架构（不冲突）

```text
输入
  ├─ 原始问题
  ├─ RAG 检索文档（主知识源）
  └─ PLE 可选局部 n-gram 先验（弱，可关闭）
       │
       ▼
Base 0.8B backbone（可冻结/可低秩训练）
       │
       ├─ base logits
       └─ memory/teacher logits（log-likelihood-ratio head）
            │
            ▼
      learned router（logit-level 融合）
            │
            ▼
      输出
```

PLE 被设计为可开关、可审计的 optional module，不会阻塞主路径。

---

## 4. 后续开发计划

### Phase R1：把证据做硬（当前优先）

目标：

- 建立可用的多任务评测协议；
- 测量 B2/B3；
- 结束“PLE 到底有没有用”的争议。

任务：

1. 建立评测集：
   - rare knowledge QA（已有）
   - GSM8K / MATH subset
   - HumanEval / MBPP subset
   - synthetic long-context / needle
2. 统一 protocol：
   - no-memory / RAG / teacher-distilled / PLE optional
   - real vs control
   - 3 seeds + paired test
   - contamination/leakage audit
3. 实现并测量：
   - `estimate_ple_bounds.py`：B0/B1
   - B2：backbone Jacobian 可见子空间
   - B3：logit-space memory head
4. 门禁：
   - 如果 B3 仍接近 0，PLE 正式降级为低优先级；
   - 如果 B3 > B4，说明 hidden 通道是瓶颈，后续可做 logit-space 或 sparse backbone adaptation。

### Phase D1：RAG 产品化原型

目标：

- 把当前 BM25 baseline 升级为可靠检索 + 0.8B 推理闭环。

任务：

1. 混合检索：
   - BM25 + embedding
   - 可加 rerank
2. 检索语料构建：
   - 知识、代码、数学、长文档
   - 来源与 license 记录
3. 接入 serving/bundle：
   - 先 offline eval
   - 再 CPU/GPU serving smoke
4. 门禁：
   - RAG 在 rare/common/reasoning/code/long-context 上都 > no-context；
   - 无严重风格退化；
   - CPU 延迟可接受。

### Phase D2：教师蒸馏 / OPD / Purified OPSD

目标：

- 把更强的能力注入 0.8B，而不是只靠检索上下文。

任务：

1. 准备 teacher：
   - 优先 Qwen3.8-Flash-Next；
   - 或使用 RAG-augmented teacher 自蒸馏。
2. 离线阶段：
   - SFT/distill on reasoning/code/long-context mixture；
   - 可先用 MoRA 或低秩 LoRA。
3. on-policy 阶段：
   - OPD + Purified OPSD；
   - 避免 reference shortcut。
4. 门禁：
   - 蒸馏后通用能力不退；
   - 推理/代码/长上下文相对 RAG baseline 有提升；
   - 如果使用 PLE，仍须 real>control。

### Phase D3：产品化 / CPU 100 tok/s

目标：

- 最终可部署 0.8B + RAG（PLE 可选）。

任务：

1. 量化/编译/推理后端：
   - vLLM/SGLang/CompileForge；
   - 或轻量 CPU runtime。
2. 记忆/检索 offload/prefetch；
3. bundle + manifest + e2e；
4. CPU 同机 A/B，目标 ≥100 tok/s。

### Phase PLE-Final（低优先级）

只在以下条件下继续投入：

- B3 logit-space PLE head 出现显著 real>control；
- 或 RAG/蒸馏之外需要局部 n-gram 先验。

否则：

- 将 PLE 记录为可选局部语言先验；
- 不进入大规模 PLE MoRA/GaLore/RL。

---

## 5. 如何更稳地前进

### 5.1 负面结果也作为资产

- P1 负结果已经帮助完成 pivot；
- 后续每个阶段都有明确 gate，不靠“感觉”推进。

### 5.2 先小后大

- 每个实验从 10–100 条/small run 开始；
- 有正信号后再放大；
- 不做一次性 5M–20M 无门禁训练。

### 5.3 所有能力提升都要可归因

- 必须能区分：
  - backbone 能力提升
  - RAG 上下文收益
  - 蒸馏 teacher 收益
  - PLE 真实/控制差异
- 使用消融 + 同口径评测。

### 5.4 保持工程可复现

- 新脚本进 CI；
- 配置/路径写入 manifest；
- 数据 provenance 记录；
- 每次实验输出 JSON + 文档。

---

## 6. 停止条件 / 转折点

| 情况 | 行动 |
|---|---|
| RAG 在多任务稳定提升 | 继续主路径 |
| RAG 只提升知识问答，不提升推理/代码 | 加强蒸馏，不把 RAG 当万能 |
| 蒸馏导致通用能力退化 | 回退，减少 OPD 强度，加入 replay/KL |
| B3 PLE 仍为 0 | 永久将 PLE 降级为 optional |
| CPU 达不到 100 tok/s | 先产品化最简 RAG/蒸馏模型，再优化 |
| 资源不足 | 缩小任务集，不做大规模 RL |

---

## 7. 一句话总结

我们现在的真正目标不是“让 PLE 成功”，而是：

> **用所有可审计的低资源手段，把 0.8B 做成一个真正更好用、可部署、可复现的模型。**

PLE 只是其中一个尚未证明有任务级价值的可选模块。
