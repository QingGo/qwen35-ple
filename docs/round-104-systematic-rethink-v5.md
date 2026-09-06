# Round 104：系统性复盘 v5——终极目标、技术债、开发计划与借鉴矩阵

> 日期：2026-09-06  
> 状态：基于 P0 证据修复、Purified OPSD 实跑和 CI 修复后的再次校准  
> 定位：在“PLE 在真实局部任务上通过 code/name、Purified OPSD 有正提升”的新证据下，重新安排后续工作。

---

## 1. 终极目标

> 以可审计、可寻址的 PLE/外部稀疏记忆为核心创新，构建低资源、可复现、可部署的 0.8B 混合智能系统；在真实局部任务和正式风格基准上证明各组件价值，最终实现 CPU 100 tok/s 的端到端产品。

### 三层目标

| 层 | 目标 | 当前状态 |
|---|---|---|
| 科学 | 明确 PLE 能做什么、不能做什么 | code/name 正，number 仍待改进 |
| 创新 | PLE + 任务 router + Purified OPSD 成为系统一等公民 | 组件均已可运行，尚未联合验证 |
| 产品 | 0.8B + PLE + RAG + adapter 可部署 | 工具已就绪，实际 CPU 100 tok/s 未验证 |

---

## 2. 本轮 session 成果

1. P0 PLE 证据修复：
   - 真实局部任务：代码 next-token / name / number；
   - 同域 bank + real/control + per-task 校准；
   - 3-seed：
     - code：真实融合 +0.421，real-control +0.402；
     - name：真实融合 +0.291，real-control +0.279；
     - number：真实融合 -0.022，未获绝对正收益。
2. Per-task 校准已持久化到 serving。
3. 正式评测工具：
   - GSM8K-like / MATH-like / HumanEval-like / MBPP-like；
4. Purified OPSD 实跑：
   - 过滤 138/160；
   - Purified MoRA-80：
     - CAP-1 held-out 较 base +0.085；
     - GSM8K-like +0.289；
     - HumanEval-like +0.096；
     - MBPP-like +0.124；
     - MATH-like -0.048。
5. CI 已修复并通过。

---

## 3. 技术债

### 3.1 科学/证据债

| # | 技术债 |
|---|---|
| S1 | number 任务仍无绝对正收益，尚未找到合适数字 bank / 任务定义 |
| S2 | name 任务 seed 1 为负，样本量仍不足，稳定性不够 |
| S3 | 正式基准仍是合成风格，不是真实 GSM8K/MATH/HumanEval/MBPP |
| S4 | 尚无 PLE + Purified OPSD + RAG + MoRA 的联合 3-seed 评测 |
| S5 | 尚未做新正式基准的污染审计 |
| S6 | 尚无 multi-seed Purified OPSD，只有 seed0 80 步一个点 |

### 3.2 架构/方法债

| # | 技术债 |
|---|---|
| M1 | 只有规则 router，在线 Hedge/mirror descent 尚未实现 |
| M2 | Per-task serving 参数已写入，但未做端到端 serving 验证 |
| M3 | 未做 LoRA / QLoRA / MoRA / Purified MoRA 的多 seed 对照 |
| M4 | Purified OPSD 只训练 80 步，未做步数/数据规模扫描 |
| M5 | PLE 的 number 任务虽已配置为 scale=0，但任务分类仍可能误开 |
| M6 | 没有把 PLE 局部收益与 Purified OPSD 能力收益合并到同一个系统评测 |

### 3.3 工程/产品债

| # | 技术债 |
|---|---|
| E1 | CPU 100 tok/s 未实际测量/优化 |
| E2 | 未做量化、GGUF、ExecuTorch |
| E3 | bundle/manifest/e2e 未完整跑通 |
| E4 | 远程训练产物未归档到可复现 manifest |
| E5 | serving 未包含 per-task PLE 与 adapter 联合路由的 e2e 测试 |

---

## 4. 开发计划

### Phase P1a：Purified OPSD 多 seed 与组件对照

1. 多 seed 训练：
   - Purified MoRA-80/160；
   - seed 0 / 1 / 2；
2. 对照：
   - base；
   - LoRA-160；
   - QLoRA-160；
   - MoRA-160；
   - Purified MoRA-80/160；
3. 评估：
   - CAP-1 held-out；
   - 正式风格四类基准；
   - 3-seed 均值 ± 标准差。

### Phase P1b：联合系统评测

1. 在正式风格基准上跑：
   - base；
   - +RAG；
   - +PLE（per-task）；
   - +Purified MoRA；
   - +all；
2. 在 PLE 局部任务上跑：
   - real vs control；
   - 3-seed；
3. 输出统一的 per-task/per-source 表格。

### Phase P1c：在线 Router 与正式评测

1. 实现 Hedge/mirror descent 在线 router：
   - 每个任务维护源权重；
   - 根据验证 log-loss 更新；
   - 输出可解释的 per-task 权重。
2. 接入真实数据集（如果能获取）或继续用结构化合成基准并明确标注。
3. 污染审计。

### Phase PROD：产品化

1. 量化 / GGUF / ExecuTorch；
2. CPU benchmark：
   - 目标 100 tok/s；
   - 当前基线脚本已就绪。
3. bundle + manifest + e2e；
4. serving e2e：
   - RAG + PLE + adapter + per-task router。

---

## 5. 可借鉴但不冲突的项目

| 项目/方向 | 借鉴什么 | 不拿什么 | 为什么与 PLE 不冲突 |
|---|---|---|---|
| Purified OPSD | 验证过滤 + 自蒸馏 | 不追求大规模 RL | 提升参数化能力，与 PLE 互补 |
| MoRA / DoRA | 高秩结构化参数更新 | 不把 adapter 当记忆 | 参数化能力提升 |
| kNN-LM reliability | 何时用检索/记忆才有效 | 不把 n-gram 当语义 | PLE 门控理论来源 |
| NGM | 免训练即插即用记忆 | 不复制其 embedding | 可作 PLE 接口 |
| MemSFT | 冻结 backbone + token router | 不让外部参数替代 PLE | 分布层融合 |
| TokenMem | 独立记忆通道 + conflict gate | 不塞满长上下文 | 与稀疏寻址互补 |
| RAGRouter / L-RAG | 查询路由 / 不确定性触发 | 不照搬训练成本 | 可落为规则+Hedge |
| Log Opinion Pool | 先校准再融合 | 不认为多源必然好 | 防止 PLE 负收益 |
| Rate-Distortion Memory | 记忆容量/成本权衡 | 不无限扩容量 | bank 规模决策 |
| Hedge / Expert Advice | 在线源权重 | 不替代正式评测 | 轻量 router |
| Optimal Control / MDP | 检索/记忆作为控制输入 | 不做复杂在线规划 | 激活决策 |
| ROME / MEMIT | 直接知识编辑 | 不用于通用能力 | 知识修复可选 |

---

## 6. 停止条件

1. 如果 Purified OPSD 多 seed 均值在正式基准没有稳定正提升 → 回到数据/验证过滤；
2. 如果 PLE 在真实局部任务 real-control 不显著 → 不再作为主创新；
3. 如果联合系统 +PLE 在语义/能力任务导致退化 → 关闭 PLE；
4. 如果 CPU 100 tok/s 无法达到 → 产品化未完成，需要量化/编译优化。

---

## 7. 一句话

> 本轮最大的进展是：**PLE 在真实局部任务拿到了证据，Purified OPSD 在正式风格任务拿到了正提升。**  
> 下一阶段不是继续堆方法，而是：
> 1. 把 Purified OPSD 做稳（多 seed、多 adapter）；
> 2. 把 PLE、RAG、Purified OPSD 放到同一个系统里做联合评测；
> 3. 最后推进 CPU 100 tok/s 产品闭环。
