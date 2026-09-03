# Round 47：系统性思考——终极目标、技术债、借鉴矩阵与开发计划

> 日期：2026-09-03
> 状态：战略复盘
> 目的：厘清终极目标，沉淀本轮技术债，制定更稳、更接近目标的开发计划。

---

## 1. 终极目标

我们不是“为了证明 PLE 有用而做实验”，而是：

> **在可复现、可审计、可产品化的前提下，尽可能提升 Qwen3.5-0.8B + Qwen3.8-Flash-Next PLE 的端到端智能水平，同时保持 CPU 100 tok/s 的可用性。**

可拆成四个层次：

| 层次 | 目标 | 验收 |
|---|---|---|
| 科学 | 证明 PLE 提供真实、因果、任务相关的记忆信息 | real > control，且差异来自 E 内容而非格式 |
| 工程 | 四仓库可复现闭环；训练/推理数值一致；资产可重建 | golden、bundle、e2e |
| 产品 | 0.8B 模型在知识/长尾/推理任务上有可测量提升 | 固定评测集 + 3 seeds |
| 过程 | 每个决策有证据、有门禁、可回滚 | Pre-registration + 门禁 |

---

## 2. 本轮 session 的主要发现

| 发现 | 影响 |
|---|---|
| PLE 线性增量 R² 很小 | 不能直接放大 |
| PLS 低秩方向优于 PCA | 应该用监督方向 |
| 稀有 token 增量约为常见 token 2 倍 | 应做 rare-token 任务与门控 |
| MLP 非线性比线性好 3–4 倍 | 需要非线性 value |
| learned h_to_e 会退化 | 必须约束 E_perp |
| MLP(E_perp) 单独无效 | 必须联合 H |
| differential 注入被随机噪声淹没 | 当前 E 特有信号太弱 |
| contrastive hinge 发散 | 需要稳定对比目标 |
| Loss 下降不代表智能提升 | 必须用任务指标 |

---

## 3. 本轮发现的技术债

### 3.1 高优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-1 | 没有“记忆需求任务”专门评测集 | 未解决 |
| TD-2 | 没有在真实知识任务上测 \(I(Y_{\text{task}};E\mid H)\) | 未解决 |
| TD-3 | learned h_to_e 退化未约束 | 已发现，未根治 |
| TD-4 | 没有 backbone adaptation（LoRA/部分解冻）实验 | 未解决 |
| TD-5 | 没有稳定 contrastive value 训练方法 | 已尝试但发散 |
| TD-6 | 没有 RAG / 教师蒸馏的同口径对照 | 未解决 |
| TD-7 | 没有 real-vs-control 在 rare-task 上的置信区间/3 seeds | 未解决 |

### 3.2 中优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-8 | 训练规模太小（1024 token） | 未解决 |
| TD-9 | 没有 Memory Grafting / XMemTransfer 规模参考实验 | 未解决 |
| TD-10 | 没有 CPU 100 tok/s 的 PLE 注入性能闭环 | 未解决 |
| TD-11 | 没有完整 serving/bundle 端到端测试 | 部分解决 |
| TD-12 | WSL 与本地环境漂移 | 存在 |
| TD-13 | 缺乏 3-seed 显著性 | 未解决 |

### 3.3 低优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-14 | 文档偶有重复/格式问题 | 已清理一部分 |
| TD-15 | 新脚本测试覆盖不足 | 部分覆盖 |
| TD-16 | 缺少对 MLPValueReader 的 registry roundtrip 测试 | 未做 |
| TD-17 | 缺少 RAG baseline 评测脚本 | 未做 |

---

## 4. 借鉴矩阵

| 项目/方向 | 借什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| XMemTransfer / Memory Grafting | 大规模 target-side reader 训练、5M–20M 量级 | 不直接照搬模型 | 只有在 real>control 后才放大 |
| DeepSeek Engram / engram-peft | 条件记忆、gate、ShortConv、训练集成 | 不引入第二套存储 | 复用现有 PLE 表 |
| NGM / MLP Memory | 训练无关/更语义化的记忆表示 | 不替代当前 PLE 主路径 | 作为 parallel baseline |
| RAG / ReAugKD | 外部检索、教师蒸馏 | 不把项目变成纯 RAG | 作为“能否更好”的同口径对照 |
| Hierarchical Memory | 常见知识 vs 长尾知识分离 | 不重训整个模型 | 支持 rare-token gating |
| Selective Memory / MemFlow / MemPO | 小模型记忆编排、选择性读取 | 不照搬智能体框架 | 补充 gate/RL 方法 |
| SR-TTT | surprise-aware residual | 不引入 test-time 训练 | 可用作 gate 信号 |
| Storage–Retrieval Gap | 诊断 adapter 是否只是输出条件化 | 不改变存储 | 用于识别假阳性 |
| Scaling Law 研究 | Loss 代理的适用条件 | 不把 loss 当真理 | 提醒我们使用任务指标 |
| EngramDB / CompileForge | 证据库、golden、契约、推理性能 | 不重复实现存储/编译 | 工程底座 |

---

## 5. 开发计划

### Phase A：任务与指标重构（1–2 周）

目标：先解决“我们到底要提升什么”。

1. 构造 rare-token 知识评测集；
2. 测任务级 \(\Delta I(Y_{\text{task}};E\mid H)\)；
3. 建立 real/control/random/zero 的 CATE 与置信区间；
4. 固定评测协议，禁止再以 val loss 作代理。

门禁：

- 完成 rare-task 指标；
- 能回答：PLE 在真实知识任务上是否 > control。

### Phase B：Reader 结构稳定（2–4 周）

1. 固定 E_perp + MLP(H,E_perp)；
2. differential injection；
3. rare-token condition gate；
4. 稳定 contrastive value loss（InfoNCE/triplet + 谱归一化）；
5. 小规模 3 seeds。

门禁：

- rare-task 上 real > control；
- BoolQ 不显著退化；
- value 的 E 特有占比上升。

### Phase C：Backbone 适配与规模（4–8 周）

1. LoRA / 部分解冻；
2. 100k–1M token 记忆增强数据；
3. 稀有 token 过采样；
4. SFT 初步；
5. 与 RAG baseline 同口径对比。

门禁：

- 端到端 rare-task 提升显著；
- 通用任务不退化；
- 有 3 seeds。

### Phase D：RL/混合记忆（可选）

1. DPO/GRPO 奖励正确使用记忆；
2. 或 RAG / 教师蒸馏；
3. 比较 PLE、PLE+RAG、RAG、蒸馏。

门禁：

- real > control；
- 有可复现的正向结果才继续。

### Phase E：产品化与性能（并行）

1. vLLM/SGLang/CompileForge serving；
2. Store-P/access-order；
3. CPU 100 tok/s；
4. bundle 端到端测试。

---

## 6. 停止条件 / 转向条件

如果 Phase B/C 后：

```text
real 在 rare-task 上仍不显著 > control
```

则：

- 将 PLE 定位为“局部语言模式增强”；
- 转为 RAG / 蒸馏 / 更语义化记忆；
- 记录为可审计负面结果；
- 不进入大规模 RL。

---

## 7. 当前最有利的资产

- 已有完整数学证明框架；
- 已有增量 R²/PLS/rare-token/oracle MLP 诊断工具；
- 已有 MLPValueReader 原型；
- 已有 CI 和文档体系；
- 已有 WSL 可复现实验环境；
- 已有“不用 val loss 作代理”的共识。

---

## 8. 下一步行动（按优先级）

1. 建 rare-token 知识评测集；
2. 固定 E_perp + MLP(H,E_perp) + differential + rare gate；
3. 对比 LoRA/backbone adaptation；
4. 加 RAG baseline；
5. 做完再决定是否放大训练/RL。
