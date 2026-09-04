# Round 50：系统性复盘与后续开发计划

> 日期：2026-09-04
> 状态：Phase A 已完成，进入 Phase B/C 的方案设计与关键技术选型
> 目的：在 Phase A 负结果和本轮调研基础上，重新校准终极目标、技术债、借鉴矩阵和开发顺序。

---

## 1. 终极目标

我们的终极目标不是“证明 PLE 能用”，而是：

> **在可复现、可审计、低资源、可产品化的前提下，尽可能提升 Qwen3.5-0.8B + 冻结 Qwen3.8-Flash-Next PLE 的端到端智能水平，同时保持 CPU 100 tok/s 的可用性。**

可进一步拆成四层：

| 层 | 目标 | 验收 |
|---|---|---|
| 科学 | 证明 PLE 在真实下游任务上是否提供 causal 增益 | real > control，稀有/推理/代码/长上下文分层 |
| 工程 | 四仓库可复现闭环；训练/推理数值一致 | golden、bundle、e2e、CI |
| 产品 | 0.8B + PLE 在知识、推理、代码、长上下文有可测量提升 | 固定评测 + 3 seeds |
| 过程 | 每个决策有证据、有门禁、可回滚 | pre-registration + 门禁 + 文档 |

---

## 2. 本轮 session 的核心发现

### 2.1 Phase A：现有 PLE 嫁接没有下游收益

- 纯 PLE 特征有极小的因果增量：
  \[
  \Delta R^2 \approx 10^{-3}
  \]
- real > control 在所有子集上稳定；
- 但 reader 中介的 rare 知识任务：
  - simple reader：real−control ≈ −0.017；
  - MLP reader：real−control ≈ +0.001；
- 生成式 exact-match 小样本：real/control/no-reader 无差异；
- 结论：**PLE 有微弱条件信息，但当前 reader/backbone 没有能力把它转化为任务收益。**

### 2.2 原论文的收益不能直接移植

- DeepSeek Engram 的推理/代码/长上下文收益来自：
  - 早期层静态重建卸载；
  - 注意力容量释放；
  - 有效深度增加；
- Qwen3.8-Flash-Next 自己的 N-gram ablation 显示：
  - loss 下降和 downstream accuracy 不同步；
  - 知识/中文任务收益较好；
  - 代码收益较小；
  - 部分任务会饱和/波动；
- 因此，冻结 PLE + 小 reader 不能自动继承这些收益。

### 2.3 Qwen PLE 的硬约束

- Qwen3.8-Flash-Next PLE 只有 2-gram 和 3-gram；
- `ngram_size = 3`，`heads_per_ngram = 8`；
- 没有原生 4-gram；
- 如果想用 4-gram，需要自己构建外部 exact 4-gram bank。

### 2.4 关键相关技术

| 方向 | 代表工作 | 可借鉴点 |
|---|---|---|
| 外部记忆嫁接 | Memory Grafting / XMemTransfer | exact longest-match、frozen memory bank、projection+gate+ShortConv、Engram fallback、post-training 可用 |
| 记忆分布模块 | MLP Memory / MemSFT | 用 memory 模块模仿 retrieval teacher，router 融合，避免 catastrophic forgetting |
| 独立知识通道 | TokenMem | 独立 cross-attention，避免与 self-attention 竞争 |
| 长上下文 | PERK | 用 test-time LoRA 编码长上下文，低成本复现长上下文推理 |
| 高秩 PEFT | MoRA | 同参数量更高秩，适合 memory / continual pretraining |
| 全参数低显存 | GaLore | 梯度低秩投影，全参数更新但只存低秩优化器状态 |
| 多轮低秩 | ReLoRA | 多轮合并 LoRA，提高总更新秩 |
| Muon + LoRA | sMuon / Riemannion | 不要用朴素 per-factor Muon |
| on-policy 蒸馏 | OPD / Purified OPSD | 学生轨迹 + teacher 逐 token 监督；去除 reference shortcut |
| 层次化记忆 | Hierarchical Memory | 长尾知识放外部记忆，小 backbone 做 anchor，保留推理能力 |

---

## 3. 技术债

### 3.1 高优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-1 | 没有“能真正使用 PLE”的记忆模块 | 未解决 |
| TD-2 | 没有 exact longest-match PLE bank | 未解决 |
| TD-3 | 没有独立 cross-attention 记忆通道 | 未解决 |
| TD-4 | 没有 distribution-level memory / router fusion | 未解决 |
| TD-5 | 没有 MoRA / GaLore / ReLoRA 实验 | 未解决 |
| TD-6 | 没有 OPD / Purified OPSD 蒸馏流程 | 未解决 |
| TD-7 | 没有推理/代码/长上下文 real-vs-control 评测 | 未解决 |
| TD-8 | 没有 RAG / 教师蒸馏同口径 baseline | 未解决 |

### 3.2 中优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-9 | rare-task benchmark 不够完整 | 部分解决 |
| TD-10 | 没有 3-seed 任务级显著性 | 部分解决 |
| TD-11 | 没有 4-gram 外部记忆 | 未解决 |
| TD-12 | 训练规模仍太小 | 未解决 |
| TD-13 | 没有 CPU serving / 100 tok/s 闭环 | 未解决 |
| TD-14 | 没有完整记忆模块的 registry/roundtrip 测试 | 未解决 |

### 3.3 低优先级

| # | 技术债 | 状态 |
|---|---|---|
| TD-15 | 文档重复/格式 | 已清理一部分 |
| TD-16 | 新脚本测试覆盖不足 | 部分覆盖 |
| TD-17 | WSL/本地环境漂移 | 存在 |

---

## 4. 借鉴矩阵：怎么借而不冲突

| 来源 | 借什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| Memory Grafting / XMemTransfer | exact longest-match、frozen memory、projection+gate+ShortConv、fallback、post-training 用法 | 不重训大表 | 它是 PLE 的读取接口，不改变 PLE 表 |
| DeepSeek Engram / Qwen PLE | 早期层注入、contextual gating、prefetch/offload、容量重分配 | 不重训 125B/51B | 只借用架构位置和 memory 接口思想 |
| MLP Memory / MemSFT | 让 memory 模块模仿 retrieval teacher，router 融合，防遗忘 | 不替代 PLE 表 | 它和 PLE 可共存：PLE 提供特征，memory 模块负责使用 |
| TokenMem | 独立 cross-attention 通道、conflict-aware gate、两阶段课程 | 不改 backbone self-attention 结构 | 它只新增外部通道 |
| PERK | test-time LoRA 编码长上下文 | 不把它当作主训练路径 | 用于补长上下文短板 |
| MoRA | 高秩更新，适合 memory/continual pretraining | 不需要替代全部 LoRA | 它是 parameterization，不是架构 |
| GaLore | 梯度低秩投影、全参数低显存 | 不需要 full FT | 可作为 MoRA 的替代/互补 |
| ReLoRA + sMuon | 多轮低秩累积、Muon 正确用于低秩 | 不需要每轮全调参 | 适合极小资源 |
| OPD / Purified OPSD | 学生轨迹 + teacher 监督、PMI 去除 shortcut | 不直接照搬 OPSD 到长 CoT | 作为 post-training，不改变 PLE 模块 |
| RAG / ReAugKD | 同口径对照和兜底 | 不把项目变成纯 RAG | 用于判断 PLE 是否真的更好 |
| Hierarchical Memory | 长尾知识与 anchor 推理分离 | 不重训整个模型 | 支持 small anchor + external memory |
| MemLoRA | 任务专用 LoRA expert | 不照搬记忆系统框架 | 可用于知识/推理/代码分阶段适配 |

---

## 5. 后续开发计划

### Phase P1：构建真正可用的 PLE 记忆模块

目标：

- 不训练 backbone，先把“PLE 怎么被使用”这个问题解决。

具体任务：

1. 构建 **exact longest-match PLE bank**：
   - 2/3/4-gram 高频 key；
   - 每个 key 对应一个冻结 PLE 特征；
   - miss 时 fallback 到原始多哈希 PLE。
2. 实现 **TokenMem 式独立 cross-attention 通道**：
   - memory 有独立 softmax；
   - 不和 backbone self-attention 竞争。
3. 实现 **distribution-level memory + router**：
   - 参考 MLP Memory / MemSFT；
   - memory 模块输出 token distribution；
   - router 决定和 backbone 融合比例。
4. 训练：
   - 先从 retrieval teacher 或 PLE-augmented teacher 蒸馏 memory 模块；
   - 冻结 backbone；
   - 只训练 memory 模块和 router。

门禁：

```text
rare knowledge: real > control
boolq/通用: 不显著退化
```

### Phase P2：加入低资源 backbone adaptation

目标：

- 让 backbone 真正学会使用 PLE。

具体任务：

1. 首选 **MoRA**，因为它最适合 memory / continual pretraining；
2. 对照 **GaLore**，看 full-parameter low-memory 是否有额外收益；
3. 如果资源极紧，用 **ReLoRA + sMuon** 做多轮；
4. 数据：
   - rare knowledge 1M–5M；
   - GSM8K/MATH 1M–5M；
   - HumanEval/MBPP 1M–5M；
   - synthetic long-context 0.5M–2M；
5. 训练中保留：
   - real/control 对比；
   - 防遗忘 KL 或 replay。

门禁：

```text
rare knowledge:      real > control
GSM8K / MATH:        real > control
HumanEval / MBPP:    real > control
long-context:        real > control
```

### Phase P3：加入 OPD / Purified OPSD 蒸馏

目标：

- 用 Qwen3.8-Flash-Next 作为 teacher，把推理/代码/长上下文能力迁移到小模型。

具体任务：

1. 离线 teacher 数据蒸馏，先稳定；
2. 再 OPD：
   - 学生自己采样；
   - teacher 逐 token 给分布；
   - 用 stop-gradient Top-K KL 稳定；
3. OPSD 只用于共享规则：
   - 例如“何时使用 PLE”；
   - 不要用于 instance-specific reference；
4. 如果用 OPSD 做推理，使用 **Purified OPSD / PMI residual**。

门禁：

```text
蒸馏后通用能力不退化
推理/代码任务相对 SFT 提升
PLE real > control 仍然成立
```

### Phase P4：对照与决策

- 跑同口径：
  - PLE + MoRA；
  - PLE + MoRA + OPD；
  - RAG baseline；
  - 纯教师蒸馏 student；
- 如果 PLE 始终不贡献 real>control，则：
  - 把 PLE 定位为局部语言模式增强；
  - 转向 RAG / 蒸馏 / 更语义化记忆；
  - 不进入大规模 RL。

### Phase E：产品化

- Store-P / access-order；
- vLLM / SGLang / CompileForge；
- CPU 100 tok/s；
- bundle e2e；
- memory offload / prefetch。

---

## 6. 当前最优先的实验

我认为下一个最值得做的事情不是大规模训练，而是：

> **先做一个“exact longest-match PLE bank + TokenMem style cross-attention + MLP Memory style router fusion”的最小原型，backbone 完全冻结。**

理由：

- 成本最低；
- 直接解决我们观察到的问题：“reader 无法区分 real/control”；
- 可以尽早判断 PLE 是否真的可以通过更好的读取接口被使用；
- 如果这一步都失败，就不需要进入 MoRA/GaLore/OPD 的大规模实验。

---

## 7. 风险与停止条件

| 风险 | 应对 |
|---|---|
| PLE 本身信息太弱 | 先用 exact bank + cross-attention 试，再决定是否转 RAG |
| 蒸馏后通用能力退化 | 保留 backbone 冻结阶段、防遗忘 KL、replay |
| OPSD 导致推理退化 | 使用 Purified OPSD，或只用 OPD |
| 资源不足 | 先小原型，门禁再做大规模 |
| 4-gram 不是原生 | 需要外部 bank，且不能指望 PLE 表直接提供 |

停止条件：

- exact bank + cross-attention + router 后，rare real−control 仍不显著；
- 或 Phase P2 后所有真实任务 real 仍不显著 > control；
- 或通用能力显著退化且无法修复。

满足任一条件，就进入 RAG / 蒸馏 / 更语义化记忆的转向路线。

---

## 8. 总结

一句话：

> **现在的问题不是“PLE 没有信息”，而是“我们没有正确的读取/使用接口”。**

下一阶段应该：

1. 冻结 backbone，先做最好的 memory interface；
2. 用 MoRA 做低资源高秩 backbone adaptation；
3. 用 OPD / Purified OPSD 做 Qwen3.8 到小模型的能力迁移；
4. 用 RAG/蒸馏作为同口径对照；
5. 每一步都有 real-vs-control 门禁。

这样我们才能更稳、更省地接近终极目标。
