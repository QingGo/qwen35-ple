# Round 64：0.8B 端到端提升路线与 PLE 的正确用法

> 日期：2026-09-04
> 状态：基于 P1/B3/RAG 证据重新定位 PLE 的可行角色
> 结论：PLE 不应继续作为“语义知识记忆”使用；若要用起来，应改成“局部 n-gram/低熵先验”，并用新门禁验证。

---

## 1. 我们能走哪几条技术路线

按预期收益从高到低：

| 路线 | 核心机制 | 当前证据 | 预计作用 |
|---|---|---|---|
| RAG | 外部检索上下文直接进入输入 | rare Δ≈+0.85，common Δ≈+2.07 | 知识问答、事实回忆 |
| 教师蒸馏 / OPD / Purified OPSD | 用更强 teacher 迁移推理/代码/格式 | 尚未实施，但方向明确 | 推理、代码、长上下文、生成质量 |
| 低资源 backbone adaptation | MoRA / GaLore / ReLoRA+sMuon | 尚未实验 | 让模型学会使用新数据和检索/记忆信号 |
| 多任务/长上下文后训练 | 长上下文数据 + 压缩/回忆 | 尚未系统验证 | 长文档、代码、工具使用 |
| PLE 局部 n-gram 先验 | 低成本 exact 2/3/4-gram 先验 | 当前 rare QA 无 real>control | **只可能用于局部低熵/格式/补全** |
| RL / 工具使用 | 强化学习或 API/工具 | 尚未到该阶段 | 复杂任务，但资源/证据不足 |

综合判断：

> **大程度提高 0.8B 端到端表现的主路径是 RAG + 教师蒸馏 + 低资源后训练，而不是 PLE。**

---

## 2. 为什么 PLE 目前用不起来

已有直接证据：

- Phase A：task-level ΔR² 只有约 1e-4 ~ 1e-3；
- P1 hidden injection：rare real−control ≈ +0.00013，不显著；
- B3 logit-space：3 seeds rare real−control ≈ **−0.00117 ± 0.00008**，全部为负；
- 而 RAG 的 rare Δ ≈ **+0.851**。

结论：

> PLE 的问题不是“读取方式不够好”，而是它本身在 rare 知识任务上的条件信息太低。
> 即使直接改 logits、绕过所有 hidden 瓶颈，也无法放大。

---

## 3. 如何才能真正把 PLE 用起来

### 3.1 重新定义 PLE 的角色

PLE 是 **n-gram 查表记忆**：

- 擅长：局部共现、常见短语、名字/数字接续、代码 token 补全、格式；
- 不擅长：跨句推理、世界知识、语义检索。

所以应把 PLE 从“知识记忆”改为：

> **局部低熵 token 先验 / 低成本小 n-gram 专家。**

### 3.2 建立新的 PLE 门禁

不要再只测 rare QA real−control。应测 PLE 真正可能有用的任务：

| 新任务 | 检查什么 |
|---|---|
| 低熵 token 预测 | 在 given 前文下，real > control 的 next-token logprob |
| 名字/实体接续 | 专有名词、人物、地点后续 token |
| 代码补全 | 括号、关键字、常见 API 序列 |
| 数字/日期格式 | 日期、数量、单位等局部模式 |
| 长尾短语 | exact 2/3/4-gram 在 bank 中命中时的表现 |

如果这些任务上有显著 real>control，PLE 才有资格作为“局部专家”。

### 3.3 具体实现方式

1. **PLE 作为 logit-level 专家**
   - 训练 `PureLogitMemoryModule` 或更好的 `LogLikelihoodRatioHead`；
   - 只在局部低熵任务上训练；
   - 输出 logit 偏移：
     \[
     \delta(y)=\log P(y|H,E)-\log P(y|H)
     \]
2. **Router 决定何时信任 PLE**
   - 只在以下情况激活：
     - exact bank 命中长 n-gram；
     - 当前 token 属于代码/日期/专名等低熵类别；
     - base model uncertainty 高且 PLE 专家置信度高。
3. **与 RAG/蒸馏分层**
   - RAG 提供语义证据；
   - 蒸馏模型提供推理/格式；
   - PLE 只作为可选局部先验，不能覆盖前两者。
4. **最终架构**：

```text
输入
 ├─ 可选 RAG 上下文
 │
 ▼
Base 0.8B
 ├─ base logits
 ├─ teacher/RAG logits
 └─ PLE local expert logits（仅在低熵/命中时激活）
      │
      ▼
 learned router
      │
      ▼
 output
```

### 3.4 可能需要 backbone adaptation

如果新的局部任务上 PLE 有正信号，但仍然不足以影响输出，可以：

- 用 MoRA / LoRA 小规模适配 backbone；
- 仅用 PLE 命中数据；
- 保持 real vs control 门禁；
- 在知识推理任务上验证不退化。

但注意：

- 这不应成为主路径；
- 只有先看到局部任务 real>control 才值得做。

---

## 4. 不建议的路径

| 不建议 | 原因 |
|---|---|
| 把 PLE 当知识检索 | 实测无 real>control |
| PLE + 大规模 MoRA/GaLore 主训练 | 在没有正信号前投入过大 |
| 用 PLE 替代 RAG | RAG 已验证远优于 PLE |
| 用 hidden 注入继续调 reader | B3 证明 logit 层都无法放大 |
| 把 PLE loss 下降当智能提升 | 已被 round-46 和 B3 否定 |

---

## 5. 最可能带来大提升的执行顺序

1. **D2 教师蒸馏 / OPD / Purified OPSD**
   - 先做离线 SFT/distill；
   - 再做 on-policy；
   - 这是当前最可能大幅提升推理/代码/格式的路线。
2. **RAG 产品化**
   - 已有 hybrid + HTTP serving 原型；
   - 升级 sentence embedding / rerank；
   - 接入生产推理后端。
3. **低资源后训练**
   - 用 MoRA 在高质量混合数据上继续训练；
   - 如果 RAG/蒸馏证明有效，再考虑把检索或 teacher 信号融入训练。
4. **PLE 局部先验**
   - 新门禁通过后，作为可选专家加入；
   - 不通过则正式降级/归档。

---

## 6. 一句话

> PLE 不是不能用，而是被用错了角色。
>
> 把它从“知识记忆”改成“局部 n-gram/低熵先验”，用新的局部任务门禁重新验证；如果通过就作为可选专家，如果不通过就记录为已证伪的局部记忆方案。
