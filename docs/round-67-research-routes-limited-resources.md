# Round 67：有限资源下可以尝试的技术路线全景

> 日期：2026-09-04
> 状态：多轮调研汇总
> 资源：8GB GPU、约 15GB RAM、0.8B student、可获取 Qwen3.8-Flash-Next teacher（需高 RAM/云）
> 目标：在有限资源下，把每条可能提升 0.8B 的路线按投入/收益/风险排序。

---

## 1. 先说结论

| 优先级 | 路线 | 理由 |
|---|---|---|
| P0 | RAG + 检索增强自蒸馏 | 已有实证 RAG 收益大，且不需要大 teacher |
| P0 | 高质量数据筛选 + QLoRA/MoRA | 成本低，直接改善 0.8B |
| P1 | Qwen3.8 离线 teacher 蒸馏 | 需要高 RAM/云，一旦拿到 teacher 数据收益潜力最大 |
| P1 | 自生成 + 过滤 + 自我训练 | 不依赖外部 teacher，适合推理/代码 |
| P2 | PERK/测试时 LoRA | 长上下文低成本适配 |
| P2 | 多 LoRA 合并 / 模型合并 | 把多个小域适配器合并成一份 |
| P2 | PLE 局部 n-gram 专家 | 仅当新门禁通过才投入 |
| P3 | 量化/CPU 推理优化 | 产品化，不明显提升智能 |

---

## 2. 详细路线

### 2.1 高质量数据筛选

在训练前最便宜、最容易见效的一步。

做法：

- 从 M1–M5 / distilled CoT 中：
  - 去除低质量、成人、噪音数据；
  - 保留 math / code / reasoning / long-context；
  - 去重；
  - 污染审计；
- 控制 mix ratio；
- 小模型训练数据质量比数据量更关键。

### 2.2 QLoRA / LoRA / MoRA / GaLore

在 8GB GPU 上：

- **QLoRA**：4-bit base + LoRA，能训练更久；
- **LoRA**：最简单；
- **MoRA**：同参数量更高秩，适合记忆/持续预训练；
- **GaLore**：全参数低显存更新；
- **ReLoRA + sMuon**：极小资源多轮累积。

建议：

1. 先用 LoRA/QLoRA 跑通；
2. 再对照 MoRA；
3. 不要一上来全参数。

### 2.3 RAG + 检索增强自蒸馏

当前最优低成本路线：

- teacher = 0.8B + RAG 上下文；
- student = 0.8B 无上下文；
- 训练目标：
  \[
  KL(P_{\text{teacher}}\|P_{\text{student}})
  \]
- 也可以直接用 RAG teacher 生成答案文本，做 SFT。

### 2.4 Qwen3.8-Flash-Next 离线 teacher 蒸馏

如果拿到高 RAM 机器或云：

1. 下载 Qwen3.8-Flash-Next；
2. 导出 teacher answers / logits；
3. 本地 0.8B 只做 LoRA 蒸馏；
4. 可按任务分：
   - math
   - code
   - long-context
   - general chat
5. 之后可做 OPD / Purified OPSD。

### 2.5 自生成 + 过滤 + 自我训练

不需要 teacher：

1. 用当前 0.8B 或 RAG-augmented 0.8B 生成候选答案；
2. 用规则/自洽性/验证器过滤：
   - math：数值验证；
   - code：运行测试；
   - knowledge：检索验证；
3. 用过滤后的数据做 LoRA SFT；
4. 迭代 2–3 轮。

### 2.6 PERK / Test-Time LoRA

长上下文/测试时适配：

- 在测试时用少量上下文训练临时 LoRA；
- 适合 long-context retrieval；
- 不改推理模型主权重。

### 2.7 多 LoRA 合并 / Compress-then-Merge

- 分别训练：
  - knowledge LoRA
  - math LoRA
  - code LoRA
  - long-context LoRA
- 用 compress-then-merge 合并成单 adapter；
- 比多模型低成本，保留多能力。

### 2.8 PLE 局部 n-gram 专家

只有在新门禁通过后：

- 低熵 token
- 代码补全
- 专名接续
- 数字/日期
- 作为 logit-level optional expert，不与 RAG/蒸馏冲突。

### 2.9 量化 / CPU 推理

- Q4/Q5 GGUF；
- ExecuTorch / llama.cpp / SGLang CPU；
- 对 0.8B 目标是 100 tok/s；
- 对智能提升帮助有限，但决定产品可用性。

---

## 3. 推荐组合策略

```text
数据质量筛选
    ↓
RAG self-distillation / Qwen3.8 offline teacher
    ↓
LoRA / QLoRA / MoRA 训练（数学、代码、通用）
    ↓
多 LoRA 合并
    ↓
RAG + optional PLE 局部专家
    ↓
量化 + CPU 100 tok/s serving
```

## 4. 在 8GB GPU + 15GB RAM 下最值得先做的三件事

1. **RAG self-distillation**：
   - 不需要大 teacher；
   - 已有 RAG 基础设施；
   - 能验证“检索能力能否蒸馏进 0.8B”。

2. **高质量 CoT 数据筛选 + LoRA SFT**：
   - 已有 `run_lora_distill.py`；
   - 直接用 distilled CoT / M 混合语料。

3. **建立多任务门禁**：
   - rare-kb / arithmetic / code-output / math / code / long-context；
   - 3 seeds；
   - 污染审计。

## 5. 相关参考

- [flash-next-8gb](https://github.com/lna-lab/flash-next-8gb)
- [Unsloth Qwen3.8-Flash-Next](https://unsloth.ai/docs/zh/mo-xing/qwen3.8-next)
- [MoRA](https://huggingface.co/papers/2405.12130)
- [PERK](https://github.com/eric11eca/perk)
- [Compress then Merge](https://github.com/ZhengbaoHe/compress-then-merge)
- [DRAG: Distilling RAG for SLMs](https://aclanthology.org/2025.acl-long.358/)
- [SlimMoE](https://ar5iv.labs.arxiv.org/html/2506.18349)
- [Hierarchical memory pretraining](https://huggingface.co/papers/2510.02375)
