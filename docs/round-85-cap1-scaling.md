# Round 85：CAP-1 扩展到 199 条 RAG self-distill + 39 条 held-out 评测

> 日期：2026-09-05  
> 状态：完成  
> 结论：在更大数据上，LoRA 和 QLoRA 都带来更明显的 held-out 正收益。

---

## 1. 数据规模扩展

- 使用 `distilled_corpus_400k_with_cot-filtered.jsonl` 作为 source 和 corpus；
- 新增加 `--exclude-source` 防止检索到自身答案，避免作弊；
- 生成 199 条 RAG self-distill 数据；
- 切分：
  - 训练：160 条
  - held-out 评测：39 条

文件：

```text
data/cap1-rag-distill-200.jsonl
data/cap1-rag-distill-160.jsonl
data/cap1-rag-distill-eval39.jsonl
```

---

## 2. 训练

### LoRA 160 / 50 步
- 输出：`outputs/cap1-lora-160`
- trainable：540,672
- 训练完成

### QLoRA 160 / 50 步（4-bit NF4）
- 输出：`outputs/cap1-qlora-160`
- bitsandbytes 可用
- 训练完成

---

## 3. Held-out 39 条评测结果

| Model | mean answer logprob | Δ vs base |
|---|---:|---:|
| base | -1.33702 | — |
| LoRA-160 | -1.24270 | **+0.0943** |
| QLoRA-160 | -1.25187 | **+0.0851** |

相对提升：

- LoRA：约 **7.05%**（0.0943 / 1.337）
- QLoRA：约 **6.36%**（0.0851 / 1.337）

---

## 4. 解读

- 从 10 条 held-out 的小样本提升（约 +0.005）扩大到 39 条后，提升幅度变得显著；
- LoRA 略优于 QLoRA，但 QLoRA 显存占用更低，适合 8GB GPU 部署/训练；
- RAG self-distillation 数据确实让 0.8B 在同类数学题答案建模上有可测提升；
- first-token hit 仍为 0，说明评测指标还需扩展到精确匹配/生成质量。

---

## 5. 下一步

1. 继续扩大数据到 500–1000 条并对更多领域（code/math）训练；
2. 用完整多任务评测（知识/算术/代码输出）验证泛化；
3. 将训练后的 LoRA 与 PLE/RAG 混合系统联合评测；
4. 尝试 MoRA 或高秩参数化替代（当前 peft 无 MoRAConfig）；
5. 延长训练 + 更稳定超参。

---

## 6. 一句话

> CAP-1 已从“入口跑通”推进到“可测的 held-out 能力提升”：39 条评测上 LoRA +0.0943 logprob、QLoRA +0.0851 logprob。
