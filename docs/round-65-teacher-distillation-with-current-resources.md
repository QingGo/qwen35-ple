# Round 65：用现有资源跑教师蒸馏

> 日期：2026-09-04
> 状态：已跑通离线 teacher-text LoRA 蒸馏 smoke
> 目标：说明在当前资源下，哪些教师蒸馏路线可行，如何执行。

---

## 1. 现有资源盘点

| 资源 | 状态 |
|---|---|
| Qwen3.5-0.8B | 本地/远程可用 |
| GPU | GTX 1070 8GB |
| PEFT / datasets / TRL | 已安装 |
| 现有语料 | M1–M5、rare-kb、wikitext、Alpaca、distilled CoT 语料 |
| Qwen3.8-Flash-Next 完整 teacher | **当前环境未确认可用** |
| 网络下载 teacher | 当前不可用/未验证 |

因此：

> **目前最可行的教师蒸馏不是“加载一个更大的 teacher 做 logit 蒸馏”，而是用现有数据做 teacher-text 蒸馏 / RAG-augmented self-distillation。**

---

## 2. 三条可行路线

### 路线 A：离线 teacher-text LoRA 蒸馏（已跑通）

含义：

- “teacher” = 已有的高质量 CoT/solution 文本；
- “student” = Qwen3.5-0.8B + LoRA；
- 训练目标：让 student 模仿 teacher 的答案格式和推理过程。

命令：

```bash
python scripts/run_lora_distill.py \
  --model data/models/Qwen3.5-0.8B \
  --data data/sources/distilled_corpus_400k_with_cot-filtered.jsonl \
  --output outputs/lora-distill \
  --steps 200 \
  --lr 1e-4 \
  --max-length 512 \
  --device cuda
```

已 smoke：

```text
examples=30
trainable params=540672
step 10 loss ≈ 1.76
saved adapter to outputs/lora-distill-smoke
```

特点：

- 实现简单；
- 不需要更大 teacher；
- 可以先用小数据验证。

### 路线 B：RAG-augmented self-distillation（推荐下一步）

含义：

- teacher = **同一个 0.8B 但输入中带 RAG 上下文**；
- student = **同一个 0.8B 但不带 RAG 上下文**；
- 训练目标：让 student 在无检索时也能模仿带检索的 teacher 分布。

数学上就是：

\[
\delta^*(y)=\log\frac{P_{\text{teacher}}(y|q,D)}{P_{\text{base}}(y|q)}
\]

这是 ReAugKD 思路，且不需要额外大模型。

实现要点：

1. 用已有 `HybridRetriever` 检索；
2. 对每个训练问题分别 forward：
   - teacher 分布：`context + question`
   - student 分布：`question`
3. 训练 LoRA 或 logit correction head；
4. 损失：
   \[
   KL(P_{\text{teacher}}\|P_{\text{student}})
   \]
5. 评估：RAG/无 RAG/蒸馏后 student 三条件。

### 路线 C：真 logit/ON-policy 蒸馏（需要 teacher 模型）

如果后续能拿到 Qwen3.8-Flash-Next 或一个可用 teacher，可以：

1. 离线 teacher logits；
2. 学生 forward 后计算 KL；
3. 再做 OPD：
   - 学生采样；
   - teacher 给逐 token 分布；
4. 长 CoT 使用 Purified OPSD 或只使用 OPD。

---

## 3. 怎么选

| 目标 | 路线 |
|---|---|
| 最快看到效果 | 路线 A：离线 CoT SFT |
| 想验证“RAG 知识能不能蒸馏进模型” | 路线 B |
| 想最大化能力迁移 | 路线 C，但需要 teacher 模型 |
| 想同时提升推理/代码/格式 | A 或 C |

---

## 4. 建议的第一步实验

1. 用 `data/sources/distilled_corpus_400k_with_cot-filtered.jsonl` 中 math/code 子集；
2. 跑 LoRA 蒸馏 100–500 步；
3. 用 `scripts/run_multi_task_eval.py` 评测：
   - 蒸馏前 vs 蒸馏后；
   - rare-kb；
   - arithmetic；
   - code-output；
4. 如果有效，再做 RAG-augmented self-distillation。

---

## 5. 产物

- `scripts/run_lora_distill.py`
- WSL smoke：
  - `outputs/lora-distill-smoke`
  - `data/teacher-distill-smoke.jsonl`
