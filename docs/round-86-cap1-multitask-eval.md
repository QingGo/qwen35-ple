# Round 86：CAP-1 多任务评测——LoRA 在 code/arithmetic 提升，knowledge 略降

> 日期：2026-09-05  
> 状态：完成  
> 结论：RAG self-distill LoRA 在代码输出和算术 logprob 上有正向提升，但知识类 logprob 略降；需按任务 gate/混合。

---

## 1. 评测设置

- 新增：`run_multi_task_eval.py` 支持 `--adapter`
- 模型：Qwen3.5-0.8B
- Adapter：`outputs/cap1-lora-160`
- 数据：
  - knowledge：`data/rare-kb-v1.json` 前 10 条
  - arithmetic：10 条生成题
  - code-output：10 条生成题
- 生成：greedy，max new tokens 8
- 指标：答案 logprob、first-token hit、exact match

---

## 2. 结果

| Task | Base logprob | LoRA-160 logprob | Δ | Base exact | LoRA exact |
|---|---:|---:|---:|---:|---:|
| knowledge | -1.979 | -2.079 | -0.100 | 0.100 | 0.100 |
| arithmetic | -7.287 | -7.280 | +0.007 | 0.000 | 0.000 |
| code-output | -14.250 | -14.134 | +0.116 | 0.000 | 0.000 |

---

## 3. 解读

- **code-output 提升最明显**：logprob +0.116；
- arithmetic 有微小正向提升：+0.007；
- **knowledge 略降**：−0.100，说明 RAG self-distill 偏数学/代码语料，可能会轻微影响知识类建模；
- exact match 全部仍为 0（生成任务难，当前评测集太小且 greedy 不匹配长答案）。

### 启示

- CAP-1 的 RAG self-distill 更适合作为“代码/数学/格式能力”的 adapter；
- 对通用知识任务，需要任务条件 router/混合，或使用更均衡的数据；
- 不能宣称“全面增强 0.8B”，应表述为“在 math/code 相关任务上有可测提升”。

---

## 4. 后续

1. 扩大 knowledge/数学/代码正式评测集；
2. 训练更均衡的混合数据；
3. 与 PLE/RAG 联合，按任务路由；
4. 完成 MoRA 对照（当前 peft 无 MoRAConfig）。

---

## 5. 一句话

> CAP-1 已产生可复现的多任务证据：LoRA 提升 code/arithmetic，knowledge 略降；下一步应是任务条件混合而非单一 adapter 全量替换。
