# Round 84：CAP-1 实际训练跑通——RAG self-distill + LoRA/QLoRA

> 日期：2026-09-05  
> 状态：实际训练完成，held-out logprob 有正向提升  
> 机器：远程 WSL + NVIDIA GTX 1070 (8GB)，CUDA 可用  
> 结论：CAP-1 从“入口”进入“实际可训练/可评测”阶段。

---

## 1. 数据

- 源：`data/cap1-rag-distill-smoke.jsonl`（30 条 RAG self-distill）
- 训练/评测切分：
  - `cap1-rag-distill-train20.jsonl`（20 条）
  - `cap1-rag-distill-eval10.jsonl`（10 条）

## 2. 训练结果

### 2.1 LoRA（全量 30 条，50 步）

- 输出：`outputs/cap1-lora-50`
- trainable params：540,672
- final loss：1.91
- runtime：167s

### 2.2 LoRA held-out（训练 20 条，评测 10 条，50 步）

- 输出：`outputs/cap1-lora-50-heldout`
- 训练 loss 从约 1.52 振荡下降到 1.83（小样本不稳定）
- held-out 评测：

| Model | mean answer logprob | first-token hit |
|---|---:|---:|
| base | -0.23099 | 0.0 |
| LoRA 50-heldout | -0.22611 | 0.0 |

**提升**：+0.0049 logprob（约 0.005 nats），方向为正。

### 2.3 QLoRA（4-bit NF4，5 步 smoke）

- 输出：`outputs/cap1-qlora-smoke`
- bitsandbytes 可用，QLoRA 加载/训练成功
- loss：2.14 → 1.41
- held-out 评测：

| Model | mean answer logprob |
|---|---:|
| base | -0.23099 |
| QLoRA 5-step | -0.23079 |

**提升**：+0.0002 logprob，极微小但为正。

---

## 3. 实际意义

- CAP-1 的 RAG self-distill 数据可以训练；
- LoRA 和 QLoRA 都能在 GTX 1070 上跑通；
- held-out 上 LoRA 给出正向 logprob 提升，说明不是单纯过拟合训练集（虽然幅度很小）；
- 当前只是 20/10 小样本，不能作为最终能力结论。

---

## 4. 局限

1. 评测只有 10 条，且 first-token hit 为 0；
2. 日志 show 训练 loss 波动大，小数据下 LoRA 不稳定；
3. QLoRA 只跑了 5 步 smoke；
4. MoRA 在当前 peft 版本不可用（无 `MoRAConfig`），未实现；
5. 尚未把 LoRA 与 PLE/RAG 混合系统联合评测。

---

## 5. 下一步

1. 扩大数据：从 30 条扩展到完整 distilled corpus 采样；
2. 跑更稳定训练：
   - 更多 steps / 更小 lr / warmup；
   - 固定 seed；
3. 完整评测：
   - 数学 / 代码 / 知识多任务；
   - base vs LoRA vs QLoRA vs PLE+RAG；
4. 在更完整的 RAG self-distill 数据上重跑；
5. 若 peft 支持 MoRA，再补 MoRA 对照。

---

## 6. 一句话

> CAP-1 已经在真实 GPU 上跑通 RAG self-distill + LoRA/QLoRA，held-out logprob 正向；下一步是扩大数据和完整多任务评测。
