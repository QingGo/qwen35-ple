# Round 66：在有限 GPU 下运行 Qwen3.8-Flash-Next 作为 teacher

> 日期：2026-09-04
> 状态：调研完成，尚未在本机实际部署
> 结论：当前 8GB GPU + 15GB RAM 不足以完整加载 176B/6B MoE；需要一台高 RAM 机器或改用离线/API teacher。

---

## 1. Qwen3.8-Flash-Next 规模

- 总参数约 176B；
- 激活参数约 6B（MoE）；
- 上下文 256K；
- 附带 PLE n-gram/Engram 表；
- ModelScope 已开源。

这意味着：

> 即使激活参数只有 6B，完整权重仍需要大量内存/显存。

---

## 2. 在 8GB GPU 上运行的公开方案

### 2.1 llama.cpp / GGUF + CPU offload

参考 [lna-lab/flash-next-8gb](https://github.com/lna-lab/flash-next-8gb)：

- 实测：6.6 GiB VRAM；
- 需要约 **47.8 GiB RAM**；
- 速度约 **34–35 tok/s**；
- n-gram/PLE 表放在磁盘；
- expert 权重放在 CPU/内存。

参考 [Unsloth Qwen3.8-Flash-Next 本地运行文档](https://unsloth.ai/docs/zh/mo-xing/qwen3.8-next)。

### 2.2 vLLM / SGLang + disk-backed PLE

- vLLM 社区已有 disk-backed PLE offload 支持；
- 适合有较大 CPU RAM 的服务器；
- 对 8GB GPU 仍建议配合 CPU offload。

### 2.3 结论

8GB GPU **可以跑**，但前提是：

- 至少约 48GB RAM；
- 或者把 weight 放在 disk + CPU 流式读取，速度会更慢；
- 当前 WSL 环境约 15GB RAM，**不足以直接跑完整 Qwen3.8-Flash-Next**。

---

## 3. 我们项目该怎么用

关键思路：

> **teacher 推理和 student 训练可以解耦。**

不需要在本地 8GB GPU 上实时跑 teacher。

### 方案 A：离线 teacher logits / teacher answers

1. 在 ModelScope 下载权重；
2. 在另一台高 RAM 机器或云 GPU 上：
   - 用 vLLM / SGLang / llama.cpp 跑 teacher；
   - 批量生成我们训练集的 teacher 输出；
   - 或导出 teacher logits；
3. 把 teacher 数据保存到磁盘；
4. 回到本地 8GB GPU：
   - 训练 Qwen3.5-0.8B LoRA；
   - 使用保存的 teacher logits / text 做 KL 或 CE。

#### 下载 ModelScope 权重示例

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.8-Flash-Next
```

#### 导出 teacher 数据

对每个样本：

```text
question + retrieved context
        ↓
teacher model forward
        ↓
teacher_answer_text 或 teacher_logits
        ↓
保存 JSONL / npz
```

#### 本地 student 训练

```bash
python scripts/run_lora_distill.py \
  --model data/models/Qwen3.5-0.8B \
  --data teacher_outputs.jsonl \
  --output outputs/lora-distill \
  --steps 200
```

---

### 方案 B：ModelScope / 云 API teacher

如果无法本地跑，也可以：

- 使用 ModelScope 的在线推理 API；
- 或租一台高 RAM 云服务器；
- 只做离线 teacher logits 导出；
- 学生训练仍在本地。

---

### 方案 C：RAG-augmented self-distillation（完全不需要大 teacher）

如果暂时没有高 RAM 环境，最快路径是：

- teacher = 0.8B + RAG 上下文；
- student = 0.8B 无上下文；
- 蒸馏 RAG 的能力到 student。

这条路线当前资源即可跑。

---

## 4. 我们的落地顺序

1. 先确认能否获得一台 ≥48GB RAM 的机器；
2. 若无：
   - 先跑 RAG-augmented self-distillation；
3. 若有：
   - 下载 Qwen3.8-Flash-Next；
   - 导出 teacher logits/answers；
   - 本地训练 0.8B LoRA；
4. 评估：
   - 蒸馏前后多任务；
   - 特别关注 math/code/long-context。

---

## 5. 引用

- [flash-next-8gb](https://github.com/lna-lab/flash-next-8gb)
- [Unsloth 本地运行文档](https://unsloth.ai/docs/zh/mo-xing/qwen3.8-next)
- [Qwen3.8-Flash-Next GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)
