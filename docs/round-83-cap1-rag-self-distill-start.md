# Round 83：CAP-1 启动——RAG self-distillation 数据与 LoRA 训练入口

> 日期：2026-09-05  
> 状态：数据管线 + 训练入口完成；CPU 训练 smoke 未能在本回合跑完  
> 结论：CAP-1 已开始，下一步是在 GPU/高 RAM 或后台长跑中执行实际训练与评测。

---

## 1. 新增

### `scripts/build_cap1_rag_distill_data.py`

构建 RAG self-distillation 数据：

- 输入：teacher-style 求解语料；
- 用 BM25 检索相似问题/解答作为上下文；
- 输出：
  ```json
  {"category": "...", "problem": "...", "context": "...", "solution": "..."}
  ```

已生成：

```text
data/cap1-rag-distill-smoke.jsonl  (30 条)
```

### `scripts/run_lora_distill.py`

增强 `_format_example`：

- 若数据带 `context`，训练文本变为：

```text
Question: ...
Context: ...
Answer: ...
```

- 这使现有 LoRA 蒸馏 runner 可以直接吃 RAG self-distill 数据。

---

## 2. 尝试与限制

- 在本地 CPU 上用：
  ```bash
  python scripts/run_lora_distill.py --device cpu --steps 2 --max-length 128 ...
  ```
- 模型加载成功，LoRA trainable params = 540,672；
- 但单步前向/反向在 CPU 上极慢，本回合未能完成 smoke step；
- QLoRA 需要 `bitsandbytes`，当前 engram-peft venv 未安装；
- 因此实际训练需：
  - GPU 8GB 机器；
  - 或后台长跑 CPU；
  - 或安装 bitsandbytes 后走 QLoRA。

---

## 3. CAP-1 后续执行清单

1. 在 GPU 上运行：
   - `run_lora_distill.py`
   - 数据：`data/cap1-rag-distill-smoke.jsonl` → 扩展到更多源数据
2. 新增 QLoRA 路径：
   - `BitsAndBytesConfig(load_in_4bit=True)`
   - `PEFT LoraConfig`
3. 新增 MoRA 实验入口（如依赖可用）
4. 训练后跑多任务评测：
   - 知识 / arithmetic / code-output
   - 对比 baseline vs LoRA + RAG context
5. 将 CAP-1 输出接入最终混合系统：
   - base + RAG + PLE + teacher/distilled

---

## 4. 当前 CAP-1 完成度

| 项目 | 状态 |
|---|---|
| RAG self-distill 数据生成 | ✅ |
| LoRA 训练数据格式支持 context | ✅ |
| 小规模数据 smoke 产物 | ✅（30 条） |
| 实际 LoRA 训练完成 | ⏳ 待 GPU/长跑 |
| QLoRA/MoRA | ⏳ 待环境依赖 |
| 训练后评测 | ⏳ |

---

## 5. 一句话

> CAP-1 已从“计划”进入“可运行数据 + 可运行训练入口”阶段；实际训练需要 GPU 或长时后台资源。
