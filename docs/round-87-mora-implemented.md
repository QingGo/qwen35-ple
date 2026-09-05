# Round 87：MoRA 已实现并完成 160 条数据训练与评测

> 日期：2026-09-05  
> 状态：完成  
> 结论：MoRA 在 39 条 held-out 上取得当前最优 logprob，且 code-output 多任务提升显著。

---

## 1. MoRA 来源与集成

- 获取 peft-mora fork：https://github.com/kongds/MoRA
- 将 `peft-mora/src/peft` vendor 到本仓库：
  ```text
  vendor/peft-mora/src/peft
  ```
- 使用方式：
  ```bash
  PYTHONPATH=vendor/peft-mora/src \
    python scripts/run_lora_distill.py \
      --use-mora --mora-type 1 ...
  ```
- `run_lora_distill.py` 新增参数：
  - `--use-mora`
  - `--mora-type`（1/2/3/4/6）

---

## 2. MoRA 160 条训练

- 数据：`data/cap1-rag-distill-160.jsonl`
- 输出：`outputs/cap1-mora-160`
- trainable：540,570
- 50 步完成

## 3. 39 条 held-out 对比

| Model | mean answer logprob | Δ vs base |
|---|---:|---:|
| base | -1.33702 | — |
| LoRA-160 | -1.24270 | +0.0943 |
| QLoRA-160 | -1.25187 | +0.0851 |
| **MoRA-160** | **-1.23607** | **+0.10095** |

MoRA 在 held-out 上略优于 LoRA/QLoRA。

## 4. 多任务评测（30 题）

| Task | Base | LoRA-160 | MoRA-160 |
|---|---:|---:|---:|
| knowledge logprob | -1.979 | -2.079 | -2.019 |
| arithmetic logprob | -7.287 | -7.280 | -7.285 |
| code-output logprob | -14.250 | -14.134 | **-13.317** |

- MoRA 在 code-output 提升约 **+0.93 logprob**，明显优于 LoRA 的 +0.12；
- knowledge 退化也小于 LoRA；
- exact match 仍为 0/0/0，需更大正式评测。

---

## 5. 意义

- 原先的“MoRA 不可用” blocker 已解除；
- 已在本仓库 vendor 依赖，保证可复现；
- CAP-1 现在同时具备：
  - LoRA
  - QLoRA
  - MoRA
  - RAG self-distillation
  - held-out 与多任务证据

---

## 6. 下一步

1. 用更多 seed 平均 MoRA vs LoRA vs QLoRA；
2. 将 MoRA 与 PLE/RAG 混合系统联合评测；
3. 扩大正式生成评测（pass@k / exact match）。
