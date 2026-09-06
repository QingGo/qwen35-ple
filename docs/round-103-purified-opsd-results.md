# Round 103：正式评测 + Purified OPSD 实跑结果

> 日期：2026-09-06  
> 状态：P1 关键实跑完成  
> 结论：Purified OPSD MoRA-80 在多数正式风格基准上优于 base，并在 CAP-1 held-out 上取得正提升；CI 已修复并通过。

---

## 1. CI 修复

- `da1c35c fix(ci): pass ruff 0.16 checks`
- 修复项：
  - 新脚本可执行位；
  - import sorting；
  - RUF046 / BLE001 / S112；
  - UP035 / RUF012 / TRY004 / RUF022；
- 本地用 ruff 0.16 验证通过；
- 当前 GitHub CI：**success**。

---

## 2. Purified OPSD 数据构建

```bash
python scripts/run_purified_opsd.py \
  --input data/cap1-rag-distill-160.jsonl \
  --output data/purified-opsd-train.jsonl
```

结果：

| 指标 | 数量 |
|---|---:|
| 输入 | 160 |
| 通过验证 | 138 |
| 拒绝 | 22 |
| 拒绝原因 | no numeric answer 18、no parseable code 3、syntax error 1 |

---

## 3. Purified OPSD 训练

使用过滤后的 138 条数据训练：

```bash
python scripts/run_lora_distill.py \
  --data data/purified-opsd-train.jsonl \
  --output outputs/cap1-purified-mora-80 \
  --steps 80 --use-mora --mora-type 1 --seed 0
```

训练 80 步完成，CAP-1 held-out 39 条：

| 模型 | mean answer logprob |
|---|---:|
| base | -1.3370 |
| 原始 MoRA-160 | -1.2361 |
| **Purified MoRA-80** | **-1.2522** |

Purified MoRA-80 相比 base 提升约 0.085 nats，略低于原始 MoRA-160，但只用 80 步且数据经过验证过滤。

---

## 4. 正式风格基准

| 基准 | base | 原始 MoRA-160 | Purified MoRA-80 | Purified vs base |
|---|---:|---:|---:|---:|
| GSM8K-like | -8.781 | -9.394 | **-8.492** | **+0.289** |
| MATH-like | -4.590 | -4.727 | -4.638 | -0.048 |
| HumanEval-like | -1.560 | -1.768 | **-1.464** | **+0.096** |
| MBPP-like | -1.496 | -1.569 | **-1.373** | **+0.124** |

解读：

- Purified OPSD 在 GSM8K-like / HumanEval-like / MBPP-like 上均优于 base；
- MATH-like 略微下降；
- 原始 MoRA-160 在这些合成基准上反而略差，说明 **验证过滤 + 更针对性的数据** 对正式风格任务更有效；
- 这支持 v4 判断：不能盲目扩大 self-distill，必须 Purified OPSD。

---

## 5. 后续

1. 增加多 seed：
   - Purified MoRA 80/160 步，seed 0/1/2；
2. 增加 QLoRA/LoRA 对照；
3. 把正式基准与 PLE per-task 融合联合评测；
4. 继续推进 CPU 100 tok/s / bundle / e2e。
