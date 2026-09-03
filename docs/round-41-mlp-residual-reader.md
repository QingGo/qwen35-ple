# Round 41：MLP Value Reader 原型与残差监督实验

> 日期：2026-09-03
> 状态：原型已实现，结果需要进一步解释
> 理论背景：`docs/round-40-mlp-reader-theory.md`

---

## 1. 原型实现

新增：

- `src/qwen35_ple/reader.py`：`MLPValueReader`
  - 用 \(E_\perp=E-W_{he}H\) 作为 value 输入；
  - value 路径为 3 层 MLP + GELU；
  - 简单 gate。
- `src/qwen35_ple/reader_registry.py`：注册 `mlp_value_v1`。
- `scripts/run_phase0.py`：支持 `--reader mlp`。
- `scripts/train_mlp_value_reader.py`：用 LM 梯度残差 \(R_t=-\partial L/\partial h_t\) 监督训练 value 路径。

---

## 2. 训练结果

### 2.1 MLP 残差监督训练

- 数据：1024 token，layer 8；
- 验证 R²：
  - best = **0.288**；
  - final = 0.121（过拟合）；
- 保存的是 best-epoch权重；
- 与 Oracle MLP H+E⊥ 的 0.278 接近。

说明：
> 残差监督训练可以学到与 oracle 相当的 value 映射。

### 2.2 BoolQ 8 题 logit patching

| 条件 | mean answer logprob | mean entropy |
|---|---:|---:|
| no-reader | -9.35 | 1.19 |
| real | -8.95 | 1.55 |
| control | -8.91 | 1.57 |
| random | -8.94 | 1.54 |
| zero | -8.94 | 1.54 |

关键发现：

- MLP 残差 reader 显著改善 answer logprob；
- 但 **real / control / random / zero 几乎完全一样**；
- 说明当前原型注入的不是“真实 PLE 内容特有信号”，而更像一个**全局通用的 hidden 校正**。

---

## 3. 解释

### 为什么能提升但 real=control

可能的数学原因：

1. **`h_to_e` 可能把 H 的信息编码进了 \(E_\perp\)**，使得 value 主要响应 H；
2. 或者 value MLP 学到的映射主要预测“由 H 决定的大部分梯度”，而不是 E 专门新增的部分；
3. Gate 被固定为全开，缺少按 token 选择性；
4. 当前残差监督目标 \(R\) 中，E 的增量部分本来就很小（线性 ΔR²≈0.006，MLP ΔR²≈0.02），因此 real/control 差异在 logit 层面自然很小。

### 理论含义

- 单纯提高 value 的 R² 不足以保证 real−control 差异；
- 真正需要的是 **value 对 E 中“H 之外增量部分”的响应**；
- 需要同时训练：
  - value 与 \(E_\perp\) 的耦合；
  - gate 的选择性；
  - 或者一个显式“只对 E 特有信息响应”的正则。

---

## 4. 下一步

1. 检查 `h_to_e` 是否过度把 H 注入 \(E_\perp\)；
2. 训练时让 value 对 real/control/random 的响应差异变大：
   - 增加 real-vs-control 判别损失；
   - 或把 value 与 E 的耦合用显式条件化实现；
3. 恢复 gate 训练，允许模型选择“不用记忆”；
4. 用稀有 token 子集而不是全部 BoolQ 评测。

---

## 5. 产物

```text
scripts/train_mlp_value_reader.py
outputs/reader-mlp-residual.pt
outputs/train-mlp-residual.json
outputs/mech-logit-mlp-res-boolq8.json
docs/round-41-mlp-residual-reader.md
```
