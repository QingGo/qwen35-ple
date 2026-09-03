# Round 34：理论指标实证——增量 R² 与正交化诊断

> 日期：2026-09-03
> 状态：首批实证
> 目标：验证 round 29–33 的理论预测是否与真实数据一致。

---

## 1. 理论预测回顾

根据完整证明：

1. 任意 reader 的增益上界是 \(I(Y;E\mid H)\)；
2. 线性 reader 的最优增益是 \(R\) 在 \(E_\perp\) 上的投影；
3. Value 使用 \(E_\perp=E-\Pi_H E\) 不损失线性增量信息；
4. 几何对齐既不充分也不必要。

因此我们应看到：

- \(\Delta R^2(Y;E\mid H)\) 应该是正的，但可能很小；
- \(\Delta R^2(Y;E_\perp\mid H)\) 应接近 \(\Delta R^2(Y;E\mid H)\)；
- 真实 reader 的 real−control 优势也应当很小，除非任务本身有强记忆需求。

---

## 2. 实验 1：next-token embedding 线性增量

用 PLE 记忆向量 \(E\)、Qwen layer-8 hidden \(H\)、下一 token 的输入 embedding \(Y\) 做岭回归诊断。

### 2.1 \(\lambda=1.0\)，2048 token

| 指标 | 数值 |
|---|---:|
| R²(H) | -0.209 |
| R²(H,E) | -0.199 |
| R²(H,E⊥) | -0.202 |
| ΔR²(E 给定 H) | +0.010 |
| ΔR²(E⊥ 给定 H) | +0.007 |

### 2.2 \(\lambda=0.1\)，2048 token

| 指标 | 数值 |
|---|---:|
| R²(H) | -0.720 |
| R²(H,E) | -0.614 |
| R²(H,E⊥) | -0.699 |
| ΔR²(E 给定 H) | +0.106 |
| ΔR²(E⊥ 给定 H) | +0.021 |

- 低正则时 raw E 的“增量”明显更大，但这更像高维过拟合，不是稳定信号；
- 高正则时增量很小但为正。

---

## 3. 实验 2：LM 梯度残差增量（更贴近理论）

用真实 backprop 信号：

\[
R_t = -\frac{\partial L}{\partial h_t}
\]

作为任务残差，测量 E 对 R 的增量线性可解释性。

### 3.1 \(\lambda=1.0\)，512 token

| 指标 | 数值 |
|---|---:|
| R²(H) | 0.314 |
| R²(H,E) | 0.316 |
| R²(H,E⊥) | 0.317 |
| ΔR²(E 给定 H) | +0.0024 |
| ΔR²(E⊥ 给定 H) | +0.0031 |

### 3.2 \(\lambda=1.0\)，1024 token

| 指标 | 数值 |
|---|---:|
| R²(H) | 0.232 |
| R²(H,E) | 0.238 |
| R²(H,E⊥) | 0.237 |
| ΔR²(E 给定 H) | +0.0058 |
| ΔR²(E⊥ 给定 H) | +0.0051 |

---

## 4. 与理论的一致性

1. **满足定理 A/B**：E 确实带来正的增量 R²，但非常小；
2. **满足定理 C**：\(E_\perp\) 保留了大部分增量（1024 token 时约保留 88%），几乎不损失；
3. **与实证 QA 结果一致**：
   - real−control 总体 logprob 只有 +0.10；
   - BoolQ 上 real−control 约 +0.47；
   - scale=1.0 时优势最大，2.0 时 control 反超；
   - 这些与“记忆增量信息很小”一致。
4. **不支持大规模 5M–20M**：当前任务残差中，PLE 能线性解释的比例只有约 0.2%–0.6%，不足以支撑直接放大训练规模。

---

## 5. 下一步

先做以下实验，再决定是否进入大规模：

1. **Oracle reader 上界**：训练一个 MLP\((H,E)\) 预测梯度残差，看非线性增量 R² 是否显著大于线性；
2. **PCA/PLS 压缩**：把 E 降到 32/64/128 维，再测 \(\Delta R^2\)，判断高维噪声是否稀释信号；
3. **高残差子集**：只在梯度范数大的 token 上测增量 R²，看记忆是否对“难 token”更有用；
4. **Key/Value 分工商**：使用 \(E_\perp\) 作为 value、对齐 key 作为 gate，重复 scale sweep。

---

## 6. 产物

```text
outputs/mech-incr-r2-lam1.json
outputs/mech-incr-r2-lam01.json
outputs/mech-grad-r2-lam1.json
outputs/mech-grad-r2-lam1-1024.json
```

脚本：

```text
scripts/mechanism_incremental_r2.py
scripts/mechanism_gradient_r2.py
```
