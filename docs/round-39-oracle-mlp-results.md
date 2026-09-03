# Round 39：Oracle MLP 非线性上界结果

> 日期：2026-09-03
> 状态：已完成
> 预注册：如果 MLP 显著高于线性，说明需要非线性 reader；如果接近，说明容量不是瓶颈。

---

## 1. 实验设置

- 数据：`ple-books-160k` 1024 token；
- 目标：真实 LM 梯度残差 \(R_t=-\partial L/\partial h_t\)；
- 模型：3 层 MLP，hidden=128，120 epochs；
- 对比特征：
  - H；
  - H+E；
  - H+E⊥；
  - H+PLS64(E)。

---

## 2. 结果

### 2.1 线性基线

| 特征 | R² | ΔR² vs H |
|---|---:|---:|
| H | 0.2324 | — |
| H+E | 0.2382 | +0.0058 |
| H+E⊥ | 0.2375 | +0.0051 |
| H+PLS64 | 0.2398 | +0.0074 |

### 2.2 MLP 上界

| 特征 | R² | ΔR² vs H |
|---|---:|---:|
| H | 0.2556 | — |
| H+E | 0.2762 | +0.0206 |
| H+E⊥ | **0.2784** | **+0.0228** |
| H+PLS64 | 0.2678 | +0.0122 |

---

## 3. 与预注册对照

### 结论：非线性显著增益

- 线性 H+E：ΔR² = +0.0058；
- MLP H+E：ΔR² = +0.0206；
- 约 **3.6 倍**。

### E⊥ 在 MLP 下反而最好

- MLP H+E⊥：ΔR² = +0.0228；
- 略高于 H+E 的 +0.0206。

这说明：

> 非线性 reader + E⊥ 去相关可能比当前官方 reader（线性 bridge + gate + ShortConv）更有潜力。

### PLS64 在 MLP 下低于全维度 MLP

- MLP H+PLS64：+0.0122；
- 低于 H+E 的 +0.0206。

说明：

- PLS 对线性模型有帮助；
- 但对非线性模型来说，64 维 PLS 可能丢弃了非线性可用的额外信息；
- 或许需要更大 r 或混合 PLS + 非线性。

---

## 4. 科学含义

1. **当前线性增量小，不代表记忆没有用**；
2. **非线性 reader 可以提取约 3–4 倍的增量信息**；
3. **E⊥ 去相关在非线性下仍是安全且更好的选择**；
4. **容量瓶颈确实存在**，当前 reader 可能不够非线性；
5. 下一步应实现：
   - 使用 \(E_\perp\) 的 Value 路径；
   - 加入非线性 MLP / 更深的 gate；
   - 或者保留 PLS 作输入特征，同时增加非线性。

---

## 5. 下一步实验

1. 实现 **MLP Value Reader**：
   \[
   \Delta = g(h,z)\cdot \mathrm{MLP}(z_\perp)
   \]
2. 对比：
   - 当前 official reader；
   - 线性 + PLS；
   - MLP + E⊥；
   - MLP + PLS；
3. 用稀有 token 子集和 BoolQ 退化作为主要指标；
4. 若 MLP Value 在稀有 token 上 real−control 提升，再进入更大规模训练。

---

## 6. 产物

```text
outputs/mech-oracle-mlp-1024.json
scripts/mechanism_oracle_mlp.py
docs/round-39-oracle-mlp-results.md
```
