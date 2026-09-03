# Round 42：MLP Value Reader 深入实验与关键负结果

> 日期：2026-09-03
> 状态：已完成
> 目标：验证 MLPValueReader 是否能真正利用 E 特有信息。

---

## 1. 实验摘要

我们实现了 `MLPValueReader` 并尝试了四种设置：

1. 可训练 h_to_e + value_mlp(E_perp)；
2. 固定 h_to_e + value_mlp(E_perp)；
3. 固定 h_to_e + value_mlp(H, E_perp)；
4. 在 3 的基础上做 differential injection：
   \[
   v_{\text{diff}} = v(H,E_\perp)-v(H,0)
   \]

---

## 2. 结果

### 2.1 可训练 h_to_e 的退化

- best val R² = **0.288**；
- 但 BoolQ patching real/control/random/zero 几乎相同；
- 依赖度诊断：
  - cos(real, zero) = 0.952；
  - cos(real, control) = 0.914。

说明：

> 可训练 h_to_e 把 H 信息“走私”进了 E_perp，value 主要响应 H，而不是 E。

### 2.2 固定 h_to_e + value_mlp(E_perp only)

- best val R² = **-0.001**；
- 说明：只用真正的 E_perp，几乎无法预测梯度残差。

这非常重要：

> 纯 E_perp 本身几乎没有可预测 R 的信息；
> 必须结合 H 才能形成有用的非线性条件预测。

### 2.3 固定 h_to_e + value_mlp(H, E_perp)

- best val R² = **0.275**；
- 恢复到了接近 Oracle 的水平；
- 说明正确的 value 输入应该是：
  \[
  \mathrm{MLP}(H,E_\perp)
  \]
  而不是 \(\mathrm{MLP}(E_\perp)\)。

### 2.4 Differential injection

- 只注入：
  \[
  v_{\text{diff}}=v(H,E_\perp)-v(H,0)
  \]
- BoolQ 8 题：
  - real：-9.29；
  - control：-9.32；
  - random：-9.18；
  - no-reader：-9.35。

结论：

> real 只比 control 好 0.03，random 反而最好。
> E 特有差异太小，且被随机噪声淹没。

---

## 3. 关键教训

1. **“预测 R 能力强”不等于“能区分 real/control”**；
2. **可学习的 h_to_e 会退化**：它会把 H 编码进 E_perp；
3. **E_perp 单独几乎不含 R 信息**，必须有 H 一起作为 value 输入；
4. **只注入 E 特有微分项，当前信号太弱**；
5. **contrastive real-vs-control 训练会导致发散**，需要更稳定形式。

---

## 4. 理论更新

数学上，最优 value 应该是：

\[
\Delta^*(H,E)
=
\mathbb{E}[R\mid H,E]-\mathbb{E}[R\mid H]
\]

这本身是一个**同时依赖 H 和 E** 的函数。

所以：

\[
\mathrm{MLP}(H,E_\perp)
\]

是更合理的 value 结构；
而：

\[
\mathrm{MLP}(E_\perp)
\]

是不充分的。

同时，E 特有部分的绝对信息量：

\[
\Delta R^2_{\text{MLP}}(E \text{ over } H) \approx 0.02
\]

相对于 H 本身的 R²≈0.25 仍然很小，因此 dfferential injection 被随机噪声淹没是可以预期的。

---

## 5. 下一步候选

1. 做 **rare-token 条件化 gate**：只在稀有 token 上放行 differential value；
2. 用 **更大训练集**（例如 20k–160k token）训练 value，可能提高 E 特有信号；
3. 用 **真实任务 loss**（QA/rare-token completion）而不是梯度 R² 作为训练目标；
4. 如果稀有 token 上仍无 real−control，则记录为当前 PLE 表对知识增强不足的负面证据。

---

## 6. 产物

```text
outputs/train-mlp-residual.json
outputs/train-mlp-residual-fixed.json
outputs/train-mlp-residual-concat.json
outputs/train-mlp-contrastive.json
outputs/mech-mlp-dependence.json
outputs/mech-mlp-dependence-concat.json
outputs/mech-logit-mlp-diff-boolq8.json
outputs/mech-logit-mlp-diff10-boolq8.json
docs/round-42-mlp-reader-experiments.md
```
