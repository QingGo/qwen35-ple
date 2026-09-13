# Round 168 Stage 1.5e 预注册：互补性判据（"融合能否超过骨干"）

> 日期：2026-09-13
> 前置：`docs/round-168-stage1.5c-preregistration.md`（margin 分布）、
> `docs/round-167-design-corpus-selection-by-marginal-advantage.md`（§6.3 的 replace ≠ complement）
> **本文在任何真实数据的融合数值被计算之前写成。** 规则与阈值冻结于此，实现见
> `src/qwen35_ple/fusion_probe.py`。

---

## 0. 为什么必须做这一步：1.5c 的判据有一个类别错误

1.5c 测的是

```text
margin(t) = L_bb(t) − L_cnt(t)     ← "替换式"记忆的价值
```

而天花板论证关心的是

```text
I(Y ; w_t | h_t) = H(Y | h_t) − H(Y | h_t, w_t)   ← "互补式"信息
```

**两者不等。** 一个单独很弱的预测器完全可以携带骨干没有的信息。**四十年来的失败恰恰是
替换式框架造成的**——所有 reader 都被要求*复现*骨干的预测，而不是*补充*它。

因此：**1.5c 判 `EMPTY` 时不得关闭整条线；由 1.5e 裁决**（见 §5 对 1.5c 预注册的修订）。

---

## 1. 仪器：留出插值权重（Jelinek–Mercer）

$$\text{p\_mix}(w) = \lambda\, p_{bb}(w) + (1-\lambda)\, p_{cnt}(w)$$

关键性质：**p_mix 是合法分布（分区函数恒为 1）**，所以它在真实 token 上的概率就是两个标量
概率的凸组合。**因此这一步不需要任何 GPU/前向传播**——只需要 1.5c 已经存下的两个 NLL 数组：

```text
p_bb  = exp(−bb_lens)   （主，注入点 logit lens）或 exp(−bb_final)（次，端到端）
p_cnt = exp(−cnt_nll)
```

$\lambda$ 用 2 折**交叉拟合**（在另一折上选，再在被留出折上评）。

---

## 2. 陷阱一：收缩效应（shrinkage）——**这是本阶段最重要的发现**

因为 p_mix 在**概率空间**是线性的，恒有

$$\mathbb{E}[p\_mix(y)] = \lambda\,\mathbb{E}[p_{bb}(y)] + (1-\lambda)\,\mathbb{E}[p_{cnt}(y)]$$

**与联合分布无关。** 所以平均 NLL 的下降**不可能**来自边缘分布，只能来自 `log` 的凹性
对**联合形状**的敏感。而最大的那一类就是**重标定**：logit lens 分布过度自信
（很多位置 $10^{-3}$、少数位置 $0.9$），此时把它和**任何**中等的东西混合，都会抬高那些
极小的值，从而换来巨大的 Jensen 收益——**但没有发生任何信息传递**。

**实现这一模块时的第一次测试就撞上了它**：一个**纯独立噪声**的计数器（与骨干完全无关）
在 20000 个位置上给出 **raw gain = 1.32 nats**，而初版判定规则（以 raw gain 为准）
会把它标成 `MARGINAL`。**若不修，本阶段会稳定地产生假阳性。**

### 2.1 控制手段：置换零假设（permutation null）

置换 `p_cnt` **保持其边缘分布、破坏位置配对**，所以收缩部分的收益在置换后**依然存在**，
而互补部分的收益**消失**。因此：

```
决策量 = excess = gain(real) − gain(permuted null)
```

- `p_cnt` 为常数（纯收缩）⇒ excess **恰为 0**；
- `p_cnt` 在骨干崩溃处恰好救援 ⇒ excess **大**。

**零假设是保守的**：与"同边缘的独立副本"混合比与"相关的副本"混合更容易，所以在两个模型
一致的（常见）情形下，真实 excess 可能为负。因此 **excess ≤ 0 只能读作"没有超出独立重标定的
互补性"，不能读作"完全没有互补性"**。

### 2.2 陷阱二：单次置换的有限样本噪声与阈值同量级

同样在测试中撞上：单次置换的 gain 抖动约 **0.003 nats**，而冻结阈值是 0.005/0.02。
**因此零假设必须平均 `n_perm = 16` 次**，并要求

$$\text{excess} > 3\cdot\frac{\text{null\_std}}{\sqrt{n\_perm}}$$

（`noise_floor`）。这一条让"多跑几次置换"变成有回报的动作，也让单次置换无法凭噪声给出确信判定。

---

## 3. 冻结的判定规则

$$\text{excess} = \text{gain} - \text{null\_gain},\qquad \text{noise\_floor} = 3\cdot \text{null\_std}/\sqrt{n\_perm}$$

```text
IF excess <= max(0.005, noise_floor)                    => COMPLEMENTARITY_ABSENT
ELSE IF excess > max(0.02, noise_floor, 0.01)           => COMPLEMENTARITY_CONFIRMED
ELSE                                                    => COMPLEMENTARITY_MARGINAL
```

**主 frame = lens（注入点）**；final 一并报告。

阈值来历（与 1.5c 同源，先写下来）：本项目测到的嫁接效应量级是 0.001–0.01 nats；
**0.02 nats 的 excess 高于我们历史上任何一次观测**；0.005 是那个带的上沿。

### 3.1 分组：把"平均"拆成"哪一条频带"

设计文档的假设是**有用区间是一条窄带**，一个全局 $\lambda$ 会把它平均掉。
因此按 **trigram 上下文在计数训练流中的出现次数**（`COUNT_EDGES`，与 1.5c 完全相同的桶）
**每组各拟合一个 $\lambda$**（仍然交叉拟合），并报**等权 pooled excess**
（不让大组把小藏起来）。

---

## 4. 每个结局之后做什么（冻结）

| 结局 | 动作 |
|---|---|
| `COMPLEMENTARITY_CONFIRMED` | **替换式框架被证伪**。read-out 的目标改写为"复现该 $\lambda$ 融合 / 学残差方向"；进入 Stage 3，目标是**匹配计数模型的融合增益**（一个有明确数值的靶子） |
| `COMPLEMENTARITY_MARGINAL` | 先看**分组表**：若只有某几条频带有正 excess，Stage 3 改为**按该频带选样**；否则按 ABSENT 处理 |
| `COMPLEMENTARITY_ABSENT` | 线性融合族只买到重标定。**此时才允许把"记忆线"作为能力主张关闭**，并写明：这是"线性融合族 + 独立边缘零假设"下的结论，**log-linear（乘积）族更强，本结论不覆盖它** |

**无论哪个结局，都要同时报 oracle（逐位置最优 $\lambda$）**：若 oracle 能动而拟合 $\lambda$ 不能动，
缺的是"**何时该信记忆**"的预测器——这比一个零结果有用得多。

---

## 5. 对 1.5c 预注册的修订（**在任何 margin 数值被计算之前**）

`docs/round-168-stage1.5c-preregistration.md` §4.1 原写：

> `MEMORY_CEILING_EMPTY` ⇒ 关闭整条"冻结嫁接能力线"，不做 Stage 3。

**修订为**：

> `MEMORY_CEILING_EMPTY` / `NO_POSITIVE_POSITION` ⇒ **只关闭"替换式"路线**；
> 是否关闭整条线由 **Stage 1.5e** 裁决。理由：margin 是替换式价值，不是
> $I(Y;w_t|h_t)$（§0）。

**修订的时点声明**：写下这句话时，1.5c 的 `counts` 阶段已完成、但**任何 margin 数值都尚未计算**
（analyze 阶段还在等 GPU 锁）。已看到的只有计数模型自身的统计量
（wiki $L_{cnt}$ = 5.3759，与 round-158 逐位一致）。因此这是**先于数据的修订**，不是事后合理化。

---

## 6. 承限（现在写）

1. **族不够强**：线性混合弱于 log-linear 乘积族。`ABSENT` 只排除线性族。
2. **零假设保守**：见 §2.1。
3. **标量摘要的信息损失**：本仪器只用"真实 token 的概率"。如果 trigram 的信息表现为
   "**换一个 token 的排序**"而不是"抬高真 token 的概率"，标量摘要看不出来——这正是
   `cmi_probe`（含容量曲线）要补的位置，也是 §4 里"oracle 与拟合的差距"要指出的东西。
4. **计数代理偏悲观**：与 1.5c 相同；代理越弱，excess 越被低估（对"存在互补性"这一侧保守）。
5. **域面板**：wiki / stem / code 三个域都跑，若结论不一致必须报出来。

---

## 7. 复现

```bash
for tag in wiki stem code; do
  python scripts/round168_fusion_probe.py --tag "$tag" \
      --workdir outputs/round168/margin --n-perm 16
done
```

产物：`outputs/round168/margin/<tag>-fusion.{json,md}`。
**无 GPU、无训练**；依赖 1.5c 的 `<tag>-counts.npz` 与 `<tag>-backbone.npz`。
