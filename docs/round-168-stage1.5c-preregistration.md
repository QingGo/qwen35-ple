# Round 168 Stage 1.5c 预注册：边际优势（margin）分布

> 日期：2026-09-13
> 前置：`docs/round-168-ultimate-goal-tech-debt-and-plan-v3.md`（计划 v3，Stage 1.5c）、
> `docs/round-167-design-corpus-selection-by-marginal-advantage.md`（设计）、
> `docs/round-158-ngram-reference-frame.md`（计数参照系）、`docs/round-167-...-preregistration.md`（体例）
> **本文在任何 margin 数值被计算之前写成。** 规则与常数冻结于此，实现见
> `src/qwen35_ple/margin.py`（`MARGIN_QUANTILES` / `MIN_CONTEXT_COUNT` /
> `VERDICT_FLOOR_NATS` / `VERDICT_ROOM_NATS` / `VERDICT_Q_CEILING`）。

---

## 0. 要回答的问题

四十轮实验测的都是"**能不能把表读出来**"（受解码器限制，TD-9），
从没测过"**那里到底有多少可读**"（与解码器无关，TD-10）。
Stage 1.5c 用不需要训练、不需要解码器的量回答后者：

```text
margin(t) = L_bb(t) − L_cnt(t)
```

`L_bb` = 纯骨干在位置 t 的 NLL（**不挂 reader、不注入**）；
`L_cnt` = **trigram** 计数语言模型在同一位置的 NLL。

**为什么是 trigram，而不是更强的模型**：round-167 已证 PLE 行 id 只依赖 ≤3 个 token
（`k·n=12` 字符，`r=9`）。因此对**这个设计**的任何记忆，

```text
I(Y ; e_{t−r:t} | h_t)  ≤  I(Y ; w_t | h_t),   |w_t| ≤ 3 tokens
```

即 trigram 后验是**设计上**的天花板。计数模型是对它的估计。

**`margin > 0` 的含义**（要小心，不要过度声称）：单靠 trigram 就比单靠骨干预测得好，
所以 trigram 在这一点上**不与骨干状态冗余**。**`margin ≤ 0` 处处成立**的含义是强结论：
没有任何"把 trigram 的答案读出来替换预测"的记忆能改善任何位置，整条线可**免费关闭**。

---

## 1. 两个参照系，都必须报

| 参照 | 定义 | 为什么要 |
|---|---|---|
| **lens**（主） | 注入点残差流的 logit lens：`hidden_states[layer+1]`，`layer=2` | 这才是界的不等式所写的 `h_t`。后层能从 `h_t` 算出的信息**本来就不需要记忆** |
| **final** | 纯骨干最终层的 NLL | 端到端视角：嫁接实际要打败的数字 |

恒有 `L_bb^lens ≥ L_bb^final`，所以 **lens 是对记忆更有利的一侧**。
**判定用 lens**（更宽松的一侧）；final 一并报告，用于说明"即使按注入点的宽松口径也……"。

---

## 2. 冻结的常数

| 项 | 值 | 来源 |
|---|---|---|
| 阶数 | **3**（trigram） | round-167 的界 |
| `V_uni` | **248047** | round-158（= max token id + 1） |
| 平滑 | **mkn**（interpolated modified Kneser-Ney） | round-158/160 |
| 计数的训练流 | **与评测流同域**（见 §3） | 域匹配 |
| `min_context_count` | **5** | 设计文档条件 3（Nishida 2025 的行质量警告） |
| q 网格 | **1, 2, 5, 10, 20, 50, 100 %** | 冻结，事后不得增删 |
| 判定允许的最大 q | **20%**（`VERDICT_Q_CEILING`） | 50% 语料的记忆化不是语料级论证 |
| 频率桶边界 | **0,1,2,3,5,10,50,200,1000** | 冻结 |

### 2.1 oracle gain 的定义

```text
Δ(q) = (1/N_all) · Σ_{t ∈ top-q% by margin} max(0, margin(t))
```

按**全部位置**归一（不是按被选位置），所以不同 q 之间可直接比较：
单位是"整份语料每位置省下的 nats"。

**它是上界，但是哪一种上界**：Δ(q) 是"**替换式**记忆"（读出 trigram 的答案并用它）
的上界。它**不是**"**互补式**记忆"的上界 —— 两个预测器组合可以同时打败两者，
真正的上限是 `I(Y;w_t|h_t)`，那要靠 `cmi_probe`（TD-9 已示其解码器受限）。
**这一条必须写在结果文档里，不许省。**

### 2.2 两条曲线

* `all`：全位置池。
* `ctx_count>=5`（**realizable**）：先把 trigram 上下文在计数训练流中出现 <5 次的位置剔除，
  **再**在这个池里取 q 分位。因为这样的行不可能被估准，选它没有意义。
  判定用**这条**。

---

## 3. 域面板（三个，域匹配）

| 域 | 计数训练流 | 评测流 | 备注 |
|---|---|---|---|
| wiki | `PURE_WIKI` (990,455 tok) | `wikitext-heldout-decon` (1,152,891) | 与 round-158 的 5.3759 可比 |
| stem | `PURE_STEM` (2,314,464 tok) | `PURE_STEM-heldout-decon` | 小（~89k 位置），只看分布形状 |
| code | `PURE_CODE` (2,004,570 tok) | `PURE_CODE-heldout-decon` (1,030,673) | |

**为什么要有域面板**：WikiText 几乎肯定在 Qwen 的预训练数据里（round-158 §7.3），
骨干在那里是**乐观**的（记忆显得更没用）。三个域里若结论一致，
"没有正尾"就不是 Wiki 污染造成的；若只在 wiki 上为负，那必须说出来。

---

## 4. 判定规则（冻结）

```text
正尾 = share(margin > 0)                       （primary frame = lens）

IF 正尾 == 0                        => NO_POSITIVE_POSITION
ELSE IF best Δ_realizable < 0.05    => MEMORY_CEILING_EMPTY
ELSE IF best Δ_realizable < 0.20    => MEMORY_CEILING_THIN
ELSE                                => MEMORY_CEILING_ROOM
```

其中 `best Δ_realizable = max over q ∈ {1,2,5,10,20} of Δ`（realizable 池）。

**阈值的来历（写下来，免得事后合理化）**：

* 本项目测到的嫁接效应量级是 **0.001–0.01 nats**。所以 0.05 nats 已是它一个数量级以上。
* 0.8B 在这类文本上的 NLL 约 2–3 nats。**0.20 nats ≈ 8% ppl**，是肉眼可见的收益；
  0.05 nats ≈ 2% ppl，是"边界"。
* 两者都是**语料级平均值**，不是被选位置上的均值 —— 后者会被 q 放大。

### 4.1 每个结局之后做什么（也冻结）

| 结局 | 动作 |
|---|---|
| `NO_POSITIVE_POSITION` | **只关闭"替换式"路线**（见下方修订）。是否关闭整条线由 Stage 1.5e 裁决 |
| `MEMORY_CEILING_EMPTY` | 同上，但结论写成"替换式价值低于本项目可分辨的效应量" |
| `MEMORY_CEILING_THIN` | 先做 **1.5a**（用 ≫1M token 把计数代理练强，看 Δ 是否随代理变强而上升）再决定 Stage 3 |
| `MEMORY_CEILING_ROOM` | 启动 **Stage 3**：按 top-q% margin 选样共训小表，检验 PR / cos(real,shuf) / NEC |

**注意 1.5a 是单向的**：更强的计数代理只会让 `L_cnt` 更低、`margin` 更大（对记忆更有利）。
所以若弱代理下已判 `EMPTY`，1.5a 只能把结论往 `THIN` 推，**不可能**推出 `ROOM`；
反之若强代理下仍是 `EMPTY`，结论最硬。

---

### 4.2 修订（2026-09-13，**在任何 margin 数值被计算之前**）

本表初版把 `EMPTY` 写成"关闭整条线"。这是**类别错误**：`margin` 测的是*替换式*价值
$H(Y|h_t)-H(Y|w_t)$，而天花板论证关心的是*互补式*信息
$I(Y;w_t|h_t)=H(Y|h_t)-H(Y|h_t,w_t)$，两者不等。

**修订：`EMPTY` / `NO_POSITIVE_POSITION` 只关闭"替换式"路线；整条线是否关闭由
`docs/round-168-stage1.5e-preregistration.md` 的融合判据裁决。**

时点声明：写下这段时 counts 阶段已完成、**任何 margin 数值都尚未计算**
（analyze 仍在等 GPU 锁）；已看到的只有计数模型自身统计量（wiki $L_{cnt}$ = 5.3759，
与 round-158 逐位一致）。

---

## 5. 承限（现在写，不许事后删）

1. **代理偏悲观**：`L_cnt` 来自一个 1–2M token 的计数模型；真实可达的 trigram 后验
   （round-158 拟合渐近 NLL ≈ 4.94）更低。所以**负结论强、正结论弱**。
2. **replace ≠ complement**（§2.1）。
3. **Wiki 污染**：见 §3。
4. **两阶段的定位必须对齐**：`L_bb(t)` 与 `L_cnt(t)` 必须来自**同一批位置**。
   实现里 backbone 阶段在每个 chunk 的首位置留 NaN，analyze 阶段取两者都有限的交集；
   覆盖率与小节都写进 JSON（若不写，这个仪器唯一的静默失效模式就是错位）。
5. **`replace ≠ complement` 已由 1.5e 接手**（§4.2）。
6. **计数模型是 CPU 的、骨干是 GPU 分钟级**：本文档首版把 1.5c 说成"无 GPU"，
   **更正为"无训练、GPU 分钟级"**（见 `run_round168_margin.sh` 的 util 记账）。

---

## 6. 复现

```bash
# 每个域三次；counts 是 CPU（可与 GPU 任务并行），backbone 是 GPU 分钟级
for D in "wiki PURE_WIKI wikitext-heldout-decon" \
         "stem PURE_STEM PURE_STEM-heldout-decon" \
         "code PURE_CODE PURE_CODE-heldout-decon"; do
  set -- $D
  python scripts/round168_margin_distribution.py --stage counts \
      --tag "$1" --train-npy "data/phase1/$2/tokens.npy" \
      --eval-npy "data/phase1/$3/tokens.npy" --workdir outputs/round168/margin
  python scripts/round168_margin_distribution.py --stage backbone \
      --tag "$1" --eval-npy "data/phase1/$3/tokens.npy" \
      --workdir outputs/round168/margin --model <Qwen3.5-0.8B>
  python scripts/round168_margin_distribution.py --stage analyze \
      --tag "$1" --workdir outputs/round168/margin
done
```

产物：`outputs/round168/margin/<tag>-{counts,backbone}.npz` + `-margin.{json,md}`。
