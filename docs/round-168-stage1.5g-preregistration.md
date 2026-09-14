# Round 168 Stage 1.5g 预注册：阶③对齐复测（行内容）

> 日期：2026-09-14
> 前置：`docs/round-168-stage1.5-results.md`（①②④已对齐）、`docs/round-161-table-probe-verdict.md`（旧 59%）、
> 现成仪器：`scripts/probe_table_next_token.py`（自带预注册判定规则）
> **本文在任何对齐数值被计算之前写成。**

---

## 0. 为什么必须做，以及我为什么拖了

能力阶梯的第③级（"冻结行里有多少可回收的下一 token 信息"）目前挂在
**round-161 的 ~59%** 上，而那个数字来自：

* **不同的位置采样**（该脚本自己的 `--n-eval 60_000`、自己的 seed）；
* **不同的候选集**（`--topk 5000`，不是 1.5c 用的全 token 集）；
* **不同的切分**。

而 ①②④ 现在都已经对齐到同一批位置上（1.5c/1.5e/1.5f 共用 1,152,327 个位置）。
**所以阶梯有内部空洞，而阶梯是能力线关闭之后论文剩下的主要贡献。**
拿"能力主张已死"当"阶梯可以缓"的许可证，是把因果关系弄反了。

**另一个必须记下的事实**：这一级**不需要 GPU**——它是
"冻结行 → 下一 token"的探针，纯 **CPU + 48 GB 行表随机访问 I/O**。
所以它既便宜又低风险，我之前用"高方差"来解释延后，**站不住**。

---

## 1. 仪器：复用，不新造

`scripts/probe_table_next_token.py` 已经实现了需要的一切，且**自带预注册规则**：

| 预测量 | 说明 |
|---|---|
| `majority_train_prior` | 训练流类先验（地板） |
| `count_bigram` | 显式计数 $P(y \mid t)$ |
| `count_trigram` | 显式计数 $P(y \mid t-1, t)$ ← **阶梯①在本级的参照** |
| `probe_raw_rows` | ridge 线性探针 $2560 \to K$，输入是 16 行拼接的 `e_t` |
| MLP 探针 | 可选（`--no-mlp` 关闭）→ **容量曲线**（TD-9 要求） |

它还有 `verify_causality`（经验复核"行只依赖前 3 个 token"）与污染检查。
**本级不新造仪器，只做对齐**——这是刻意的：新造仪器会引入新的解码器自由度，
正好是 TD-9 警告的东西。

---

## 2. 要对齐什么（冻结）

| 项 | 值 | 理由 |
|---|---|---|
| 位置集 | **1.5c 的同一批位置**：`wiki-counts.npz` ∩ `wiki-backbone.npz`，**1,152,327** 个 | 阶梯四级同轴 |
| 评测流 | `data/phase1/wikitext-heldout-decon/tokens.npy` | 与 1.5c/1.5f 同 |
| 行表 | `qwen38-rows`（47.7 GiB，128 shards），`scale=0.00019931793212890625` | 冻结表本体 |
| 切分 | **按位置连续 60/40**（前 60% 训探针，后 40% 评） | 与 1.5f 同约定；避免同一 trigram 跨半泄漏 |
| 候选集 $K$ | **5000（主，与 round-161 可比）** + **1000（敏感性）** | 比值对 $K$ 敏感，必须报 |
| 对照 | **shuffled rows**（行置换）+ `majority_train_prior` | **没有这个对照，"59%"不可解读** |
| 容量曲线 | 线性 + MLP（`--mlp-hidden 512`，报出宽度） | TD-9 四元组 |

### 2.1 要报的数（阶梯第③级的那一个）

$$\text{recovery} = \frac{\text{probe\_raw\_rows top-1}}{\text{count\_trigram top-1}}$$

**在完全相同的候选集与位置上**。这个数取代 round-161 的 59%，进入论文 Fig 1。

---

## 3. 冻结的判定规则

记 $R$ = recovery（同一 $K$、同一位置集），$S$ = shuffled-row 探针的 top-1。

```text
IF probe_raw_rows top-1 <= max(S, majority) + 0.005
    => ROWS_DEAD               （行内容没有可回收信号；阶③为空，损失全在④）

ELSE IF R >= 0.50
    => ROWS_CARRY_THE_TRIGRAM  （行忠实承载 trigram 先验；与 round-157 的"先验而非知识"一致）

ELSE IF R >= 0.20
    => ROWS_DEGRADED           （行有信号但明显弱于显式计数）

ELSE
    => ROWS_NEARLY_DEAD
```

阈值来历（先写下）：round-161 报 **~0.59**，所以 `0.50` 是"与历史一致"的下沿；
`0.20` 是"还有实质信号"的下沿；`+0.005` 是相对 shuffled 对照的可分辨增量
（与 1.5e 的 0.005 nats 门槛同量级，但这是准确率不是 nats）。

### 3.1 每个结局之后做什么（冻结）

| 结局 | 动作 |
|---|---|
| `ROWS_CARRY_THE_TRIGRAM` | 阶梯③填入对齐后的 $R$；论文的结论变成**"损失定位在④（读出）"**，且①→②→③ 都是有内容的 |
| `ROWS_DEGRADED` | 阶梯③填入 $R<0.5$；结论是**"行本身已经损失了一部分"**，④ 不是唯一瓶颈 |
| `ROWS_DEAD` / `ROWS_NEARLY_DEAD` | **阶梯③为空**：这是对"共训小表"路线的**正面否决**（连行内容都没有），并且**解释了为什么四十个零不需要归因于读出** |

**注意这一级不能反推的东西**：$R$ 高**不**意味着记忆有用——它只说明**行里有东西**。
"有没有用"已由 1.5c（替换上界 THIN/EMPTY）与 1.5e（互补性 8/8 ABSENT）裁决。

---

## 4. 承限（现在写）

1. **$R$ 依赖 $K$**：候选集越小，显式计数越弱、探针越容易接近它。**必须报 $K=5000$ 与 $K=1000$ 两个数**，
   若两者给出不同判定则以**更保守者**为准。
2. **探针是受解码器限制的**（TD-9）：线性探针的 $R$ 是**下界**；MLP 是更强的下界；
   两者都不构成"行里没有"的证明——所以 `ROWS_DEAD` 才要求**两个探针都不超过 shuffled 对照**。
3. **行表是 fp8**：量化误差是表的一部分，不是测量误差，因此不修正。
4. **探针的输入是 16 行拼接（2560 维）**，与生产 reader 看到的一致；本级的结论**不**适用于
   "只用某几个头"的变体。
5. **污染**：沿用脚本自带的检查；WikiText 在 Qwen 预训练数据里这件事影响的是**骨干**评价，
   对本级（行 → token）不适用，因为计数侧已在 round-158 去污染。

---

## 5. 复现命令（对齐层）

```bash
# 1) 从 1.5c 产物导出对齐位置集（确定性，无随机）
python scripts/round168_rung3_aligned.py --export-positions \
    --counts outputs/round168/margin/wiki-counts.npz \
    --backbone outputs/round168/margin/wiki-backbone.npz \
    --out outputs/round168/rung3/positions-wiki.npy

# 2) 复用现成仪器，在**该位置集**上跑（CPU + 行表 I/O，无 GPU）
python scripts/probe_table_next_token.py \
    --positions-npy outputs/round168/rung3/positions-wiki.npy \
    --train-tokens data/phase1/PURE_WIKI/tokens.npy \
    --eval-tokens  data/phase1/wikitext-heldout-decon/tokens.npy \
    --topk 5000 --mlp-hidden 512 \
    --out outputs/round168/rung3/probe-k5000.json
```

**待实现**：`scripts/round168_rung3_aligned.py`（导出位置集 + 汇总 $R$ + 应用 §3 规则）。
`probe_table_next_token.py` 需要**新增** `--positions-npy` 与 `--eval-tokens` 两个参数
（契约纪律：**只新增，不改既有语义**）；若它的采样逻辑无法在不改变默认行为的前提下接受外部位置集，
则对齐层改为**导出位置集后用其 `--seed`/`--n-eval` 复现同一子集**，并在结果文档里写明
这一步引入了多少位置偏差。
