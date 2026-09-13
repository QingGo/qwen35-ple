# Round 168 Stage 1.5 结果：天花板测出来了，而且互补性不存在

> 日期：2026-09-13
> 预注册：`docs/round-168-stage1.5c-preregistration.md`（含 §4.2 修订）、
> `docs/round-168-stage1.5e-preregistration.md`
> 实现：`src/qwen35_ple/{margin,fusion_probe}.py`、`scripts/round168_{margin_distribution,fusion_probe}.py`
> 产物：`outputs/round168/margin/<tag>-{margin,fusion}.{json,md}`（wiki / code / stem）
> **回答**：这个通道里到底有多少可读的东西？它是不是骨干所没有的？

---

## 0. 结论（三句话）

1. **预注册规则在三个域上都判 `MEMORY_CEILING_ROOM`**（注入点 lens 框架，最好可实现降幅
   0.70 / 0.87 / 0.84 nats）。**但那个 `ROOM` 是探针偏置，不是记忆的性质**（§2）。
2. **端到端（final）框架是 `THIN` / `EMPTY` / `THIN`**：一个**替换式**记忆的上界是
   wiki **0.091 nats**（≈3.2% ppl）、code **~0.02**、stem **~0.05**。
3. **互补性测试：六个（框架 × 域）组合全部 `COMPLEMENTARITY_ABSENT`，excess 全为负**
   （−0.003 到 −0.390）。**而且如果没有那个置换零假设，每一次都会报出正的"增益"**
   （raw 最大 8.60 nats）。

**按 1.5c §4.2 的修订（`EMPTY` 只关替换式路线，整条线由 1.5e 裁决）：
1.5e 判 ABSENT ⇒ 能力线可以关闭。**

---

## 1. 逐域数字

### 1.1 1.5c：`margin = L_bb − L_cnt`（全部位置可评，99.95% 两阶段对齐）

| 域 | 框架 | mean L_bb | mean L_cnt | mean margin | median | 正 margin 占比 | 预注册判定 |
|---|---|---|---|---|---|---|---|
| wiki | **lens** | 13.9726 | 5.3760 | **+8.5966** | +8.8437 | 0.9549 | `ROOM` |
| wiki | **final** | 2.8298 | 5.3760 | **−2.5462** | −1.8263 | 0.1834 | `THIN` |
| code | **lens** | — | — | — | — | — | `ROOM`（0.8723 @ q=20%）|
| code | **final** | — | — | — | — | — | **`EMPTY`** |
| stem | **lens** | — | — | — | — | — | `ROOM`（0.8380 @ q=20%）|
| stem | **final** | — | — | — | — | — | `THIN` |

**oracle 增益曲线（wiki，realizable 池 = 上下文计数 ≥ 5）**：

| q | n | mean margin | corpus 降幅（lens） | corpus 降幅（final） |
|---|---|---|---|---|
| 1% | 2,522 | +20.92 / +11.73 | 0.0458 | 0.0257 |
| 5% | 12,610 | +18.36 / +5.61 | 0.2009 | 0.0614 |
| 10% | 25,219 | +17.17 / +3.66 | 0.3757 | 0.0801 |
| 20% | 50,437 | +15.93 / **+2.08** | 0.6975 | **0.0911** |
| 50% | 126,092 | +13.74 / +0.57 | 1.5031 | 0.0913 |

`L_cnt` 的 wiki 均值 **5.3760** 与 round-158 的 **5.3759** 逐位一致（管子对准了）。
`final` 框架的 oracle 降幅在 q≈20% 之后**饱和**（0.0911 → 0.0913），
因为再往外扩的位置 margin 已经 ≤ 0。

### 1.2 1.5e：互补性（留出插值权重 + 置换零假设，n_perm=16）

| 域 | 框架 | raw gain | **null gain** | null_std | **excess** | 判定 |
|---|---|---|---|---|---|---|
| wiki | lens | 8.6015 | 8.6084 | 0.0002 | **−0.0069** | `ABSENT` |
| wiki | final | 0.0763 | **0.4664** | 0.0008 | **−0.3901** | `ABSENT` |
| code | lens | 8.7960 | 8.7990 | 0.0003 | **−0.0030** | `ABSENT` |
| code | final | 0.0163 | 0.1319 | 0.0009 | **−0.1156** | `ABSENT` |
| stem | lens | 8.2095 | 8.2168 | 0.0016 | **−0.0073** | `ABSENT` |
| stem | final | 0.0474 | 0.1467 | 0.0032 | **−0.0993** | `ABSENT` |

**读法（这是全篇最重要的表）**：

- **raw gain 全部为正**（lens 上 8.2–8.8 nats！）。若按初版规则（以 raw 为准），
  我们会报告"发现 8.6 nats 的互补信息"。
- **置换零假设给出更大的增益**（final 上 null 0.13–0.47 > raw 0.02–0.08）。
  也就是说：**把计数器和一个同边缘的独立副本混合，比把它和真实的骨干配对更好。**
  这是 §2.1 预告的"零假设保守"，在这里表现为 **excess 大幅为负**。
- **机制解释**：骨干分布与计数分布在**位置层面正相关**（都在同样的"容易位置"上自信），
  所以两者冗余；而独立副本带来的是**纯重标定**收益。
- 因此：**trigram 携带的信息对骨干是冗余的，不是互补的。**

---

## 2. 必须记录的**预注册设计缺陷**：lens 框架是偏置探针

1.5c 预注册把 **lens（注入点 logit lens）设为判定用的主框架**，理由写得很明确：
> "这才是界的不等式所写的 `h_t`。后层能从 `h_t` 算出的信息本来就不需要记忆。"

**理由错在把"logit lens 的输出"当成了"`h_t` 里有什么"的合理代理。实测否证：**

```text
mean L_bb(lens) = 13.9726 nats      >   ln(248320) = 12.4249 nats   ← 均匀分布
```

**layer-3 的 logit lens 比均匀分布还差**（它对 248k 词表是"自信地错"）。
也就是说：**lens 是一个反信息的读出**，它**低估** `h^{l3}` 里的信息，
因而**高估**"记忆能补多少"。三个域**全部**出现 `lens = ROOM` + `final = THIN/EMPTY`，
这就是同一个偏置的三次复现，不是三次独立发现。

**修订（在任何规模结论被引用之前）**：
* `lens` 框架的 `ROOM` 判定**不得**作为"有空间"的证据；
* 决策相关框架是 `final`（端到端），并由 1.5e 的互补性判据裁决；
* 若将来要重新使用注入点框架，**必须换一个非偏置的探针**
  （例如 `cmi_probe` 的容量曲线探针，而不是 logit lens）。

**这条修订不改变任何原始产物**：`<tag>-margin.json` 里的 pre-registered 判定原样保留
（`lens=ROOM`），本节只是**解释**它。

---

## 3. 由此得到的判定链

```text
1.5f: 寻址无损（三域精确枚举）        ⇒ 容量不是瓶颈
1.5c(final): 替换式上界 THIN/EMPTY    ⇒ oracle 上界 ≤0.091 nats（wiki）
1.5c(lens): ROOM                      ⇒ 【偏置，见 §2】
1.5e: 互补性 ABSENT（六/六）           ⇒ 连"组合两个预测器"也没有超额
--------------------------------------------------------------
⇒ 能力主张：关闭。理由是两台独立仪器 + 两个正确设计的零假设，
  而不是四十次模糊的尝试。
```

**这是本项目四十轮以来第一次可以"用证据关闭"而不是"用疲惫关闭"。**

---

## 4. 诚实的限定

1. **计数代理偏悲观 + 覆盖率约束**：wiki 上 **60.9%** 的位置其 trigram 上下文在
   990k 训练流里从未出现，只有 **21.9%** 满足 `ctx_count ≥ 5`。realizable 池因此很小
   （q=20% 时 50,437 个位置）。**更强的计数代理只会让 margin 更大**（对记忆更有利），
   所以负结论是稳的、正结论是弱的——方向对我们有利。
2. **WikiText 几乎肯定在预训练数据里** ⇒ `final` 的 2.8298 nats 偏乐观（记忆显得更没用）。
   code/stem 是对冲，**结论三域一致**。
3. **线性融合族弱于 log-linear 乘积族**：`ABSENT` 排除线性族，**不覆盖乘积族**。
   但注意 **raw gain 在 final 上只有 0.016–0.076 nats**，所以即使乘积族更强，
   它的活动空间也被 oracle 上界（0.091 nats）压住了。
4. **零假设保守**：excess 为负只说明"没有超出独立重标定的互补性"。
   但由于 raw 本身也远小于阈值，这里的负结论比零假设的保守性所能解释的更强。
5. **端到端口径是 NLL 而非 QA 准确率**：这台仪器测的是语言建模损失，
   不是"模型答对率"。QA 上的结论由 Stage 2.2 / round-166/167 覆盖。

---

## 5. Stage 2.2 顺带完成（同一轮）

12/12 评测 + 6/6 gate 全部完成，判决脚本（`scripts/round167_stage2_verdict.py`）给出：

```text
verdict = PARTIAL
O(scalar) = 0.5957    O(per_dim) = 0.5387    → 降幅 0.057（门槛 0.20，未达）
gain vs scalar = −0.0042      gain vs control = +0.0033
```

* **病复现了**（O(scalar) = 0.5957 > 0.5；但低于 round-146 报告的 0.77–0.98，值得记一笔）；
* **把标量归约换成逐维（门控条目 4 → 10,240）几乎不改变开启率**
  ⇒ **饱和不是标量归约造成的**，这从机制清单上划掉了"sum(-1) 是原因"这一条，
  与 1b 的"塌缩是架构性的"一致。
* `PARTIAL` 是冻结规则里的 catch-all；逐臂数字已写入
  `outputs/round167/stage2-gate/verdict.{json,md}`。

---

## 7. 追加：规模对比（用**能力相关**指标替换被撤回的那一个）

`docs/round-168-retraction-scale-trend-unidentifiable.md` 撤回了基于**格式描述子**
（对长度单调）的规模趋势。这里补上**同一台仪器**在 4B 上的读数——
**单次前向，无生成，因此没有生成上限**，指标是与长度无关的语言建模 NLL。

**三次单次前向**（0.8B / 2B / 4B，同一评测流、同一个计数模型；无生成，因此没有生成上限）：

| 规模 | 位置数 | 耗时 | 峰值 RSS | final mean L_bb | mean margin | **正 margin 占比** | **oracle@q=20%** | lens mean L_bb |
|---|---|---|---|---|---|---|---|---|
| 0.8B | 1,152,328 | 195.7 s | 5.02 GiB | 2.8298 | −2.5462 | **0.1834** | **0.09112** | 13.9726 |
| 2B | 1,152,328 | 216.4 s | 11.35 GiB | 2.6446 | −2.7315 | **0.1609** | **0.07879** | ~13.8 |
| 4B | 1,151,765 | 422.3 s | **24.19 GiB** | 2.5832 | −2.7927 | **0.1606** | **0.10257** | 13.6812 |

（`wiki{2b,4b}-counts.npz` 是 `wiki-counts.npz` 的副本——同训练流、同评测流，
计数模型**逐位相同**，复制是正确且省时的。）

**读法：两个量行为不同，所以"规模趋势"这个说法对一半、错一半。**

1. **"计数器占优的位置比例"单调下降并饱和**：0.1834 → 0.1609 → **0.1606**。
   这正是压缩论证预测的方向（更大的模型自己就会查 n-gram），
   而且这次是**与长度无关**的指标。**这是被撤回的那条主张的正确替代品。**
2. **语料级可用上界不下降**：0.0911 → 0.0788 → **0.1026**（非单调）。
   所以**替换式上界在三个规模上都停在 `THIN`（0.05–0.20 nats）**。
   两个效应（能赢的位置变少 / 赢的位置赢得更多）相互抵消。
3. **lens 框架在三个规模上都是 `ROOM`，且三个规模的 lens 都比均匀分布差**
   （13.97 / ~13.8 / 13.68，均 > ln 248320 = 12.42）——
   **§2 的探针偏置是规模不变的**，进一步确认 `ROOM` 是探针性质而非记忆性质。

**诚实结论**：**能声称"计数器占优的位置比例随规模下降且饱和"；
不能声称"可用上界下降"**（它不降）。这比被撤回的那条强得多：它有正确的指标、
三个规模点、没有生成上限。

**1.5e 在 4B 上（补充记录）**：lens raw 8.3047 / null 8.3095 / excess **−0.0047** → `ABSENT`；
final raw 0.0916 / null 0.4129 / excess **−0.3213** → `ABSENT`。
**加上 4B，互补性判据的完整记录是 8/8 全部 `ABSENT`，excess 全为负。**

**顺手记录一个运维事实**：4B fp32 的峰值 RSS 24.19 GiB 意味着
**`--chunk-tokens 1024` 是 24 GiB 卡上的极限**；2048 会 OOM。这写进复现命令。

---

## 6. 复现

```bash
# 计数阶段（CPU，秒级）
for d in "wiki PURE_WIKI wikitext-heldout-decon" "code PURE_CODE PURE_CODE-heldout-decon" \
         "stem PURE_STEM PURE_STEM-heldout-decon"; do set -- $d
  python scripts/round168_margin_distribution.py --stage counts --tag $1 \
    --train-npy data/phase1/$2/tokens.npy --eval-npy data/phase1/$3/tokens.npy \
    --workdir outputs/round168/margin; done

# 骨干阶段（GPU，实测 5,800 tok/s，峰值 RSS 5.0 GiB）
python scripts/round168_margin_distribution.py --stage backbone --tag wiki \
  --eval-npy data/phase1/wikitext-heldout-decon/tokens.npy \
  --workdir outputs/round168/margin --model <Qwen3.5-0.8B> --chunk-tokens 2048

python scripts/round168_margin_distribution.py --stage analyze --tag wiki --workdir outputs/round168/margin
python scripts/round168_fusion_probe.py --tag wiki --workdir outputs/round168/margin --n-perm 16
```

**GPU 记账**：wiki 1,152,328 位置 / 195.7 s；code 515,069 / 87.5 s；stem 44,555 / 8.1 s，
全部在 **100% 利用率**下完成（此前评测任务的 CPU 相位只占 29%，被这三个任务填满）。
