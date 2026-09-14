# Round 169 Stage B1 预注册:只重训行(冻结骨干)

> 日期:2026-09-14
> 前置:`docs/round-168-ultimate-goal-tech-debt-and-plan-v3.md`(TD-11 存活版)、
> `docs/round-168-stage1.5g-preregistration.md`(阶③仪器)
> **本文在任何重训数值被计算之前写成。**

---

## 0. 这一级问什么

TD-11 的碰撞版本已被 1.5f 证伪(寻址无损)。**存活版本是「梯度预算错配」**:

> 冻结行是在预训练里形成的,那时梯度**按 token 频率**分配,而不是按**这个骨干的需要**分配。
> 所以行内容不是"这个骨干最想要的",而是"语料里最常出现的"。

嫁接实验(L0)无法证伪这一点,因为嫁接根本没有给行任何梯度。

**B1 直接测它**:冻结骨干与冻结 reader,只把梯度回传到**行**,
看行内容能否被**这个骨干的损失**重塑。

这一级**不需要预训练语料**,用现有的 PURE_WIKI 即可,因为问的不是"规模行为",
而是"梯度能不能移动行"。

---

## 1. 设计:冻结初始化本身就是对照组(配对设计)

这是本级的核心,也是最强的部分:

```text
1. 取 PURE_WIKI 的 667,404 个不同三元组
2. 从 47.7 GiB 表里取它们的 e_t,记作 E0   ← 冻结状态的快照
3. E := E0,只训练 E;骨干与 reader 全程冻结
4. 在**留出流**上评测,并且**只取三元组上下文出现在训练流里的位置**
   => 同一批位置,同时有 E0(冻结)与 E*(训练后)两条读数
   => 不需要 real/control 配对臂:初始状态就是控制
```

**为什么必须限定「seen」位置**:留出流里 60.9% 的位置其三元组在 PURE_WIKI 中
**从未出现**(实测 `context_counts == 0` 有 701,882 / 1,152,891)。这些位置取不到训练过的行,
混进来只会稀释效应。可用位置约 **450,009** 个,统计功效充足。

**这不是一个漏洞**:把它写成"未训练行"的天然对照,单独报告(见 §4)。

---

## 2. 冻结的超参数

| 项 | 值 | 理由 |
|---|---|---|
| 骨干 | `Qwen3.5-0.8B`,**全冻结** | 与 1.5c/1.5e/1.5g 同 |
| reader | `outputs/round162-0.8B-nosft/reader-wiki-seed0.pt`,**全冻结** | 见 §2.1 修正 |
| 注入层 | `layer=2` | 全项目一致 |
| 训练流 | `data/phase1/PURE_WIKI/tokens.npy`(990,455) | 与计数基线、候选集同源 |
| 评测流 | `data/phase1/wikitext-heldout-decon/tokens.npy` | 与 1.5c/1.5f/1.5g 同 |
| 评测位置 | 1.5g 的对齐集 ∩ `context_counts >= 1` | 同一根轴 |
| 可训练参数 | `E ∈ R^(667404 × 2560)` | = 16 头 × 160 维拼接 |
| 优化器 | AdamW,lr ∈ {1e-3, 1e-4},wd=0 | **两档 lr 即容量曲线的粗版** |
| epoch | 1 与 4 | 容量曲线 |
| 序列长度 | 2048,batch-tokens 8192 | 与既有 GPU 运行同 |
| 种子 | 0,1,2 | |

---

## 2.1 修正(在任何训练数值产生之前写下)

**原文冻结的 reader 是 `data/official_ple_reader.pt`,这个选择让本级在结构上不可能成立。**

实现时先跑了冒烟(5 步,`--max-steps 5`),读数是:

```text
grad_rms = 0.000e+00      changed rows 0/667,404
```

不是"很小",是**精确的 0**。原因是 `OfficialSourceQwenReader.from_official_checkpoint`
的默认参数 `zero_init_out=True`,而 `out_proj` **不在**从官方 checkpoint 加载的张量清单里
(`load_source_state` 只加载 `key_proj` / `value_proj` / 三个 norm / `conv1d`)。实测:

| 张量 | `data/official_ple_reader.pt` | `round162-0.8B-nosft/reader-wiki-seed0.pt` |
|---|---|---|
| `out_proj` | **norm = 0.0(精确)** | `out_proj.0` norm 19.67,`out_proj.2` norm 2.56 |
| `key_proj` | 70.45 | 有 |
| `value_proj` | 39.14 | 有 |

读出恒等于零 ⇒ `∂L/∂e_t ≡ 0` ⇒ **行永远不动,与行内容无关**。
这正是预注册 §3 里 `GRADIENT_BLOCKED` 分支存在的理由,但它这次是**配置**造成的,
不是读出的架构性质造成的,所以可以修正而不是终止。

**修正**:reader 改用 `outputs/round162-0.8B-nosft/reader-wiki-seed0.pt`
(registry 格式 `qwen35-ple-reader-v1` / `official_source_qwen_v1`),
即 round-162/165/167 这一线读出实验实际使用的、**输出投影已训练**的 reader。

**这次修正不改任何阈值、不改判定规则、不改位置集**;它只把一个结构上不可能的配置
换成一个可能的配置。§3 的 `GRADIENT_BLOCKED` 分支保留:若换了 reader 之后梯度仍为零,
那个结局照原样成立。

**顺带记下一条陷阱**(应进 playbook):默认的 official reader 是**零输出读出**,
所以任何"通过它训练行/探测行"的实验都会**静默地**得到零梯度 ——
与 round-161 的"注入从未发生 ⇒ 所有臂逐位相同"是同一族错误。

---

## 3. 判定规则(冻结)

记

* `L_frozen` / `L_trained` = 冻结行 / 训练后行 在 **seen 留出位置**上的下一 token 交叉熵
* `Δ = L_frozen − L_trained`(正 = 训练有用)
* `SE` = 逐位置配对差的标准误
* `Δ_shuf` = **打乱目标**对照臂的同一个量(把下一 token 标签置换后训练行)

```text
IF  rms(g) < GRAD_FLOOR                          => GRADIENT_BLOCKED
ELSE IF Δ >= 0.05 AND Δ > 3*SE AND Δ_shuf <= 0.01 => ROWS_RESHAPEABLE
ELSE IF Δ <= -0.05 AND |Δ| > 3*SE                => ROWS_HARMED
ELSE                                              => ROWS_STUCK
```

**阈值来历(先写下)**:

* `0.05 nats` 沿用 `src/qwen35_ple/margin.py` 的 `VERDICT_FLOOR_NATS = 0.05` —— 本仓库既有的
  "可判读的最小效应"约定,不新造标尺;
* `3*SE` 沿用 1.5e 的噪声地板(`noise_floor = 3·null_std/√n_perm`);
* `Δ_shuf <= 0.01` 是**必须的对照**:没有它,"训练有用"可能只是"给行加了任何扰动都让
  损失下降"(例如 reader 的 gate 被推离饱和区),与内容无关 —— 与 Stage 1a 的
  `real ≡ shuf` 教训同型;
* `GRAD_FLOOR`:第一批量上 `rms(∂L/∂E)` 的绝对地板,取 `1e-8`。
  **低于它说明梯度被读出挡死**,这个结局的解读是"坏在读出",不是"行不可训练"。

### 3.1 每个结局之后做什么(冻结)

| 结局 | 动作 |
|---|---|
| `ROWS_RESHAPEABLE` | **TD-11 存活版被证实**:冻结行确实按频率而非按需要分配 ⇒ 共训(L2–L4)有理由;同时给 L4 的预注册写下"应当出现同类增益"的预测 |
| `ROWS_STUCK` | 行内容在**这个骨干的这个读出下**不可移动 ⇒ 共训的收益不能来自"重塑行内容",只能来自读出共适应或骨干让出计算 |
| `ROWS_HARMED` | 训练把行推坏了 ⇒ 说明冻结行**已经**接近这个读出能用的最优,负结果归因反过来指向读出 |
| `GRADIENT_BLOCKED` | 读出把梯度吃掉了 ⇒ **L1 无法回答 L1 的问题**,直接升级到 L2(放开 reader) |

---

## 4. 预注册的第二个预测(可证伪,且是本级最有信息量的部分)

TD-11 说错配来自"**梯度 ∝ 频率**"。那么增益应当是**频率的非单调函数**:

```text
高频上下文:  已经估得很准  => 增益 ~ 0
中频上下文:  最欠估        => 增益最大        ← 预测的峰在这里
低频/未见:   取不到训练过的行 => 增益 = 0(结构性)
```

**按 `context_counts` 分桶报告 Δ**(沿用 `COUNT_EDGES = (0,1,2,3,5,10,50,200,1000)`)。

* 若峰**不在中频**、而是随频率单调(或均匀)⇒ **TD-11 的频率机制解释被证伪**,
  即便 `ROWS_RESHAPEABLE` 成立,也说明增益来自别的东西。

这一条是先写下的,因此它是一个真正的检验,而不是事后叙事。

---

## 5. 必须同时给出的四读数(沿用会计法)

| 读数 | 本级怎么给 |
|---|---|
| **解码器族** | 阶③探针(线性 + MLP)在 `E0` 与 `E*` 上的 top-1,两个都给 |
| **容量曲线** | 1 vs 4 epoch、lr 1e-3 vs 1e-4(4 个组合) |
| **零假设超额** | `Δ_shuf` 臂(**不是**原始 Δ) |
| **天花板对照** | 同位置的 `count_trigram` NLL 与 top-1;以及 ≤3-token 的界 |

**外加两个闸**:可识别性(损失不得钉在顶/底)、寻址多样性(1.5g 已通过)。

---

## 6. 承限(现在写)

1. **只训 1.5% 的表**:667,404 / 3.2e8 行。所以本级测的是"**这些行**能否被重塑",
   不构成"整张表可训练"的证明。
2. **留出流里 60.9% 的位置取不到训练过的行**,它们不进主判定。
3. **基线是冻结骨干**,不是从零共训的骨干 ⇒ 即使 `ROWS_RESHAPEABLE`,
   也**不能**推出"共训一定更好";它只证明"冻结行不是最优的"。
4. **lr 未调**:两档是容量曲线的粗版,不是最优搜索。若两档都不动,
   结论是"在这个 lr 量级下不动",不是"不可训练"。
5. **行只被这一个读出评判**:读出塌缩(1.5c ④)意味着梯度可能被结构性地削弱,
   这正是 `GRADIENT_BLOCKED` 分支存在的理由。

---

## 7. 复现命令(实现后填)

```bash
# 1) 取 E0 快照(CPU + 行表 I/O,无 GPU)
python scripts/round169_row_snapshot.py --train-tokens data/phase1/PURE_WIKI/tokens.npy \
    --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows --out outputs/round169/E0-wiki.pt

# 2) 训练行(冻结骨干,GPU)
python scripts/round169_train_rows.py --arm real --epochs 4 --lr 1e-3 \
    --snapshot outputs/round169/E0-wiki.pt --out outputs/round169/E-real-e4-lr1e-3.pt

# 3) 评测(冻结骨干,GPU):Δ 与阶梯③探针
python scripts/round169_eval_rows.py --frozen outputs/round169/E0-wiki.pt \
    --trained outputs/round169/E-real-e4-lr1e-3.pt --aligned outputs/round168/rung3/positions-wiki.npy
```

**待实现**:三个脚本,均不存在。
