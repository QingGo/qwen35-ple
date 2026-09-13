# Round 167 Stage 0 结果：还债

> 日期：2026-09-13
> 计划：`docs/round-167-ultimate-goal-tech-debt-and-plan-v2.md` §4 Stage 0
> 状态：0.1 / 0.2 / 0.3 / 0.4 / 0.5 **全部完成**
> 产物：`outputs/round167/metric-nuisance-audit.{json,md}`、
> `src/qwen35_ple/metric_audit.py`、`scripts/audit_metric_nuisance.py`、
> `docs/unexecuted-controls.md`、`docs/live-claims.md`

---

## 0. 一句话

**Stage 0 最重要的产出不是把债清掉，而是 0.1 当场抓到了两个东西：
一个我写错的指标声明，以及论文里 0.8B 那一半格式论断的失效。**

---

## 1. Stage 0.1 —— 测量效度债（TD-1）

### 1.1 建了什么

| 组件 | 内容 |
|---|---|
| `src/qwen35_ple/metric_audit.py` | 配对分解 `d_i = c + b·l_i`（`d` = 臂差，`l` = 滋扰差），`c` = 等滋扰效应，`b` = 每 token 斜率，并给出 `nuisance_share = (raw − c)/raw` |
| `tests/test_metric_audit.py` | **64 项**。核心是**证伪测试**：`synth_length_only` 必须在 300 个种子上 100% 判为 `NUISANCE_SENSITIVE`；`synth_content_only` 必须 100% 判为 `SURVIVES_ADJUSTMENT` |
| `scripts/audit_metric_nuisance.py` | 在真实臂上跑，输出 JSON + markdown，`--strict` 在"测得混淆但未声明"时失败 |

### 1.2 判定规则的一个设计决定（值得记录）

只用"等滋扰效应在 2σ 内"作判据时，纯长度对照的检出率实测是 **95.47%**（1000 种子），
即普通 2σ 的 5% 漏检率。**对一道纪律闸门来说这太漏**，所以加了第二条判据：

> `nuisance_share ≥ 0.5` —— "你报的数里有一半以上是滋扰"。

这条既更可解释，又是**确定性**的。加入后：

| 判据 | 纯长度检出 | 纯内容保留 |
|---|---|---|
| 2σ（仅第一条） | 95.47% | 100% |
| **+ share ≥ 0.5** | **100%** | **100%** |

这条取舍写进了模块 docstring 与测试注释，不让它变成事后调参。

### 1.3 真实臂上的发现

`nuisance = n_generated`。三制度 × 五臂对 × 五指标（`outputs/round167/metric-nuisance-audit.md`）。

**（a）论文的 4B 格式论断稳固，且被校正加强**

```
4B-sft  ple-off|wiki  chat_scaffold_rate
        raw +0.9583   adjusted +0.9736   share -0.016   => SURVIVES_ADJUSTMENT
```

对应论文的 "4B 零注入 95.8% 脚手架 → 注入后 0.0%"。**长度校正后效应不但没削弱，
还略强**（斜率 −0.0005/token，长度解释不了任何部分）。这条**活**。

**（b）论文的 0.8B 格式论断失效**

```
0.8B-sft ple-off|wiki  chat_scaffold_rate
        raw +0.1953   adjusted -0.0306   share +1.157   => NUISANCE_SENSITIVE
```

19.53 pp 的"格式增益"**全部由生成长度解释**，校正后符号反转。
对应论文的 "0.8B chat_scaffold 0.195 → 0.000（p=1.3e−88）"。
**这条必须改写**（见 `docs/live-claims.md` D1）。

**（c）同一个指标在不同制度上判定不同 —— 框架不是在无差别报警**

`0.8B-nosft600 ple-off|wiki`：raw +0.1767 → adjusted +0.0987，share +0.441 → **存活**。
同一个 `chat_scaffold_rate`，三个制度三种判定，方向与 round-166 对 `prose` 的结论一致。

### 1.4 一个我自己写错的声明被当场抓住

初版 registry 把 `chat_scaffold_rate` 声明为 `requires_adjustment=False`，
理由是"per-item 格式标签不是累加计数"。审计在真实臂上三次反驳了它。机制很清楚：
**生成必须够长才有机会吐出标记**。

按纪律，我没有把声明改成"能过关"的值，而是**改声明并记录证据与机制**
（`METRIC_DECLARATIONS["chat_scaffold_rate"].note` 以 `REVISED` 开头，
`test_revised_declarations_record_the_evidence` 强制该字段存在）。

同样的过程也推翻了"位置 0 前缀对照是免疫的"这一猜想：
在自回归模型里**开头 token 与最终长度相关**，所以没有结构性免疫的指标。
**这正是本机制的真正结论**：审计的价值不是给指标发免检证，而是**逐对照量化**。

---

## 2. Stage 0.2 —— 未执行对照登记册（TD-4）

`docs/unexecuted-controls.md`：三份历史 TODO 合并去重，分栏
**未执行 12 项 / 已完成 14 项 / 已否决 4 项**，每项带成本与优先级。

关键结论：

* 未执行里**成本最低的三项（U1 effective depth、U11 比例扫描、U12 正交性审计）**
  恰好覆盖两个一级子目标；U11/U12 **不需要 GPU**。
* **U2/U3 在 Stage 0.3 之前不是"未做"，是"不可达"**（同址的 TD-3a）。
* **R1/R2 两条否决在事后看来站不住**：TokenMem 在诊断出 gate 饱和之前被否决；
  Memory Grafting 被记反（它的核心是**换 memory value 的内容**）。
  两条都附上了"能推翻该否决的实验"。

---

## 3. Stage 0.3 —— 配置空间可扰动（TD-3a）

| 改动 | 文件 | 契约影响 |
|---|---|---|
| `zero_init_out` 由字面量 → CLI 字段 | `src/qwen35_ple/reader_registry.py`、`scripts/run_phase0.py` | **纯新增**：默认仍为 `True`，历史队列与 checkpoint 行为不变 |
| `None` 视为"未设置"→ 回落到默认 | 同上 | 防止 `bool(None) == False` 静默翻转默认 |

`tests/test_reader_config_perturbable.py`（9 项）**双向**钉死：新旋钮可达 **且**
默认未变。读出的深度旋钮（`--out-mlp` / `--out-hidden`）本就贯通，无需改。

> 重要事实（Stage 1b 的前提）：**生产配方用的就是 `--out-mlp` + 零初始化**，
> 即一个**零初始化的双层 MLP**读出 —— 正是三条理论共同预测会塌缩的配置。

---

## 4. Stage 0.4 —— 文献更正（TD-5）

| 项 | 原记录 | 更正 |
|---|---|---|
| Memory Grafting | "读取接口，不改变 PLE 表" | 核心是**把 grafting model 的隐状态存为 memory value**；2.8B/100B 下 51.95(MoE)/52.43(Engram) → **53.86** |
| TokenMem | "不照搬 cross-attention channel；我们仍是 residual PLE" | 它治的正是 `round-146` 后来测出的病（gate 饱和常开 0.77–0.98）；专用通道 + **两阶段课程**，去掉第二阶段遵从度→0 |

两条都以"**能推翻该否决的实验**"形式记入 `unexecuted-controls.md` §C。

---

## 5. Stage 0.5 —— 活结论清单（TD-7 / TD-8）

`docs/live-claims.md`：每条结论带**配置四元组**
`{表冻结|可训练} × {主干冻结|共训} × {单层|双层} × {饱和|非饱和}`。

| 分栏 | 数量 | 要点 |
|---|---|---|
| A 有证明背书 | 3 | A1 的界**与四元组无关**；A3 明确界**只覆盖知识轴** |
| B 负结论 | 7 | 全部限定在 `F · F · 1 · S/U` |
| C 正结论 | 6 | **C4（4B 格式迁移，被校正加强）与 C2（`code>wiki` 三骨干存活）是最强的活正结论** |
| D 撤回/需重述 | 7 | 含 D1（0.8B 格式）、D5（"PLE 无效"作为方向级论断） |

---

## 6. Stage 0 出口检查

| # | 出口判据 | 状态 |
|---|---|---|
| 0.1 | 每个 lockedsuite 指标都有"只动滋扰变量"的测试并通过 | ✅ 64 项；`--strict` 干净 |
| 0.2 | 三份 TODO 合并、分栏、带成本 | ✅ |
| 0.3 | `zero_init_out` / 注入层 / out_proj 深度可配置（纯新增） | ✅ 读出初始化与深度已可扰动；**注入层数留待 Stage 2.1**（需改 forward，非纯配置） |
| 0.4 | 文献更正 + 每个否决附可推翻实验 | ✅ |
| 0.5 | `live-claims.md` + 范围四元组 | ✅ 论文摘要重述留待结果定稿后一并做 |

**测试**：本地 `261 passed, 24 skipped`；远端新增 95 项全过。
**GPU 实验**：Stage 0 期间**未启动任何 GPU 实验**（遵守 0.1 先行的纪律）。

---

## 7. 对 Stage 1 的直接影响

1. **Stage 1b 的假设现在有了具体形状**：生产读出是"零初始化 + 双层 MLP"，
   即三条理论共同预测的必塌配置；四个变体（`×{零初始化,真初始化} × {1层,2层}`）
   正是把这个预测变成可证伪。
2. **Stage 1a 的解释要多留一档**：若深度变化由"注入任意 PLE 形状向量"驱动，
   `shuf` 会与 `real` 等强 → 判 `PERTURBATION_ONLY`，这**不能**证伪 Engram
   （官方共训，主干会把注入纳入计算）。该限定已写进预注册 §6。
3. **论文需要两处改写**：0.8B 格式论断（D1）与"PLE 无效"的方向级表述（D5）。
