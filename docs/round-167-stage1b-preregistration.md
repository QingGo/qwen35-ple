# Round 167 Stage 1b 预注册：read-out 的塌缩是初始化的、深度的，还是架构的？

> 日期：2026-09-13
> 状态：判定规则在**读取任何 PR / 深度数字之前**冻结。
> 说明：四个变体的**训练已经跑完**（队列自动执行），但本文件写成时
> **尚未读取任何一个 provenance 或 effective-depth 数字**。
> 前置：`docs/round-167-stage1a-effective-depth-results.md`、
> `docs/round-167-ultimate-goal-tech-debt-and-plan-v2.md` §3 TD-3 / TD-3a
> 队列：`scripts/run_round167_stage1b.sh`
> 分析：`scripts/round167_stage1b_analysis.py`

---

## 0. 被检验的三个理论预测

`round-165A` 测得读出把有效维数 **136.71** 的输入压成 **1.020**
（`value_proj` 14.387 → `branch_sum` **2.676** → `c_t` **1.020**）。
三条独立结果都预测**恰好这个配置**会塌缩：

| 来源 | 预测 |
|---|---|
| 官方 Engram 配置 `conv_zero_init: True`；本仓库另加 `out_proj` 零初始化 | 零初始化是塌缩的种子 |
| arXiv 2510.06954：小初始化 → condensation → **渐近秩塌缩** | 零初始化是"小初始化"的极限情形 |
| ICML 2026《The Implicit Bias of Depth》：深度带来**隐式低秩偏置**（低秩矩阵传播范数更高效） | 层数越多越容易塌 |

**而生产配方用的正是"零初始化 + 双层 MLP"读出**——三者共同预测必塌的配置。

## 1. 四个变体（其余一切相同）

同语料（`PURE_WIKI/tokens.npy`）、同步数（500）、同 LR、同 seed、同骨干、同注入层（layer 2）。

| 变体 | 初始化 | 读出深度 | 参数 |
|---|---|---|---|
| `prod` | 零初始化 | 2 层 MLP | `--out-mlp` |
| `nozero` | **真初始化** | 2 层 MLP | `--out-mlp --no-zero-init-out` |
| `linear` | 零初始化 | **1 层** | （无） |
| `linear-nozero` | **真初始化** | **1 层** | `--no-zero-init-out` |

> 注意：本队列**不是** round-162 设计。四个变体训练在**同一语料**上，
> 没有内容操纵。任何输出泄漏进 round-162 的判定表都是错误。

## 2. 测量

| 量 | 来源 | 位置 |
|---|---|---|
| `PR(value_proj)` / `PR(branch_sum)` / `PR(c_t)` | `round165_collapse_provenance.py` | **offset 12**（`e_t` 逐条内容丰富的那个位置，生产值 PR 136.71）与 offset 0 |
| `PR(e_t)` | 同上 | 同上（应≈136.7，作为健全性检查） |
| effective depth（三仪器） | `round167_effective_depth.py` | 答案位置 |

## 3. 反空转先决条件

* 若任一变体的 `PR(e_t)` 在 offset 12 **< 50**，该变体的数据不可信 → 该变体判 `INVALID`。
* 若 `prod` 变体的 `PR(c_t)` 明显不同于历史值 **1.020**（超出 ±0.05），
  说明重训没有复现基线 → 判 `BASELINE_NOT_REPRODUCED`，不出结论。

## 4. 预注册判定规则（在看数字之前冻结）

```text
基准 = PR(c_t) @ offset 12, 变体 prod            （历史值 1.020）

IF 反空转条件不满足
    => INVALID / BASELINE_NOT_REPRODUCED

ELSE IF 存在变体 v 使 PR(c_t)[v] >= 5.0 * PR(c_t)[prod]  且 PR(c_t)[v] >= 5.0
    => COLLAPSE_IS_RECIPE        （塌缩是训练配方的产物，不是架构极限）

ELSE IF max over v of PR(c_t)[v] < 2.0
    => COLLAPSE_IS_ARCHITECTURAL （四配置全塌 → 结构性）

ELSE
    => PARTIAL                   （逐变体报告，不合并叙述）
```

辅助读数（**不参与判定**，仅描述）：
* `PR` 在 `value_proj → branch_sum → c_t` 三级上各自被砍掉多少倍；
* 各变体的 effective depth，用于回答"修好 PR 之后深度读数是否随内容变化"。

## 5. 三种结局的含义（写定，避免事后解释）

| 结局 | 含义 | 对计划的影响 |
|---|---|---|
| `COLLAPSE_IS_RECIPE` | 换个初始化/深度就能把 PR 从 1 提到 ≥5 | **论文"不可修"的表述必须收窄**；Stage 1a 的 `DEPTH_NULL` 要改述为"读出缺陷掩盖了深度效应"；Stage 2/3 优先级下降，先修读出 |
| `COLLAPSE_IS_ARCHITECTURAL` | 四种配置全塌在 PR < 2 | "塌缩是结构性的"成立；`round-165B` 的"推理期修不好"升级为"训练期也修不好"；Stage 3（共训）的动机进一步加强 |
| `PARTIAL` | 部分变体部分缓解 | 逐变体报告；需要新的机制假设，不许合并叙述 |

## 6. 反方预测（诚实记录）

* **我预期 `PARTIAL` 或 `COLLAPSE_IS_ARCHITECTURAL`**。理由：`branch_sum` 的 PR 只有
  2.676，而它是 `out_proj` 的**输入** —— 塌缩在 `out_proj` 之前就已经发生了，
  所以只改 `out_proj` 的初始化/深度**未必**能把 `c_t` 的 PR 拉起来。
  若出现 `COLLAPSE_IS_RECIPE`，说明 `out_proj` 确实是主因，这与我此前的定位相反。
* **本实验的真实风险**：500 步、单 seed、batch 1 的配置下，不同初始化可能只是
  收敛速度不同而非可达秩不同。若 `nozero` 的 PR 明显更高但 loss 更差，
  必须同时报告 loss，不得只报 PR。

## 7. 规模与成本

| 项 | 值 |
|---|---|
| 骨干 | Qwen3.5-0.8B（fp32） |
| 变体 | 4 |
| 训练 | 500 步 × 4，实测约 4 分钟/变体 |
| provenance | offset 0,12；200 条 |
| 深度 | 200 条 |
| 预计 | 约 20 分钟（已含在队列内） |

**不做的事**：不调阈值、不加第五个变体、不在看到 PR 后更换 offset。
