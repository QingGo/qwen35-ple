# Round 167 Stage 1a 预注册：PLE 嫁接是否改变骨干的有效深度？

> 日期：2026-09-13
> 状态：**在任何数字产生之前写定**。判定规则与仪器操作化在此冻结。
> 前置：`docs/round-167-ultimate-goal-tech-debt-and-plan-v2.md` §4 Stage 1
> 脚本：`scripts/round167_effective_depth.py`
> 测试：`tests/test_round167_effective_depth.py`（24 项，仪器方向性已钉死）

---

## 0. 为什么是这个实验

我们**全部**的负结果都测在**知识轴**上，而那条轴恰好是：

1. `round-157` §1.1 的可证明界说**必然为零**的那条；且
2. 官方 Engram 报告收益**最小**的那条（MMLU +3.4 vs BBH +5.0、ARC-C +3.7、HumanEval +3.0）。

Engram 自己的机制主张是别的：记忆 *"relieves the backbone's early layers from
static reconstruction, **effectively deepening the network**"*。这是一条关于
**有效深度**的主张，而本仓库里 `logit lens` 只以 TODO 的形式出现过（round-26 V185）。

**所以这个实验问的是：我们一直在 Engram 承认收益最小的轴上得到零，
那我们有没有在它声称收益所在的轴上测过？答案是从来没有。**

---

## 1. 仪器与操作化（冻结）

四件仪器取自 Csordás et al.（arXiv 2505.13898），有效深度的操作化取自 arXiv 2512.14064。
**在答案位置（prompt 最后一个 token）测量**，每条目一个值，跨条目取均值。

| 仪器 | 定义 | 有效深度的判定 |
|---|---|---|
| logit-lens KL | `KL(p_final ‖ p_layer)`，逐层 | **首个** `KL ≤ 0.5 · max(KL)` 的层 |
| top-5 overlap | 该层 top-5 与最终 top-5 的交集 / 5 | **首个** `overlap > 0.3` 的层 |
| 残差余弦（层） | `cos(h_{l+1} − h_l, h_l)` | **首个**"连续两层为正"的层 |
| 残差余弦（attn / mlp） | `cos(a_l, h_l)`、`cos(m_l, h_l + a_l)` | 仅作描述性报告，不参与判定 |

阈值 `0.5 / 0.3 / 连续两层` 在看数据之前固定。
余弦规则用"连续两层为正"而非滑动平均，是为了把滞后压到 ≤1 层 —— **滑动平均的 2 层滞后
有可能正好抹掉本实验要找的小幅深度位移**（`tests/test_round167_effective_depth.py`
为此单列了一项测试）。

## 2. 条件

| 条件 | 含义 |
|---|---|
| `off` | 不注入（`model._current_ple_e_t = None`，与 `run_phase0` 的 ple-off 同一开关） |
| `real` | 冻结的 wiki 行 |
| `shuf` | 同样的行做条目内置换（round-162 的对照） |

## 3. 反空转先决条件

**若 `mean TV(off, real) ≤ 1e-6`，判定为 `UNDERPOWERED`，不出任何结论。**
（round-161 的教训：注入钩子有两个静默 early return，整条臂会变成逐位相同的 no-op。）

## 4. 预注册判定规则（在看数字之前固定）

记 `gain_real = ED(off) − ED(real)`，`gain_shuf = ED(off) − ED(shuf)`。
**深度的"有效位移"定义为 ≥ 2 层**（1 层在 24–32 层的模型上不可与噪声区分）。

```text
IF mean TV(off, real) <= 1e-6
    => UNDERPOWERED                      （不出结论）

ELSE IF 至少 2 件仪器上 gain_real >= 2 且 gain_shuf < gain_real
    => DEPTH_FREED                       （Engram 的机制主张在冻结嫁接下复现）

ELSE IF gain_real >= 2 且 gain_shuf >= gain_real
    => PERTURBATION_ONLY                 （深度变化来自扰动，不来自内容）

ELSE IF 所有仪器 |gain_real| < 2
    => DEPTH_NULL                        （冻结加性嫁接在它被归功的机制上也是零）

ELSE
    => MIXED                             （逐仪器报告，不合并叙述）
```

## 5. 三种结局的含义（写定，避免事后解释）

| 结局 | 含义 | 对计划的影响 |
|---|---|---|
| `DEPTH_FREED` | 冻结嫁接**确实**释放了深度，只是不在知识轴上 | **目标重定位**为"深度释放 + 表面先验"；论文从负结果翻为正机制；Stage 2/3 优先级下降 |
| `DEPTH_NULL` | 冻结加性嫁接在 Engram 自己归功的机制上也是零 | 支持"必须共训"；Stage 3 获得强动机；论文的负结果**定位变准**（从"在弱轴上为零"变为"在它自己的轴上为零"） |
| `PERTURBATION_ONLY` | 深度变化来自注入扰动而非内容 | 与 round-162 的 `PERTURBATION_ARTIFACT` 一致；进一步支持 read-out 缺陷是约束 |
| `MIXED` | 仪器间不一致 | 逐仪器报告；不写合并结论 |

## 6. 反方预测（诚实记录，便于事后核对谁对）

* **我预期 `DEPTH_NULL`**。理由：`round-165A` 已测得答案位置的 `e_t` 是**逐字节常量**
  （每任务 1/200 条不同行），所以"内容"在答案位置不存在；而有效深度是在答案位置测的。
  若这里出现 `DEPTH_FREED`，那说明深度效应**不需要逐条内容**、只需要"注入了这个形状的向量"，
  这将是一个比机制主张更强的结果。
* **该实验的真实风险**：如果深度效应确实由"注入任意 PLE 形状向量"驱动，
  那么 `shuf` 会与 `real` 等强 → 判为 `PERTURBATION_ONLY`，而这**不能**证伪 Engram
  （官方是共训的，主干会把注入纳入计算）。这条限定必须写进结果文档。

## 7. 规模

| 项 | 值 |
|---|---|
| 骨干 | Qwen3.5-0.8B / 2B / 4B |
| 条目 | `data/qa-standard/eval-600b.jsonl`，前 200 条 |
| 读数位置 | 答案位置（prompt 末 token） |
| 层 | 24（0.8B/2B）/ 32（4B） |
| 预计成本 | 每骨干约 200 次前向 × 3 条件 ≈ 数分钟 |

**不做的事**：不调阈值、不加仪器、不在看到结果后更换测量位置。
若三件仪器方向不一致，按 §4 判 `MIXED`，不做多数投票。
