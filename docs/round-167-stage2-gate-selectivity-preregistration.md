# Round 167 Stage 2.2 预注册：gate 选择性

> 日期：2026-09-13
> 状态：判定规则在产生任何数字之前冻结。
> 前置：`round-146`（诊断出 gate 饱和）、`round-167-stage1b-results`（塌缩的衰减结构）
> 代码：`src/qwen35_ple/reader.py`（`gate_mode`）、`scripts/run_phase0.py --gate-mode`
> 测试：`tests/test_gate_selectivity.py`（17 项）

---

## 0. 动机：这是一个**已经被观测到的病**

`round-146` §3 的诊断（原话）：

> "The current gate saturates open after SFT (mean **0.77–0.98**), which fits this
> failure mode" —— always-on injection 使 chat 的 TriviaQA **−0.032**、NQ **−0.028**。

结构原因在代码里可直接读到（`reader.py` 旧 forward）：

```python
score = (key_normed * query_normed).sum(-1, keepdim=True) / sqrt(d_source)
gate  = sigmoid(sharpen(score))        # [B, T, hc, 1]
```

**`sum(-1)` 把 2560 个维度压成一个标量**，所以每个分支只有一个旋钮来控制整个
2560 维 value 向量 —— **结构上无法"选哪些维度通过"**。`hc=4` 分支 ⇒ 每 token 只有 4 个旋钮。

## 1. 干预：只改归约方式，别的一律不动

```python
product = key_normed * query_normed
gate = sigmoid(sharpen(product / sqrt(d)))    # per_dim: [B, T, hc, 2560]
```

| 模式 | 归约 | 门控条目数/token |
|---|---|---|
| `scalar`（官方，默认） | `sum(-1)` | 4 |
| `per_dim` | 逐元素 | **4 × 2560 = 10,240** |

**唯一变量是归约方式**：不加偏置、不改初始化、不改结构、不改层、不改语料。
`scalar` 路径经 `test_scalar_gate_reproduces_the_official_formula` 与官方公式逐位对拍。
默认仍为 `scalar`（契约"只允许新增"）。远端全量测试 **391 passed / 10 skipped**，官方 golden 未破。

## 2. 实验设计

| 因子 | 取值 |
|---|---|
| gate 模式 | `scalar` / `per_dim` |
| 行 | `real`（wiki）/ `control`（同 reader，行置换） |
| seed | 0, 1, 2 |

其余全部对齐 round-162 配方（同语料、同步数、同 LR、同注入层 layer 2，mixed50）。
评测：`data/qa-standard/eval-600b.jsonl`，**raw 与 chat 两个模板**（chat 是 always-on 受害的制度）。

## 3. 反空转先决条件

* 若任一臂的 `ple_injected == 0` → `UNDERPOWERED`（round-161 的静默 no-op 教训）。
* 若 `scalar` 臂的 `gate_open_frac` 不在 round-146 报告的区间内（< 0.5）→
  说明重训没有复现那个病，**判 `DISEASE_NOT_REPRODUCED`**，不出结论。
  （这是必须的：如果 gate 本来就不饱和，这个干预就没有靶子。）

## 4. 预注册判定规则（在看数字之前冻结）

记 `O(mode)` = `gate_open_frac_all_entries`（**原始门控条目**中 > 0.5 的比例，
两种模式同一定义），`M(mode, arm)` = chat 制度下 TriviaQA 与 NQ 的均值。

```text
IF 反空转条件不满足
    => UNDERPOWERED / DISEASE_NOT_REPRODUCED

ELSE IF O(per_dim) >= 0.90
    => GATE_STILL_SATURATED        （换成 per-dim 也照样全开 → gate 不是机制）

ELSE IF O(per_dim) <= O(scalar) - 0.20
        AND M(per_dim, real) - M(scalar, real) >= +0.02
        AND M(per_dim, real) - M(per_dim, control) >= +0.02
    => GATE_SELECTIVITY_HELPS      （选择性恢复，且增益是内容相关的）

ELSE IF O(per_dim) <= O(scalar) - 0.20
        AND |M(per_dim, real) - M(scalar, real)| < 0.02
    => GATE_SELECTIVITY_NO_EFFECT  （确实更有选择性，但指标不动）

ELSE
    => PARTIAL                     （逐臂报告，不合并叙述）
```

阈值 `0.20 / 0.02 / 0.90` 在看数据之前固定。`+0.02` 与本仓库既有的
"产品结论要求 ≥0.05 nat / 2 点"一致（`round-153` §3.2）。

## 5. 三种结局的含义（写定）

| 结局 | 含义 | 对计划的影响 |
|---|---|---|
| `GATE_SELECTIVITY_HELPS` | gate 饱和**就是** always-on 伤害的原因，per-dim 修好了它 | **负结果首次出现"可修"的正面机制**；论文可加一节"选择性可恢复"；Stage 3 可与选择性叠加 |
| `GATE_SELECTIVITY_NO_EFFECT` | gate 更有选择性，但输出不变 | 饱和是**症状**不是**病因**；与 Stage 1b 的"塌缩在冻结投影"一致 → 动机进一步转向 Stage 3 |
| `GATE_STILL_SATURATED` | per-dim 也全开 | gate 不是变量；`round-146` 对 gate 的定位需要收窄 |
| `DISEASE_NOT_REPRODUCED` | scalar 臂不饱和 | `round-146` 的 gate 统计口径与重训配置不同 → 先对齐口径，不出结论 |

## 6. 反方预测（诚实记录）

* **我预期 `GATE_SELECTIVITY_NO_EFFECT` 或 `PARTIAL`**。理由：Stage 1b 已显示
  最大的一段衰减（6.53×）发生在**冻结的 `value_proj`**，而 gate 作用在它**之后** ——
  把 2560 维的 gate 打得再开闭，也救不回上游已经丢掉的信息。
  若出现 `GATE_SELECTIVITY_HELPS`，说明选择性对**输出分布**的影响可以独立于秩而存在
  （即 always-on 的扰动本身就是伤害，与信息量无关）。这与 round-162 的
  `PERTURBATION_ARTIFACT` 相容，值得单独记一笔。
* **本实验的真实风险**：per-dim gate 只有一次前向的统计可比性；
  若 `O(per_dim)` 因初始化而系统性偏低（例如 sigmoid 输入分布不同），
  那么"选择性"可能只是幅度差异而非结构差异。因此**必须同时报 `gate_mean`**，
  不得只报 open fraction。

## 7. 规模与成本（按 round-167 利用率标准设计）

| 项 | 值 |
|---|---|
| 训练 | 2 模式 × 2 臂 × 3 seed = **12** 次，各约 4 分钟 |
| 并发 | **3 路并发**（探针实测单臂约 5 GiB / 24 GiB，23% 利用率 → occupancy-bound） |
| 评测 | eval-600b，raw + chat |
| 目标利用率 | ≥ 50%（`scripts/gpu_util_probe.py --require 50`） |
| 预计 | 含评测约 1–1.5 小时 |

**不做的事**：不调阈值、不加第三模式、不在看到 open fraction 后更换评测集。
