# Round 100：P0 PLE 证据修复结果——真实局部任务 + 同域 bank + per-task 校准

> 日期：2026-09-05  
> 状态：P0 核心证据闭环完成  
> 结论：把 PLE 放到真正的局部任务、同域 bank 和 per-task 校准后，**代码续写和姓名拼写出现 real > control 的正收益**；数字格式仍未取得绝对正收益，应继续修正或关闭。

---

## 1. 实验设置

- 模型：`Qwen3.5-0.8B`
- 可选 adapter：`outputs/cap1-mora-160`
- 代码任务：仓库内 Python 源码 next-token；
- 姓名/数字任务：Wiki 文本 next-token（按首个字符分类）；
- PLE memory：
  - 代码 bank：代码文件 80% 训练 / 20% 评测；
  - Wiki bank：Wiki 文档 80% 训练 / 20% 评测；
  - 每组同时构建 **real** 与 **shuffled control**。
- Per-task 校准：在每个任务上单独网格搜索 `(scale, bias)`；
- 种子：0 / 1 / 2；
- 指标：next-token base logprob、n-gram logprob、fused logprob、top-1；
- 核心判据：
  \[
  \Delta_{\text{fused,real}}>0,\quad
  \Delta_{\text{fused,real}}-\Delta_{\text{fused,control}}>0
  \]

---

## 2. 基础模型 3-seed 结果

| 任务 | 真实融合收益 | 控制融合收益 | real-control 融合差 |
|---|---:|---:|---:|
| code | **+0.421** | +0.018 | **+0.402** |
| name | **+0.291** | +0.011 | **+0.279** |
| number | -0.022 | -0.914 | +0.893 |

逐 seed：

| 任务 | seed 0 real | seed 1 real | seed 2 real | seed 0 rc | seed 1 rc | seed 2 rc |
|---|---:|---:|---:|---:|---:|---:|
| code | +0.548 | +0.383 | +0.331 | +0.333 | +0.387 | +0.486 |
| name | +0.680 | **-0.979** | +1.170 | +0.672 | **-1.082** | +1.247 |
| number | -0.013 | +0.054 | -0.106 | +0.112 | +2.297 | +0.268 |

---

## 3. MoRA adapter 3-seed 结果

| 任务 | 真实融合收益 | 控制融合收益 | real-control 融合差 |
|---|---:|---:|---:|
| code | **+0.422** | +0.037 | **+0.385** |
| name | **+0.377** | +0.029 | **+0.348** |
| number | -0.010 | -0.896 | +0.886 |

MoRA 在 code/name 上与 base 接近，没有明显额外帮助。

---

## 4. 解读

### 4.1 代码续写：PLE 通过
- 真实融合相对 base 平均 +0.42 nats；
- control 只有 +0.02；
- real-control 融合差 +0.40；
- 所有 3 个 seed 均为正。

这证明：

> 当使用同域代码 bank + per-task 校准时，PLE 可以在真实代码续写上提供可测增益。

### 4.2 姓名拼写：均值通过，但 seed 不稳定
- 平均真实融合 +0.29；
- 平均 real-control +0.28；
- 但 seed 1 为负，表明当前 name bank/评测仍不稳定。

需要进一步：

- 扩大 name 样本；
- 使用更明确的实体拼写语料；
- 或增加更强 control；
- 目前可以允许 PLE 进入 name 任务，但要持续监测。

### 4.3 数字格式：未通过绝对正收益
- 真实融合均值 -0.02，接近零；
- control 更差，所以 real-control 为正；
- 但“正 real-control”不是“真实正收益”。

因此：

> 数字任务当前不应作为 PLE 的已证明优势领域，继续保留为待改进。

### 4.4 与上一轮 P0 对比

上一轮 P0 在“code-output Q&A / 算术计算”上 PLE 为负。  
本轮换成：

- 真实代码续写；
- 同域 bank；
- per-task 校准；

结果发生反转：

- code from 0 → +0.42；
- name from 0 → +0.29；
- number 仍无绝对正收益。

这验证了 v4 系统复盘的判断：

> PLE 之前没有正收益，不是因为 PLE 没有信息，而是因为任务、记忆 bank、gate、校准全部不匹配。

---

## 5. P0 完成情况

- [x] 真实局部任务评测：代码 next-token / name / number；
- [x] 同域 PLE bank：代码 bank + Wiki bank；
- [x] real vs control；
- [x] 3-seed；
- [x] per-task 校准；
- [x] PLE+MoRA 对比；
- [ ] HumanEval/MBPP 正式代码评测；
- [ ] 更强 name bank；
- [ ] number 正收益；
- [ ] 将 per-task 校准参数写入 serving router。

---

## 6. 下一步

1. 把 code/name 的 per-task 校准参数持久化到 `configs/ngram-fusion-router.json`；
2. 在 serving router 中按任务加载不同 `(λ,β)`；
3. 增加 HumanEval/MBPP next-token 评测；
4. 强化 name bank / 增加实体语料；
5. 对 number 任务暂时关闭 PLE 或继续寻找专用数字 bank；
6. 进入正式评测 + Purified OPSD。
