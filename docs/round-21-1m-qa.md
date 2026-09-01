# Round 21：1M token QA exact-match 正式评测（WSL）

> 日期：2026-09-01
> 状态：已完成
> 结论：PPL 强正信号保持；小规模 QA exact-match 也出现 real > control > no-reader 的平均排序，但样本量不足以单独定论。

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-0.8B |
| 记忆表 | Qwen3.8-Flash-Next PLE / `qwen38-rows` |
| Reader | `OfficialSourceQwenReader` + 2 层 MLP bridge / out_proj |
| 注入层 | layer 8 |
| 训练 | 1M tokens live-store，500 steps，seq_len 128，lr 1e-4 |
| Seeds | 0 / 1 / 2 |
| 三线 | no-reader / real / control |
| QA | 9 题 TriviaQA / NQ / BoolQ 风格 |
| 指标 | answer log-likelihood + greedy exact-match（16 new tokens，归一化匹配） |
| 设备 | WSL + GTX 1070 |
| 输出 | `outputs/phase0-live1m-qa.json` |

## 2. PPL 结果

| 线 | val loss | PPL |
|---|---:|---:|
| no-reader | 2.9896 | 19.88 |
| control | 2.8776 | 17.77 |
| **real** | **2.8174** | **16.73** |

关键差距：

```text
real − control = −0.0602
real − no-reader = −0.1721
control − no-reader = −0.1120
```

与上一轮 1M PPL 基本一致，结论保持。

## 3. QA exact-match 结果

### 3.1 总体

| 线 | QA EM mean | 3 seeds |
|---|---:|---|
| no-reader | 44.44% | 44.4 / 44.4 / 44.4 |
| control | 48.15% | 44.4 / 33.3 / 66.7 |
| **real** | **51.85%** | 66.7 / 44.4 / 44.4 |

```text
real − control = +3.70pp
real − no-reader = +7.41pp
control − no-reader = +3.70pp
```

### 3.2 分 task

| task | no-reader | control | real |
|---|---:|---:|---:|
| TriviaQA | 100% | 77.8% | 100% |
| NQ | 33.3% | 55.6% | 55.6% |
| BoolQ | 0% | 11.1% | 0% |

### 3.3 每 seed / 每 line

| seed | no-reader | control | real |
|---|---:|---:|---:|
| 0 | 4/9 | 4/9 | 6/9 |
| 1 | 4/9 | 3/9 | 4/9 |
| 2 | 4/9 | 6/9 | 4/9 |

## 4. 解读

1. **PPL 信号稳健**：
   - real 2.8174，control 2.8776，no-reader 2.9896。
   - 3 seeds 下 real 全部优于 control。

2. **QA exact-match 给出方向性正信号**：
   - real 平均 51.85% > control 48.15% > no-reader 44.44%。
   - real 在没有低于 no-reader 的 seed，且 2/3 seeds 超过 control。
   - 差距主要来自 NQ：real 55.6% vs no-reader 33.3%。

3. **不能过度解读**：
   - 只有 9 题，种子级波动大。
   - seed 2 上 control 66.7% 反超 real 44.4%。
   - BoolQ 上 real 为 0%，control 反而有 11.1%，说明小样本噪声明显。

## 5. 下一步

1. 把 QA 集扩大到标准 TriviaQA / NQ / BoolQ 子集（建议每任务 ≥50 题）。
2. 跑 5M token 正式矩阵：
   - 3 seeds
   - real / control / no-reader
   - PPL + 扩大版 QA exact-match
3. 如果 5M 保持 real > control，则可进入 4B / SFT / CPU 100 tok/s 产品化阶段。
