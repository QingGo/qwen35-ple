# Phase A 科学闭环：WSL 真实模型 1M real/control/3-seed 结果

> 本文件是 Phase A 的正式实验结果记录。
> 实验在 WSL 实机完成，使用真实 Qwen3.5-0.8B、真实 PLE 行表和 1M 真实语料 token。

## 1. 实验环境

- 主机：Windows + WSL2 Ubuntu
- Python：qwen35-ple `.venv`（Python 3.13）
- EngramDB Python：0.2.9（WSL 当前安装版本）
- 模型：`data/models/Qwen3.5-0.8B`（1.7GB safe tensors）
- PLE 行表：`/home/zeng/qwen38-rows`
- 语料：`data/wet-1m-tokens.npy`，取前 1,000,000 token
- 训练配置：
  - `--live-store`
  - `--steps 500`
  - `--seq-len 128`
  - `--seeds 0 1 2`
  - `--modes real control`
  - 另外单独跑 `no-reader` 3 seeds 作为基线
  - reader：official / simple

## 2. 结果汇总

| Arm | n_seeds | val_loss_mean | val_loss_std | val_ppl_mean |
|---|---:|---:|---:|---:|
| no-reader | 3 | 2.98956 | 0.00000 | 19.8771 |
| real PLE | 3 | 2.81671 | 0.00719 | 16.7217 |
| control shuffled | 3 | 2.87380 | 0.00174 | 17.7042 |

### 关键差值

| 比较 | Δ val_loss | 相对改善 |
|---|---:|---:|
| real − control | -0.0571 | -1.99% |
| real − no-reader | -0.1729 | -5.78% |
| control − no-reader | -0.1158 | -3.87% |

## 3. 解读

- **真实 PLE 有效**：
  - real 优于 control，说明不是“单纯注入额外向量”带来的收益；
  - real 优于 no-reader，说明真实 PLE 记忆特征对 1M 语料微调有正信号。
- **收益幅度中等**：
  - 相对 no-reader 约 5.8% loss 改善；
  - PPL 从 19.88 降到 16.72。
- **control 也优于 no-reader**：
  - 说明 reader/训练结构本身也有一定收益；
  - 但 real 仍然显著优于 control，真正的 PLE 语义信号存在。

## 4. 结论（Go/No-Go）

**建议：Go，继续放大到 5M–20M token。**

理由：

1. 真实 PLE 在 1M 规模、3 seeds 上稳定优于 control 和 no-reader；
2. 方差很小（real std ≈ 0.007），不是偶然；
3. 当前使用的是 Store-I live-store 路径；后续可使用 Store-P + access-order 进一步降低 I/O 影响。

## 5. 局限与后续

- 本次实验没有逐窗口 fetch timing 记录；读取性能数据另见 WSL Store-P lazy 基准（1M ≈ 23.9s）。
- 当前实验使用 store-I 直读；下一步可用 `--store-p-slot-index` + `--access-order` 复跑，确认磁盘路径不改变科学结论。
- 需要补充 QA exact-match / 长上下文评测后再做最终放大决策。

## 6. 产物

- `outputs/phase0-live1m-mlp500-3seed.json`
- `outputs/phase0-live1m-noreader-3seed.json`
- 原始日志：
  - `/home/zeng/live1m-mlp500-3seed.log`
  - `/home/zeng/live1m-noreader.log`
