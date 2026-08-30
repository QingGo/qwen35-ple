# Phase 0 WSL/GPU 正式三线结果（2026-08-31）

## 环境

| 项 | 值 |
|---|---|
| 主机 | Windows + WSL2 Ubuntu |
| GPU | NVIDIA GeForce GTX 1070 8GB（本次未显式使用 CUDA，跑在 WSL CPU/torch 路径） |
| Python | 3.12.3 |
| torch | 2.6.0+cu124 |
| transformers | 5.16.1 |
| engram-peft | `dc74c85` |
| qwen35-ple | `c7f2381` |
| 数据 | 从 Mac 拷贝的旧预计算 `qwen35-ple-data`（约 46k tokens，scale 用 0.0002） |
| 步骤 | 10 步 / seed，seq_len=64，lr=1e-4 |
| 分割 | 90% train / 10% val，固定位置分割 |

## 命令

### 简单 Reader

```bash
cd ~/qwen35-ple
.venv/bin/python scripts/run_phase0.py \
  --model data/models/Qwen3.5-0.8B \
  --features /mnt/c/Users/minam/qwen35-ple-data \
  --steps 10 --seq-len 64 \
  --seeds 0 1 2 \
  --modes no-reader real control \
  --output outputs/phase0-wsl.json
```

### 忠实 Enagram Reader（Phase 1）

```bash
cd ~/qwen35-ple
.venv/bin/python scripts/run_phase0.py \
  --model data/models/Qwen3.5-0.8B \
  --features /mnt/c/Users/minam/qwen35-ple-data \
  --steps 10 --seq-len 64 \
  --seeds 0 1 2 \
  --modes no-reader real control \
  --reader engram --zero-init-v \
  --output outputs/phase0-wsl-engram.json
```

## 结果（held-out loss）

### Simple Reader

| 线 | val_loss_mean | val_loss_std | val_ppl_mean |
|---|---:|---:|---:|
| no-reader | 3.794245 | 0.0 | 44.4447 |
| real | 3.794319 | 0.000083 | 44.4480 |
| control | 3.794269 | 0.000111 | 44.4457 |

### QwenEngramReader（zero-init）

| 线 | val_loss_mean | val_loss_std | val_ppl_mean |
|---|---:|---:|---:|
| no-reader | 3.794245 | 0.0 | 44.4447 |
| real | 3.794197 | 0.000026 | 44.4425 |
| control | 3.794188 | 0.000010 | 44.4421 |

## 解读

1. **Phase 0 协议已跑通**：3 seeds × 三线，一条命令可复现。
2. **当前数据没有 PLE 增益**：
   - simple reader：real > control > baseline，差异在 1e-4 量级；
   - engram reader：baseline ≈ control ≈ real，差异也在 1e-4 量级。
3. **这不能作为科学结论**：
   - 只用了 46k token / 10 步；
   - 数据还是旧预计算；
   - 还没接入 live Store 真实表；
   - 没有 QA 评测。
4. 下一阶段必须：
   - 把真实 PLE Store 放到 WSL/GPU 可访问位置；
   - 用 live DiskPleNGramEmbedding 训练；
   - 用至少 1M–5M token 语料；
   - 加入 PPL + QA 双评测。

## 下一步

- [ ] 将真实 PLE 行表或远程存储挂载到 WSL，验证 live 训练路径。
- [ ] 下载 / 准备 1M–5M token 语料。
- [ ] 跑 Phase 2 正式消融，不再使用旧预计算。
