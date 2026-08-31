# 第十九轮完整汇总：WSL Phase 0 / Live 路径 / 真实行表复制

> 时间：2026-08-31
> 范围：从“WSL Phase 0 正式三线”到“真实 qwen38-rows 复制到 WSL”
> 状态：本轮已沉淀，传输仍在进行

---

## 1. 本轮目标

1. 在 WSL/GPU 上跑正式 Phase 0：
   - 三线：no-reader / real / control
   - 3 seeds
   - 一条命令可复现
2. 用 live Store 路径替代旧预计算 e_t。
3. 用 `QwenEngramReader` 做正式 reader 实验。
4. 判断 WSL 磁盘空间是否足够，并将真实 `qwen38-rows` 完整复制到 WSL。
5. 为 Phase 2 的 1M / 5M token 实验铺路。

---

## 2. 本轮计划

- [x] 搭建 WSL/GPU 环境
- [x] 下载 Qwen3.5-0.8B 到 WSL
- [x] 拷贝 46k token 预计算特征到 WSL
- [x] 跑 Phase 0 三线 + 3 seeds（Simple Reader）
- [x] 跑 Phase 0 三线 + 3 seeds（QwenEngramReader）
- [x] 验证 live `DiskPleNGramEmbedding` 与预计算一致
- [x] 检查 WSL 磁盘空间
- [x] 启动真实 PLE 行表完整复制
- [ ] 等待复制完成并校验
- [ ] 将 live Store 接入训练循环
- [ ] 准备 1M / 5M token 语料
- [ ] 跑 Phase 2 正式消融

---

## 3. 完成的内容

### 3.1 WSL/GPU 环境

| 项 | 值 |
|---|---|
| 主机 | Windows + WSL2 Ubuntu |
| GPU | NVIDIA GeForce GTX 1070 8GB |
| Python | 3.12.3 |
| torch | 2.6.0+cu124 |
| transformers | 5.16.1 |
| engram-peft | `dc74c85` |
| EngramDB | 本地开发路径 |
| Qwen3.5-0.8B | `/home/zeng/qwen35-ple/data/models/Qwen3.5-0.8B` |

### 3.2 Phase 0 正式实验

已新增并验证：

- `scripts/run_phase0.py`
- `scripts/run_phase0.sh`
- `src/qwen35_ple/reader.py`
  - `EngramReader`
  - `QwenEngramReader`
  - `QwenShortConv`
- `docs/phase0-protocol.md`
- `docs/phase0-wsl-results.md`

### 3.3 Live 数值一致性

- `scripts/run_live_vs_precomputed.py`
- 实测：
  ```text
  live DiskPleNGramEmbedding  ==  fetch_e_t 路径
  max_abs_diff = 0.0
  allclose = true
  ```

### 3.4 真实行表复制

- WSL 可用空间：**833 GB**
- `qwen38-rows` 大小：**48 GB**
- 结论：空间充足
- 已启动流式复制到：
  ```text
  /home/zeng/qwen38-rows
  ```
- 当前进度：持续增长中

---

## 4. 做的尝试

1. 用 SSH 访问 Windows/WSL 主机。
2. 在 WSL 使用 `uv sync --all-groups` 安装完整依赖。
3. 从 ModelScope 下载 Qwen3.5-0.8B。
4. 将 46k token 预计算特征从 Mac 拷贝到 Windows 再给 WSL 使用。
5. 跑 Phase 0 Simple Reader。
6. 跑 Phase 0 QwenEngramReader + zero-init。
7. 用 `DiskPleNGramEmbedding` 做 live 读取比较。
8. 写后台 Python 脚本启动 48GB 行表流式复制。
9. 尝试直接 `scp`、`tar | ssh`、后台进程等多种传输方式。

---

## 5. 发现

### 5.1 Phase 0 结果

Simple Reader：

| 线 | val_loss_mean |
|---|---:|
| no-reader | 3.794245 |
| real | 3.794319 |
| control | 3.794269 |

QwenEngramReader + zero-init：

| 线 | val_loss_mean |
|---|---:|
| no-reader | 3.794245 |
| real | 3.794197 |
| control | 3.794188 |

结论：

- Phase 0 协议已跑通。
- 当前没有可检测的 PLE 增益。
- 不能作为科学否定，因为训练量、数据、live 路径都还没到位。

### 5.2 重要事实

- live `DiskPleNGramEmbedding` 与当前预计算读取路径完全一致。
- 旧预计算 e_t 文件与当前 Store 不一致，应弃用。
- WSL 有充足磁盘空间放真实行表。
- WSL 环境可以跑通 Qwen3.5 + 自定义 reader + 三线实验。

---

## 6. 踩过的坑

### 6.1 SSH / 网络
- Tailscale 域名解析偶尔失败。
- 后来发现可通过 Tailscale IP `100.78.250.122` 直接访问 Windows/WSL。
- Windows OpenSSH 默认是 cmd，不是 bash，跨层命令要显式调用 `wsl.exe`。

### 6.2 环境安装
- WSL 默认 Python 没有 pip / ensurepip。
- 使用 WSL 已有的 `uv` 完成依赖安装。
- 全量 `uv sync` 会安装 CUDA torch，体积较大，但可行。

### 6.3 数据迁移
- 直接 `scp -r` 48GB 不适合在当前会话中同步等待。
- `tar | ssh` 流式写入 WSL 可行。
- 后台任务需要 `start_new_session=True` 才能真正脱离当前 shell。
- 直接后台 `nohup ... &` 在当前工具环境中容易导致会话卡住。

### 6.4 数据集
- WSL 上 `datasets.load_dataset("wikitext", ...)` 解析失败。
- 后续语料准备需要换用更稳定的数据源或手动下载 raw 文本。

### 6.5 数值
- 旧预计算特征没有乘 `weight_scale`，且部分行与当前 Store 不一致。
- live 路径必须作为后续主要数据面。

---

## 7. 未完成 / 技术债

1. **真实 PLE 行表复制尚未完成**
   - 当前 48GB 传输中
   - 完成后需要：
     - 128 个 shard 数量校验
     - 总大小校验
     - EngramDB Store 可打开
2. **live Store 尚未接入训练循环**
   - 目前只做了数值一致性，没有在 `run_phase0.py` 中直接读取 Store。
3. **Phase 0 尚未跑 QA**
4. **1M / 5M token 语料尚未准备**
5. **未使用 GPU tensor 训练**
   - 当前 WSL 实验仍是 CPU 路径。
6. **旧预计算特征需要重新生成或彻底弃用**
7. **QA 只有 log-likelihood，没有 exact-match 生成式评测**
8. **缺少语料 provenance / manifest**
9. **没有完整资产 checksum**

---

## 8. 未来计划

### 立即
- [ ] 等待 `qwen38-rows` 复制完成
- [ ] 校验：
  - `ls /home/zeng/qwen38-rows | wc -l` == 128
  - `du -sh` ≈ 48GB
  - EngramDB Store 可打开
- [ ] 在 WSL 上跑：
  ```bash
  .venv/bin/python scripts/run_live_vs_precomputed.py \
    --features /mnt/c/Users/minam/qwen35-ple-data \
    --rows-dir /home/zeng/qwen38-rows \
    --model-dir /path/to/qwen38-ple
  ```
- [ ] 将 live Store 读取接入 `run_phase0.py`

### Phase 2a：1M token pilot
- [ ] 准备 1M token 语料
- [ ] live Store + `QwenEngramReader`
- [ ] 三线 + 3 seeds
- [ ] PPL + QA 双评测

### Phase 2b：5M token 可比实验
- [ ] 5M token 训练
- [ ] 正式评测集
- [ ] Go / No-Go

### Phase 3+
- [ ] Backbone 策略矩阵
- [ ] SFT/RL
- [ ] CPU 100 tok/s

---

## 9. 可借鉴项目（互不冲突）

| 项目 | 借什么 | 不拿什么 |
|---|---|---|
| XMemTransfer | 5M/20M 训练预算、target-side reader、多分支/双层 | 不拿它的表/模型 |
| Qwen / Flash-Next | 官方 PLE 结构、weight_scale、key/value/norm/conv、hc_count、ShortConv | 不重训 51B 表 |
| DeepSeek Engram / engram-peft | ContextAwareGating + ShortConv + PEFT/TRL | 不引入第二套存储 |
| Memory Grafting | 离线冻结记忆、精确 n-gram、轻量 projection/gating | 不放弃 PLE |
| Prometheus Mind | 冻结模型忽略信号、stage-wise、部分解冻 | 不复制记忆提取 |
| EngramDB | Store-I/Store-P、DiskPleNGramEmbedding、C ABI、bit-exact | 不修改存储核心 |
| vLLM / SGLang | 磁盘 PLE offload、预取、serving | 现在不引入 serving |
| PWC / 标准评测 | WikiText-103、TriviaQA、NQ、BoolQ、OpenBookQA 等口径 | 不追榜单 |

---

## 10. 关键提交

| commit | 说明 |
|---|---|
| `d5a5621` | Phase 0 harness、忠实 reader、live gate、engramdb store config bridge |
| `1496e46` | official Qwen4Exp custom loader |
| `a5ca602` | lightweight real FP8 PLE e2e runner |
| `9cac640` | WSL Phase 0 三线结果 + live gate 记录 |
| `e840f67` | 第十九轮系统复盘 |
