# 第二十轮完整汇总（2026-09-01）

> 范围：从“1M live-store 三线实验”到“CI 升级 uv / pre-commit / README 更新”。
> 状态：本轮已完成科学关键确认（1M real > control > no-reader），工程底座开始稳定。

---

## 1. 本轮目标

1. 用懒加载 live-store 在 WSL 上跑通 1M token 实验，避免 10GB `e_t.npy` 与 OOM。
2. 验证官方 Qwen3.8 PLE reader + 2 层 MLP bridge/out_proj 在 1M 规模下的真实效果。
3. 补跑 1M no-reader baseline，形成完整三线对比。
4. 修复 CI 中发现的 ruff / pytest 问题。
5. 把 CI 升级为 uv 管理，并配置 pre-commit。
6. 更新 README，沉淀当前进展。

---

## 2. 本轮计划

- [x] 使用 `LiveETStore` / `LiveETView` 懒加载，不物化全量 e_t。
- [x] 在 WSL 上跑 1M token real / control / 3 seeds。
- [x] 补跑 1M no-reader baseline。
- [x] 修复 live_store / test_cross_repo 的 CI 问题。
- [x] CI 改用 `uv sync --all-groups` + `uv run ruff/pytest`。
- [x] 新增 `.pre-commit-config.yaml` 与本地 hook 使用说明。
- [x] 更新 README 当前状态和最新实验结果。

---

## 3. 本轮发现

### 3.1 科学结果：首次出现强正信号

1M tokens / live-store / 官方 reader + 2 层 MLP / 500 steps / 3 seeds：

| 线 | val loss | PPL |
|---|---:|---:|
| no-reader | 2.9896 | 19.88 |
| control | 2.8738 | 17.70 |
| **real** | **2.8167** | **16.72** |

关键差距：

```text
real − control = −0.0571
real − no-reader = −0.1729
control − no-reader = −0.1158
```

结论：

- 3 seeds 全部 real > control。
- real 也超过 no-reader baseline。
- 这是目前最强的科学正信号。

### 3.2 工程结论：懒加载路径正确

- `LiveETStore` 只保存 `[T,16]` rowids。
- 每个训练/评测窗口只读取 `seq_len × 16` 行。
- 1M token 不会因为 10GB e_t 而 OOM。
- 1M live-store 实验已在 WSL 上完整跑通。

### 3.3 CI 问题根因

本轮多次 CI 失败：

- `typing_extensions` 缺失 → 改为 Python 3.11 标准库 `typing.Self`。
- `torch` 无 `float8_e4m3fn` / `uint8` → 增加 dtype fallback。
- `test_cross_repo_hash_golden` 的 torch stub 污染全局 `sys.modules` → 增加 `finally` 恢复。
- `UP037` 引号类型注解 → 根据 `from __future__ import annotations` 去掉引号。

根本原因：

> 本地没有真实执行与 CI 一致的完整流程，很多问题只能等 CI 暴露。

---

## 4. 做的尝试

### 4.1 1M 全量 e_t 预计算（已被否）

- 尝试生成 1M token 的完整 `e_t.npy`。
- 实测需要约 10GB 内存/磁盘。
- WSL 上全量 `Store.fetch` 慢且可能 OOM。
- 结论：放弃全量预计算作为主路径。

### 4.2 20k 级读取方法对比

| 方法 | 20k tokens / 320k rows | 说明 |
|---|---:|---|
| `PleDiskGather.fetch` | 约 9.6s（首次） | 旧 Python 字节展开路径 |
| `fetch_e_t_tensor` 首次 | 约 6.2s | 首次有初始化成本 |
| `fetch_e_t_tensor` 热态 | 约 0.55s | 稳定后很快 |
| 直接 `Store.fetch` | 约 0.43s | 最快原始读取 |

结论：当前 `engramdb.fetch_e_t_tensor` 已经是正确路径，但仍需 Store-P / 批量读进一步优化。

### 4.3 WSL Store-P 路径

- 已确认 WSL 上 Store-P 比 Store-I 快约两个数量级。
- 1M Store-P 懒加载约 24s。
- 本轮科学实验仍走 Store-I 懒加载，因为真实模型训练读取量小；后续大规模应切 Store-P。

### 4.4 CI / 工程

- 创建临时 venv 安装 ruff。
- 用 `uvx ruff check src tests` 本地复现 lint。
- 更新 `.github/workflows/ci.yml` 为 uv。
- 新增 `.pre-commit-config.yaml`。

---

## 5. 踩过的坑

1. 之前误以为 1M 全量 `--live-store` 可行，实际上会 OOM。
2. 本地 Mac 没有 `ruff`，最初无法本地复现 CI。
3. `uv sync --all-groups` 在当前 Mac 因 `bitsandbytes` 无 macOS x86_64 wheel 失败。
4. `pre-commit` 在当前沙箱无法连接 GitHub，无法完整运行。
5. `typing_extensions` 不是 CI 默认依赖，导致 pytest collect 失败。
6. `test_cross_repo_hash_golden` 注入的 torch stub 污染了其他测试。
7. `LiveETViewStore._fetch` 默认依赖 `torch.float8_e4m3fn`，在不支持 FP8 的 torch 环境崩溃。
8. 本地 Mac Python 3.9 / torch numpy bridge 不兼容，不能完整复现 CI。
9. 远端新增文件后本地没有立即重新跑 `ruff check`，导致多次 CI 才发现。

---

## 6. 完成的内容

- [x] `OfficialSourceQwenReader` 官方 reader 权重复用。
- [x] 可切换 1 层 / 2 层 MLP `query_bridge` / `out_proj`。
- [x] `run_phase0.py --live-store` 懒加载模式。
- [x] `LiveETStore` / `LiveETView` / `LiveETDataset` / `LiveETViewStore`。
- [x] 1M token live-store real / control / 3 seeds。
- [x] 1M no-reader baseline。
- [x] `fetch_e_t_tensor` 快速路径接入。
- [x] CI 升级 uv 管理。
- [x] pre-commit 配置。
- [x] README 更新。

---

## 7. 未完成 / 技术债

| # | 技术债 | 影响 |
|---|---|---|
| 1 | 1M QA exact-match 已接入 harness，尚未在 WSL 跑正式结果 | PPL 之外还需要知识类任务证据 |
| 2 | 5M token 正式实验未跑 | XMemTransfer 显示 5M 才是可比规模 |
| 3 | dual-layer / multi-layer reader 未系统测试 | 可能进一步提升读取能力 |
| 4 | reader + LoRA / 部分解冻未测 | 冻结 backbone 可能忽略信号 |
| 5 | Qwen3.5-4B 未测 | 4B hidden=2560，官方 reader 可更完整复用 |
| 6 | WSL Store-P 训练路径未端到端验证 | 1M 科学实验仍走 Store-I 懒加载 |
| 7 | 完整 CI 未在本地/云上跑通 | 仍存在本地环境偏差 |
| 8 | pre-commit 未本地 install | 提交前仍未强制 |
| 9 | corpus / asset manifest 与 checksum 缺失 | 复现和审计不足 |
| 10 | CPU 100 tok/s 推理闭环未开始 | 产品目标未验证 |
| 11 | SFT/RL 未开始 | 最终交付模型未形成 |

---

## 8. 未来计划

### 第一阶段：科学确认加固（最重要）

1. 1M QA exact-match：
   - TriviaQA / NQ / BoolQ
   - real / control / no-reader
   - 3 seeds
2. 1M reader 变体：
   - dual-layer / multi-layer
   - reader + LoRA
   - reader + 部分解冻
3. 固定 5M 实验协议：
   - 数据、split、seeds、指标、manifest 全部固化

### 第二阶段：5M 正式判定

```text
5M tokens
3 seeds
real / control / no-reader
PPL + QA exact-match
```

- 4090 预计 18–45 小时
- A100 预计 9–27 小时

Go / No-Go：

- Go：3 seeds 下 real 稳定 > control，且至少一个正式指标 > no-reader。
- No-Go：记录负结果，停止放大。

### 第三阶段：产品闭环

- 4B 模型验证
- CPU 100 tok/s 推理
- SFT / RL
- 最终交付

---

## 9. 可借鉴项目

| 项目 | 借什么 | 不拿什么 |
|---|---|---|
| XMemTransfer | 5M–20M 训练量、target-side reader、dual-layer、QA 评测 | 不拿记忆表/模型 |
| DeepSeek Engram | ContextAwareGating、ShortConv、多分支、确定性 hash | 不引入第二套存储 |
| Memory Grafting | 离线冻结记忆、轻量 projection/gating | 不替换 PLE 表 |
| Prometheus Mind | 冻结模型可能忽略信号、stage-wise、部分解冻 | 不复制其记忆提取 |
| Qwen3.8-Flash-Next | 官方 PLE 结构、reader 权重、FP8 scale、hash | 不重训 51B 表 |
| EngramDB | Store-I/Store-P、rowid、C ABI、磁盘优先、manifest | 不改变核心存储 |
| vLLM / SGLang | 磁盘 PLE offload、异步预取、批量读 | 不引入 serving 依赖 |
| LLM-CompileForge | 编译器、外部权重源、100 tok/s 验收 | 不绑定科学实验 |
| 多模态 Aligner | 2 层 MLP connector、表示空间对齐、冻结 encoder + adapter | 不照搬视觉架构 |

---

## 10. 关键提交

| commit | 说明 |
|---|---|
| `4295584` | lazy per-window e_t fetch for 1M memory-limited runs |
| `7033f8b` | fix lint + pre-commit hooks |
| `38b79f4` | fix I001 import group |
| `048d249` | use stdlib typing.Self |
| `0271def` | fallback raw dtype for torch without fp8 |
| `ce605a5` | isolate torch stub in cross-repo hash test |
| `c8a2a5b` | CI use uv |
| `6533be8` | remove quotes from LiveETViewStore.view return |
| `80f4fbb` | README update with 1M results |

---

## 11. 当前结论

> 1M live-store 实验已经给出强正信号：real 稳定优于 control，且优于 no-reader。
> 下一步的关键不是继续无限调参，而是：
> **1M QA → 5M 正式判定 → 如果 Go 再做 4B / SFT / CPU 推理。**
> 同时把 CI、pre-commit、Store-P、manifest 等工程底座补稳，避免每次都在 CI 才发现问题。
