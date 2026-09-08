# Round 138：4090 正式 Phase 2 启动后的系统性思考

> 日期：2026-09-08  
> 背景：4090 上正式 Phase 2 全量矩阵已启动，约 04:00 自动关机  
> 目的：在真实远程实验环境中重新校准终极目标、技术债、开发计划和外部借鉴边界

---

## 0. 当前状态

- 4090 环境、代码、Qwen3.5-0.8B、qwen38-rows 已准备好；
- Phase 2 全量矩阵正在运行：
  - 6 语料 × 3 seeds × 500 steps × real/control/no-reader × 150 QA；
  - 预计 03:55 左右全部完成；
  - 04:00 自动关机，已完成语料的 JSON 会保留。
- 第一轮全量曾因 QA OOM 中断，已修复：
  - token-budget QA batching；
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
  - 每批 `empty_cache()`。

---

## 1. 终极目标（不变）

> **把 Qwen3.8-Flash-Next 的冻结 PLE 记忆表，通过纯 PLE 嫁接 + 少量训练，显著提升 Qwen3.5-0.8B 的通用性能，并保持低资源、可审计、可复现。**

拆成可检验的形式：

```text
1. 真实 PLE 行必须带来增益：
   real > control 且 real > no-reader
2. 增益必须体现在任务级指标，而不只是 PPL
3. 增益必须可归因于“真实 PLE 内容”，不是背训练语料
4. 训练成本必须小，不能变成全参 CPT
5. 系统必须低资源、可审计、可复现
```

RAG、蒸馏、通用 adapter、LoRA/MoRA 都不是核心。  
它们只有在“纯 PLE 嫁接被严格证伪后”才作为 fallback 手段。

---

## 2. 本 session 发现了什么

### 2.1 正向发现

1. **ModelScope FP8 镜像可以替代 48GB 本地上传**
   - `Qwen/Qwen3.8-Flash-Next-FP8` 的 33 个 checkpoint shard 包含全部 128 个 PLE n-gram shard；
   - 新增 `download_qwen38_fp8_rows.py` 直接抽取为 EngramDB Store-I；
   - 与本地 `qwen38-rows` 的 `shard_000` / `shard_127` sha256 完全一致；
   - 抽取耗时约 30 分钟，避免 3 小时本地上传。

2. **4090 可以完整跑通真实 PLE 路径**
   - torch 2.6.0+cu124 + transformers 5.16.1 + engramdb-python 0.2.12；
   - Qwen3.5-0.8B 可在 4090 上加载、前向、训练 reader；
   - `/dev/shm` 60GB RAM 盘可以容纳 48GB Store-I 行表。

3. **训练侧方向初步一致**
   - PURE_WIKI seed 0：
     - real 训练 loss 3.3077 < control 3.3286
   - PURE_WIKI seed 1：
     - real 训练 loss 3.2243 < control 3.2810
   - 与之前 3 步 pilot、unseen-KB pilot 的方向一致。

### 2.2 工程/技术债

| 编号 | 债务 | 影响 | 优先级 |
|---|---|---|---|
| TD-1 | 只能按 corpus 续跑，不能按 seed/mode 续跑 | 中断最多丢 48 分钟 | P0 |
| TD-2 | 训练 GPU 利用率约 22% | 费用/时间浪费 | P0 |
| TD-3 | QA 仍使用全序列重算，未用 KV cache | 推理慢 3–4 倍 | P1 |
| TD-4 | 结果只在 corpus 结束时写 JSON | 中途崩溃无结果 | P0 |
| TD-5 | `/dev/shm` 行表实例重启后丢失 | 每次重启重抽约 30 分钟 | P1 |
| TD-6 | 无远程 bootstrap/一键复现脚本 | 环境漂移风险 | P1 |
| TD-7 | no-reader 与 seed 无关，却按 seed 重复计算 | QA 浪费约 1/3 | P1 |
| TD-8 | 下载链路对 HTTP/2/LFS 不稳定 | 曾中断抽取 | P1 |
| TD-9 | matrix runner 曾缺 device/scale/bridge/out 参数 | 可能跑错配置 | P0（已修） |
| TD-10 | 评估任务仍以 QA 为主 | 通用性能证据不足 | P0 |
| TD-11 | 无细粒度自动备份/上传 | 远程故障可能丢结果 | P1 |
| TD-12 | 成本管理依赖人工定时关机 | 可能中断 mid-corpus | P2 |
| TD-13 | 未正式跑 layer2 vs layer8 / reader 变体 | 机制归因不足 | P2 |
| TD-14 | 未进入 5M/20M | scaling 叙事缺失 | P1 |
| TD-15 | 未跑 unseen-KB 正式多 split | 外部记忆泛化证据不足 | P1 |

### 2.3 本 session 修复/新增

- `scripts/run_phase1_matrix.sh`：
  - 增加 `--device cuda`；
  - 增加 `--scale`；
  - 默认启用 `--bridge-mlp --out-mlp`；
  - 增加 `--qa-batch-size` / `--qa-batch-max-tokens`。
- `scripts/run_phase0.py`：
  - QA 批处理；
  - token-budget batching；
  - OOM 修复；
  - 与 batch=1 生成结果对拍一致。
- `scripts/download_qwen38_fp8_rows.py`：
  - 从 ModelScope FP8 checkpoint 抽取 Store-I 行；
  - HTTP/1.1 + retry-all-errors；
  - 支持断点续传与已存在 shard 跳过。
- 远程实验记录：
  - `docs/round-137-phase2-full-launch.md`。

---

## 3. 开发计划

### 3.1 Phase 2 收尾（今天/明天）

1. 04:00 关机后，重新开机；
2. 重新抽取 `/dev/shm/qwen38-rows`（约 30 分钟）；
3. 重新执行相同 matrix 命令，已完成语料自动跳过；
4. 收集所有 `phase1-*.json`；
5. 运行：

```bash
python scripts/summarize_phase1_matrix.py \
  --files outputs/phase2-full/phase1-*.json \
  --output outputs/phase2-full/summary.json \
  --markdown outputs/phase2-full/summary.md
```

6. 按预注册门禁判定。

### 3.2 Phase 2 门禁

```text
PPL 门禁：
  至少 5/6 语料、2/3 seeds：
    real PPL < control PPL
    real PPL < no-reader PPL

任务级门禁：
  至少一个 QA 任务：
    real > no-reader
  且 real >= control（或 real > control）
  且没有明显任务级回退

审计门禁：
  训练语料与 QA 集合无答案/问题/文档级污染
```

不满足门禁时：

- 不进入 5M/20M；
- 先做诊断：layer 2 vs layer 8、reader 变体、训练步数、语料配比、QA 评分方式。

### 3.3 基础设施加固（门禁通过前并行做）

1. **细粒度 resume**
   - 每完成一个 seed/mode 写 partial JSON；
   - 保存 reader checkpoint；
   - matrix 启动时扫描 partial，跳过已完成；
   - 目标：中断最多丢一个 seed/mode，而不是一个 corpus。

2. **自动备份**
   - 每个 corpus 完成后，JSON 复制到独立备份目录；
   - 定期 rsync 回 Mac/对象存储；
   - 避免单文件损坏或实例回收导致结果丢失。

3. **远程 bootstrap**
   - 一条命令完成：
     - 安装 venv/依赖；
     - clone/checkout 代码；
     - 下载 Qwen3.5；
     - 抽取 qwen38-rows；
     - 校验 sha256；
   - 输出 `remote-manifest.json`。

4. **下载工具统一**
   - ModelScope/GitHub 强制 HTTP/1.1；
   - retry-all-errors；
   - partial resume；
   - 下载后 sha256 校验。

5. **成本管理**
   - 定时关机前自动检查：
     - 如果当前 corpus 已完成，直接关机；
     - 如果 mid-corpus，等待下一个 corpus 完成再关机，或接受损失；
   - 避免整晚跑但 GPU 空转。

### 3.4 效率优化（门禁通过后）

#### 训练

- micro-batch：每步采样 4–8 个窗口；
- gradient accumulation；
- `torch.compile`；
- bf16/autocast 做 A/B，确认不改变结论；
- 目标 GPU 利用率从 22% 提升到 60%+。

#### QA 推理

- KV cache + reader 增量 ShortConv 状态缓存；
- 与完整重算做生成结果对拍；
- 如果一致，QA 预期提速 3–4 倍；
- 如果增量 reader 数值不通过，保留完整重算作为正式路径。

#### 框架选择

- 当前：PyTorch + Transformers，正确性优先；
- 后续服务化再考虑 vLLM / SGLang；
- 但必须先解决 custom PLE 行注入与 reader hook 的集成问题；
- 不为了提速牺牲 real/control/no-reader 的可审计性。

### 3.5 科学路线

```text
Phase 2 门禁通过
  -> 5M scaling（选 1–2 个最优语料）
  -> 5M 通过
  -> 20M scaling
  -> unseen-KB 正式多 split + QA
  -> layer2 vs layer8
  -> reader 变体诊断
  -> 机制分析（gate、CKA、patching）
  -> 论文/artifact 发布
```

如果 5M 饱和：

- 记录 scaling 边界；
- 不盲目上 20M；
- 转向 reader 结构或语料质量。

如果任务级失败：

- 先诊断，不直接上 RAG/蒸馏；
- 保留纯 PLE 负结果；
- 只有当纯 PLE 被严格证伪后，才启动 fallback 路线。

---

## 4. 外部借鉴矩阵

| 项目/方向 | 可以借什么 | 不能借什么 / 冲突边界 |
|---|---|---|
| XMemTransfer | matched corpus、control、1M→5M→20M、memory transfer 设计 | 不照搬模型规模/数据规模 |
| Memory Grafting | 条件记忆、外部记忆注入、gate 设计 | 不重训 PLE 表 |
| DeepSeek Engram | 条件记忆、rowid、disk offload、reader 结构 | 不改 EngramDB 存储核心 |
| Qwen PLE | FP8 表、rowid 契约、official reader、weight_scale | 不重训 Qwen3.8 |
| Smol Training Playbook | 分阶段数据混合、小模型训练经验 | 不照搬 11T 语料/全参训练 |
| Data Mixing Laws | 语料配比实验设计 | 不追求全局最优公式 |
| vLLM / SGLang | 高效 serving、continuous batching | 不作为科研核心；需先解决 PLE 注入 |
| FlashAttention / xFormers / torch.compile | kernel/fusion 加速 | 只在正确性验证后启用 |
| PEFT / LoRA / MoRA | fallback 能力迁移手段 | 不进入纯 PLE 核心路径 |
| RAG / BM25 | 外部知识上界、baseline | 不替代 PLE 记忆 |
| kNN-LM / NGM | 失败模式、retrieval baseline | 不替代核心 |
| OPD / Purified OPSD | 蒸馏 fallback | 不把蒸馏当核心创新 |
| DVC / MLflow / W&B | 实验追踪、artifact 管理 | 保持轻量，不引入重依赖 |
| AutoDL / 云调度 | 数据盘、定时关机、快照 | 不为省钱牺牲结果完整性 |
| EngramDB / engram-peft | ABI、测试、CI、契约 | 不让实验仓承担核心库职责 |

### 不冲突原则

1. 核心指标永远是：

```text
real / control / no-reader
```

2. 任何外部方法只能作为**可选模块**，不能替换核心三线。
3. 一次只改一个变量；改完先过 control。
4. 先证明科学，再优化效率；先优化效率，再扩规模。
5. 表必须冻结；只训练 reader / bridge / gate。
6. RAG、蒸馏、adapter 只能在纯 PLE 失败后进入 fallback。
7. 所有结果必须带 manifest、命令、seed、数据 hash、审计。

---

## 5. 近期最重要的 5 件事

1. **让 04:00 前的 Phase 2 结果安全落地**：收集已完成 JSON，备份到 Mac。
2. **补 resume 能力**：partial JSON + reader checkpoint，避免下次中断重跑整个 corpus。
3. **做门禁判定**：PPL + QA + 审计三张表。
4. **优化训练/推理效率**：micro-batch + KV-cache/incremental reader。
5. **门禁通过后再上 5M**：不要跳步。

---

## 6. 一句话结论

> 这轮 session 最大的价值，不只是“4090 跑起来了”，而是把远程实验链路从“人工、易中断、低利用率”推进到了“可复现、可续跑、可审计”的工程基础。
>
> 下一步的核心不是立刻上更大模型，而是：
>
> **用正式 Phase 2 的任务级结果决定是否值得 scaling，同时把断点续跑和 GPU 利用率补上，让 5M/20M 跑得更稳、更便宜、更可信。**
