# qwen35-ple 战略路线图（Roadmap）

> 作者视角：工程负责人复盘。本文件与 `docs/qwen35-ple-design.md` 配套，
> 是“目标 → 技术债 → 借鉴 → 阶段计划”的战略层文档。
> 状态：v0.1（2026-08-30，仓库初始化后的第一次系统盘点）。

---

## 1. 终极目标（北极星）

**一句话：用最小可复现实验证明“Qwen3.5 主干 + 冻结 Flash-Next PLE 记忆表”的嫁接是否成立；
若成立，交付 0.8B SFT/RL 模型 + CPU 100 tok/s 可复现推理闭环；若不成立，留下可审计的负结果。**

展开为四条可验收的轴：

| 轴 | 定义 | 口径 |
|---|---|---|
| 科学 | A1（PLE 嫁接）相对 A0（无 PLE 同 CPT）是否有增益 | 小规模消融报告，不因负结果而隐瞒 |
| 工程 | 四仓库按契约 v1 形成可复现闭环 | `view verify` 位级一致；golden 对拍；e2e 可跑 |
| 产品 | 0.8B 变体 CPU 推理 ≥100 tok/s | CompileForge 同机 A/B；PLE 尾差 ≤2% |
| 过程 | 每个决策有证据、可重建、可回滚 | 门禁 + 文档 + 资产 manifest 三件套 |

**当前最重要的战略约束：不要先冲 4B / 50B，也不要先铺开 A2-A5。先用 0.8B 的 A0/A1、golden、e2e
消掉“嫁接是否成立”这一最大不确定性。**

---

## 2. 当前真实状态

- [x] 仓库初始化、设计文档、四仓库交互契约 v1 冻结
- [x] 纯 Python `PLE_QWEN_V1` golden 参考对拍（`src/qwen35_ple/ple_hash.py`）
- [x] Store-P 视图构建/校验脚本骨架（`scripts/build_table_assets.sh`）
- [x] YAML 配置加载与契约校验（`src/qwen35_ple/config.py`）
- [x] engram-peft C2 字段 + `QwenPleHashMapping` + 跨仓 golden
- [x] M0 quick 磁盘版 MultiHeadEmbedding 自检
- [x] M0 合成表磁盘注入完整 forward/generate 闭环（`scripts/run_m0_smoke.py --synthetic-e2e`）
- [x] Qwen3.5-0.8B + engram-peft PLE-lite CPU e2e（`scripts/run_qwen35_e2e.py`，合成内存表）
- [x] M1 hc=1 PLE-lite 前向 golden（与 Qwen PLE 参考数学 4096 token 对拍）
- [x] Qwen 官方 `refs/qwen4_exp_modeling.py` 快照 + SHA-256 manifest 固定
- [x] 官方 PLE 前向 4096 token golden fixture（`tests/golden/official_ple_forward_4096.*`）
- [x] A0/A1 评测对比入口（`scripts/run_eval.py` + `eval/protocol.py`）
- [x] 最小知识召回/长上下文/推理评测执行器（`scripts/run_ablation_eval.py`）
- [x] M2 CPT 训练冒烟（`scripts/run_cpt_smoke.py`）：A0/A1 均可反向训练
- [x] Qwen3.5-0.8B A0/A1 极小消融脚本（`scripts/run_qwen35_ablation.py`）：10 步 pipeline 已跑通
- [x] 第一轮极小 A0/A1 结果：A1 ≈ A0，未见增益（负结果已记录，不视为最终结论）
- [x] 真实 PLE `e_t` 预计算 + 知识探针：test acc 72.7% vs random 16.7%（正信号）
- [x] CI：lint + 基础单元测试（`.github/workflows/ci.yml`），已修复本轮 ruff 失败
- [x] qwen35-ple 已推送到 GitHub（`451b046` / `cbf640c` / `aad9bec` / `f86fd0f` / `91a032f`）
- [x] engram-peft 已推送到 GitHub（`5fc90d2` + `272166a`）
- [ ] M0 真表 e2e（TinyLlama/Qwen + 完整 engram-peft + 50GB PLE 表环境）
- [ ] CPT 消融（A0/A1）
- [ ] 100 tok/s 推理闭环

---

## 3. 技术债清单

### 3.1 契约与实现的剩余缺口

- C2 的 `engine` / `table_spec` / `table_source` 已进入 `EngramConfig`，`QwenPleHashMapping` 已落地。
- M1 已有本地 PLE-lite 前向参考，并且已把 Qwen 官方文件作为快照固定到
  `refs/qwen4_exp_modeling.py`（含 SHA-256 manifest），避免上游漂移。
- `table_source="engramdb:view"` 的推理侧读取尚未在 engram-peft 中接通。

### 3.2 缺少 golden/位级一致性防线

- 已建立本仓 golden + engram-peft 跨仓 golden（`tests/test_cross_repo_hash_golden.py`）。
- 已建立 4096 token 的 PLE-lite 前向 golden（`tests/test_ple_forward_golden.py`）。
- 已增加官方 PLE 前向 golden：`tests/golden/official_ple_forward_4096.*`，
  由 `src/qwen35_ple/official_ple_snapshot.py`（冻结的官方代码 AST 抽取）生成。
- `refs/qwen4_exp_modeling.py` 已从 EngramDB 拷贝并固定 checksum；本仓可离线复现。

### 3.3 缺少完整可运行闭环

- 已有 M0 quick 磁盘自检。
- 已有合成表 `--synthetic-e2e` 完整 forward/generate 闭环：引擎、磁盘注入、无 NaN 均验证。
- 真表 e2e 仍需要 50GB PLE 表与 FP8 行解码/磁盘读取路径。
- 配置样例仍主要在“初值/编排”层面，不是完整训练入口。

### 3.4 资产与可重建性

- 有 Store-P 构建脚本，但尚未固化真表资产的本地路径/校验报告。
- 没有语料 provenance/许可记录。
- `engram_vocab_size_per_ngram` 初值已修正为每个 n-gram 8 头总桶数；仍需在真实表上验证。

### 3.5 评测与决策机制

- 已有 A0/A1 JSON 对比协议、报告入口和最小知识召回执行器（`run_ablation_eval.py`）。
- 仍缺长上下文与基础 reasoning 的正式评测执行器。
- 没有“A1 负增益即止损”的正式门禁。

### 3.6 工程过程

- 已有基础 CI workflow（lint + 单元测试）；跨仓 golden/重依赖 e2e 尚未进 CI。
- 兄弟仓库已有成熟“门禁 + 文档同步”习惯，本仓继续对齐。
- 本 session 暴露了跨仓 git 权限问题：某些环境下不能直接写 engram-peft `.git`，
  需要用可写镜像 + rebase 完成提交；后续应把跨仓变更流程标准化为 patch/镜像流程。

---

## 4. 借鉴矩阵（分层、不冲突）

| 来源 | 借鉴什么 | 明确不拿 | 为什么不冲突 |
|---|---|---|---|
| EngramDB | 证据库、golden/bit-exact、可重建资产、roadmap/复盘方法 | 存储、IO、视图内部实现 | 存储面已由它完成；本仓只消费和验证 |
| engram-peft | PEFT 接口、`train_mode`、TRL/SFT 集成、测试纪律 | 引擎前向/训练核心实现 | 本仓通过契约 C2 消费，改动走“只增字段” |
| LLM-CompileForge | 契约驱动、子项目独立、TDD、E2E 最后验证 | 编译器、runtime、MLIR | 推理目标由它实现；本仓只做资产对接和验收 |
| Qwen/DeepSeek 官方 | 精确实义、golden 生成、PLE 层参考实现 | 不自创哈希、不重训 51.2B 表 | 官方语义是事实标准，必须逐位对齐 |
| Memory Grafting 等研究 | “表独立训练后嫁接到新主干”的实验设计 | 不照搬模型/代码 | 只借方法论，不引入第二套存储或训练实现 |

一句话：**EngramDB 教我们“证据与可重建性”，engram-peft 教我们“训练集成与工程纪律”，
LLM-CompileForge 教我们“契约驱动与性能验证”，Qwen/DeepSeek 教我们“位级精确”。**

---

## 5. 开发计划（按“先证伪、再放大”排序）

### Phase 0：门禁与对齐（进行中）

- 纯 Python golden 参考与测试（✅ 已落地）
- Store-P 构建脚本骨架（✅ 已落地）
- CI / README / 文档同步（进行中）
- 固定 `refs/qwen4_exp_modeling.py` 引用（✅ 已落地：快照 + SHA-256 manifest）
- 官方 4096 token 前向 golden（✅ 已落地：`tests/golden/official_ple_forward_4096.*`）

**Gate：** `make check` 绿；golden 测试可复现。

### Phase 1：M0 最小纵切

- 用 EngramDB 真表/合成表构建 Store-P 视图。
- 接入 `engramdb.integrations`，跑 TinyLlama 或 Qwen3.5-0.8B 磁盘注入前向/生成。
- ✅ 合成表路径已闭环：`scripts/run_m0_smoke.py --synthetic-e2e` 一条命令可跑，forward/generate 无 NaN。
- ⏳ 真表路径仍待 50GB 表 + FP8 行读取。

**Gate：** 一条命令可复现；e2e 可跑（合成表已过，真表待跑）。

### Phase 2：M1 实现 `engine="qwen_ple"`

- engram-peft 只增 `engine` / `table_spec` / `table_source` 字段（✅ 已完成并推送）。
- 实现 Qwen 原生日志映射（`PLE_QWEN_V1`）与 PLE-lite `hc=1` 层（✅ 哈希映射已完成）。
- ✅ 与 Qwen PLE 参考数学做 4096 token 数值对拍（`tests/test_ple_forward_golden.py`）。
- ✅ 已固定 `refs/qwen4_exp_modeling.py` 官方快照并生成 4096 token 官方前向 golden。
- 保持 `engine="deepseek"` 全量回归（⏳ 待完整环境验证）。

**Gate：** 位级一致；DeepSeek 路径不回归。

### Phase 3：M2 小规模消融（决策点）

- 实现 config/data/train/eval 编排。
- ✅ 训练冒烟已通：`scripts/run_cpt_smoke.py --ple` 可反向更新 PLE 层。
- ✅ 已跑第一轮极小 A0/A1（Qwen3.5-0.8B + 10 步 + 小语料），结果 A1 ≈ A0，无可见增益。
- ✅ 真实 PLE 知识探针通过：线性分类 acc=72.7% vs random=16.7%，说明 `e_t` 含语义信号。
- ✅ 完整实验矩阵（layer 1/8 × branch 1/4 × short_conv）：所有组合 real 均优于 control，最佳为 layer8+b1+无shortconv（after 4.851），但仍高于 no-reader baseline 4.428。
- ⏳ 需要减小初始扰动/更长训练/换任务/部分解冻 backbone，或直接用官方 PLE gating 结构。
- 输出知识 recall、长上下文、基础 reasoning 报告（当前为迷你 probe）。

**Gate：** 报告 + go/no-go；A1 不优于 A0 则停止放大。当前第一轮证据不足以下结论，但未观察到增益。

### Phase 4：后训练 + 推理闭环

- A1 为正时进入 SFT/RL（M3）。
- 与 CompileForge P0-P5 并行；本仓负责模型/视图资产和 A/B 验收协议。

**Gate：** 100±10 tok/s；PLE 开销 ≤2%；训练/推理数值一致。

---

## 6. 稳健前进三原则

1. **先最小纵切，再规模化。** 每一环节没有可复现命令和 gate 就不进入下一阶段。
2. **跨仓改动只走契约。** 只增字段/符号，旧行为冻结，避免四仓互相踩脚。
3. **决策留下证据。** 无论正负结果都写入 roadmap/session log，附可复现命令。

---

## 7. 2026-08-30 第十七轮战略更新

> 详细复盘见 `docs/session-log.md` 第十七轮；本轮只记录战略层增量。

### 7.1 当前最重要的判断

- 我们已经有“真实 PLE 内容包含语义信号”的初步证据（知识探针 + real>control）。
- 但还**没有**“嫁接能带来净增益”的证据。
- 当前最可能的原因：
  1. 训练量只有 46k token，而 XMemTransfer 需要 5M–20M；
  2. reader 仍未对齐官方 hc_count=4 + ShortConv + 官方 gate；
  3. 评测任务/协议不够严格。
- 因此战略重心仍然是 **先科学证伪，再产品放大**，不提前进入 SFT/RL/100 tok/s。

### 7.2 新增已解决问题

- [x] EngramDB v0.2.8 接入：`rowids_for_seq` / `discover_ple` / `load_ple_weight_scale`
- [x] 修复 e_t 未乘 `weight_scale` 的数值缺陷
- [x] 清理 ruff RUF046，CI lint 恢复
- [x] 新增 EngramDB v0.2.8 消费冒烟脚本

### 7.3 下一阶段只做三件事

1. **Phase 0**：正式 held-out + 多 seed + 最小 QA 评测 + 环境固化。
2. **Phase 1**：忠实 reader（官方/engram-peft gating + live Store 读取）。
3. **Phase 2**：把训练量推到 1M → 5M token，用正式指标做 Go/No-Go。

若 Phase 2 仍无稳定正增益：记录负结果，停止放大。
若 Phase 2 有正增益：才进入 backbone 策略矩阵和产品化。
