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

---

## 8. 2026-09-01 Track A：通用懒加载数据流

> 详细复盘见 `docs/session-log.md` 的“Session 33 Track A”章节。

### 8.1 已完成

- [x] 将 `LiveETStore` / `LiveETView` 从 `run_phase0.py` 提炼到
  `src/qwen35_ple/live_store.py`。
- [x] 实现 `LiveETDataset`（IterableDataset 兼容）：
  - 每个 `__iter__` 只取一个窗口；
  - 支持 `control`、`shuffle`、`worker_id` / `num_workers` 分片；
  - 支持 `torch.utils.data.DataLoader(num_workers=N)`，每个 worker 自动按
    `get_worker_info()` 分片并重新打开自己的 Store 句柄；
  - 记录 `LiveETBatch.fetch_seconds` / `rows`，Store 级统计 `FetchStats`。
- [x] `LiveETStore` 支持 pickle / unpickle，为 DataLoader 多进程子进程重开 Store。
- [x] 新增 `LiveETViewStore`：Store-P 物化视图读取器，为 Track B 提供同构 A/B 路径。
- [x] 新增 `scripts/bench_store_vs_view.py`：Store-I 与 Store-P 同 token 集 A/B 骨架 + CSV / 阈值。
- [x] 新增 `scripts/bench_lazy_windows.py`：逐窗口懒加载基准，输出每窗口 CSV + 百分位；本机 1M Store-P 已跑通（约 7.1s）。
- [x] `run_phase0.py --live-store` 改用统一模块，不再在脚本内维护私有类。
- [x] 新增 `scripts/run_live_et_dataset_smoke.py` 冒烟入口。
- [x] 新增 `tests/test_live_store.py`（8 个测试通过）。
- [x] README 增加三行接入示例。

### 8.2 下一个最高优先

- Track B：WSL Store-P 构建 + Store-I / Store-P / lazy / full-memory 同口径 A/B。
- Track C：用 `LiveETDataset` 跑 1M token real/control/3-seed，并输出每窗口
  fetch 时间 / CSV。
- Track D/E：Store 连接池、服务化、CI nightly 入 live-store smoke。

---

## 9. 2026-09-01 第二十轮系统性思考（Session 34）

> 完整版见 EngramDB `docs/roadmap.md` Section 24 与本仓 `docs/session-log.md`。

### 9.1 当前定位

- 终极目标不变：磁盘优先的 PLE/Engram 记忆表基础设施。
- Track A 已完成；Track B/C 完成了“读取基准”，但尚未完成“真实模型实验”。
- 已确认 Store-P 是 WSL 随机 IO 的正确出路；已实现 rowid→slot 语义映射和访问序调度。
- 已新增有限语料的 access-order Store-P builder：`scripts/build_corpus_store_p_view.py`。
- 已让 `run_phase0.py --store-p-view` / `--store-p-slot-index` / `--access-order` 直接走 Store-P 训练路径。

### 9.2 新增技术债（V123–V132）

- ~~V123 rowid-tuple → Store-P slot 语义映射~~ ✅ 已完成：`SlotIndex` + `--slot-index-out` + `run_phase0 --store-p-slot-index`
- ~~V124 access-order 视图/调度~~ ✅ 已完成：`LiveETViewStore(access_order=True)` + `LiveETDataset(access_order=True)`
- V125 真实模型 1M real/control/3-seed
- V126 WSL golden 漂移
- V127 serving A/B
- V128 懒加载基准门禁
- V129 StorePool 与 LiveET 深度集成
- V130 Arrow IPC 验证
- V131 WSL 复现脚本
- V132 WSL 全表 Store-P 构建策略

### 9.3 下一阶段

1. P0 剩余：真实模型 1M 三线实验（WSL/模型侧单独推进，已部分完成）。
2. P1：性能门禁 + WSL 复现 + golden 对齐。
3. P2：serving / Arrow / 连接池深化。
4. P3：全表 Store-P + 三仓同步 + 发布。

---

## 10. 2026-09-01 第二十一轮系统性思考（Session 35）

> 完整版见 EngramDB `docs/roadmap.md` Section 25 与本仓 `docs/session-log.md`。

### 10.1 本轮坐标

- ✅ v0.2.11 已发布。
- ✅ V123/V124 P0 语义索引和访问序调度代码完成。
- ✅ qwen35-ple 全量测试 34 passed / 7 skipped。
- ✅ DiskSlotIndex / access-order CI / 全表批式构建工具已推进。
- ✅ V125 真实模型 1M real/control/3-seed 已在 WSL 完成，结果见 `docs/phase-a-1m-result.md`。

### 10.2 本仓新增/关注技术债

| # | 债 |
|---|---|
| V133 | SlotIndex 全表 320M 无法纯内存承载 |
| V134 | SlotIndex 在 EngramDB 与 qwen 两仓重复实现 |
| V135 | `engramdb view build` 尚未原生生成 slot index |
| V136 | access-order 调度缺正式 A/B 与门禁 |
| V137 | numpy 依赖/降级语义未完全理清 |
| V138 | `LiveETDataset(access_order=True)` 窗口重排对顺序敏感实验需单独建模 |
| V139 | 两仓 SlotIndex 无 cross-repo contract test |

### 10.3 下一阶段

1. ✅ Phase A：真实模型 1M real/control/3-seed 已完成，结论为 Go，建议继续到 5M–20M。
2. Phase B：SlotIndex 统一到 EngramDB canonical + 磁盘化 + `engramdb view build` 原生输出（已大部分完成）。
3. Phase C：access-order 基准门禁 + WSL 复现 + golden（合成门禁已入 CI）。
4. Phase D：serving / Arrow / 全表 Store-P。
5. Phase E：依赖与跨仓治理。

---

## 11. 2026-09-01 第二十二轮系统性思考（Session 36）

> 完整版见 EngramDB `docs/roadmap.md` Section 26。

### 11.1 坐标

- ✅ Phase A：1M real/control/3-seed 完成，real < control < no-reader，Go。
- ✅ DiskSlotIndex、全表批式构建、StorePool 遥测、合成 CI 门禁已落地。
- ✅ qwen35-ple 34 passed / 7 skipped。
- ⚠️ Phase A2：Store-P/access-order 复跑 + fetch timing 尚未做。

### 11.2 本仓关注技术债

| # | 债 |
|---|---|
| V140 | Phase A 未用 Store-P/access-order 复跑 |
| V141 | DiskSlotIndex 无 320M 实测 |
| V142 | bucket 文件数过多 |
| V143 | qwen 保留本地 SlotIndex fallback |
| V144 | CLI 未原生生成 slot index |
| V145 | Phase A 无 fetch timing |
| V146 | WSL golden 漂移（CI 已对官方前向 golden 使用 xfail，待主动重建） |
| V147 | CI 只有合成门禁 |
| V148 | 新功能未发布 |

### 11.3 下一阶段

1. Phase A2：Store-P + access-order 复跑 1M，记录 loss + fetch timing。
2. Phase B2：DiskSlotIndex 全表实测 + 单文件/原生化。
3. Phase C2：真表性能门禁 + golden 修复。
4. Phase D2：Arrow / serving / 全表实际构建。
5. Phase E2：v0.2.12 发布。


---

## 12. 2026-09-02 第二十三轮：上游版本收口与后续计划

> 详细评估见 `docs/round-23-upgrade-assessment.md`。

### 12.1 坐标

- ✅ engram-peft v1.2.7 发布：Qwen PLE engine + `table_source="engramdb:store"` 自动消费。
- ✅ EngramDB v0.2.12 发布：DiskSlotIndex v3、PleMemory / Bundle / TargetReader、通用 Engine Adapter。
- ✅ qwen35-ple 最低版本收口：pyproject / uv.lock / CI tag / WSL 脚本。
- ✅ reader_registry / bundle / checkpoint / serving adapter 第一版落地：`TargetReaderRegistry` 薄封装 + `BundleManifest` 兼容 bundle + `run_phase0 --save-reader/--load-reader` + `QwenReaderServingAdapter`；`ShortConv` 也已纳入 checkpoint。
- ⚠️ 剩余：真实 vLLM/SGLang 引擎适配、Phase A2 收尾。

### 12.2 下一阶段

1. ✅ 接入 EngramDB `TargetReaderRegistry` / `BundleManifest`，qwen35 只做具体 reader 注册与实验逻辑。
2. ✅ `run_phase0.py` 增加 reader 保存/加载；下一步用于跑完 150 题 QA。
3. Phase A2 Store-P/access-order 结果收尾。
4. 5M 三线 Go/No-Go；随后再进入 vLLM/SGLang/CompileForge serving 闭环。



---

## 13. 2026-09-02 第二十四轮：三线 QA、语料混比与系统性收口

> 详细总结见 `docs/round-24-full-summary.md`。

### 13.1 坐标

- ✅ 工程闭环：reader checkpoint / bundle / serving adapter / 三线 QA 已完成。
- ✅ 150 题三线结果：no-reader 53.3% / real 42.0% / control 30.7%。
- ✅ val loss：real 2.7892 < control 2.9159 < no-reader 2.9895。
- ⚠️ 科学结论：PLE 有信号，但当前未跑赢 no-reader。
- ⚠️ 当前语料不是 Qwen3.5 任务格式，需要构建 chat/CoT/tool/agent + Wikipedia 混比语料。

### 13.2 下一阶段

1. 构建 1M token 混比实验：通用 / chat / wiki / CoT+tool。
2. 跑 M1–M5，每组 real / no-reader / control。
3. 做严格污染审计，确保评测题不在语料中。
4. 选最优 mix 跑 3 seeds；如果 positive，再进入 5M–20M。
5. 真实 vLLM / SGLang / CompileForge 产品化放在科学确认之后。


---

## 14. 2026-09-02 第二十五轮：M1–M5 混合语料与严格污染审计

> 详细见 `docs/round-25-mix-corpus.md`。

### 14.1 坐标

- ✅ `build_mix.py`：五类来源按 token 比例混合，支持 Qwen tokenizer、manifest、`--exclude-qa`。
- ✅ 本地 ModelScope 语料已落地：alpaca-cleaned / wikitext / Opus CoT / MSAgent-Bench dev。
- ✅ M1–M5 1M token 混合语料已生成，均在 `data/mixes/`（gitignore）。
- ✅ 严格污染审计：150 题全部 low，无 critical/high。
- ✅ `run_phase0.py`：数字归一化 + 逐题 QA 进度日志。
- ✅ `run_mix_batch.sh`：WSL 批量三线 QA 入口。

### 14.2 下一步

1. WSL 跑 `scripts/run_mix_batch.sh --mixes M1 M2 M3 M4 M5 --seeds 0`。
2. 对比各 mix 的 real / control / no-reader 150 QA EM。
3. 选择最优 mix 后跑 3 seeds。
4. 只有在新做对题目仍不在语料中（污染审计已保证）且 real 稳定 > no-reader 时，才进入 5M–20M。

---

## 15. 2026-09-03 第二十六轮：机制分析优先与语义对齐证据

> 详细见 `docs/round-26-systematic.md`。

### 15.1 坐标

- ✅ M1 三线 150 QA 已完成：real 50.7% / control 52.7% / no-reader 53.3%。
- ✅ val loss 上 real < control < no-reader，但任务级 PLE 无净收益。
- ✅ control 证明“注入扰动”本身会伤害 BoolQ。
- ✅ 发现 control 也能做对 Newton/Shakespeare 等“知识型”题，单纯“答案不在语料中”不足以证明 PLE 语义对齐。
- 🔄 M2–M5 后台运行中。
- ⚠️ 下一阶段从“继续堆 mix”转向“机制与可解释性分析”。

### 15.2 阶段调整

1. Phase A：机制分析（最高优先）
   - reader 参数有效性；
   - CKA / probe / activation patch / logit lens；
   - BoolQ 退化定位；
   - 语义对齐证据报告。
2. Phase B：固定外部评测
   - 固定 LM probe；
   - 固定 QA/BoolQ；
   - 不再用各自语料 val loss 选 mix。
3. Phase C：训练与门控修正
   - 调整注入层/scale/gate；
   - 增加 BoolQ/QA 格式数据；
   - 确认后再进入 5M–20M。
4. Phase D：RL 门禁
   - 必须满足 real > control、real > no-reader、有 real 独有非记忆新做对、BoolQ 不退化，才做 RL。
5. Phase E：产品化
   - serving A/B + CPU 100 tok/s。

---

## 16. 2026-09-03 第二十七轮：暂停 M2–M5，转向流形对齐与机制验证

> 详细见 `docs/round-27-manifold-alignment.md`。

### 16.1 坐标

- ✅ M2–M5 已暂停，不再继续混比微调。
- ✅ 方向调整：优先机制验证和 case 分析。
- ✅ 完成流形/语义空间对齐调研：
  - Procrustes / CCA；
  - Manifold alignment；
  - Gromov-Wasserstein / Optimal Transport；
  - Contrastive / InfoNCE / MMD；
  - CKA / local neighbor overlap / intrinsic dimension。
- ⚠️ 下一步先测 PLE e_t 与 Qwen hidden 的可对齐性。

### 16.2 下一步

1. CKA / Procrustes / kNN overlap 诊断。
2. reader 参数有效性与 activation patching。
3. BoolQ 错误分类与 logit lens。
4. layer/scale/gate 扫描。
5. 设计 manifold alignment / contrastive / KL 约束 loss。

---

## 17. 2026-09-03 第二十八轮：第一批机制验证结果

> 详细见 `docs/round-28-mechanism.md`。

### 17.1 已完成

- ✅ 新增 `mechanism_alignment.py`：CKA / Procrustes / kNN overlap / intrinsic dimension / reader 参数与 gate 统计。
- ✅ 新增 `mechanism_logit_patch.py`：logit-level activation patching，no-reader / real / control / random / zero。
- ✅ PLE e_t 与 Qwen hidden 的全局线性对齐和局部邻域对齐都很弱：
  - CKA 约 0.15–0.22；
  - Procrustes alignment 约 0.01–0.05；
  - kNN overlap 约 0.068–0.084，随机基线 0.039；
  - PLE intrinsic dimension 约 766，Qwen 约 37–78。
- ✅ real 与 control 都会提高 next-token entropy；random/zero 接近 no-reader。
- ✅ 当前效应主要来自“注入 PLE 类向量”，而不是“真实 token 顺序的语义内容”。

### 17.2 已完成后续补充

- ✅ 完整 150 题 logit-level patching 已完成：real 总体 logprob -7.94，control -8.04，real 仅 +0.10，逐题 76:74。
- ✅ BoolQ 上 real 优势较明显（+0.47），NQ/Trivia 上 control 略优。

### 17.2.5 Scale sweep 结果（BoolQ 50 题）

| scale | real logprob | control logprob | real-control | real entropy |
|---:|---:|---:|---:|---:|
| 0.25 | -9.51 | -9.64 | +0.14 | 0.89 |
| 0.5 | -8.78 | -9.16 | +0.38 | 1.20 |
| 1.0 | -7.62 | -8.21 | +0.59 | 2.23 |
| 2.0 | -7.46 | -7.19 | -0.27 | 3.95 |

- real 优势在 1.0 附近最大，2.0 时 control 反超。
- 0.5 是“低破坏 + 仍有真实信号”的候选，但优势仍不足。
- 结论：单纯加大注入强度不是正解，需要 layer/gate/训练目标层面的改进。

### 17.3 下一步

1. 增加 zero/random reader、gate/layer 扫描（scale sweep 已完成）。
2. 设计并验证 contrastive / neighbor / KL 约束 loss 是否能提高对齐指标。
3. 完成 BoolQ logit lens 与错误分类。
4. 在此之前不进入 5M–20M 和 RL。

---

## 18. 2026-09-03 第二十九轮：数学推导“最有效对齐”

> 详细见 `docs/round-29-alignment-math.md`。

### 18.1 核心观点

- 不应以 CKA / Procrustes 高为目标；
- 应以 **条件增量可解释性** 为目标：

\[
\Delta R^2(Y; E \mid H) = R^2(Y; H,E) - R^2(Y; H)
\]

- 任意 reader 的增益上界是 \(I(Y; E \mid H)\)；
- 最优 reader 应逼近 \(\mathbb{E}[R \mid H,E]\)，其中 \(R = Y - \mathbb{E}[Y \mid H]\)。

### 18.2 建议实验

1. 测量 \(R^2(Y;H)\)、\(R^2(Y;H,E)\)、\(R^2(Y;H,E_\perp)\)；
2. 将 reader 拆成：
   - Key/gate：与 H 局部对齐；
   - Value：与 H 去相关、与任务残差可解码。
3. 增加 loss：
   - \(L_{\text{align}}(K(E), Q(H))\)
   - \(L_{\text{task}}(V(E), R)\)
   - \(\|\mathrm{Corr}(V(E),H)\|^2\)
   - gate 稀疏度 / entropy
   - KL(reader_on || reader_off)
4. 只有看到明确的正 \(\Delta R^2\) 与 real-specific 新正确题，才进入 5M–20M / RL。

---

## 19. 2026-09-03 第三十轮：多视角数学推导

> 详细见 `docs/round-30-multimath-alignment.md`。

### 19.1 核心

从 8 个数学视角重新定义“最有效对齐”：

- 信息论：最优 reader 逼近条件期望，增益上界是 \(I(Y;E\mid H)\)；
- 线性代数/谱：Value 用 \(E_\perp\)，Gate 用共享子空间；
- 随机矩阵/高维统计：高维 E 噪声会稀释信号，需 ridge/PCA/PLS；
- 最优传输/度量几何：应该做加权/条件对齐，而不是全局对齐；
- RKHS/核方法：测条件 HSIC，判断是否需要非线性 reader；
- 图谱/谱聚类：Gate 需要局部图对齐，可测 SpecAlign；
- 优化动力学：检查 reader 输出是否真的与任务残差相关；
- 流形假设：先压缩到低维记忆流形，再对齐。

### 19.2 后续实验排序

1. 增量 R² 诊断；
2. PCA/PLS 噪声压缩诊断；
3. 高残差子集上的条件对齐诊断；
4. gate/value 分工商；
5. 正信号确认后才进入 5M–20M / RL。

---

## 20. 2026-09-03 第三十一轮：更深数学分支推导

> 详细见 `docs/round-31-deeper-math.md`。

### 20.1 新增视角

- 统计决策论：最小最大风险，防止 BoolQ 退化；
- 因果推断：real 相对 control 的 CATE，避免把相关性当因果；
- 贝叶斯/GP：把 memory 看成后验更新，用核有效维度解释样本需求；
- 微分几何：注入应落在 Qwen hidden 切空间，测投影比例 \rho；
- 最优控制：按层梯度范数选注入层；
- 拓扑/持久同调：看记忆是否引入新结构；
- 信息几何：测记忆方向与 task score 的匹配度。

### 20.2 统一形式

\[
\text{有效对齐}
=
\text{条件信息}
+
\text{门控可对齐性}
+
\text{切空间兼容性}
+
\text{去冗余}
+
\text{低维压缩}
\]

### 20.3 后续实验

1. 线性/核 \(\Delta R^2\)；
2. 切空间投影比例 \rho；
3. 分层梯度范数 \(c_l\)；
4. real−control 的 CATE；
5. 新 reader：低维瓶颈 + 局部窗口 gate + 去相关 value + 稀疏门控。

---

## 21. 2026-09-03 第三十二轮：第一性原理——对齐的本质

> 详细见 `docs/round-32-first-principles-alignment.md`。

### 21.1 核心

- 对齐的本质是**条件充分性**，不是 CKA / Procrustes 几何相似；
- 最优 reader 应逼近：
  \[
  \Delta^*(H,E)=E[Y|H,E]-E[Y|H]
  \]
- 几何相似既不充分也不必要。

### 21.2 四命题

- A：任意 reader 增益上界是条件互信息；
- B：线性 reader 增益上界是线性增量 R²；
- C：在线性残差意义下，Value 去相关不损失增量信息；
- D：几何对齐既不充分也不必要。

### 21.3 下一步

1. 定义记忆任务族；
2. 训练 oracle reader 得到上界；
3. 测线性/核增量 R²；
4. 测 E_perp 增量；
5. 若 oracle 也无法提升，则记录为负面证据。

---

## 22. 2026-09-03 第三十三轮：完整证明

> 详细见 `docs/round-33-proofs.md`。

已形式化证明：

- A：任意 reader 增益上界是 \(I(Y;E|H)\)；
- B：线性 reader 增益上界是 \(E_\perp\) 上的增量 R²，最优 reader 是残差投影；
- C：Value 去相关不损失线性增量信息；
- D：几何对齐既不充分也不必要；
- S：存在理论最优表示 \((H,E)\)；
- H：最优注入是 Hilbert 空间正交投影到新增信息子空间。

---

## 23. 2026-09-03 第三十四轮：理论指标实证

> 详细见 `docs/round-34-empirical-theory.md`。

### 23.1 结果

- 梯度残差增量 R²：
  - 512 token：ΔR²(E|H)=+0.0024，ΔR²(E⊥|H)=+0.0031；
  - 1024 token：ΔR²(E|H)=+0.0058，ΔR²(E⊥|H)=+0.0051。
- E⊥ 保留了约 88% 增量，支持“Value 去相关不损失信息”。
- 与 real−control 很小、scale=1.0 后反超等现象一致。
- 当前记忆线性增量很小，尚不支持直接进入 5M–20M。

### 23.2 下一步

1. Oracle MLP 非线性上界；
2. PCA/PLS 压缩诊断；
3. 高残差 token 子集诊断；
4. Key/Value 分工商实验。

---

## 24. 2026-09-03 第三十五/三十六轮：预注册与实测

> 预注册：`docs/round-35-preregistered.md`
> 实测：`docs/round-36-experiment-results.md`

### 24.1 预注册判读

- PCA 压缩若保持 → 低秩信号；若快速下降 → 高维分散；若全零 → 记忆本身无信息。
- 高残差子集若更大 → 记忆帮助难 token；若低残差为负 → 简单 token 受伤。

### 24.2 实测结果

- PCA：r=256 仅 +0.002，远低于全维度 +0.0058，信号分散在高维；
- 高梯度 token：ΔR² = -0.0048；
- 低梯度 token：ΔR² = +0.0149；
- 说明当前 PLE 主要提供高频局部 n-gram 信号，而不是长尾知识信号。

### 24.3 决策

- 不能直接无监督 PCA 压缩；
- 需要 PLS / 监督方向或重新定义记忆任务；
- 当前证据仍不足以进入 5M–20M / RL。

---

## 25. 2026-09-03 第三十七/三十八轮：PLS 与稀有 token

> 预注册：`docs/round-37-preregistered.md`
> 实测：`docs/round-38-results-pls-rare.md`

### 25.1 结果

- PLS r=64 时 ΔR²=+0.0074，超过全维度 +0.0058；
- 稀有 token ΔR²=+0.0152，约为常见 token +0.0079 的两倍；
- 说明记忆信号可用监督低秩方向提取，且更偏向长尾 token。

### 25.2 决策

- 放弃无监督 PCA；
- 改用 PLS/有监督低秩记忆瓶颈；
- 用稀有 token 子集重测 real vs control；
- 当前仍不足以进入 5M–20M / RL。

---

## 26. 2026-09-03 第三十九轮：Oracle MLP 非线性上界

> 详细见 `docs/round-39-oracle-mlp-results.md`。

### 26.1 结果

- 线性 H+E：ΔR²=+0.0058；
- MLP H+E：ΔR²=+0.0206；
- MLP H+E⊥：ΔR²=+0.0228；
- MLP H+PLS64：ΔR²=+0.0122。

### 26.2 结论

- 非线性可将记忆增量提高约 3–4 倍；
- E⊥ 在非线性下表现最好；
- 当前 reader 容量不足，应实现非线性 Value 路径。
- 当前 reader 容量不足，应实现非线性 Value 路径。

---

## 27. 2026-09-03 第四十轮：MLP Value Reader 数学理论

> 详细见 `docs/round-40-mlp-reader-theory.md`。

### 27.1 核心数学

\[
\Delta R^2_{	ext{mlp}} / \Delta R^2_{	ext{lin}}
pprox
\|g\|^2 / \|g_{	ext{lin}}\|^2
\]

- 实验比值 3.55，说明线性只提取约 28% 信号；
- E⊥ 与 R 正交补相关，去冗余且不损失信息；
- PLS 优化线性相关，对非线性模型次优。

### 27.2 设计指导

\[
\Delta=g(h,e)\cdot \mathrm{MLP}(E_\perp)
\]

建议加 aux loss：

\[
\|\mathrm{MLP}(E_\perp)-\widehat R\|^2
\]

并用稀有 token / 高不确定性 gate。

---

## 28. 2026-09-03 第四十一轮：MLP Value Reader 原型实验

> 详细见 `docs/round-41-mlp-residual-reader.md`。

### 28.1 结果

- 新增 MLPValueReader / `--reader mlp` / 残差监督训练脚本；
- 残差监督 best val R² = 0.288，接近 Oracle；
- BoolQ 8 题：real/control/random/zero 几乎无差异；
- 说明“value 能预测 R”不等于“value 会针对真实 PLE 内容选择性注入”。

### 28.2 结论

- 需要训练 value 对 E 特有信息的响应；
- 需要恢复 gate 训练；
- 需要 real-vs-control 判别损失或条件化耦合。
