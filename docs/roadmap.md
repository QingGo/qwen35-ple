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
