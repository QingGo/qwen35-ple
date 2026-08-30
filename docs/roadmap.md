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
- [x] A0/A1 评测对比入口（`scripts/run_eval.py` + `eval/protocol.py`）
- [ ] M0 e2e（TinyLlama/Qwen + 完整 engram-peft 环境）
- [ ] CPT 消融（A0/A1）
- [ ] 100 tok/s 推理闭环

---

## 3. 技术债清单

### 3.1 契约与实现的剩余缺口

- C2 的 `engine` / `table_spec` / `table_source` 已进入 `EngramConfig`，`QwenPleHashMapping` 已落地。
- 仍缺：完整 PLE-lite 前向与 `refs/qwen4_exp_modeling.py` 的逐位/数值对拍。
- `table_source="engramdb:view"` 的推理侧读取尚未在 engram-peft 中接通。

### 3.2 缺少 golden/位级一致性防线

- 已建立本仓 golden + engram-peft 跨仓 golden（`tests/test_cross_repo_hash_golden.py`）。
- 待补：与 `refs/qwen4_exp_modeling.py` 的官方前向对拍。
- `refs/qwen4_exp_modeling.py` 目前仍存放在 EngramDB，需要固定跨仓引用或拷贝以避免漂移。

### 3.3 缺少完整可运行闭环

- 已有 M0 quick 磁盘自检，但完整模型 e2e 需要安装 peft/transformers 依赖后运行
  `scripts/run_m0_smoke.py --e2e`。
- 配置样例仍主要在“初值/编排”层面，不是完整训练入口。

### 3.4 资产与可重建性

- 有 Store-P 构建脚本，但尚未固化真表资产的本地路径/校验报告。
- 没有语料 provenance/许可记录。
- `engram_vocab_size_per_ngram` 初值已修正为每个 n-gram 8 头总桶数；仍需在真实表上验证。

### 3.5 评测与决策机制

- 已有 A0/A1 JSON 对比协议与报告入口，但还没有真实评测执行器。
- 没有“A1 负增益即止损”的正式门禁。

### 3.6 工程过程

- 还没有 CI workflow；有 session log 与跨仓 golden 测试。
- 兄弟仓库已有成熟“门禁 + 文档同步”习惯，本仓需要对齐。

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
- 固定 `refs/qwen4_exp_modeling.py` 引用

**Gate：** `make check` 绿；golden 测试可复现。

### Phase 1：M0 最小纵切

- 用 EngramDB 真表/合成表构建 Store-P 视图。
- 接入 `engramdb.integrations`，跑 TinyLlama 或 Qwen3.5-0.8B 磁盘注入前向/生成。
- 验证 `view verify` 位级一致、e_t 形状、无 NaN。

**Gate：** 一条命令可复现；e2e 可跑。

### Phase 2：M1 实现 `engine="qwen_ple"`

- engram-peft 只增 `engine` / `table_spec` / `table_source` 字段。
- 实现 Qwen 原生日志映射（`PLE_QWEN_V1`）与 PLE-lite `hc=1` 层。
- 与 `refs/qwen4_exp_modeling.py` 做 4096 token 位级对拍。
- 保持 `engine="deepseek"` 全量回归。

**Gate：** 位级一致；DeepSeek 路径不回归。

### Phase 3：M2 小规模消融（决策点）

- 实现 config/data/train/eval 编排。
- 先跑 A0 vs A1（0.2-1B tokens 量级）。
- 输出知识 recall、长上下文、基础 reasoning 报告。

**Gate：** 报告 + go/no-go；A1 不优于 A0 则停止放大。

### Phase 4：后训练 + 推理闭环

- A1 为正时进入 SFT/RL（M3）。
- 与 CompileForge P0-P5 并行；本仓负责模型/视图资产和 A/B 验收协议。

**Gate：** 100±10 tok/s；PLE 开销 ≤2%；训练/推理数值一致。

---

## 6. 稳健前进三原则

1. **先最小纵切，再规模化。** 每一环节没有可复现命令和 gate 就不进入下一阶段。
2. **跨仓改动只走契约。** 只增字段/符号，旧行为冻结，避免四仓互相踩脚。
3. **决策留下证据。** 无论正负结果都写入 roadmap/session log，附可复现命令。
