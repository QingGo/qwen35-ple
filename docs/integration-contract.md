# 四仓库交互契约 v1（integration contract）

> 权威位置：`qwen35-ple/docs/integration-contract.md`
> 参与仓库：**EngramDB**（存储）、**engram-peft**（模型/训练）、**qwen35-ple**（实验编排）、
> **LLM-CompileForge**（推理编译/runtime）。
> 冻结日期：2026-08-30。**本契约是唯一权威；各仓库的实现与文档以本文件为准。**

---

## 0. 版本与变更纪律

| 规则 | 内容 |
|---|---|
| 语义 | 契约版本：`v1`（当前）。三档：`+`（additive，字段/枚举/新符号）、`!`（违反即不兼容）、`~`（注释性澄清）。 |
| 只增不改 | **禁止**修改既有字段语义、删除符号、改变既有枚举值含义。 |
| ABI 演进 | 新函数一律用新符号（后缀 `_v2`）；旧符号冻结，永不改变行为。 |
| 数据格式 | manifest/json 字段只增不改；读端对未知字段必须忽略（向前兼容）。 |
| 变更流程 | ① 提案（在本文件「变更日志」起草，标注 owner）；② 四仓库 owner 确认；③ 实现分两步：先支持新/旧双读，再统一；④ bump 一档并更新 changelog。 |
| 守门 | 每个契约有 owner 测试（见 §5）；跨仓库 PR 必须附带「契约影响：`+`/无」标注。 |

**依赖方向（严格无环）**：

```
EngramDB ──► engram-peft（可选磁盘注入，仅 Python 侧）
EngramDB ──► LLM-CompileForge（仅 C ABI / 视图文件，无 Python 依赖）
engram-peft ──► qwen35-ple（消费方）
EngramDB ──► qwen35-ple（消费方）
```

---

## 1. 术语（全局唯一语义）

| 术语 | 定义 |
|---|---|
| gram / ngram | 一个连续 token 元组（`ngram_sizes=[2,3]` → bigram 与 trigram）。 |
| head | 每个 ngram 尺寸下的哈希头（`n_head_per_ngram=8`；bigram 8 头 + trigram 8 头 = 16 头）。 |
| rowid | 单个 (ngram 尺寸, head, gram) 的确定性哈希，`u64`，取值域见 `PLE_QWEN_V1` 规格。 |
| 行 / row | rowid 对应的固定宽向量。Qwen 口径 = **160 字节 FP8（E4M3，无打包）**。 |
| e_t | 单 token 的 16 行拼接（**head 0..7 = bigram，head 8..15 = trigram，行主序**），2560 维。 |
| 记录 / record | Store-P 视图中的定长槽（默认 2560B，`--slot` 可 4096 对齐填充）。记录 i 对应 keys 文件第 i 个 gram 的 16 行。 |
| PleSpec | 行语义注册表 id：`PLE_QWEN_V1`（Qwen 官方）/ `ENG_DEEPSEEK_V1`（论文口径）。 |

---

## 2. 契约 C1：存储契约（EngramDB → 使用方）

### C1.1 行语义（冻结）

- 参考实现：`EngramDB/crates/engramdb-keygen`（`PleSpec::real` = `PLE_QWEN_V1`）。
- 接口：`rowids_for_seq(ids: &[u32]) -> [ [u64; 16]; T ]`（每 token 16 行，head 序）。
- **golden**：EngramDB 仓库测试内固定向量（min 4096 token 级）；跨仓库对拍由
  `qwen35-ple/tests` 与 `LLM-CompileForge/tests` 引用同一 golden 文件（路径由
  EnGramDB markdown 导出，见 §5）。
- 反量化：FP8 字节布局与官方 `weight_scale` 的应用语义由 EngramDB 的
  `view verify` / `bitwise_check.py` 实现并作为**唯一位级真源**；使用方只消费
  「反量化后的数值」或「FP8 原始字节」，不得自行定义。

### C1.2 视图（Store-P）文件格式（已存在即契约）

- 文件对：`<view>.bin` + `<view>.manifest.json`（manifest 字段：`grans`、`heads`、
  `slot_bytes`、`record_bytes`、`build_seconds`、`build_mb_s`、`rows`、`source`）。
- 记录索引：`物理槽位 i（0-based）× slot_bytes`；读取方禁止假设 4KB 对齐步长，
  一律用 manifest 的 `slot_bytes`。
- **keys 交付要求（+）**：供训练/推理消费的视图**必须**附带 keys 文件
  （每行 1 个 rowid，每 gram 16 行、head 序，与记录顺序一一对应）。
  CLI 现状（`engramdb view build <rows> <n> <view> <keys>`）已满足；
  未来若 manifest 含 LCG 重建参数（`build_view` 的确定性 LCG 固定种子），
  keys 文件仍优先，属可选冗余。
- 模式区分：`build_view`（LCG 随机 gram 流，bench/重建用）与
  `build_view_from_keys`（按 keys 订制的热集视图，**推理用**）。

### C1.3 C ABI（已存在即契约，符号冻结）

`crates/engramdb-python`（cdylib，无 pyo3 依赖）：

| 符号 | 签名 | 错误码 |
|---|---|---|
| `engramdb_store_open` | `(const char* dir, uint64 shards, uint64 rows_per_shard, uint64 width) -> *mut StoreHandle` | NULL = 失败 |
| `engramdb_store_fetch` | `(handle, const u64* rowids, size_t n, u8* out, size_t out_cap) -> i32` | 0 成功；-1 空指；-2 容量；-3 IO；输出 = n×width 字节（行主序） |
| `engramdb_store_width` | `(handle) -> u64` | — |
| `engramdb_store_close` | `(handle)` | — |
| `engramdb_view_open` | `(const char* path) -> *mut ViewHandle` | NULL = 失败 |
| `engramdb_view_read_record` | `(handle, size_t index, u8* buf, size_t buf_cap) -> i32` | 0 成功；-1 空指；-2 容量；-3 IO |
| `engramdb_view_len` | `(handle) -> size_t` | — |
| `engramdb_view_slot_bytes` | `(handle) -> u64` | — |
| `engramdb_view_close` | `(handle)` | — |

**新增（v1 提案，符号冻结规则同表）**：

| 符号 | 签名 | 说明 |
|---|---|---|
| `engramdb_rowids_for_seq` | `(const u32* ids, size_t len, u64* out, size_t out_cap, uint32 ple_spec) -> i32` | 输出 16×len 的 u64（head 序）；`ple_spec`：1=`PLE_QWEN_V1`，2=`ENG_DEEPSEEK_V1`；错误码同上 |
| `engramdb_abi_version` | `() -> u32` | 返回 1；调用方校验 ≥1 |

### C1.4 Python API（已存在即契约）

`engramdb`（pkg `engramdb-python` v0.2）：`Store(path, shards, rows_per_shard, width)`；
`store.fetch(keys) -> [B, 160*dtype]`；`View`；
`install_disk_multi_head_embedding(store)`（位于 `engramdb.integrations`，
**唯一磁盘注入点**，必须在 `get_engram_model` 前调用）。

---

## 3. 契约 C2：模型契约（engram-peft → qwen35-ple）

### C2.1 已存在即契约（冻结）

- `EngramConfig`（`PretrainedConfig` 子类，kw_only dataclass）：现有字段
  （`engram_vocab_size_per_ngram`、`ngram_sizes`、`n_head_per_ngram`、`embedding_dim`、
  `target_layers`、`hc_mult`、`combine_mhc`、`conv_kernel_size`、`conv_dilation`、
  `conv_zero_init`、`gating_zero_init`、`learning_rate_multiplier`、`weight_decay`、
  `tokenizer_name_or_path`、`compressed_vocab_size`、`pad_id`、`seed`、`hidden_size`、
  `clip_grad_per_group`、`enable_telemetry`、`entropy_loss_weight`、
  `backbone_freeze_steps`、`engram_dtype`、`use_sparse_embeddings`、
  `engram_version`、`train_mode`、`wrap_peft`）**语义不再变化**。
- `get_engram_model(model, config, tokenizer, train_mode="engram_only")` 签名冻结。
- `EngramLayer` 构造契约、`MultiHeadEmbedding(primes, embedding_dim_per_head, sparse=True)`
  构造契约冻结（磁盘注入依赖 `sparse=True` 路径，见 C1.4）。
- `train_mode` 取值：`engram_only` / `preserve_trainable` / `full_finetune` 语义冻结。
- 保存格式：`engram_version`（当前 "1.2.4"）状态文件；新版本只增 sidecar 文件。

### C2.2 新增（v1，全部带默认值，不得改既有字段）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `engine` | `str = "deepseek"` | `"deepseek"`（论文对齐路径，行为不变）\| `"qwen_ple"`（PLE-lite，hc=1 等变体） |
| `table_spec` | `str \| None = None` | `"PLE_QWEN_V1"` / `"ENG_DEEPSEEK_V1"`；None = engine 默认 |
| `table_source` | `str = "memory"` | `"memory"` \| `"engramdb:store"` \| `"engramdb:view"` |
| `prime_sizes` | `list[int] \| None = None` | 仅开发/合成表用；提供 16 头素数可绕过真实 320M 行表；生产置 None |
| `table_store_path` | `str \| None = None` | `table_source="engramdb:store"` 时的 Store-I 目录 |
| `table_model_dir` | `str \| None = None` | 真实 Qwen checkpoint 目录，用于自动读取 FP8 `weight_scale` |
| `table_shards` | `int \| None = None` | Store 分片数；`PLE_QWEN_V1` 默认为 128 |
| `table_rows_per_shard` | `int \| None = None` | 每分片行数；`PLE_QWEN_V1` 默认为 2_500_012 |
| `table_width` | `int \| None = None` | 行字节宽；`PLE_QWEN_V1` 默认为 160 |
| `table_dtype` | `str = "float32"` | Store 行 dtype：`"float32"` / `"float8_e4m3fn"` |
| `table_scale` | `float \| None = None` | 显式 FP8 `weight_scale`；与 `table_model_dir` 二选一 |
| `table_cache_size` | `int = 4096` | Disk MultiHeadEmbedding LRU 容量 |

> **自动消费**：当 `table_source="engramdb:store"` 时，`get_engram_model()` 会自动打开
> EngramDB Store 并调用 `engramdb.integrations` 注入 Disk MultiHeadEmbedding，
> 调用方无需手动 import `install_disk_multi_head_embedding` / `install_real_qwen_ple_embedding`。


约束：`engine="qwen_ple"` 时（`table_spec="PLE_QWEN_V1"`）的语义由 C1 定义；
`engine="deepseek"` 的所有行为与现有版本**完全一致**（回归保障）。
引擎行为由 config 字段控制——**不新增 `get_engram_model` 参数**（签名冻结是
「尽可能不变」的核心）。

### C2.3 hash 映射

- `NgramHashMapping` 现有实现 = `ENG_DEEPSEEK_V1`（冻结）。
- `PLE_QWEN_V1` 的新映射类：算法以 engramdb-keygen 为准并带 golden；
  建议在 engram-peft 引入 `HASH_SPECS` 注册表（id → 映射工厂），默认注册
  既有实现（id 名 `"deepseek_v1"` 与表 spec 对齐）。
- 双仓库对拍：`qwen35-ple/tests/test_hash_golden.py` 载入 EnGramDB 导出 golden，
  与 engram-peft 的 `PLEC_QWEN_V1` 映射逐值比较（CI 门禁）。

---

## 4. 契约 C3：推理契约（LLM-CompileForge ↔ EngramDB）

### C3.1 ABI/元数据（已有即契约 + 新增）

`include/sfa_abi.proto`（proto3，**只增字段**）：

- 新增（+）：
  - `enum SfaWeightSource { SFA_WEIGHT_HF_SAFETENSORS = 0; SFA_WEIGHT_EMBEDDED = 1; SFA_WEIGHT_ENGRAMDB_VIEW = 2; }`
  - `SfaWeightEntry { ... uint32 source = 3; string external_ref = 4; }`（`source=2` 时
    `external_ref` = 视图文件路径，相对 artifact 根）。
  - `dtype_code` 增加 `7 = fp8_e4m3`（`SfaConstant.dtype_code` 与权重 dtype 共用）。
- 读取方规则：未知字段忽略；`source` 缺省 = `0`（行为不变）。

### C3.2 视图作为外部权重源（语义）

- 权重形状声明：编译期在 artifact 元数据中声明 `[rows, cmp_slot]`（`cmp_slot` =
  manifest `slot_bytes`）；运行时打开视图文件并按 `slot_bytes` 步长寻址。
- mmap 与 C ABI 读取可并存：mmap 路径要求视图文件页对齐（slot 2560B 时由
  `--slot 4096` 保证）；C ABI 路径（`engramdb_view_*`）无对齐要求。
- **两条路径输出必须位级一致**；默认推荐 C ABI（不依赖 mmap 细节）。

### C3.3 运行时职责边界

| 职责 | 归属 |
|---|---|
| rowid 计算（gram → 16 行 rowid） | 引擎侧（keygen 参考实现/`engramdb_rowids_for_seq`） |
| 行/记录读取 | EngramDB（C ABI / mmap 视图） |
| 预取窗口与计算重叠（PrefetchPlanner） | CompileForge runtime（输入 = rowid 流；输出 = 进入 staging 的记录） |
| 反量化（FP8→BF16/F32） | 视图侧数值一致性由 EnGramDB 保证；编译侧 `sf.fp8_dequant` 或 HAL 内核 |

- 预取语义：`fetch(keys) -> records` 无副作用、可乱序、可并发；CompileForge
  承诺「PLE 读取 ≤2% 尾差」作为集成验收指标（推理设计文档 §7）。
- **e_t 数值一致性**：编译产物中 e_t 的 head 顺序必须为 C1 冻结序；
  CompileForge 的 golden 测试引用 EnGramDB 导出向量。

### C3.4 dylib ↔ EngramDB 连接方式

- 编译后 dylib **不静态链接** EngramDB（保持 artifact 可移植）；runtime 通过
  `dlopen/dlsym` 解析 C1.3 符号；找不到符号 = 降级为纯内存权重（无 PLE）或报错
  （由模型配置的 `table_source` 决定）。

---

## 5. 契约 owner 与守门测试

| 契约 | owner | 守门 |
|---|---|---|
| C1 | EngramDB | `view verify` 位级一致；golden 向量导出（min 4096 token）；C ABI 冒烟单测 |
| C2 | engram-peft | 既有全量测试（保证 `engine="deepseek"` 回归）；`qwen_ple` 引擎的 golden 对拍（qwen35-ple 侧） |
| C3 | LLM-CompileForge | cos_sim ≥ 0.9999（数值路径）；A/B tg 曲线（100 tok/s 验收）；`generate_golden_outputs` 引用真实 PLE 行 |
| C4 | qwen35-ple | schema 一致性冒烟（`docs/` 与实现同步检查） |

---

## 6. 变更日志

| 版本 | 日期 | 内容 |
|---|---|---|
| v1 | 2026-08-30 | 初始冻结。**已存在即契约**：C1.1 行语义、C1.2 视图/manifest/keys、C1.3 C ABI 全表、C1.4 Python API、C2.1 config/签名/train_mode、C3.1 现有 sfa_abi.proto。**本次新增（+）**：`PLE_QWEN_V1` 注册、keys 交付要求、`engramdb_rowids_for_seq`/`engramdb_abi_version`、C2.2 三字段、C2.2 开发字段 `prime_sizes`、C3.1 `SfaWeightSource`/`external_ref`/dtype 7、C3.3 职责边界与 e_t 一致性。 |
