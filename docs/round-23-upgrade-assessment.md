# Round 23：engram-peft 1.2.7 / EngramDB 0.2.12 升级评估与后续计划

> 日期：2026-09-02
> 触发：engram-peft 发布 v1.2.7，EngramDB 发布 v0.2.12。
> 结论：**qwen35-ple 需要升级，且本轮升级主要是“收口/协议对齐”，不是科学结论变更。**

---

## 1. 上游新增了什么

### 1.1 engram-peft v1.2.7

- `EngramConfig.engine="qwen_ple"`、`table_spec="PLE_QWEN_V1"` 正式发布。
- `table_source="engramdb:store"` 由 `get_engram_model()` 自动消费，无需手动调用
  `install_disk_multi_head_embedding` / `install_real_qwen_ple_embedding`。
- 新增磁盘表配置字段：`table_store_path`、`table_model_dir`、`table_shards`、
  `table_rows_per_shard`、`table_width`、`table_dtype`、`table_scale`、`table_cache_size`。
- 新增 `QwenPleHashMapping` 与 `create_hash_mapping()` 工厂。
- `engine="deepseek"` 保持默认且行为不变，符合契约 C2 的“只增不改”。

### 1.2 EngramDB v0.2.12

- 存储/索引：
  - `DiskSlotIndex` v3 单文件 + offset table；
  - 原生 `engramdb slot-index build|verify` 与 `view build --slot-index`；
  - `view build --keys-stream`、manifest `keys_out`。
- Serving 协议：
  - `PleMemory` / `PleSequence` / `PleSequenceStore`：按请求/序列维护 n-gram history，
    统一 Store-I 与 Store-P 读取；
  - `BundleManifest` / `TargetReaderRegistry`：跨仓统一 reader 加载协议；
  - `PleMemoryAdapter` / `TargetReaderHook` / vLLM/SGLang 注入别名：面向 serving 的
    PyTorch 通用适配层。
- 验证与发布：
  - 真表 Arrow IPC、serving A/B、真表性能门禁；
  - v0.2.12 已发布，CI/release 全绿。

---

## 2. qwen35-ple 当前是否兼容

| 面 | 状态 |
|---|---|
| 配置桥接 | ✅ `EngineConfig` → `EngramConfig` 已支持 C2 新字段 |
| hash/前向 golden | ✅ 已引用 `QwenPleHashMapping`，跨仓 golden 在 CI 中 |
| Store-P / SlotIndex | ✅ 已 re-export EngramDB `SlotIndex` / `DiskSlotIndex`，保留轻量 fallback |
| live-store / 训练 | ✅ `LiveETStore` / `LiveETViewStore` / `run_phase0` 已在用 |
| 通用 serving 协议 | ✅ 已接入 `TargetReaderRegistry` / `BundleManifest` 薄封装（qwen35 只注册具体 reader） |
| reader checkpoint | ✅ `run_phase0 --save-reader` / `--load-reader` 已支持；`short_conv` 尚未并入 checkpoint |
| 本地 SlotIndex fallback | ⚠️ 生产入口仍保留本地实现，按 EngramDB 路线图应收敛为 canonical |

因此：**没有破坏性不兼容；升级是“值得做”而非“被迫重构”。**

---

## 3. 本仓已做的升级收口

- `pyproject.toml`：`engram-peft>=1.2.7`，`engramdb-python>=0.2.12`。
- `uv.lock`：本地 path 依赖版本记录同步到 1.2.7 / 0.2.12。
- CI：EngramDB checkout `v0.2.12`，engram-peft checkout `v1.2.7`，避免漂移到未发布 master。
- `scripts/wsl_repro.sh`：默认 EngramDB 版本改为 0.2.12。
- 配置注释更新：`table_source` 已由 engram-peft 1.2.7 自动消费。
- 新增 `src/qwen35_ple/reader_registry.py`：
  - 基于 `engramdb.TargetReaderRegistry` 注册 `official_source_qwen_v1` / `engram_v1` / `simple_v1`；
  - 提供 `save_reader()` / `load_reader()`，checkpoint 内保存 reader 类型、版本、config 与 `state_dict`。
- 新增 `src/qwen35_ple/serving/bundle.py`：
  - `make_bundle` / `save_bundle` / `load_bundle` / `open_bundle_memory`，兼容 `engramdb-bundle-v1`。
- 新增 `src/qwen35_ple/serving/adapter.py`：
  - `QwenReaderServingAdapter` / `install_qwen_reader_adapter`；
  - `install_qwen_reader_adapter_from_bundle`：从 bundle + registry 直接装配 memory + reader + hooks；
  - `install_vllm_reader_from_bundle` / `install_sglang_reader_from_bundle`：vLLM/SGLang 风格别名。
- `run_phase0.py` 新增 `--save-reader` / `--load-reader` / `--save-bundle`：
  - 支持 `{mode}` / `{seed}` 路径模板；
  - loaded checkpoint 模式跳过训练，直接跑 val/QA；
  - `simple` reader 的 `ShortConv` 状态也已纳入 `extra_state` 保存/加载；
  - `--save-bundle` 可同时生成 EngramDB `BundleManifest` 兼容的部署 bundle。
- 新增 `tests/test_reader_registry.py`：覆盖 config、save/load、short_conv extra、bundle、serving adapter。
- 本地全量测试：`43 passed, 4 skipped`。

---

## 4. 后续开发计划（按优先级）

### Phase 1：工程收口（建议本周）

1. ✅ **接入 EngramDB serving 协议，不重复造轮子**
   - `qwen35_ple/reader_registry.py` 已落地为薄封装：
     - 内部使用 `engramdb.TargetReaderRegistry`；
     - 已注册 `official_source_qwen_v1` / `engram_v1` / `simple_v1`；后续扩展 `dual_layer_v1` / `multi_layer_v1` / `lora_v1`。
   - `qwen35_ple/serving/bundle.py` 已落地，使用兼容 `engramdb.BundleManifest` 的 v1 schema：
     - 记录 backbone 路径、PLE 表/视图描述、reader 类型/版本/checkpoint、
       memory 参数（head_dim/num_heads/scale/slot_index）。
2. ✅ **`run_phase0.py` 支持 `--save-reader` / `--load-reader`**
   - 保存 reader `state_dict` + reader 版本/config；
   - 加载后直接跑 val/QA，不再每次重训；
   - `simple` 的 `ShortConv` 也已通过 `extra_state` 完整保存/加载。
3. **完成 Phase A2 收尾**
   - 汇总 Store-P + access-order 1M 三线结果与 fetch timing；
   - 与 Store-I Phase A 结果对比，确认磁盘路径不改变科学结论。

### Phase 2：科学确认（决策点）

1. 跑完 `assets/qa-expanded-150.json` 三线 exact-match：
   - no-reader / control / real；
   - 3 seeds；
   - 记录 PPL + QA EM。
2. 如果有条件进入 5M token 正式实验：
   - 3 seeds；
   - real / control / no-reader；
   - reader 变体：
     - `hc_mult=4`（对齐官方 PLE）
     - 双/多层注入
     - reader + LoRA / 部分解冻
3. Go / No-Go 门槛（沿用 round-22）：
   - Go：3 seeds 下 real 稳定超过 control，且 PPL 或 QA 至少一项超过 no-reader；
   - No-Go：5M 下仍无稳定正增益，则记录负结果并停止放大。

### Phase 3：产品化与推理

1. 用 `PleMemoryAdapter` / `TargetReaderHook` 写 vLLM / SGLang 薄适配层。
2. 先让 no-reader 基线跑在 vLLM/SGLang 上，验证加速与数值一致性。
3. 再把 `OfficialSourceQwenReader` + bundle 接入，做真表 serving A/B。
4. 与 LLM-CompileForge 的 CPU 100 tok/s 目标并行，推进 C3 契约验收。

---

## 4.5 WSL 实际操作结果（2026-09-02）

已通过 Tailscale SSH + Windows 计划任务在 WSL 完成：

1. **保存 1M real reader**
   - 配置：1,000,000 token，`reader=official`，layer 8，500 steps，lr 1e-4，seed 0
   - 产物（在 WSL `/home/zeng/qwen35-ple/outputs/`）：
     - `reader-real-seed0.pt`（183 MB）
     - `bundle-real-seed0.json`
     - `phase0-live1m-real-save.json`
   - 保存时 val loss = 2.7892，PPL = 16.27

2. **用 `--load-reader` 跑 150 题 QA（免重训）**
   - 命令加载上述 reader，未训练，直接 running exact-match
   - 产物：`outputs/phase0-live1m-qa150-loaded.json`
   - 结果：

| 指标 | 值 |
|---|---:|
| QA EM（150 题） | 42.0% |
| TriviaQA | 66.0% |
| NQ | 0.0% |
| BoolQ | 60.0% |

说明：这是 seed 0 single-real-reader 的 loaded-QA 结果；样本仍不足以单独定论，但验证了
“保存 reader → bundle → 免重训跑 QA”的工程闭环。

---

## 5. 风险与注意事项

- **不要同时引入两套 reader registry/bundle 协议**：EngramDB 0.2.12 已提供 canonical
  协议，qwen35 只做具体 reader 的注册和实验逻辑。
- **保持只增契约**：如果需要在 `BundleManifest` 中扩展 qwen35 专有字段，应先更新
  `docs/integration-contract.md` 变更日志，再改实现。
- **本地 SlotIndex fallback**：保留作为无 EngramDB 的轻量测试/开发回退；生产入口
  应统一使用 EngramDB canonical，避免双实现漂移。
- **golden（V126）**：升级 engram-peft 到正式 tag 后应重新评估官方前向 golden；
  若仍漂移，需决定重建 golden 还是维持 xfail，不要静默掩盖。

---

## 6. 结论

- 需要升级：✅ 已完成版本收口以及 reader/bundle/checkpoint 第一版工程落地。
- 下一步重点是把 checkpoint 用于 150 题 QA 和 5M 三线实验；
- 继续扩展 reader 变体、真实 vLLM/SGLang serving 适配；
- 科学上仍以 150 题 QA 和 5M 三线为 Go/No-Go 核心；serving/100 tok/s 排在正收益确认之后。
