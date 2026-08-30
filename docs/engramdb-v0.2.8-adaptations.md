# EngramDB v0.2.8 适配清单（2026-08-30）

> 本轮检查了 EngramDB 新更新（v0.2.8 / master `b04ab1b`），
> 并同步修正了 qwen35-ple 中一个此前未注意到的数值问题：
> **真实 PLE FP8 行必须乘 `weight_scale`**。

---

## 1. EngramDB 新增的可消费能力

| 能力 | 入口 | 对我们的用途 |
|---|---|---|
| 官方 PLE rowid | `engramdb.rowids_for_seq(tokens)` | 替代本仓旧 `PleSpec.rowids_for_seq`，统一走 EngramDB/Rust/C ABI |
| 真实 PLE 元数据发现 | `engramdb.discover_ple(model_dir)` | 读取 `ple_embed_dim`、`hc_count`、`ngram_size`、`ple_layer_ids` 等 |
| FP8 `weight_scale` | `engramdb.load_ple_weight_scale(model_dir)` | 修正 e_t 反量化，必须乘 scale |
| 真实 FP8 行读取 | `engramdb.Store` + `PleDiskGather` | 当前预计算路径已用 |
| 磁盘版 n-gram embedding | `engramdb.ple_adapter.DiskPleNGramEmbedding` | 之后可做 live 读取/训练，不必落 50GB e_t 数组 |
| engram-peft 实时注入 | `engramdb.integrations.install_real_qwen_ple_embedding` | 让 engram-peft 的 `ContextAwareGating + ShortConv` 直接用真实 PLE 表 |
| Store-P 物化视图 | `engramdb.View` | 外盘已有 48GB 全量视图；训练流/推理低延迟候选 |
| 多表/服务 | `Database` / `EngramDBBinaryServer` / `EngramDBClient` | 未来推理服务/多表资产面 |
| vLLM/SGLang 插件 | `engramdb.vllm_plugin` / `engramdb.sglang` | 不直接用于当前 0.8B 实验，但为推理闭环保留 |

---

## 2. 本轮直接完成的适配

### 2.1 修正 FP8 `weight_scale`

**问题**：之前 `src/qwen35_ple/real_ple.py::fetch_e_t` 只把 FP8 转成 float32，
没有乘真实 `weight_scale`。

**影响**：
- 后续如果直接加载官方 PLE 的 `key_proj` / `value_proj` / `conv1d`，
  输入量纲会错误；
- 与 EngramDB 的 `DiskPleNGramEmbedding` / 官方数值路径不再一致。

**修复**：
- `fetch_e_t(..., scale=1.0)` 现在会做：
  ```python
  fp8 = arr.float().numpy()
  return (fp8 * scale).reshape(...)
  ```
- `resolve_ple_weight_scale()` 支持：
  1. 显式传 `--scale`
  2. 从 Qwen3.8 checkpoint 读 `weight_scale`
  3. 回退到 EngramDB 已知的 0.0002

### 2.2 统一 rowid 来源

`rowids_from_tokens()` 现在优先调用 `engramdb.rowids_for_seq()`，
失败时才回退本仓的冻结 `real_spec`。

已验证：
- `engramdb.rowids_for_seq` 与本仓 `PleSpec` 对 5 个真实 token 完全一致。

### 2.3 新脚本：EngramDB v0.2.8 消费冒烟

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_engramdb_v028_smoke.py
```

当前输出：

```text
rowids_for_seq: ok
discover_ple:   ok (ple_embed_dim=2560, scale=0.000199318)
store_fetch_e_t: ok
view:           ok (len=20000096, slot=2560)
ENGRAMDB_V028_SMOKE_OK
```

### 2.4 预计算/探针/adapter 支持 scale

- `scripts/precompute_real_ple_features.py`
  - 新增 `--model-dir` / `--scale`
  - 输出的 `meta.json` 记录 `weight_scale`
- `scripts/run_ple_knowledge_probe.py`
  - 新增 `--model-dir` / `--scale`
  - 使用真实 scale 计算 e_t，并写入 metadata
- `scripts/run_ple_adapter.py`
  - 新增 `--model-dir` / `--scale`
  - 如果旧 feature 目录的 `meta.json` 没有 `weight_scale`，会自动乘 scale
  - 新 feature 目录如果已有 `weight_scale`，不会二次缩放

---

## 3. 值得注意的资产

- 真实 Qwen3.8 PLE 表：
  - Store-I：`/Volumes/My Passport/qwen38-rows`
  - 完整 Store-P 视图：`/Volumes/My Passport/p4view-full-2560.bin`（48GB，`len=20000096`）
  - Qwen3.8 checkpoint：`/Volumes/My Passport/qwen38-ple`
- 官方 PLE 层非 embedding 权重可以从小权重的 safetensors 中读取：
  - `model.language_model.layers.1.ple.key_proj.weight` `[10240, 2560]`
  - `...value_proj.weight` `[2560, 2560]`
  - `...conv1d.weight` `[10240, 1, 4]`
  - `...norm_key/norm_query/norm_conv.weight` `[10240]`
- EngramDB 已有 `scripts/ple_layer_bit_exact.py` 证明：
  - Store 读行 + 官方 PLE 前向 = 原始 safetensors 直读，bit-exact。

---

## 4. 下一步可以进一步适配

### 4.1 用官方 PLE 权重构造“源空间 target-side reader”
- 把 `key_proj` / `value_proj` / `conv1d` / norms 作为**冻结的官方 PLE 读取器**；
- 只训练：
  - target hidden -> source query 的 bridge
  - source PLE 输出 -> target hidden 的 out_proj
- 这可能比从零训练简化 reader 更接近 Qwen 原生 PLE 行为。

### 4.2 用 `DiskPleNGramEmbedding` 做 live 训练
- 避免为 5M token 预计算 50GB+ e_t；
- 训练时直接从 Store-I 行读取 + LRU cache；
- 与 `install_real_qwen_ple_embedding` 路线二选一或并行验证。

### 4.3 Store-P 视图接入
- 已有全量视图，但没有完整 keys/index 映射；
- 若做成“访问序视图”，可把 IOPS 从 16:1 降到 1:1；
- 可以作为训练流 DataLoader 或 CPU 推理低延迟路径。

### 4.4 engram-peft 真实表闭环
- 用 `install_real_qwen_ple_embedding` 替换 MultiHeadEmbedding；
- 用 engram-peft 自带 `ContextAwareGating + ShortConv`，而不是继续手写简化 reader。

---

## 5. 结论

EngramDB v0.2.8 已经把我们的“数据面”补到可以进入真实表 live 实验的程度：
- 官方 rowid、元数据、FP8 scale、磁盘 embedding、Store-P 视图全部可用；
- 之前最关键的缺口是 **e_t 预计算没有乘 weight_scale**，本轮已修正；
- 下一步实验应优先使用官方 PLE 读取器/engram-peft 磁盘注入，并扩大训练量。
