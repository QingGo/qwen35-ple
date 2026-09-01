# qwen35-ple

**Qwen3.5 主干 + Qwen3.8-Flash-Next PLE 记忆表**的实验项目：继续预训练（CPT）与后训练
（SFT/RL），目标是一个知识/长上下文更强的模型，并在 CPU 上以 100 tok/s 推理。

本仓库是四仓库协作中的「编排者」，自身不实现存储与模型核心：

```
engram-peft  (模型库: DeepSeek Engram 一致性实现, 记忆层/训练/TRL 基础设施)
EngramDB     (存储: PLE/Engram n-gram 行表, badge 布局, Store-P 视图, C ABI)
   ▲              ▲
   └──────┬───────┘
    qwen35-ple  (本仓库: 嫁接实验编排, 依赖以上两者)
       ▲
LLM-CompileForge (推理: MLIR 编译 .dylib + Rust runtime, CPU 100 tok/s 目标)
```

## 关键文档

| 文档 | 内容 |
|---|---|
| [docs/qwen35-ple-design.md](docs/qwen35-ple-design.md) | 项目设计：嫁接方案、CPT/后训练、消融矩阵、里程碑 |
| [docs/integration-contract.md](docs/integration-contract.md) | **四仓库交互契约 v1**（存储/模型/推理/数据四条契约，冻结原则） |
| [docs/roadmap.md](docs/roadmap.md) | 战略路线图：终极目标、技术债、借鉴矩阵、阶段计划 |
| [docs/round-21-full-summary.md](docs/round-21-full-summary.md) | 本轮完整汇总：计划/发现/尝试/踩坑/完成/未完成/未来 |
| [docs/session-log.md](docs/session-log.md) | 会话复盘：完成项、发现的技术债、下一步 |

## EngramDB 配置即用（自动注入）

`table_source="engramdb:store"` 现在由 engram-peft 自动消费：在 `EngramConfig` 中配置

```python
config = EngramConfig(
    ...,
    engine="qwen_ple",
    table_spec="PLE_QWEN_V1",
    table_source="engramdb:store",
    table_store_path="/path/to/rows",
    table_model_dir="/path/to/Qwen3.8-Flash-Next",
    table_dtype="float8_e4m3fn",
)
model = get_engram_model(base_model, config, tokenizer)
```

无需手动调用 `install_disk_multi_head_embedding` / `install_real_qwen_ple_embedding`。

从 qwen35-ple YAML 配置可以直接转换：

```python
from qwen35_ple.config import load_config

cfg = load_config("configs/your.yaml")
engram_cfg = cfg.to_engram_config(
    hidden_size=model.config.hidden_size,
    compressed_vocab_size=model.config.vocab_size,
    pad_id=tokenizer.pad_token_id,
    tokenizer_name_or_path="...",
)
model = get_engram_model(base_model, engram_cfg, tokenizer)
```

真实 FP8 Store-I e2e：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_m0_smoke.py --e2e \
  --model /path/to/Qwen3.5-0.8B \
  --store-dir /path/to/qwen38-rows \
  --ple-model-dir /path/to/qwen38-ple
```

轻量版（不需要完整 engram-peft 的 TRL/datasets 依赖）：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_real_fp8_e2e.py \
  --model /path/to/Qwen3.5-0.8B \
  --store-dir /path/to/qwen38-rows \
  --ple-model-dir /path/to/qwen38-ple
```

## 与兄弟仓库的交互契约（摘要）

- **存储契约 C1**（EngramDB → 使用方）：行语义 `PLE_QWEN_V1`（16 头/160 维/320M 行）、
  视图格式（`<view>.manifest.json` + keys 文件）、C ABI 符号冻结规则。
- **模型契约 C2**（engram-peft → 本仓库）：`EngramConfig` 只增不改字段，新增
  `engine="deepseek"（默认）| "qwen_ple"`；`get_engram_model` 签名不变；
  磁盘注入单点 `install_disk_multi_head_embedding(store)`。
- **推理契约 C3**（LLM-CompileForge ↔ EngramDB）：`sfa_abi.proto` 加
  `SfaWeightSource`（只增字段）；视图可作为外部权重源；运行时 dlopen 加载 C ABI。
- **数据契约 C4**：tokenizer 唯一来源 = Qwen 官方（vocab 248320 与 Flash-Next 相同，已核实）。

契约变更纪律见契约文档 `§0`：**只允许新增，禁止改语义/删除；ABI 演进 = 新符号
（`_v2` 后缀）**。

## 快速开始

```bash
# 前置: uv、Rust 工具链（engramdb-python 需要 maturin 构建）
uv sync --all-groups

# 冒烟: 依赖可导入
uv run python -c "import engram_peft, engramdb, qwen35_ple; print('ok')"

# 本地提交前 lint 钩子
uv run pre-commit install
```

## 目录结构

```
src/qwen35_ple/   实验编排代码（config / engine / data / train / eval / infer）
configs/          训练与推理配置（初值样例，消融矩阵见设计文档 §6）
scripts/          一次性脚本（数据构建、表资产、评测）
docs/             本仓库文档（设计 + 契约，契约以本仓库为准）
tests/            一致性冒烟测试（golden 对拍）
```

## 当前状态

- [x] 仓库初始化（2026-08-30）
- [x] 设计文档 + 四仓库契约 v1（冻结）
- [x] 战略路线图（`docs/roadmap.md`）
- [x] `PLE_QWEN_V1` 纯 Python golden 参考与测试
- [x] Store-P 视图构建/校验脚本骨架
- [x] YAML 配置加载与契约校验（`src/qwen35_ple/config.py`）
- [x] engram-peft 按契约 C2 新增字段 + `QwenPleHashMapping` + 跨仓 golden
- [x] M0 磁盘版 MultiHeadEmbedding quick 自检
- [x] Qwen3.5-0.8B + engram-peft PLE-lite CPU e2e
- [x] 官方 `refs/qwen4_exp_modeling.py` 快照 + 4096 forward golden
- [x] 真实 PLE FP8 `e_t` 预计算（EngramDB `fetch_e_t_tensor` 快速路径）
- [x] live-store 直接读取（`run_phase0.py --live-store`，无需 10GB `e_t.npy`）
- [x] 真实 PLE 知识探针（线性分类 72.7% vs 16.7%）
- [x] XMemTransfer 风格 reader 完整实验矩阵
- [x] 官方 Qwen PLE reader 权重复用（`OfficialSourceQwenReader` + 可切换 MLP bridge/out_proj）
- [x] 1M token live-store 懒加载三线实验（real / control / no-reader）
- [x] CI 改用 uv 管理依赖与测试（`uv sync --all-groups` + `uv run ruff/pytest`）
- [x] pre-commit 已配置（ruff）
- [ ] CP/后训练正式消融与 100 tok/s 推理闭环

## 实验方法与当前结论（2026-08-30）

### 1. 真实 PLE 特征预计算

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/precompute_real_ple_features.py \
  --rows-dir "/Volumes/My Passport/qwen38-rows" \
  --tokenizer data/models/Qwen3.5-0.8B \
  --corpus data/padapter-corpus.txt \
  --output data/ple-features
```

输出：

```text
tokens.npy
keys.npy
e_t.npy
meta.json
```


#### Live-store 直接读取（推荐，避免 10GB `e_t.npy`）

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_phase0.py --live-store \
    --tokens-npy /path/to/tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --model-dir "/Volumes/My Passport/qwen38-ple"
```

可复现基准：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/bench_live_store.py \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --tokens 20000 --reps 3 --csv /tmp/live-store-bench.csv
```

核心 API：`engramdb.fetch_e_t_tensor()` / `PleDiskGather.fetch_tensor()`；
`real_ple.fetch_e_t` 已不再使用旧 Python 字节展开路径。

Store-I vs Store-P 同口径 A/B 骨架：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/bench_store_vs_view.py \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --tokens 20000 --reps 3 --csv /tmp/store-vs-view.csv
```

懒加载逐窗口基准（Track B/C，不会物化全量 e_t）：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/bench_lazy_windows.py \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --tokens 100000 --seq-len 128 --step 128 \
    --csv /tmp/lazy-100k-store.csv
```

本机 Mac 外盘实测（非 WSL 结论）：

- 100k token Store-I 懒加载：781 窗口，约 60.5s
- 100k token Store-P 懒加载：781 窗口，约 0.58s
- 1M token Store-P 懒加载：7812 窗口，约 7.1s
- 1M token Store-P 控制/置换访问：3 seeds 约 17.2–17.9s，说明访问序/顺序化仍有 2.4× 收益

WSL 真表初测：

- 20k Store-I 懒加载：156 窗口，约 22.4s
- 100k Store-P 懒加载：781 窗口，约 1.9s
- 1M Store-P 懒加载：7812 窗口，约 23.9s

Access-order A/B 基准（V136 起步）：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/bench_access_order.py \
    --view /tmp/corpus.view \
    --slot-indices-npy /tmp/corpus.slot_indices.npy \
    --tokens 100000 --seq-len 128 --step 128 --reps 3 \
    --csv /tmp/access-order.csv
```

WSL 复现环境脚本（V131）：

```bash
bash scripts/wsl_repro.sh
# 或完整测试：
bash scripts/wsl_repro.sh --full
```

#### Access-order Store-P 语义视图（P0 起步）

构建一个“语料 access-order Store-P 视图”：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/build_corpus_store_p_view.py \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --tokens-npy /path/to/tokens.npy \
    --model-dir "/Volumes/My Passport/qwen38-ple" \
    --output-view /tmp/corpus.view \
    --keys-out /tmp/corpus.keys \
    --slot-indices-out /tmp/corpus.slot_indices.npy \
    --engramdb-bin /path/to/engramdb \
    --verify
```

因为槽位顺序 = 语料 token 顺序，所以：

```python
slot_indices = np.load("/tmp/corpus.slot_indices.npy")  # arange(T)
store_p = LiveETViewStore(view, slot_indices, scale, view_path="/tmp/corpus.view")
```

本机已验证：该 access-order Store-P 与 Store-I 逐 token e_t `maxdiff=0.0`。
`run_phase0.py` 可直接用：

```bash
python scripts/run_phase0.py --live-store \
    --store-p-view /tmp/corpus.view \
    --store-p-slot-indices /tmp/corpus.slot_indices.npy \
    --tokens-npy /path/to/tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows"
```

> **推荐方式（1M token/内存受限机器）**：`--live-store` 现在不会预加载完整 10GB `e_t`，
> 而是只保留 `[T,16]` rowids，训练/评测时按当前窗口懒加载对应 PLE 行。
> 因此 1M token 也可以直接跑，只要单窗口内存足够（~seq_len × 2560 × 4B）。
> 不需要先做全量 chunk npy，也不需要全量 `e_t` 常驻内存。

#### P0 完成：通用 rowid→slot 语义索引 + 自动访问序调度

**V123 通用语义索引**：

构建器会在视图旁自动写出 `*.slot_index.npz`（也可用 `--slot-index-out` 指定）：

```python
from qwen35_ple.slot_index import SlotIndex

index = SlotIndex.load("/tmp/corpus.slot_index.npz")
slots = index.to_slots(rowids)  # 任意 token 流 -> 对应 Store-P 物理槽
store_p = LiveETViewStore.from_slot_index(view, rowids, index, scale, access_order=True)
```

`run_phase0.py` 可直接用通用索引：

```bash
python scripts/run_phase0.py --live-store \
    --store-p-view /tmp/corpus.view \
    --store-p-slot-index /tmp/corpus.slot_index.npz \
    --access-order \
    --tokens-npy /path/to/tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows"
```

**V124 自动访问序调度**：

- `LiveETViewStore(access_order=True)` 在每个窗口内按物理槽位排序读取，再散射回 token 顺序；
- `LiveETDataset(access_order=True)` 还会按窗口最小物理槽位调度窗口顺序，使跨窗口 I/O 更接近顺序读；
- `run_phase0.py --access-order` 与 `bench_lazy_windows.py --access-order` 均已接入。

新增测试：`tests/test_slot_index.py`（SlotIndex 保存/加载、重复 rowid 代表槽、keys 文件构建、access-order 调度正确性）。

#### LiveETDataset：通用懒加载数据流（Track A）

任意实验脚本只需三行即可接入 live-store：

```python
from qwen35_ple.live_store import LiveETStore, LiveETDataset

live = LiveETStore(store, rowids, scale, store_path=rows_dir,
                   shards=128, rows_per_shard=2_500_012, width=160)
dataset = LiveETDataset(tokens, live, seq_len=128, step=128)
for batch in dataset:
    # batch.tokens + batch.e_t are already fetched from disk, no full e_t
    ...
```

`LiveETDataset` 支持：

- 直接 `for` 迭代，或传给 `torch.utils.data.DataLoader(..., num_workers=N)`
- 每 worker 自动分片，并且每个 worker 会重新打开自己的 Store 句柄
- `control=True` 做 e_t 行乱序对照
- `LiveETStore.stats` 记录每窗口/累计 `rows`、`unique_rows`、`cache_hits`、`fetch_seconds`
- Store-P 路径可使用 `LiveETViewStore` 从 `engramdb.View` 按物化槽位直接读取，供 Track B 做 Store-I vs Store-P A/B

冒烟命令：

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_live_et_dataset_smoke.py \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --tokens-npy /path/to/tokens.npy \
    --model-dir "/Volumes/My Passport/qwen38-ple" \
    --seq-len 128 --max-batches 4
```

核心代码：`src/qwen35_ple/live_store.py`、`src/qwen35_ple/slot_index.py`、`src/qwen35_ple/real_ple.py`。

### 2. PLE 知识探针

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_ple_knowledge_probe.py \
  --rows-dir "/Volumes/My Passport/qwen38-rows" \
  --tokenizer data/models/Qwen3.5-0.8B
```

结果：

```text
test accuracy = 72.7%
random baseline = 16.7%
```

结论：真实 PLE `e_t` 含语义类别信息。

### 3. Reader 实验矩阵

支持：

```text
--layer 1 / 8
--branches 1 / 4
--short-conv
--mode real / control
```

一键跑矩阵：

```bash
bash scripts/run_full_matrix.sh
```

结果（held-out loss，baseline=4.428）：

| layer | branches | short_conv | real after | control after |
|---:|---:|---:|---:|---:|
| 1 | 1 | 无 | 5.046 | 5.921 |
| 1 | 4 | 无 | 5.437 | 6.328 |
| 1 | 4 | 有 | 5.196 | 5.993 |
| 8 | 1 | 无 | **4.851** | 5.434 |
| 8 | 4 | 无 | 5.112 | 5.664 |
| 8 | 4 | 有 | 5.047 | 5.389 |

结论：

- 所有组合中真实 PLE 都优于 shuffled control。
- 但最佳 real 仍高于 no-reader baseline。
- 当前最佳为 `layer=8, branches=1, short_conv=off`。
- 下一步应降低初始扰动、延长训练、换知识评测，或直接使用官方 Qwen PLE gating 结构。

### 4. 官方 PLE reader + MLP bridge/out_proj（最新结果）

使用 `OfficialSourceQwenReader`，复用 Qwen3.8 官方 PLE key/value/norm/conv，
只训练：

- `query_bridge`：Qwen3.5 hidden → Qwen3.8 source query 空间
- `out_proj`：source PLE output → Qwen3.5 hidden

支持 1 层线性或 2 层 MLP。

#### 160k / 500 steps / 3 seeds

| 线 | val loss | 结论 |
|---|---:|---|
| real | 3.69394 | 3 seeds 均优于 control |
| control | 3.70218 | — |

#### 1M tokens / live-store / 500 steps / 3 seeds

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

- 1M 下 real 稳定且明显优于 control：3 seeds 全部 positive。
- real 也超过 no-reader baseline。
- 说明“Qwen3.8 PLE 记忆表 + target-side reader”方向已出现较强正信号。
- 下一步：补 QA exact-match，然后上云跑 5M 正式矩阵。
