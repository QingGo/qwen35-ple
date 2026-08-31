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

核心代码：`src/qwen35_ple/real_ple.py`。

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
