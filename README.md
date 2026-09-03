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
| [docs/round-23-upgrade-assessment.md](docs/round-23-upgrade-assessment.md) | engram-peft 1.2.7 / EngramDB 0.2.12 升级评估与后续计划 |
| [docs/phase0-live1m-qa150-analysis.md](docs/phase0-live1m-qa150-analysis.md) | 1M 150 题三线结果、bad case、语料重叠分析 |
| [docs/round-24-full-summary.md](docs/round-24-full-summary.md) | 本轮系统性思考、技术债、语料混比与后续计划 |
| [docs/round-25-mix-corpus.md](docs/round-25-mix-corpus.md) | M1–M5 1M 混合语料构建、来源、比例、污染审计 |
| [docs/round-26-systematic.md](docs/round-26-systematic.md) | 系统性思考：语义对齐证据、机制分析技术债、RL 门禁、借鉴矩阵 |
| [docs/round-27-manifold-alignment.md](docs/round-27-manifold-alignment.md) | 流形/语义空间对齐调研、数学工具、机制验证与 case 分析计划 |
| [docs/round-27-full-summary.md](docs/round-27-full-summary.md) | 本轮全量总结：计划、发现、尝试、踩坑、完成/未完成、未来计划 |
| [docs/round-28-mechanism.md](docs/round-28-mechanism.md) | 第一批机制验证：CKA/Procrustes/kNN/reader 参数/activation patch |
| [docs/round-29-alignment-math.md](docs/round-29-alignment-math.md) | 数学推导：条件增量可解释性、gate/value 分工、正交化注入、实验指导 |
| [docs/round-30-multimath-alignment.md](docs/round-30-multimath-alignment.md) | 多视角数学推导：信息论/谱方法/随机矩阵/最优传输/核方法/图谱/优化动力学/流形假设 |
| [docs/round-31-deeper-math.md](docs/round-31-deeper-math.md) | 更深数学分支：统计决策/因果推断/贝叶斯GP/微分几何/最优控制/拓扑/信息几何 |
| [docs/round-32-first-principles-alignment.md](docs/round-32-first-principles-alignment.md) | 第一性原理：对齐的本质是条件充分性，不是几何相似；含命题 A-D 证明草图 |
| [docs/round-33-proofs.md](docs/round-33-proofs.md) | 完整证明：数据处理上界、线性增量 R²、正交化不损失、几何对齐不充分/不必要、Hilbert 投影 |
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
- [x] 1M QA exact-match（9题）三线评测完成（`outputs/phase0-live1m-qa.json`）
- [x] 扩大 QA 集：50 TriviaQA-style + 50 NQ + 50 BoolQ（`assets/qa-expanded-150.json`）
- [x] 150 题三线 QA 完成（seed 0：no-reader 53.3% / real 42.0% / control 30.7%；bad case 与语料分析见 `docs/phase0-live1m-qa150-analysis.md`）
- [x] M1–M5 1M 混合语料构建完成（`build_mix.py`，含 ModelScope chat/wiki/cot/tool 来源与 `--exclude-qa` 严格污染过滤；审计全部 low）
- [x] 污染审计脚本与报告（`audit_contamination.py` + `outputs/contamination-M*.json`）
- [x] 批处理入口：`scripts/run_mix_batch.sh`（WSL 批量跑 M1–M5 三线 QA）
- [x] M1 三线 150 QA 已完成：real 50.7% / control 52.7% / no-reader 53.3%；机制分析见 `docs/round-26-systematic.md`
- [x] 关键认知：混合语料 val loss 降低不等于能力提升；control 也会出现“知识型”good case
- [x] 第一批机制分析：reader 参数 / CKA / Procrustes / kNN / intrinsic dimension / logit-level patching（详见 `docs/round-28-mechanism.md`）
- [ ] 固定外部评测集与科学 mix 选择
- [ ] RL 决策门禁（当前不提前做 RL）
- [x] 依赖收口：engram-peft>=1.2.7、engramdb-python>=0.2.12（CI 同步固定正式 tag）
- [x] target-side reader checkpoint 保存/加载（`--save-reader` / `--load-reader`）
- [x] 通用 serving adapter（`QwenReaderServingAdapter`）
- [ ] 真实 vLLM/SGLang 引擎 serving 适配与 A/B
- [x] CI 改用 uv 管理依赖与测试（`uv sync --all-groups` + `uv run ruff/pytest`）
- [x] pre-commit 已配置（ruff）
- [ ] CP/后训练正式消融与 100 tok/s 推理闭环

## 最新实验结果（2026-09-03）

### M1 混合语料 1M token 三线 150 QA

| 线 | val loss | PPL | QA EM | TriviaQA | NQ | BoolQ |
|---|---:|---:|---:|---:|---:|---:|
| no-reader | 2.4563 | 11.66 | **53.3%** | 70% | 0% | **90%** |
| real | **2.3949** | **10.97** | 50.7% | **76%** | 0% | 76% |
| control | 2.4391 | 11.46 | 52.7% | 84% | 4% | 70% |

### 核心结论

1. **val loss：real < control < no-reader**
   PLE 对语言建模仍有正信号。

2. **QA EM：no-reader > control > real**
   PLE 当前没有带来任务级净收益。

3. **control 不是原版模型**
   control = Qwen3.5 + 训练后 reader + 随机打乱的 PLE e_t。
   control 也退化，说明“注入扰动 + 训练 reader”本身就会干扰 BoolQ。

4. **control 也有“知识型”good case**
   control 也能做对 Shakespeare / Newton / Rome / Poseidon，
   因此“答案不在语料中”不能单独证明 PLE 语义对齐。

5. **当前真正属于 real 独有且不在语料中的增益很弱**
   例如 Leonardo da Vinci；其余主要是 BoolQ 上的 yes/no 差异。

### 机制验证第一批结果（2026-09-03）

在 `data/ple-books-160k` 上采样 2048 token，测量 PLE e_t 与 Qwen hidden：

| layer | CKA | Procrustes alignment | kNN overlap (k=10) | hidden PR |
|---|---:|---:|---:|---:|
| 1 | 0.222 | 0.051 | 0.079 | 77.9 |
| 8 | 0.151 | 0.034 | 0.075 | 41.2 |
| 16 | 0.192 | 0.023 | 0.084 | 58.1 |
| 23 | 0.151 | 0.010 | 0.068 | 37.5 |

- PLE intrinsic dimension ≈ 765.6，Qwen ≈ 37–78。
- 随机 kNN baseline ≈ 0.039，实际仅 0.068–0.084。
- 结论：两个空间全局线性对齐弱、局部邻域接近随机；当前 reader 更像可训练投影，尚不是稳定流形对齐记忆读取器。

Logit-level activation patching（完整 150 题：50 BoolQ + 50 NQ + 50 TriviaQA）：

| 条件 | BoolQ logprob | BoolQ entropy | NQ logprob | Trivia logprob | 总体 logprob | 总体 entropy |
|---|---:|---:|---:|---:|---:|---:|
| no-reader | -10.01 | 0.84 | -6.90 | -9.57 | -8.83 | 2.45 |
| real | **-7.62** | 2.23 | -6.80 | -9.39 | **-7.94** | 3.25 |
| control | -8.09 | 2.33 | **-6.76** | **-9.26** | -8.04 | 3.36 |
| random | -9.74 | 0.91 | -6.90 | -9.58 | -8.74 | 2.49 |
| zero | -10.01 | 0.84 | -6.90 | -9.57 | -8.83 | 2.45 |

- real/control 都显著增加 next entropy，random/zero 接近 no-reader。
- real 相对 control 仅 +0.10 总体 logprob，逐题胜负 76:74，接近抛硬币；仅 BoolQ 上 real 优势较明显（+0.47）。
- 说明当前效应主要来自“注入 PLE 类向量”，而不是“真实 token 顺序的语义内容”。
- 详细报告：`docs/round-28-mechanism.md`。

额外 BoolQ scale sweep（50 题，`--inject-scale`）：

| scale | real logprob | control logprob | real-control | real entropy |
|---:|---:|---:|---:|---:|
| 0.25 | -9.51 | -9.64 | +0.14 | 0.89 |
| 0.5 | -8.78 | -9.16 | +0.38 | 1.20 |
| 1.0 | -7.62 | -8.21 | +0.59 | 2.23 |
| 2.0 | -7.46 | **-7.19** | -0.27 | 3.95 |

- real 优势在 scale=1.0 附近最大；2.0 时 control 反超且 entropy 大幅上升。
- 低强度 0.25/0.5 可降低扰动，但 real-control 优势也缩小。
- 初步认为 0.5 附近是“低破坏 + 仍有真实信号”的候选区间，但优势仍不够强。

### 当前状态

- M2–M5 已暂停，不再继续混比微调。
- 已完成第一批机制验证工具与结果：
  - reader 参数 / gate 统计；
  - CKA / Procrustes / kNN / intrinsic dimension；
  - logit-level activation patching。
- 下一阶段：
  - 增加 zero/random reader 对照；
  - layer / gate 扫描（scale sweep 已完成）；
  - 设计 contrastive / neighbor / KL 约束 loss；
  - 完成 BoolQ logit lens 与错误分类。
- 详细分析见：
  - `docs/round-26-systematic.md`
  - `docs/round-27-manifold-alignment.md`
  - `docs/round-27-full-summary.md`
  - `docs/round-28-mechanism.md`



## 推理 / Serving 现状与规划（2026-09-01）

### 当前推理现状

- 当前 `run_phase0.py --qa-exact-match` 仍是 **Transformers 手动逐 token forward**。
- 未使用 vLLM / SGLang，也没有 KV cache / continuous batching。
- 因此 150 题规模 QA 会比较慢，主要瓶颈是重复 forward + 每步 EngramDB fetch。

### EngramDB 已有的 vLLM / SGLang 适配

EngramDB 仓库中已有：

```text
engramdb/vllm_plugin.py     DiskPleEmbedding + patch_model_class_ple
engramdb/sglang.py          install_sglang_ple + IoUringReader
engramdb/vllm.py            fetch_e_t_tensor / PleDiskGather
```

但这些适配主要面向：

```text
源模型 / 源 PLE embedding 表（Qwen3.8-Flash-Next、Gemma 风格）
→ 把 nn.Embedding 换成 EngramDB 磁盘表
```

**不能直接覆盖我们的 target-side reader**：

```text
Qwen3.5 backbone
  + OfficialSourceQwenReader
  + 每步用 [T,16] rowids 取 e_t
  + 注入到 layer 8
```

### qwen35-ple 侧需要做的工程

1. ✅ `run_phase0.py` 已支持保存 / 加载训练后的 target-side reader checkpoint（`--save-reader` / `--load-reader`），并包含 `ShortConv` extra state。
2. ✅ 已建立 `reader_registry`（`src/qwen35_ple/reader_registry.py`），基于 EngramDB `TargetReaderRegistry`：
   - `official_source_qwen_v1`
   - `engram_v1`
   - `simple_v1`
   - future：dual-layer / multi-layer / LoRA
3. ✅ 已定义统一 bundle（`src/qwen35_ple/serving/bundle.py`）：
   - backbone 路径
   - PLE table 描述
   - reader config + checkpoint
   - 兼容 EngramDB `engramdb-bundle-v1`
4. ✅ 已新增通用 serving adapter（`src/qwen35_ple/serving/adapter.py`）：
   - `QwenReaderServingAdapter`
   - `install_qwen_reader_adapter`
   - `install_qwen_reader_adapter_from_bundle`
   - `install_vllm_reader_from_bundle` / `install_sglang_reader_from_bundle`
   - 待做：接入真实 vLLM / SGLang 引擎并做 A/B
5. no-reader 基线可先直接用 vLLM / SGLang 加速。

### 已完成的后续数据准备

- 已生成扩大 QA 集：
  ```text
  assets/qa-expanded-150.json
  50 TriviaQA-style + 50 NQ + 50 BoolQ
  ```

### Reader checkpoint 用法（已落地）

训练并保存 reader：

```bash
PYTHONPATH=src:../EngramDB/python:../engram-peft/src \
python scripts/run_phase0.py \
  --live-store --rows-dir /path/to/rows --model-dir /path/to/qwen38-ple \
  --tokens-npy data/wet-1m-first.npy --reader official \
  --modes real --seeds 0 \
  --save-reader outputs/reader-{mode}-seed{seed}.pt \
  --save-bundle outputs/bundle-{mode}-seed{seed}.json
```

后续 eval-only 直接加载，跳过训练：

```bash
PYTHONPATH=src:../EngramDB/python:../engram-peft/src \
python scripts/run_phase0.py \
  --live-store --rows-dir /path/to/rows --model-dir /path/to/qwen38-ple \
  --tokens-npy data/wet-1m-first.npy --reader official \
  --modes real --seeds 0 --load-reader outputs/reader-real-seed0.pt \
  --qa-exact-match --qa-file assets/qa-expanded-150.json
```


### 相关协调

- EngramDB v0.2.12 已发布：包含 `DiskSlotIndex` v3、`PleMemory` / `PleSequence`、
  `BundleManifest` / `TargetReaderRegistry`、`PleMemoryAdapter` / `TargetReaderHook`。
- engram-peft v1.2.7 已发布：`engine="qwen_ple"` + `table_source="engramdb:store"`
  自动消费已正式可用。
- qwen35-ple 已完成最低版本收口（`pyproject.toml` / `uv.lock` / CI tag / WSL 脚本），
  后续按 `docs/round-23-upgrade-assessment.md` 接入统一 reader/bundle serving 协议。


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

#### 1M QA exact-match / 9题 / 3 seeds

| 线 | QA EM mean | 3 seeds |
|---|---:|---|
| no-reader | 44.44% | 44.4 / 44.4 / 44.4 |
| control | 48.15% | 44.4 / 33.3 / 66.7 |
| **real** | **51.85%** | 66.7 / 44.4 / 44.4 |

```text
real − control = +3.70pp
real − no-reader = +7.41pp
control − no-reader = +3.70pp
```

分 task 均值（3 seeds 合并）：

| task | no-reader | control | real |
|---|---:|---:|---:|
| TriviaQA | 100% | 77.8% | 100% |
| NQ | 33.3% | 55.6% | 55.6% |
| BoolQ | 0% | 11.1% | 0% |

结论：

- PPL 三线仍然是最强信号：real 稳定优于 control，且超过 no-reader。
- QA exact-match 上也出现 real > control > no-reader 的平均排序：
  - real − control = +3.70pp
  - real − no-reader = +7.41pp
- 但 QA 集只有 9 题，种子间波动大（real 2/3 seeds 超过 control，1 seed 被 control 反超），不能单独作为决定性证据。
- 下一步仍然应该上 5M 正式矩阵，并把 QA 集扩大到标准 TriviaQA / NQ / BoolQ 子集。
