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
- [x] A0/A1 评测对比入口
- [ ] M1 完整 PLE-lite 前向 golden（`refs/qwen4_exp_modeling.py`）
- [ ] M0 e2e（需完整 engram-peft/peft 环境）
- [ ] CPT 消融（设计文档 M2）
- [ ] 100 tok/s 推理闭环（设计文档 M4，见 LLM-CompileForge/docs/qwen35-0.8b-100toks.md）
