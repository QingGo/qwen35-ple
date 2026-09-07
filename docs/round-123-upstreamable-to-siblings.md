# Round 123：可沉淀到兄弟项目的代码与 Insight

> 日期：2026-09-08
> 目的：梳理 qwen35-ple 中哪些逻辑和结论值得上移到 engram-peft、EngramDB、LLM-CompileForge，避免实验代码与核心能力长期耦合。
>
> 状态：P0 全部完成。
>
> - engram-peft `b674155`：`fusion.py`、`projector.py`、`policy.py` + 单测。
> - EngramDB `e9c0d52`：`addressable_memory.py` + 导出 + 单测。
> - LLM-CompileForge `ba12764`：`memory_fusion.py` + `docs/external-memory-fusion.md` + 单测。

---

## 1. 核心判断

qwen35-ple 目前承担了两类职责：

1. **实验编排**（数据、评测、论文、远程 GPU）——应继续留在这里；
2. **可复用的记忆/融合/审计原语**——应逐步沉淀到兄弟项目。

沉淀原则：

- 只上移“与具体实验无关”的通用能力；
- 保留实验脚本、数据集、论文、机器相关路径在本仓库；
- 上移前补齐契约、测试、文档，并与兄弟仓库“只增字段/只增符号”的演进规则兼容。

---

## 2. 建议上移到 engram-peft

engram-peft 是模型库，应成为“记忆层 + 融合策略 + 训练接口”的权威实现。

| qwen35-ple 模块 | 可沉淀内容 | 建议位置 |
|---|---|---|
| `fusion.py` | logit-space `scale * log p_m + bias`、温度、网格校准 | `engram_peft.fusion` |
| `projector.py` | `PleProjector`、特征提取、zero-init、保存/加载 | `engram_peft.projector` |
| `router.py` 中的 `TokenPlePolicy` | token 级 learned safety gate | `engram_peft.policy` |
| `router.py` 中的 `LogDensityRatioGate` | real/control 判别、memory trust gate | `engram_peft.gate` |
| `router.py` 中的 `TaskConditionedNgramLogitProcessor` | 任务路由 + 融合 + 安全门控的统一处理器 | `engram_peft.serving` |
| `configs/*.json` schema | fusion / router / token-policy 配置格式 | `engram_peft` 配置 schema |
| `scripts/train_ple_projector.py` 的训练循环 | 冻结 backbone + 只训 project 的示例/工具 | `engram_peft.examples` |

**Insight：**
> 对于小模型，logit 级可校准融合比 hidden-state 注入更可控、更可审计。

---

## 3. 建议上移到 EngramDB

EngramDB 是存储层，应成为“表和视图读取 + 访问调度 + golden”的权威实现。

| qwen35-ple 模块 | 可沉淀内容 | 建议位置 |
|---|---|---|
| `live_store.py` | LiveET Store / Store-P 视图、懒加载、FetchStats | `engramdb.live_store` |
| `slot_index.py` | rowid → slot 语义映射、访问序调度 | `engramdb.slot_index` |
| `table_assets.py` | manifest / keys / 表资产校验 | `engramdb.tables` 或 `assets` |
| `ple_hash.py` / `official_ple_snapshot.py` / `ple_reference.py` | Qwen PLE golden 参考与位级对拍 | `engramdb` golden 目录 |
| `addressable_memory.py` | 纯 Python n-gram 可审计内存参考实现 | `engramdb` 或 `engram_peft` 小模型 fallback |
| `memory/bank.py` | ExactNgramBank / 可审计 value index | `engramdb.memory` |
| `serving/bundle.py` | bundle 加载、reader registry 接入 | `engramdb.bundle` |

**Insight：**
> Store-P / access-order / lazy view 的工程结论应沉淀到存储层，上层实验只保留调用方式。

---

## 4. 建议上移到 LLM-CompileForge

LLM-CompileForge 负责推理运行时，应把 Python 侧的融合策略变成可编译、可审计的运行时契约。

| qwen35-ple 模块 | 可沉淀内容 | 建议位置 |
|---|---|---|
| `fusion.py` 公式 | `logit += scale * log p_m + bias` 的 kernel/runtime 接口 | `include/sfa_abi.proto` 或 `logit_transform` 契约 |
| `router.py` 的 feature 计算 | token policy 特征在 C/Rust 侧计算 | runtime 特征结构 |
| `serving/rag.py` 的 trace | PLE gate / projector / provenance 可审计 trace | runtime debug/log API |
| `configs/*.json` | 融合参数的标准配置格式供 runtime 读取 | runtime config 解析 |

**Insight：**
> 低资源 PLE 的核心不仅是查表，还包括“何时信任记忆、如何修正 logits”，这些必须进入编译/runtime 契约才能达到部署目标。

---

## 5. 只适合留在 qwen35-ple 的内容

- “真实 benchmark + 小模型边界”的全部实验脚本；
- HumanEval / TriviaQA / pass@k / LLM judge / sensitivity 数据；
- 论文、图表、HF artifact、README；
- 远程 WSL / GPU / 调度脚本；
- 项目独有的 MoRA + RAG + PLE 联合系统实验。

---

## 6. 可复用的 Insight（不依赖具体代码）

1. **Real/Control 是底线**：任何记忆方法必须先证明“使用了真实记忆内容”，再讨论是否超越 base。
2. **Logit 级融合优于 hidden-state 直插**：小模型下更稳、更容易审计。
3. **Learned token policy 是开放生成安全的必要条件**：不能无条件信任 n-gram prior。
4. **小而同域的 bank 优于大而分散的 bank**：对局部代码/命名任务，compact memory 更有效。
5. **Per-task calibration + task router 比单一全局参数更稳**。
6. **公开 artifact 必须脱敏**：API key、token、本地路径等要进入 release pipeline 的自动清理。
7. **多 seed paired bootstrap 是低资源记忆评测的标准**，不是可选项。

---

## 7. 沉淀优先级

| 优先级 | 事项 | 理由 |
|---|---|---|
| P0 | fusion / projector / policy 进 engram-peft | 最核心、最通用、论文主张直接依赖 |
| P0 | Store-P / slot index / live view 进 EngramDB | 底层存储能力应由存储仓统一 |
| P0 | logit correction 契约进 LLM-CompileForge | 没有 runtime 契约，部署目标无法闭环 |
| P1 | artifact 脱敏/checksum/eval card 模板进通用工具 | 所有兄弟项目都能用 |
| P2 | 论文结论作为设计原则写入各仓 AGENTS/docs | 让“可审计记忆”成为长期工程原则 |

---

## 8. 推荐动作

1. 先上移 `fusion.py`、`projector.py`、`router.py` 到 engram-peft，并补跨仓契约测试；
2. 将 `live_store.py` / `slot_index.py` 中尚未合并的部分迁入 EngramDB，qwen35-ple 只做 re-export 兼容；
3. 在 LLM-CompileForge 增加“外部 n-gram logit correction + token policy”的 runtime 接口；
4. 把 `build_hf_artifact_release.py` 的脱敏与 checksum 逻辑提炼成兄弟项目通用的 release 工具。
