# qwen35-ple 项目设计

> 版本：v0.1（2026-08-30）
> 关联文档：`docs/integration-contract.md`（四仓库契约 v1）、
> `LLM-CompileForge/docs/qwen35-0.8b-100toks.md`（推理侧设计）。

---

## 1. 背景与目标

**动机**：Qwen3.8-Flash-Next（Qwen4 预览）证明「n-gram 记忆表（PLE）」是有效的容量扩展轴，
但其权重许可与 50B+ 规模限制了研究/边端使用。Qwen3.5 家族（0.8B~397B）是同期发布的
新一代主干（混合线性注意力 + MoE + MTP）。本项目把 Flash-Next 的 51.2B 参数记忆表
**冻结嫁接到 Qwen3.5 主干**上，通过继续预训练（CPT）与后训练（SFT/RL）获得
知识/长上下文更强的模型，并让最终形态在 **CPU 上达到 100 tok/s** 推理。

**关键判断（已核实，见 §2）**：Qwen PLE 与 DeepSeek Engram 是同一架构族——
[DeepSeek Engram 论文](https://huggingface.co/papers/2601.07372)（官方实现
[deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)）与
[engram-peft](https://github.com/QingGo/engram-peft) 的
`MultiHeadEmbedding + ShortConv + ContextAwareGating` 逐项对应
`Qwen4ExpTextPLELayer`（`refs/qwen4_exp_modeling.py:1191`）。因此本项目不是从零研究
新机制，而是**复用与定制**：以 engram-peft 为训练基础设施，把记忆层换成
「Qwen 口径」的 16 头 × 160 维 + 真实 Flash-Next 表。

**目标模型形态**：

```
Qwen3.5-0.8B-Base/4B-Base（冻结或低 LR）
  └─ 记忆层 ×N（默认 1 层，零基深度 1；hc=1 的 PLE-lite 变体）
       ├─ e_t = 16 行（8 bigram + 8 trigram）× 160 维拼接 = 2560 维
       ├─ key_proj / value_proj / 门控 gate / dilated depthwise conv
       └─ 表 = Qwen3.8-Flash-Next 真表（320,001,536 行 × 160B FP8 ≈ 47.7GiB）【冻结】
```

**非目标**：不复现 Flash-Next 全模型；不重新训练 51.2B 表（v0 冻结；低秩适应仅作消融，
见 §6 A2）；不新增 tokenizer 变体。

---

## 2. 可行性结论与依据

| 项 | 结论 | 依据 |
|---|---|---|
| 架构同源 | ✔ | engram-peft `layer.py`（ShortConv/ContextAwareGating/MultiHeadEmbedding(primes, dim_per_head)）与 PLE layer 逐项对应；[NeMo 将 Flash-Next 模块命名为 `qwen3_8_flash_next.engram`](https://docs.nvidia.com/nemo/automodel/nightly/nemo-automodel/nemo_automodel/components/models/qwen3_8_flash_next/engram) |
| 词表兼容 | ✔ | Flash-Next vocab=248320 == Qwen3.5-0.8B/4B/35B-A3B 全部 248320（HF config 实测）→ ngram 键空间直接对齐，rowid 语义无需改写 |
| 行/哈希规格 | ✔ | `engramdb-keygen` 已实现 `PleSpec::real`（Qwen 官方素数/乘子/偏移/EOS 段重置）并有 golden 对拍；契约 C1 注册为 `PLE_QWEN_V1` |
| 表可冻结 | ✔ | 表 51.2B 参数冻结后无梯度存储；适配层（key/value proj + gate + conv）仅 ~几十 MB 参数量；表读 2.5KB/token（FP8），训练侧带宽 ≈ 25MB/s @10K tok/s，无压力（Engram 论文「1% 按步激活」同此设计） |
| 先例 | ✔ | [Memory Grafting (arXiv 2605.20948)](https://ar5iv.labs.arxiv.org/html/2605.20948)：离线条件记忆 + 预训练扩展，与「表独立训练、转嫁新主干」同构；DeepSeek Engram 给出 U 形缩放律（记忆/算力配比） |
| 风险 | ⚠ | 「嫁接」相对「原生预训练」的收益**无先前公开证据**（Engram-27B 增益来自从零预训练）→ 必须消融（§6） |

---

## 3. 数据流（训练与推理共用同一条行语义链）

```
语料 tokens ──► engramdb-keygen(rowids_for_seq) ──► (T, 16) rowid 流（head 序固定）
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  Store-I（160B/行，分片）  Store-P 视图（2560B/记录  + keys + manifest）  [推理默认]
        │                     │
        └────────► engram-peft engine（qwen_ple）◄─── engramdb.integrations
                              │             install_disk_multi_head_embedding(store)
                              ▼
               Qwen3.5 主干（CPT → SFT/RL）
                              │
                              ▼
        LLM-CompileForge（.dylib + PrefetchPlanner + engramdb C ABI）→ 100 tok/s
```

要点：

1. **rowid 预知性**：一行流完全由 token 序列确定 → 训练侧可流水线预取；推理侧可在
   上一计算窗口内预取（PrefetchPlanner，契约 C3）。
2. **数值一致性**：Store-I 与 Store-P 视图在同一 rowid 上必须位级一致（EngramDB
   `view verify` 门禁已有）；训练与推理因此可混用两种形态而不引入数值漂移。
3. **head 顺序冻结**：`head 0..7 = bigram，8..15 = trigram`，行主序拼接（契约 C1）。

---

## 4. 训练设计

### 4.1 主干与记忆层规格

| 参数 | Qwen3.5-0.8B | Qwen3.5-4B | 说明 |
|---|---|---|---|
| 层数 / hidden | 24 / 1024 | 32 / 2560 | HF config 实测 |
| 注意力 | 混合（`full_attention_interval=4`：3 线性 + 1 全注意力） | 同左 | 均为 GatedDeltaNet 类 |
| MTP | 1 层（`mtp_num_hidden_layers=1`） | 同左 | 官方自投机，无损保留 |
| vocab | 248320 | 248320 | 与 Flash-Next 相同 |
| max ctx | 262144 | 262144 | |
| 记忆层注入 | depth 1（零基） | depth 1 | 对照 Flash-Next `ple_layer_ids=[2]`（1-based） |
| e_t 维 | 2560 → 经 key/value proj 到 1024 | 2560 == hidden | 4B 无需额外降维 |
| 表行宽 | 160B FP8（E4M3，官方 scale 语义） | 同左 | 契约 C1 |

### 4.2 阶段

| 阶段 | 内容 | 模式 | 规模（初值） |
|---|---|---|---|
| S0 适应 | 仅适配层（记忆层 + 可选 LoRA 主干）先行，校准 gate 初值（Qwen3.5 无 hyper-connection，query/key 空间与 Flash-Next 不同） | `backbone_freeze_steps=1000` | 0.2-1B tokens |
| S1 CPT | 全主干 + 记忆层继续预训练；表冻结 | `train_mode=engram_only` → 演化到 `preserve_trainable` | 消融 1-5B；正式 20-50B |
| S2 后训练 | SFT + 可选 DPO/GRPO（经 engram-peft `trl.py`） | `full_finetune`（记忆层仍冻表） | 见 §6 |

数据：FineWeb-Edu（en/zh，复用 EngramDB `scripts/corpus_build.py` 产物）；长上下文段
（256K 窗口）占比作为消融项 S1 子变量。

### 4.3 训练预算（0.8B，BF16 主干 + 冻结表）

| 项 | 量级 |
|---|---|
| 显存 | 主干+优化器 ≈ 5-8GB（ZeRO-2 可更低）；记忆层适配器 ≪1GB；表**不在显存** |
| 表带宽 | 2.5KB/token × 10K tok/s ≈ 25MB/s（Store-I 直取；页缓存/LRU 自管理） |
| 数值 | FP8 行 → 反量化 → BF16 进入 gate 计算；反量化语义以 EngramDB 位级校验器为准（契约 C1） |

### 4.4 与 Flash-Next 的已知差异（适配时要处理）

1. **无 hyper-connection**：Qwen3.5 config 无 `hc_count` 字段 → PLE-lite 以 `hc=1` 实现
   （`key_proj 2560→hidden`、`value_proj 2560→hidden`、conv 作用于 hidden 维）。
   这是与 Flash-Next 行为可能不等价的最大风险点 → S0 必须做 gate 初值校准与
   短 CPT 敏感性测试。
2. **MTP 共存**：记忆层注入层与 MTP 头独立工作；MTP 的验证步同样读取记忆层输入
   （token 已知 → rowid 同步已知）。
3. **混合注意力**：记忆层只属于「层注入」，不替换任何注意力；注入层建议选
   全注意力层（每 4 层一个）以叠加最强信号——作为 A 系列消融的一个维度。

---

## 5. 消融矩阵（A0-A5）

| # | 变量 | 对照 | 判定 |
|---|---|---|---|
| A0 | 无记忆层（Qwen3.5-0.8B 原版 + 同 CPT） | — | 基线 |
| A1 | PLE-lite @ depth1，表冻结（主假设） | A0 | 知识/长上下文增益 |
| A2 | 表低秩适应（仅热行；需 engine 新能力） | A1 | 表偏移是否需要 |
| A3 | 多层注入（[1, 5, 10, 15]） | A1 | 层数与收益曲线 |
| A4 | 0.8B 显式 2560→1024 降维（vs 直接 proj） | A1 | 适配器结构 |
| A5 | 长上下文段配比（25%/50%） | A1 | 训练分布 |

评测：知识 recall（域外问答）、长上下文（RULER / needle 类，含 128K-256K 档）、
reasoning 小样本（GSM8K/MATH）、检索型 agent 任务（复用 EngramDB `agent_workload_stats`
的负载形态）。

---

## 6. 工程结构与里程碑

```
src/qwen35_ple/
├── config.py     # 配置加载/校验（契约 C2 字段的默认值注入）
├── engine/       # 记忆层包装：调用 engram-peft engine = "qwen_ple"
├── data/         # 语料/行流数据管线（复用 EngramDB 产物）
├── train/        # CPT/SFT 运行入口（engram-peft CLI + 本仓消融参数）
├── eval/         # 评测脚本与报告
└── infer/        # CompileForge 对接（编译命令、A/B 脚本、预取调试）
```

| 里程碑 | 交付 | 验收 |
|---|---|---|
| M0 环境与资产 | Store-I 真表 + Store-P 视图构建脚本；engram-peft 磁盘注入跑通（借 TinyLlama e2e） | `view verify` 位级一致；e2e 可运行 |
| M1 PLE 引擎 | engram-peft `engine="qwen_ple"`（契约 C2）+ golden 对拍 | 与 `refs/qwen4_exp_modeling.py` 对 4096 行、4096 token 位级一致 |
| M2 CPT 消融 | A0-A3 完成 | 评测报告 + 是否推进 4B 的决策点 |
| M3 后训练 | SFT/RL 管线 + 评估 | A1 优于 A0（目标指标） |
| M4 推理闭环 | LLM-CompileForge 编译 + 100 tok/s 验收 | 见 LLM-CompileForge/docs/qwen35-0.8b-100toks.md |

---

## 7. 风险与开放问题

1. **嫁接收益未知**（最大风险）：A1 若负增益 → 项目停止点（只保留表资产与行语义规范）。
2. **gate 初值敏感性**：S0 校准失败的表现是「记忆层输出噪声淹没主干」→ 加
   `gating_zero_init` / `conv_zero_init` 门禁（engram-peft 已提供字段）。
3. **FP8 精度**：官方 weight_scale 为张量级 → 行级量化误差在长尾高频行上不可忽视；
   A2 若失败，可作为「热行 BF16 子表」的存储优化项（需 EngramDB 新视图变体，
   走契约 C1 新增枚举）。
4. **表分布与内容**：Flash-Next 表统计分布由其语料决定；中文/代码 n-gram 覆盖率
   需在 M0 用真实语料统计（EngramDB 已有 P2 统计工具）。
5. **开放**：PLE-lite 与 4B/35B-A3B 的 MoE + MTP 组合是否保持 100 tok/s 推理 →
   计算预算（>0.8B 需 M4 Pro/8 通道服务器），见推理设计文档。
