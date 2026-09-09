# Round 144：终极目标、技术债与下一阶段路线图

> 日期：2026-09-10  
> 主题：系统性思考 pure PLE grafting 的终极目标、本轮 session 暴露的技术债、后续开发计划，以及可借鉴项目的边界  
> 状态：large SFT / standard held-out 队列运行中；本文件是战略路线图，不是实验结果报告

---

## 0. TL;DR

```text
1. 终极目标没有变：
   frozen backbone + frozen PLE table + 只训练 reader/gate，
   用少量训练显著提升 0.8B 通用性能；可审计、低资源、可 CPU 推理。

2. 本轮最重要的正面结论：
   answer-only QA SFT + corpus next-token 50/50（mixed50）
   可以修复 BoolQ 的 format/instruction-following 回退，
   并且 PPL 优于 no-reader；纯 PLE 嫁接在机制上是可行的。

3. 本轮最重要的负面/警示结论：
   在 custom 62 held-out 上的 TriviaQA 大幅提升（0.550 -> 0.917）
   没有完全泛化到标准 1500 held-out：
   标准集上 BoolQ 0.672 -> 0.772（+10 分，泛化好），
   TriviaQA 0.128 -> 0.146（+1.8 分，泛化差），
   NQ 0.062 -> 0.050（-1.2 分）。
   说明当前 88 题 SFT 的“知识增益”过拟合了 custom 数据；
   下一阶段的核心不是加 router/RL，而是扩大和多样化 SFT 数据、
   建立标准 held-out 评估、并把知识增益做成可泛化的。

4. 技术债优先级：
   eval > data > architecture fidelity > ops > mechanism > baselines > serving。

5. 下一阶段路线：
   A. 评估与复现加固（1-2 周）
   B. 配方/数据/机制锁定（2-4 周）
   C. 泛化与 scaling（4-8 周）
   D. PLE-native 与 release（可选，资源允许时）
```

---

## 1. 终极目标（North Star）

### 1.1 一句话目标

> **纯 PLE 嫁接：通过少量训练显著提升 0.8B 通用性能。**

拆解：

```text
backbone      : 冻结（Qwen3.5-0.8B）
PLE table     : 冻结（Qwen3.8-Flash-Next PLE，Store-I 128 shards）
trainable     : 只训练 reader / bridge / gate（少量参数）
training cost : 少量（目标是单卡 4090、小时级）
evaluation    : 通用、可审计、可复现
serving       : 低资源，保留 CPU 推理可能性
```

### 1.2 非目标与 fallback 边界

以下手段**不是主路径**，只能作为 baseline / fallback：

```text
RAG              : 作为 retrieval baseline / 诊断工具
distillation     : 作为 PLE-native 不可得时的替代方案
LoRA / adapter   : 作为参数高效 baseline，不进入 pure grafting 主路径
RL / DPO         : 作为 SFT 饱和后的可选阶段，不是当前优先级
backbone 继续训练 : 原则上不做；除非证明纯嫁接不可行
```

### 1.3 成功标准（预注册 gate）

下一阶段开始前，先冻结一套“成功标准”，避免事后挑指标：

```text
知识任务（BoolQ / TriviaQA / NQ / 其他 held-out KB QA）：
  real > no-reader，且 95% CI 不跨 0；
  real > control（至少在知识型任务上），证明 PLE 内容有贡献；
  至少 2/3 知识任务达标。

通用任务（ARC / HellaSwag / MMLU 子集 / 代码 / 数学）：
  无 severe regression（相对 no-reader 下降不超过预设阈值）。

语言建模：
  real PPL <= no-reader PPL（同 corpus、同 tokenizer、同 scale）。

机制：
  real reader 的 contribution norm / gate 分布可解释；
  control 的增益可以被单独量化（format/regularization vs content）。

工程：
  相同 reader 跨 corpus 可用；
  推理显存/延迟 overhead 有上限；
  全流程可复现（固定 seed、固定 split、固定 scale/layer/EOS）。
```

---

## 2. 本轮 session 发现了什么

### 2.1 正面结论

1. **pure PLE grafting 在机制上可行**
   - mixed50（corpus next-token + answer-only QA SFT 50/50）在 custom 62 held-out 上：
     - BoolQ 0.682 -> 0.848
     - TriviaQA 0.550 -> 0.917
     - PPL 37.59 -> 28.52
   - 3 seeds 稳定（BoolQ ±0.021，TriviaQA ±0.024）。

2. **问题不是 gate，而是 reader 内容**
   - layer2 corpus-only reader 在 BoolQ 上 gate mean 只有 0.037，
     但 BoolQ 仍然 0.136；gate 强制为 0 后恢复 no-reader（0.760）。
   - gate=0.5 / 1.0 时 layer2 reader 两个任务都崩；
     SFT mixed50 reader 在 gate=0.5 / 1.0 时 BoolQ 0.88 / 0.92、TriviaQA 0.94 / 0.90。
   - 结论：**corpus-only reader 的内容在任何 gate 值下都有害；
     answer-only SFT 修好的是 reader 内容，不是 gate 开关。**

3. **oracle 路由是任务条件式的**
   - layer2 oracle(real/no)=0.575 vs no-reader=0.447。
   - unique-correct：BoolQ real-only 14 vs no-reader-only 141；
     TriviaQA real-only 142 vs no-reader-only 42。
   - 结论：oracle 的收益几乎全部来自
     “BoolQ 关掉 PLE + TriviaQA 打开 PLE”；
     不需要复杂 router，只需要让现有 gate 学会按输入类型开关。

4. **模型本身能用知识**
   - oracle context（gold answer 放进 prompt）：extracted_contains 0.700；
   - no-reader closed-book：0.447。
   - 结论：瓶颈在 PLE reader/注入，不在 instruction-following。

5. **control reader 也有增益**
   - sft-mixed50 control：BoolQ 0.833、TriviaQA 0.867；
   - real > control，尤其 TriviaQA（0.917 vs 0.867）。
   - 结论：一部分收益来自 reader/format regularization；
     但 PLE 内容有真实贡献，不能全部归因于 format。

### 2.2 负面/警示结论（本轮最重要的技术发现）

标准 held-out 1500（BoolQ 500 / TriviaQA 500 / NQ-open 500）：

| Arm | BoolQ | TriviaQA | NQ | Overall |
|---|---:|---:|---:|---:|
| no-reader | 0.672 | 0.128 | 0.062 | 0.287 |
| sft-mixed50 | **0.772** | 0.146 | 0.050 | **0.323** |
| sft-mixed50-purecode | 0.770 | 0.152 | 0.044 | 0.322 |
| sft-mixed25 | 0.730 | 0.152 | 0.044 | 0.309 |

解读：

```text
BoolQ：+10.0 分，泛化好；
        说明 answer-only SFT 修复 format/instruction-following 是真实、稳健的。

TriviaQA：+1.8 分，远小于 custom 62 上的 +36.7 分；
          说明当前 88 题 SFT 的“知识增益”过拟合了 custom 数据。

NQ：-1.2 分，轻微负向；
    说明 0.8B + 当前 reader 在 NQ 上还没有真实的知识增益。

Overall：+3.5 分，主要来自 BoolQ format 修复。
```

**这是下一阶段最重要的方向修正：**

```text
当前 recipe 已经证明“format/instruction-following 可以修好”，
但还没有证明“知识增益可以泛化到标准 held-out”。
下一阶段必须把重点从“刷 custom 62”转向：
1) 标准 held-out 评估；
2) 更大、更干净、更多样的 SFT 数据；
3) 验证知识增益是否随数据规模/多样性上升。
```

---

## 3. 技术债清单

按优先级排序。

### 3.1 评估债（最高优先级）

| 问题 | 风险 | 修复 |
|---|---|---|
| custom 62 held-out 太小、分布与 SFT 相近 | 高估知识增益；结论不可泛化 | 标准 1500 held-out 已建立；继续扩到 3-5 个独立任务、每个 500-2000 题 |
| 单一 metric / 单 seed | 结论不稳 | 预注册 task-aware metric + Wilson CI + per-seed 报告 |
| 没有独立 benchmark | 无法和社区比较 | 接入 lm-evaluation-harness / HELM 子集 |
| 没有 contamination audit | 测试集可能被 PLE 表/训练数据污染 | 对 SFT train / eval / PLE 表来源做 n-gram/entity 去污 |
| 没有 pass@k / 多采样 | 只测 greedy，低估/高估真实能力 | 对关键任务加 pass@k / temperature 采样 |
| gates 报告在单臂 eval 下为空 | 报告脚本不能处理单臂 | 让 gate 脚本支持“跨文件拼 real/control/no-reader” |
| 没有 held-out task split | 任务级过拟合 | 固定 train/val/test 任务划分，test 任务只在最后评估 |

### 3.2 数据债

| 问题 | 风险 | 修复 |
|---|---|---|
| SFT 只有 88 题 | 知识增益不泛化 | 已建 6000 题 standard split；下一步扩到 10k-100k |
| custom 150 同时是 train/eval 来源 | 评估污染 | 统一使用 qa-standard train/eval；custom 150 只做诊断 |
| 数据分布单一 | 只修 BoolQ format | 加入通用 instruction / 多任务 / 多语言 / 代码 / 数学 |
| 没有 conflict/sufficiency 数据 | gate 学不到“何时关闭” | 构造 PLE 有用/无用/冲突三类反事实数据 |
| 没有数据版本/哈希/manifest | 不可复现 | 每个 split 写 manifest + sha256 + 来源 + seed |
| 没有数据质量筛选 | 噪声/长尾 | 去重、答案归一化、长度过滤、任务平衡 |

### 3.3 训练协议债

| 问题 | 风险 | 修复 |
|---|---|---|
| reader 训练曾是 full corpus next-token | 产生有害内容 | 已加 answer-only QA SFT；继续系统研究 mixing ratio |
| 没有 validation-based early stopping | 过拟合/欠拟合 | 固定 val split，按 val QA + PPL 选 checkpoint |
| 没有 curriculum 系统实验 | mixed50 只是单点 | warmup（corpus -> mixed）已实现，正在跑 |
| 没有 LR / steps / reader capacity sweep | 不知道 scaling 曲线 | 小网格 sweep，固定预算 |
| 没有 loss weighting 研究 | 只试了 gate reg | 研究 answer-only weight、prompt mask、replay ratio |
| 没有 checkpoint averaging / EMA | 可能更稳 | 作为可选技巧，先不引入复杂度 |

### 3.4 架构保真债

| 问题 | 风险 | 修复 |
|---|---|---|
| `OfficialSourceQwenReader` 是 best-effort 复用 | 与官方 PLE 语义有偏差 | 用 golden forward / bit-exact 对拍官方实现 |
| PLE layer 曾是 8，官方是 2 | 早期结论不可用 | 单一 config 源；启动时校验 `ple_layer_ids` |
| scale / EOS / tokenizer 分散在多个脚本 | 配置漂移 | 建立 `ple_spec.yaml/json`，所有脚本从它读取 |
| reader 架构不是最终形态 | 后续迁移成本 | 冻结 `OfficialSourceQwenReader` 为 baseline；新 reader 走 registry |
| 没有版本兼容检查 | 换模型/表会静默错配 | 记录 model hash / table hash / tokenizer hash |

### 3.5 机制债

| 问题 | 风险 | 修复 |
|---|---|---|
| 不知道 layer2 reader 为何有害 | 只能试错 | contribution norm、gate 分布、activation patching、CKA |
| control 增益未完全解释 | 可能高估 PLE 内容贡献 | 增加 zero-reader / random-projection / prompt-only 对照 |
| 没有 causal tracing | 无法定位注入位置 | 对 layer 2/4/8/... 做 patching sweep |
| 没有 reader output calibration | 可能 scale 不匹配 | 研究 normalization / residual scaling / value clipping |

### 3.6 复现与运维债

| 问题 | 风险 | 修复 |
|---|---|---|
| 实验靠 shell + nohup | 难追踪、易断 | 轻量 experiment registry（YAML/JSON + runner） |
| 结果 JSON schema 不统一 | 汇总脚本易碎 | 定义统一 result schema + validator |
| 没有自动 artifact pull | 本地/远程不同步 | 定时/完成标记触发 rsync + hash |
| 没有资源 guardrail | 磁盘/RAM/显存 OOM | 启动前检查 free disk/RAM/GPU，超限拒绝 |
| 本次 `shutdown -h +N` 事故 | 误关机、打断实验 | 禁止容器 `shutdown` 时间参数；只用 AutoDL 控制台定时关机 |
| 没有 GPU CI | 训练回归不可见 | 夜间小 smoke + 结果阈值检查 |
| 没有成本记录 | 预算不可控 | 记录 GPU-hours / 实验成本 |

### 3.7 baseline / 定位债

| 问题 | 风险 | 修复 |
|---|---|---|
| 没有 LoRA/adapter 同预算 baseline | 无法证明纯嫁接更优 | 在相同训练预算下对比 LoRA / adapter |
| 没有 RAG baseline | 无法证明 PLE 的独特性 | 同 backbone + 同数据量的 RAG 对照 |
| 没有 no-reader/control 之外的强 baseline | 结论可能被 format 解释 | zero-reader、random projection、prompt-only SFT |
| 没有 PLE-native upper bound | 不知道瓶颈是表还是 backbone | 资源允许时跑 Qwen3.8 PLE on/off；否则用 official reader 近似并明确标注 |

### 3.8 serving / 产品债

| 问题 | 风险 | 修复 |
|---|---|---|
| 没有 KV-cache 集成 | 无法验证推理速度 | 实现 reader 注入 + KV cache 的 serving path |
| 没有 CPU 推理验证 | 项目目标包含 CPU 100 tok/s | 量化 + llama.cpp/ONNX 路线评估 |
| 没有 bundle/manifest 发布 | 无法交付 | 复用现有 bundle 机制，补 reader/table/model hash |
| 没有 latency/memory 报告 | 无法评估“低资源” | 固定硬件上的 tokens/s、峰值显存、内存 |

---

## 4. 下一阶段路线图

### Phase A：评估与复现加固（1-2 周）

目标：

```text
把“结论是否可信”这件事做扎实，再谈 scaling。
```

任务：

1. 冻结 canonical eval suite：
   - 知识：BoolQ、TriviaQA、NQ-open、WebQuestions/其他 held-out KB QA；
   - 通用：ARC-Easy/Challenge、HellaSwag、MMLU 子集、GSM8K 子集、HumanEval 子集；
   - 每个任务 500-2000 题，固定 prompt / few-shot / seed。
2. 建立 contamination audit：
   - SFT train vs eval vs PLE 表来源的 n-gram/entity 去污；
   - 记录每个 eval 题是否在 PLE 表/训练语料中出现。
3. 统一 result schema + report：
   - per-task / per-seed / CI / effect size；
   - real/control/no-reader 三臂 + zero-reader。
4. 实验 registry：
   - 每个实验一个 YAML（数据、模型、reader、超参、seed、预算）；
   - runner 自动写 manifest、日志、结果、哈希。
5. 运维：
   - 禁止 `shutdown` 时间参数；
   - 完成标记 + 自动 artifact pull；
   - 磁盘/RAM/GPU guardrail。

Deliverables：

```text
eval-suite-v1/ + manifest + contamination report
experiment registry + runner
round-145 评估报告：custom 62 vs standard 1500 的差异分析
```

Stop/go：

```text
如果 standard 1500 上 real 不能稳定 > no-reader，
先不要做 5M/20M scaling。
```

### Phase B：配方/数据/机制锁定（2-4 周）

目标：

```text
把“format 修复”升级为“可泛化的知识增益”。
```

任务：

1. 数据 scaling / diversification：
   - SFT 从 6k 扩到 10k-100k；
   - 多任务、多来源、多语言、代码/数学；
   - 严格 train/val/test 分离。
2. mixing / curriculum：
   - mixed50 vs mixed25/75 vs warmup（corpus -> mixed）；
   - answer-only vs full loss；
   - prompt mask / replay ratio / task balancing。
3. 机制：
   - contribution norm / gate 分布 / activation patching / CKA；
   - 找出 layer2 corpus-only reader 有害的原因；
   - 研究 reader output normalization / residual scaling。
4. baseline：
   - 同预算 LoRA / adapter / RAG / zero-reader；
   - 明确 pure grafting 的独特优势。
5. 小网格 sweep：
   - LR、steps、reader capacity、seq_len、QA weight。

Deliverables：

```text
recipe-v1：可复现的 mixed50/warmup 配方
standard held-out real > no-reader，且 CI 不跨 0
机制报告：为什么 SFT reader 内容变好、gate 可以安全打开
baseline 对比表（同预算）
```

Stop/go：

```text
如果知识增益仍不泛化，优先怀疑数据分布/任务选择，
而不是加 router、RL 或 backbone adapter。
```

### Phase C：泛化与 scaling（4-8 周）

目标：

```text
从“单任务/单数据集的成功”走向“通用性能提升”。
```

任务：

1. 多任务 / 多语言 / 长上下文 / 代码 / 数学；
2. held-out KB、unseen entity、多 split；
3. pass@k / 多采样 / temperature 评估；
4. 5M / 20M rows scaling；
5. reader capacity / table layer / injection layer scaling；
6. 与 LoRA/RAG/distillation 在相同预算下的系统对比；
7. 推理效率：KV cache、batching、quantization、CPU。

Deliverables：

```text
scaling law（数据量、reader 容量、steps vs 任务性能）
通用 benchmark 报告
效率报告（tokens/s、峰值显存、内存）
```

### Phase D：PLE-native 与 release（可选）

目标：

```text
回答“如果 backbone 是 PLE-native，上界在哪里”，并交付可复现 artifact。
```

任务：

1. 资源允许时跑 Qwen3.8-Flash-Next 原生 PLE on/off；
2. 如果 pure grafting 饱和，评估 distillation / PLE-native 小模型；
3. 发布 paper / artifact / model card / reproduction scripts；
4. 安全/滥用/污染审计。

---

## 5. 可借鉴项目与不冲突原则

| 领域 / 项目 | 可以借鉴 | 不要照搬 / 冲突点 | 如何集成 |
|---|---|---|---|
| DeepSeek Engram / EngramDB | PLE/n-gram reader、ContextAwareGating、ShortConv、bit-exact golden | 不要把 Engram 的压缩 tokenizer 直接套到 Qwen PLE（Qwen 用 raw token IDs） | 用官方 reader 语义 + golden 对拍；保持 raw token ID 路径 |
| Qwen3.8-Flash-Next PLE | `ple_layer_ids=[2]`、scale、EOS、tokenizer、表布局 | 不要把 layer 8 当默认；不要把 Qwen3.8 tokenizer 生成 EOS 与 PLE EOS 混淆 | 建立单一 `ple_spec` 配置源；启动时校验 |
| RETRO / Atlas / REALM / FiD | retrieval 训练、prompt mask、reranking、attribution、评估协议 | RAG 是 baseline/fallback，不是主路径；不要引入每 token 检索开销 | 作为同预算 baseline；借鉴 prompt/reader 训练技巧 |
| LoRA / adapter / prefix tuning | SFT 稳定性、参数高效、数据混合、early stopping | 会修改 backbone 或加 backbone adapter，违反 pure grafting 红线 | 只作为 baseline；主路径只训练 reader/gate |
| FLAN / Tulu / OpenHermes / LIMA | 数据质量、answer-only SFT、curriculum、数据配比 | 不要无脑堆数据；不要忽略 contamination | 借鉴数据工程和 SFT recipe；保持 PLE 表冻结 |
| DoReMi / RegMix / DataComp / SFTMix | 数据配比、domain weighting、curriculum | 需要大量 proxy 训练；不要过早引入复杂优化 | 在 recipe-v1 稳定后做小规模 mixing 研究 |
| Memorizing Transformers / kNN-LM | memory indexing、retrieval、gating、serving | 不要把 PLE 当成可训练/可更新数据库；PLE 是冻结表 | 借鉴 serving/eval；保持表冻结 |
| ROME / MEMIT / activation patching | causal tracing、knowledge localization、layer 选择 | 编辑 backbone 权重违反 pure grafting | 只用诊断方法，不改权重 |
| HELM / lm-evaluation-harness / BIG-bench | 标准任务、few-shot、CI、污染审计 | 不要盲目追求 leaderboard；先固定小 suite | 接入子集，建立 canonical eval |
| vLLM / SGLang / llama.cpp / TensorRT-LLM | KV cache、batching、quantization、CPU 推理 | 需要 reader 注入兼容；不要牺牲可审计性 | 先做 serving prototype，验证 tokens/s |
| W&B / MLflow / DVC | 实验追踪、artifact 版本、指标面板 | 不要引入过重依赖；保持轻量 | 轻量 registry + 文件系统 + manifest |
| Ray Tune / Optuna | HPO、调度 | 不要在 recipe 未锁定时做大规模 HPO | recipe-v1 之后小网格 |
| DeepSpeed / FSDP / quantization | 显存优化、大模型训练 | 0.8B + 4090 当前不需要；不要引入复杂度 | 只在 scaling/CPU 阶段评估 |

### 5.1 不冲突原则

```text
1. 红线：backbone 冻结、PLE 表冻结、只训练 reader/gate。
   任何修改 backbone 的方法（LoRA/adapter/RL/继续训练）
   只能作为 baseline 或 fallback。

2. 单一事实源：
   PLE spec（layer/scale/EOS/tokenizer/table layout）只存一份；
   所有脚本从它读取。

3. 预注册 gate + 冻结 test set：
   先定成功标准，再看结果；禁止事后挑指标。

4. 每个实验三臂 + 机制：
   real / control / no-reader（+ zero-reader）；
   同时记录 gate/contribution 统计。

5. 分阶段：
   eval 加固 -> recipe 锁定 -> scaling -> PLE-native；
   前一步 gate 不过，不进入下一步。

6. 资源与运维纪律：
   成本、显存、磁盘、自动关机都纳入流程；
   不再使用容器 shutdown 的时间参数。
```

---

## 6. 近两周具体任务清单

```text
Week 1:
  [ ] 冻结 canonical eval suite v1（知识 + 通用 + 代码/数学子集）
  [ ] contamination audit
  [ ] 统一 result schema + report（per-task/per-seed/CI）
  [ ] experiment registry + runner
  [ ] 完成 standard 1500 + large 6000 SFT 结果分析
  [ ] 明确 custom 62 vs standard 1500 的差异原因

Week 2:
  [ ] 数据扩到 10k-100k，多任务/多来源
  [ ] mixed ratio / warmup / answer-only loss 小网格
  [ ] 机制分析：contribution norm / gate / patching
  [ ] 同预算 baseline：zero-reader / control / LoRA / RAG
  [ ] 确定 recipe-v1
  [ ] 写 round-145 报告 + 下一阶段 gate
```

---

## 7. 结论

```text
1. 目标没变：pure PLE grafting，少量训练，显著提升 0.8B 通用性能。
2. 本轮证明了“机制可行 + format 可修”；
   但还没有证明“知识增益可泛化”。
3. 当前最大瓶颈不是 router、不是 RL、不是 backbone adapter，
   而是：
   - 评估不够标准/不够大；
   - SFT 数据不够多/不够多样；
   - 机制理解不够深。
4. 下一阶段最重要的三件事：
   A. 标准 held-out eval；
   B. 大规模、多样化 SFT 数据；
   C. 机制诊断 + recipe 锁定。
5. 借鉴别人，但守住红线：
   冻结 backbone/PLE 表，只训练 reader/gate；
   其他方法只做 baseline 或 fallback。
```

---

## 附：当前正在运行的队列

```text
large SFT queue (scripts/run_large_sft_queue.sh)：
  1. standard 1500 held-out eval：已完成 4 arms；
  2. sft-large-mixed50（6000 SFT，3 seeds）：运行中；
  3. sft-large-mixed50-warmup250：排队；
  4. full 1500 held-out eval：排队；
  5. oracle + report：排队。

完成标记：
  /root/autodl-tmp/qwen35-ple/outputs/LARGE_SFT_DONE
  /root/autodl-tmp/qwen35-ple/outputs/OVERNIGHT_REPORT.md
```

> 本文件是战略路线图；具体实验结果以 `outputs/OVERNIGHT_REPORT.md` 和后续 round 报告为准。
