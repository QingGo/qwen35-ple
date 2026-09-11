# qwen35-ple

**Qwen3.5 主干 + Qwen3.8-Flash-Next PLE 记忆表**的嫁接实验仓库。本仓库是四仓库协作中的
**编排者**：实现存储与模型核心的是兄弟仓库，这里做的是消融设计、评测协议、数据与实验编排。

> **本仓库的首要产出不是"一个更好的模型"，而是一组关于「哈希 n-gram 外部记忆到底能做什么」
> 的可复现结论 —— 其中一条是证明，不是经验。** 详见下方「结论摘要」。

---

## 结论摘要（TL;DR）

经过 round 130–158 的连续实验，**「冻结表 + 冻结主干」这条嫁接路线**得出了封闭结论：
（注意范围 —— 原版三家用的是**可训练的表 + 联合训练的主干**，见第 7 条。）

**1. 正确的界是条件形式：瓶颈是「窗口」，不是「容量」。**

`PleSpec.rowids_for_seq`（`src/qwen35_ple/ple_hash.py`）只把 `shift = 0/1/2` 折进哈希，
即 `token[t]`、`token[t-1]`、`token[t-2]`。所以 16 个头**全部**是 (t-2, t-1, t) 的确定性函数。
但注意通道的实际输出不是 `e_t`，而是 `c_t = F(h_t, e_t)` —— **gate 用 `h_t` 作 query**。
所以记忆的净增量必须写成条件形式：

```text
e_t = f(token[t-2], token[t-1], token[t])              （确定性函数）
记忆的净增量 = I(future ; e_t | h_t)  ≤  I(future ; trigram | h_t)   ← 数据处理不等式
⟹ 记忆的价值 = 表的 trigram 统计 − 权重里隐含的 trigram 统计（在 h_t 未钉死的部分上）
```

这个界**与 reader 的深度、宽度、线性与否、训练量全都无关**，而且它一次解释三件事：

| 情形 | trigram 携带 | 理论预测 | 实测 |
|---|---|---|---|
| 段落条件化问答（BoolQ 300-token 段落） | 只有 `"...\nAnswer:"` 的位置/格式信息 | 只能加**格式** | ✅ **格式先验**（round 156） |
| `"The capital of France is"` | 内容 | 可加**精确统计** | ❓ 从未单独测过 |
| 需要窗口外上下文的事实 | 无 | **零** | ✅ 全部知识探针为零 |

**我们此前所有知识探针都只用了第一行那一格 —— 理论说必然为零的那一格。**
详见 `docs/round-157-why-the-graft-is-a-prior-not-a-knowledge-channel.md`。

**1b. Round 159 从另一条路走到了同一结论：值得做的分布，恰恰在窗口外。**

四语料跨分布实测（WIKI / CODE / FINEWEB / STEM，阈值先于数据固定）：

| | WIKI | **CODE** | STEM | FINEWEB |
|---|---:|---:|---:|---:|
| k=4 逐字命中率 | 0.112 | **0.484** | — | — |
| k=16 逐字命中率 | 0.0017 | **0.208**（122×） | — | — |
| 可寻址×可记忆 NLL 质量 k≥8 | 0.012 | **0.173**（14×） | 0.034 | 0.006 |

两件事同时成立：**① 限制属于分布而非方法**（代码与散文判然有别）；
**② 但代码的优势主要在 k≥8，而 trigram 寻址只能到 k≤4** ——
**记忆最值钱的区间，正好是当前寻址够不到的区间。** V4.1 的 `{2,3,4}` 也只到 k≤4。

> 所以结论是**换寻址，而不是加表**：更长的键，或查询相关寻址（kNN-LM / RETRO 那一路）。
> 详见 `docs/round-159-distribution-dependence.md`。

**2. 这条通道是「短窗口先验」，不是缺陷 —— 三家原版设计也是如此。**

| | Engram-27B | Qwen3.8-Next | V4.1-Flash | 本仓库嫁接 |
|---|---|---|---|---|
| n-gram 阶 | {2,3} | {2,3} | {2,3,4} | {2,3} |
| 记忆 / 激活算力 | 1.4× | 8.5× | 24.5× | **64×** |
| 表如何训练 | Adam 5×LR | **与主干联合训练** | momentum+Sinkhorn | **冻结、无梯度** |

上界对三家**同样成立**（V4.1 是 4-gram 界）。它界定的是**通道用途**，不是"设计有毛病"：
Engram 论文的定位是 *A New Axis of Sparsity* —— 让激活参数**不必**承担记忆负载，
并在 **iso-FLOPs（对 Dense-4B）** 下评测。收益来自算力再分配，不是知识注入。

**3. 我们量化的是"嫁接必须打败的数字"，不是通道的终极价值。**

| | NLL | top-1 | 存储 |
|---|---:|---:|---:|
| 四阶计数 n-gram（MKN，**与嫁接同源的 1M token 语料**） | **5.3368** | **25.25%** | **32 MB** |
| 该语料上的 trigram 渐近估计（R²=0.96） | ≈4.94 | ≈27–28% | — |
| 冻结 PLE 表 | ? | ? | **47.684 GiB** |

> ⚠️ **不要把 ≈4.94 当作通道天花板。** 它是从 **1M token** 学习曲线外推的，
> 而原版记忆是从**数万亿 token** 构建的；真实的 trigram 后验远好于此。
> 因此"表格最多值 0.40 nat"**只在我们这套语料+评测下成立**，不是普适结论。
> 可以确定的只是：**在这套评测上，一张 32 MB 的计数表就是 graft 必须打败的对手。**
> 计数格式每条 n-gram 密 9.4 倍（表的 320,001,536 行用精确计数表示约 5.44 GB）。
> 详见 `docs/round-158-ngram-reference-frame.md`。

**4. 它确实有一个真实、可复现的效应 —— 但是格式，不是知识。**

零注入时 4B **不遵守** raw 续写协议，会退回对话先验吐出 `<think>` 脚手架；
**注入 PLE（real 与乱序行皆然）才会把它压回 raw 格式**。这是机制性的、与表内容无关的
**格式/协议先验**，可能是这条通道唯一可产品化的东西。
详见 `docs/round-156-g0-nople-format-vs-content-and-metric-bugs.md`。

**5. 已否证清单 —— 注意每一条否证的都是「冻结表 + 未学过使用它 的主干」这一配置**：

| 假设 | 判定 | 证据 |
|---|---|---|
| 冻结嫁接可注入知识 | ✗ | round 148/149 |
| reader 容量 / 源空间对齐是瓶颈 | ✗ | round 148-B |
| oracle routing 可线性学到 | ✗ | `scripts/train_oracle_routing_probe.py` |
| 靠数据规模可解锁 | ✗ | 1M→ 更大语料无改善 |
| full-FT 可解锁（反而崩塌） | ✗ | round 149 |
| LoRA 共适应可解锁 | ✗ | round 152：+2.8–3.5 点但 real−control = **+0.0000** |
| 4B（hidden 精确对齐源空间）可解锁 | ✗ | round 155/156：那是 control 解码崩塌造成的假象 |

**6. 那条界证明了什么、没证明什么 —— 这一条最容易误读。**

```text
证明了：  e_t 关于 token[t+1] 的信息量 ≤ 最近 2–3 个 token 的信息量
   ⟹ 这条通道【不可能】做「按查询取回」—— 答案依赖全上下文的知识型任务
   ⟹ 我们对它做过的全部知识探针（QA / oracle routing / rare-entity）都在问错的问题

没有证明：这条例通道没用
```

界限制的是**信息量**，不是**有用性**。骨干权重是训练语料的**有损压缩**，
低频 n-gram 统计恰恰是被压掉的部分 —— 表能补上权重丢掉的。而 0.8B 的权重比
6B active 的权重丢掉**更多**，所以原则上小模型应该**更**受益，不是更少。

**7. 原版验证过、而我们从未测过的配置**（这是"为什么他们行"的核心）：

| 配置 | 原版三家 | 本仓库 |
|---|---|---|
| 表是否训练 | ✅ Adam 5×LR / momentum+Sinkhorn | ❌ **冻结，无梯度** |
| 主干与记忆 | ✅ **联合训练**（主干学会路由到记忆） | ❌ 主干预训练时**从未见过**记忆；只有冻结 / LoRA / full-FT |
| 记忆:激活算力 | 1.4× / 8.5× / 24.5× | ❌ **64×**（论文的 U 型分配律说存在最优比） |
| 跨空间桥 | 不需要（同源） | ❌ 需把 Qwen 源空间(2560,4分支) 译到 Qwen3.5(1024,单流)，且 4 分支求和是有损近似 |
| gate 选择性 | 论文案例显示选择性开启 | ❌ 我们的 SFT reader gate 饱和常开（0.77–0.98）= 恒开注入 = 噪声 |

> 所以：**"Engram 设计不成立"没有被证明。我们证明的是"冻结的跨模型嫁接不成立"。**
> 这两件事经常被混为一谈，包括我自己在 round-157 里也说得过重。

**8. 仍然开放的问题**：上述 4.94 的天花板是**自然文本**的。n-gram 记忆的价值强烈依赖分布
（代码、日志、结构化语料的天花板会不同）。跨分布扫描进行中，判据是
**长尾 NLL 占比 + 逐字续写命中率**，而不是天花板本身。

---

## 仓库角色与依赖

```text
engram-peft     (模型库: DeepSeek Engram 一致性实现, 记忆层/训练/TRL 基础设施)
EngramDB        (存储: PLE/Engram n-gram 行表, badge 布局, Store-P 视图, C ABI)
   ▲              ▲
   └──────┬───────┘
    qwen35-ple   (本仓库: 嫁接实验编排, 依赖以上两者)
       ▲
LLM-CompileForge (推理: MLIR 编译 .dylib + Rust runtime, CPU 推理目标)
```

依赖方向严格无环：`LLM-CompileForge → EngramDB`；`qwen35-ple → {engram-peft, EngramDB}`。
契约唯一权威是 `docs/integration-contract.md`（v1，冻结），纪律是**只允许新增**。

---

## 评测协议与纪律

这一节是本项目最可迁移的产出。**我们曾因为协议缺陷两次得出错误结论**，以下每条都由真实事故换来。

### 指标定义（务必分清）

| 指标 | 正确用法 | 已知陷阱 |
|---|---|---|
| `qa_<task>_em` | 与历史轮次可比 | **子串匹配**，会虚高脚手架输出最多 33 个点（`Nouser` 含子串 `no`） |
| `qa_em_token_mean` | **跨臂比较用这个** | token 级窗口判等，拒绝"答案被融合进更大 token"的假命中 |
| `qa_fmt_*` | **每个臂都必须带** | 脚手架率 / 空串率 / distinct 率 / BoolQ yes 率 |
| `qa_mean_nll` | 历史可比口径 | 评的是**不带前导空格**的答案分词（`'yes'`=9405），而模型实际吐 `' yes'`=9542 |
| `qa_mean_nll_spaced` | **诊断口径**（`--qa-gold-nll-spaced`） | 与上面之差 = 格式敏感度 |

### 四条硬纪律

1. **`control`（乱序行）不能替代 `no-PLE` 对照。** 前者是**主动扰动**，只回答"内容是否匹配"；
   后者才回答"有 PLE 是否比没有好"。缺 `--ple-off` 臂会让假阳性无法被内部证伪
   —— round 152 的 G0 正是这么产生了一个后来被推翻的"正结果"。
2. **任一臂偏离分布时，生成 EM 不是有效的内容探针。** 它测的是解码稳定性。
   必须同时报 teacher-forced NLL，且**两者矛盾时先排查分词/格式，而不是默认信 NLL**。
3. **没有格式指纹就不许跨臂比较任何指标。**
4. **训练与评测必须同协议。** 复用 raw 训练的 adapter 去 chat 评测，同一 adapter 的
   gold NLL 会从 2.40 变到 5.02 —— 这个差距量化的是协议不匹配，不是能力。

### 无效臂审计

`scripts/audit_reader_checkpoints.py` 强制检查：冻结源张量跨臂**逐位一致**、adapter **确实移动过**、
**跨臂范数比**在阈值内。这条纪律的直接来源是一次事故：LoRA 权重曾从未进入 optimizer，
500 步后 96/96 个 `lora_B` 全为 0，而"训练后"的臂与冻结基线逐位相同。

### 预注册

判读标准必须在看数字之前写进文档。round-155 §7 写下的方向性预测，让后续一个
**与预期相反**的结果能被立刻定位，而不是事后挑选解释。

---

## 快速开始

### 本地开发

```bash
# 前置: uv、Rust 工具链（engramdb-python 需要 maturin 构建）
uv sync --all-groups
uv run python -c "import engram_peft, engramdb, qwen35_ple; print('ok')"
uv run pre-commit install

make lint    # ruff（范围与 CI 一致）
make test    # pytest
make check   # lint + test
```

> **注意**：本机 `.venv` 不含 torch，`pytest` 会有若干 collection error；
> 权威环境是远端（见下）。

### 远端（AutoDL）

```bash
bash scripts/ssh_autodl.sh                 # 交互 shell
bash scripts/ssh_autodl.sh 'uptime; df -h' # 单条命令
QWEN35_PORT=40783 bash scripts/ssh_autodl.sh '...'   # 端口随实例变化，用 env 覆盖
```

远端布局（`/root/autodl-tmp/qwen35-ple/`）：

| 路径 | 内容 |
|---|---|
| `repo/` | 本仓库检出；`PYTHONPATH=src` 是**必需**的（venv 有 engramdb 但没有 qwen35_ple） |
| `venv/` | Python 环境（torch 2.6.0+cu124 / transformers 5.16.1 / peft 0.20.0） |
| `models/` | `Qwen3.5-0.8B` / `Qwen3.5-2B` / `Qwen3.5-4B` / `qwen38_ple` / `Qwen3.8-Flash-Next-FP8-tokenizer` |
| `qwen38-rows/` | PLE 行表，**128 个 `shard_NNN.bin`，各 400,001,920 B**（只读） |
| `outputs/` | 实验产物 |
| `logs/` | 运行日志 |

`/dev/shm/qwen38-rows` **重启即失**；需要时用
`bash scripts/ensure_qwen38_rows.sh --python /root/autodl-tmp/qwen35-ple/venv/bin/python` 重建。

---

## 复现主要结果

### 1. PLE 表的天花板（计数参照系，无需 GPU）

```bash
bash scripts/ssh_autodl.sh 'cd /root/autodl-tmp/qwen35-ple/repo && \
  /root/autodl-tmp/qwen35-ple/venv/bin/python scripts/bench_ngram_reference.py \
  --train-npy data/phase1/PURE_WIKI/tokens.npy \
  --eval-npy  data/phase1/wikitext-heldout-decon/tokens.npy \
  --orders 1,2,3,4 --smoothing mkn --uniform-vocab 248047 \
  --acc-positions 20000 --acc-candidates 5000 \
  --output outputs/ngram-reference.json'
```

约 45 秒。自带 `--self-test` 回归覆盖。

### 2. 表的可读出性探针（无需 GPU）

`scripts/probe_table_next_token.py` —— 原始 2560 维行 → 下一个 token，对照显式计数 trigram
与**乱序行控制**（必须崩到下限，否则结论作废）。自带 `scripts/selftest_probe_table.py`
（signal 模式必须检出、noise 模式必须崩塌）。

### 3. 嫁接评测（需要 GPU）

```bash
ROOT=/root/autodl-tmp/qwen35-ple
$ROOT/venv/bin/python scripts/run_phase0.py \
  --live-store --tokens-npy data/phase1/PURE_WIKI/tokens.npy \
  --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
  --model-dir $ROOT/models/qwen38_ple --model $ROOT/models/Qwen3.5-0.8B \
  --backbone-dtype bfloat16 --reader official --layer 2 --device cuda --bridge-mlp --out-mlp \
  --official-reader-path data/official_ple_reader.pt \
  --steps 0 --seeds 0 --modes real \
  --qa --qa-exact-match --qa-gold-nll --qa-gold-nll-spaced \
  --qa-file data/qa-standard/eval.jsonl --qa-max-new-tokens 32 \
  --qa-prompt-template $'Question: {question}\nAnswer:' \
  --qa-boolq-prompt-template $'Question: {question}\nAnswer with one word, Yes or No:' \
  --output outputs/arm.json
```

关键 flag：`--ple-off`（no-PLE 对照格）、`--lora`（主干共适应）、`--qa-max-items`（冒烟）。
**`--modes no-reader` 在训练前就返回**，所以 2×2 的 no-PLE 格必须用 `--ple-off`。

### 4. 汇总与审计

```bash
python scripts/summarize_arms.py    --run A=a.json --run B=b.json --pair A=B --output arms.md
python scripts/summarize_gold_nll.py --run A=a.json --baseline A --output gold.md
python scripts/audit_reader_checkpoints.py --reference real=r.pt --arm control=c.pt --init data/official_ple_reader.pt
python scripts/bench_engram_edge.py --rows-dir ... --label nvme-persistent --cold
```

---

## 工具索引

| 脚本 | 用途 |
|---|---|
| `run_phase0.py` | **主评测/训练入口**（三线 real/control/no-reader、LoRA、`--ple-off`、QA EM/gold NLL） |
| `bench_ngram_reference.py` | 计数 n-gram 参照系（MKN 1–4 阶，精度/字节） |
| `probe_table_next_token.py` | 表的可读出性探针（含乱序控制） |
| `audit_reader_checkpoints.py` | 无效臂审计（冻结一致性 / adapter 漂移 / 范数比） |
| `summarize_arms.py` / `summarize_gold_nll.py` | 配对汇总（逐 item delta + SEM） |
| `bench_engram_edge.py` | 端侧存储基准（热/冷缓存、batch sweep、RSS） |
| `build_mix.py` / `build_qa_standard_split.py` | 语料混合与标准 QA 切分（含 `--exclude-qa` 去污染） |
| `ssh_autodl.sh` | 远端入口（**端口随实例变化**） |

`scripts/` 共 125 个 Python + 31 个 shell 脚本；完整清单见 `docs/reproducibility-manifest.md`。

---

## 目录结构

```text
src/qwen35_ple/    实验编排代码（config / engine / data / train / eval / infer / serving）
  ple_hash.py      PLE 行 id 计算 —— §1 那条上界的来源
  reader.py        OfficialSourceQwenReader 等 reader 实现
  live_store.py    PLE 行表懒加载（LiveETStore / LiveETDataset）
configs/           训练与推理配置
scripts/           一次性脚本（数据构建、表资产、评测、审计）
docs/              158 篇文档（索引见下）
tests/             35 个测试文件（golden 对拍与不变量）
paper/             paper.typ + figures（编译产物 paper.pdf）
```

---

## 文档索引

`docs/` 共 158 篇。**不要通读**，按主题进入：

**起点（想快速了解现状）**

| 文档 | 内容 |
|---|---|
| `round-157-why-the-graft-is-a-prior-not-a-knowledge-channel.md` | **可证明的天花板 + 为何它只能承载先验**（建议先读） |
| `round-158-ngram-reference-frame.md` | **计数参照系：这张表必须打败的数字** |
| `round-156-g0-nople-format-vs-content-and-metric-bugs.md` | 格式 vs 知识；两个指标 bug；关机事故复盘 |
| `round-154-session-consolidation-and-handoff.md` | 会话交接总览 |
| `round-153-goal-tech-debt-and-development-plan.md` | 目标精确化、技术债、停止规则 |

**契约与设计**：`integration-contract.md`（唯一权威）、`qwen35-ple-design.md`、`roadmap.md`

**评测协议**：`round-145-*`（标准 held-out 与 chat 模板）、`round-146-*`（cross-file oracle）、
`round-148-*`（format vs content 判决）、`evaluation-card-paper.md`、`phase0-protocol.md`

**否定性结果（重要，避免重复劳动）**：`round-148` / `149` / `150` / `152` / `155` / `156` / `157` / `158`

**存储与端侧**：`round-152-g3-engram-edge-storage-benchmark.md`

**可复现**：`reproducibility-manifest.md`

**历史**：`round-19` … `round-147`（早期 PLE 探针、机制分析、RAG/蒸馏路线、多轮系统性复盘）。
其中 round 44–123 主要探索 RAG / 蒸馏 / 可寻址记忆，结论已收敛并记录在 `session-log.md`。

`session-log.md` 是按会话的连续记录，含每轮的坑与修复。

---

## 工程状态与技术债

**已落地**

- 契约 v1 冻结，CI 绿（lint + shell 语法 + pytest + 两个 synthetic gate + 论文 PDF）
- 三线评测协议、配对统计、gold NLL 分批（数值等价且 3× 加速）
- reader checkpoint / LoRA adapter 存取、bundle + serving adapter、reader registry
- 端侧存储基准、无效臂审计、计数参照系、表探针
- 自动关机 finisher（含生产者存活检测与失败即关机）

**已知技术债（按优先级）**

| 优先级 | 项 |
|---|---|
| P0 | 跨分布扫描未完成：4.94 的天花板是自然文本的，代码/结构化语料待测 |
| P0 | gold NLL 的空格主口径尚未成为默认（现为诊断 flag） |
| P1 | 真实 vLLM / SGLang 引擎 A/B 未做 |
| P1 | 端侧只测了 NVMe；USB SSD / SD 与移动端功耗未测 |
| P2 | `docs/` 158 篇缺少系列化的归档策略，检索成本偏高 |

**停止规则（修订版）**

round-153 预注册：若比例扫描在任何比例下都不显示相对同预算纯参数的劣势优势，
则把 PLE 重新定位为**格式先验 + 可热插拔的领域先验**，通用能力目标交还给参数与数据。
round-157/158 满足该条件 **对于「冻结嫁接」这一配置**。

修订后的纪律：

* ❌ **停止**再跑「冻结表 + 冻结主干」的知识型 QA 消融 —— 那条界说它必然为零，第六次不会带来新信息。
* ✅ **尚未排除**的是原版真正验证过的配置：**可训练的表 + 学过使用记忆的主干**。
  在资源允许的范围内应测：主干参与训练（LoRA 或部分解冻）、gate 选择性修复
  （可学习 per-dim q/k 权重、per-branch key），评测改为**长尾文本的语言建模**而非 QA 知识。
* ⚠️ 资源上做不到的部分（320M 行的表做 Sinkhorn 平衡、从零联合预训练）应**明确标注为未验证**，
  而不是被前述否证顺带否定。

---

## 与兄弟仓库的交互契约（摘要）

- **存储契约 C1**（EngramDB → 使用方）：行语义 `PLE_QWEN_V1`（16 头 / 160 维 / 320,001,536 行）、
  视图格式（`<view>.manifest.json` + keys 文件）、C ABI 符号冻结规则。
- **模型契约 C2**（engram-peft → 本仓库）：`EngramConfig` 只增不改；
  `engine="deepseek"（默认）| "qwen_ple"`；`table_source="engramdb:store"` 自动注入。
- **推理契约 C3**（LLM-CompileForge ↔ EngramDB）：`sfa_abi.proto` 加 `SfaWeightSource`；
  视图可作为外部权重源；运行时 dlopen 加载 C ABI。
- **数据契约 C4**：tokenizer 唯一来源 = Qwen 官方（vocab 248320，与 Flash-Next 相同，已核实）。

变更纪律：**只允许新增，禁止改语义/删除；ABI 演进用 `_v2` 新符号。**

---

## 公开 Artifact

- Hugging Face：[DefEki/qwen35-ple-auditable-ngram-memory](https://huggingface.co/DefEki/qwen35-ple-auditable-ngram-memory)
- 论文源码 `paper/paper.typ`，编译产物 `paper.pdf`
- 可复现清单 `docs/reproducibility-manifest.md`，评测卡 `docs/evaluation-card-paper.md`
