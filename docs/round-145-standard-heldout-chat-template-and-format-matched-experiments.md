# Round 145：标准 held-out、chat template 与 format-matched 实验

> 日期：2026-09-10  
> 主题：标准 held-out 评估、6000 题大规模 SFT、chat template ablation，以及 format-matched 实验计划  
> 状态：chat template ablation 完成；large-SFT 队列运行中；chat-SFT 队列已挂载等待

---

## 0. TL;DR

```text
1. 标准 1500 held-out 说明：
   - BoolQ 的 format 修复可以泛化（0.672 -> 0.772）；
   - TriviaQA 的知识增益不能泛化（0.128 -> 0.146）；
   - NQ 轻微负向（0.062 -> 0.050）。

2. 大规模 SFT（6000 题）没有改变这个结论：
   - eval-600 上 real 0.742 / control 0.715 / no-reader 0.645（BoolQ）；
   - TriviaQA real 0.157 vs control 0.155 vs no-reader 0.145，基本是噪声。

3. chat template ablation（300 held-out）：
   - chat-no-reader 显著强于 raw-no-reader：
     BoolQ 0.61 -> 0.74，overall 0.280 -> 0.333；
   - 同一个 raw-trained reader 切到 chat eval 后优势基本消失：
     chat-sft-mixed50 BoolQ 0.70 vs chat-no-reader 0.74。

4. 当前最关键的竞争假设：
   H1. reader 没有在 chat-template hidden state 上训练，OOD 导致增益消失；
   H2. raw reader 的增益本来就是 format compensation，不是知识；
   H3. PLE 内容/覆盖/容量瓶颈，chat 数据训练也修不好。

5. 判决性实验：
   chat-template SFT + chat-template eval 的 2×2 factorial；
   sft-chat-mixed50 / warmup250 队列已挂载，等待 large-SFT 结束后自动启动。
```

---

## 1. 本轮目标

```text
1. 把评估从 custom 62 扩到标准 held-out 1500；
2. 把 SFT 从 88 题扩到 6000 题；
3. 验证 PLE reader 的增益是否泛化；
4. 检查 chat template 是否是评估协议的关键混淆变量；
5. 如果 chat template 改变结论，准备 format-matched 的判决实验。
```

---

## 2. 关键发现

### 2.1 标准 1500 held-out（旧 88 题 SFT，raw prompt）

数据来源：

```text
BoolQ    500 题  (google/boolq validation)
TriviaQA 500 题  (mandarjoshi/trivia_qa rc.wikipedia validation)
NQ-open  500 题  (google-research-datasets/nq_open validation)
```

| Arm | BoolQ | TriviaQA | NQ | Overall | 备注 |
|---|---:|---:|---:|---:|---|
| no-reader | 0.672 | 0.128 | 0.062 | 0.287 | closed-book |
| sft-mixed50 | **0.772** | 0.146 | 0.050 | **0.323** | 88 题 SFT |
| sft-mixed50-purecode | 0.770 | 0.152 | 0.044 | 0.322 | 88 题 SFT，PURE_CODE corpus |
| sft-mixed25 | 0.730 | 0.152 | 0.044 | 0.309 | 88 题 SFT |

结论：

```text
BoolQ：+10.0 分，format/instruction-following 修复泛化好；
TriviaQA：+1.8 分，知识增益没有泛化；
NQ：-1.2 分，轻微负向；
Overall：+3.5 分，主要来自 BoolQ format。
```

对比 custom 62：

```text
custom 62：BoolQ 0.848，TriviaQA 0.917
标准 1500：BoolQ 0.772，TriviaQA 0.146
=> custom 62 严重高估了知识增益，不能作为主结论。
```

### 2.2 大规模 SFT（6000 题，eval-600，raw prompt）

训练数据：

```text
BoolQ    2000 题
TriviaQA 2000 题
NQ-open  2000 题
总计     6000 题
```

评估集：`data/qa-standard/eval-600.jsonl`（每任务 200 题）。

| Arm | BoolQ | TriviaQA | NQ |
|---|---:|---:|---:|
| no-reader | 0.645 | 0.145 | 0.055 |
| large mixed50 real | 0.742 | 0.157 | 0.050 |
| large mixed50 control | 0.715 | 0.155 | 0.042 |

结论：

```text
SFT 从 88 -> 6000 没有带来泛化的 TriviaQA 知识增益：
  real 0.157 vs control 0.155 vs no-reader 0.145
  差异基本在噪声范围内。

BoolQ 仍然主要是 format 修复：
  real 0.742 vs control 0.715 vs no-reader 0.645
  control 也有大部分增益。

NQ 仍然接近 floor（0.05）。
```

### 2.3 Chat template ablation（300 held-out，每任务 100 题）

| Arm | BoolQ | TriviaQA | NQ | Overall |
|---|---:|---:|---:|---:|
| raw-no-reader | 0.61 | 0.17 | 0.06 | 0.280 |
| **chat-no-reader** | **0.74** | **0.19** | **0.07** | **0.333** |
| raw-sft-mixed50 | 0.75 | 0.16 | 0.05 | 0.320 |
| chat-sft-mixed50 | 0.70 | 0.18 | 0.07 | 0.317 |
| chat-sft-mixed50-purecode | 0.73 | 0.17 | 0.06 | 0.320 |

关键观察：

```text
1. 只把 no-reader 从 raw prompt 换成 chat template：
   BoolQ +13 分，TriviaQA +2 分，NQ +1 分，overall +5.3 分。
   => 不 apply chat template 确实低估了模型 format/instruction-following 能力。

2. 同一个 raw-trained reader 切到 chat eval：
   BoolQ 0.75 -> 0.70（下降）；
   TriviaQA 0.16 -> 0.18（略升）；
   NQ 0.05 -> 0.07（略升）；
   overall 0.320 -> 0.317（基本持平）。

3. raw reader 相对 no-reader 的优势：
   raw 协议：BoolQ +0.14，TriviaQA -0.01，NQ -0.01；
   chat 协议：BoolQ -0.04，TriviaQA -0.01，NQ 0.00。
   => 在 chat 协议下，当前 reader 相对 no-reader 的优势基本消失。
```

### 2.4 昨晚机制结论（已完成）

**Layer2 corpus-only reader：**

| 指标 | 数值 |
|---|---:|
| BoolQ gate mean | 0.037 |
| TriviaQA gate mean | 0.218 |
| gate=0.0 BoolQ / TriviaQA | 0.760 / 0.480 |
| gate=0.5 BoolQ / TriviaQA | 0.000 / 0.340 |
| gate=1.0 BoolQ / TriviaQA | 0.000 / 0.060 |

结论：

```text
layer2 corpus-only reader 在 BoolQ 上 gate 本来就几乎关闭，
但少量高 gate token 仍然破坏 BoolQ；
强制 gate=0 恢复 no-reader；
gate=0.5/1.0 两个任务都崩。
=> 问题在 reader 内容，不在 gate 开关。
```

**SFT mixed50 reader：**

| 指标 | 数值 |
|---|---:|
| BoolQ gate mean | 0.767 |
| TriviaQA gate mean | 0.979 |
| gate=0.0 BoolQ / TriviaQA | 0.760 / 0.480 |
| gate=0.5 BoolQ / TriviaQA | 0.880 / 0.940 |
| gate=1.0 BoolQ / TriviaQA | 0.920 / 0.900 |

结论：

```text
SFT 之后 reader 内容变好，gate 可以安全打开；
learned gate 已经接近 forced-open 上界。
```

**Oracle routing：**

| 指标 | layer2 | layer8 |
|---|---:|---:|
| always-real | 0.422 | 0.427 |
| always-no-reader | 0.447 | 0.447 |
| oracle(real/no) | **0.575** | 0.566 |
| oracle(all) | 0.597 | 0.607 |

unique-correct（layer2，3 seeds × 50 题）：

| Task | real only | no-reader only | 解释 |
|---|---:|---:|---|
| BoolQ | 14 | 141 | 几乎只需要“关掉 PLE” |
| TriviaQA | 142 | 42 | 需要“打开 PLE” |
| NQ | 17 | 23 | 几乎没有可区分信号 |

**Oracle context：**

```text
gold answer 放进 prompt：extracted_contains 0.700
closed-book no-reader：0.447
=> 模型本身能用知识，瓶颈在 PLE reader/注入。
```

### 2.5 custom 62（旧 SFT，供对比）

| Arm | BoolQ | TriviaQA | NQ | PPL |
|---|---:|---:|---:|---:|
| no-reader | 0.682 | 0.550 | 0.050 | 37.59 |
| sft-mixed50 real | 0.848 | 0.917 | 0.067 | 28.52 |
| sft-mixed50 control | 0.833 | 0.867 | 0.083 | 31.44 |
| sft-mixed25 real | 0.742 | 0.867 | 0.067 | 26.20 |
| sft-mixed75 real | 0.742 | 0.900 | 0.050 | 30.97 |
| sft-only real | 0.742 | 0.817 | 0.100 | 41.40 |
| sft-mixed50-gate01 real | 0.803 | 0.883 | 0.050 | 28.04 |

结论：

```text
custom 62 上 mixed50 很好；
标准 held-out 说明这个“知识增益”大部分没有泛化。
```

---

## 3. 竞争假设与判决性实验设计

### 3.1 竞争假设

```text
H1. Train/eval format mismatch（reader OOD）
    reader 在 raw prompt hidden state 上训练，
    在 chat template hidden state 上推理，query/gate/out_proj 全部偏离训练分布。
    如果 H1 为主，用 chat 格式训练 reader 应该能恢复增益。

H2. raw reader 的增益主要是 format compensation
    raw 协议下 no-reader 不守格式；
    reader 的一部分增益是在“教模型守格式”，不是注入知识。
    chat template 已经把 format 修好，这部分增益自然消失。
    如果 H2 为主，chat-trained reader 仍然不会明显超过 chat-no-reader。

H3. PLE 内容/覆盖/容量瓶颈
    PLE 表里可能没有标准 TriviaQA/NQ 的答案，
    或者 0.8B backbone + 15.2M reader 无法取出知识。
    如果 H3 为主，chat 数据训练也修不好，
    需要 coverage audit / layer sweep / reader capacity / PLE-native upper bound。
```

### 3.2 2×2 factorial design

|  | eval raw | eval chat |
|---|---|---|
| **train raw** | A（已有） | B（已有） |
| **train chat** | D（待补） | C（队列中） |

```text
A = raw-trained reader, raw eval
B = raw-trained reader, chat eval
C = chat-trained reader, chat eval   ← 决定性
D = chat-trained reader, raw eval     ← 可选，测反向 OOD
```

关键比较：

```text
C vs chat-no-reader：
  明显 > => H1 为主；
  约等于 => H2/H3 为主。

C vs chat-control：
  明显 > => PLE 内容有真实贡献；
  约等于 => 增益主要是 format/regularization。

C vs B：
  C > B => chat 格式训练有效，format mismatch 是主因；
  C <= B => chat SFT 设置可能有问题（EOS/mask/special token）。
```

### 3.3 需要控制的变量

```text
1. EOS：
   chat 模式答案以 <|im_end|>（248046）结束，
   不是 <|endoftext|>（248044）。

2. Special token PLE 注入：
   <|im_start|> / <|im_end|> / <think> / </think> 的 PLE row
   可能不在 Qwen3.8 训练分布里；
   需要一个“跳过 special token PLE 注入”的消融。

3. Think block：
   Qwen3.5 非 thinking chat template 仍会插入空
   <think>\n\n</think>\n\n；
   这是 native non-thinking 格式，作为主协议。

4. Prompt length / truncation：
   chat prompt 比 raw prompt 长；
   保证 SFT 和 eval 用同一 `apply_chat_template` 配置。

5. 三臂 + control：
   real / control / no-reader 必须同 seed、同数据、同 steps。
```

---

## 4. 本轮做的尝试

### 4.1 实验

```text
1. layer2 全量诊断（PURE_WIKI / PURE_CODE / FW_STEM × 3 seeds × 3 modes）
2. gate ablation（gate=0 / 0.5 / 1.0）
3. oracle routing upper bound（layer2 / layer8 / 各 SFT config）
4. oracle context（gold answer 放进 prompt）
5. gate 统计（BoolQ vs TriviaQA vs NQ）
6. answer-only QA SFT：sft-only / mixed25 / mixed50 / mixed75 / gate01
7. mixed50 在 PURE_CODE 上的泛化
8. 标准 held-out 1500（BoolQ / TriviaQA / NQ-open）
9. 6000 题大规模 SFT（large mixed50）
10. chat template ablation（raw vs chat，300 题）
11. chat-template SFT 路径 smoke test
12. 两阶段 curriculum（warmup250）实现与排期
```

### 4.2 工程实现

| 文件 | 内容 |
|---|---|
| `scripts/run_phase0.py` | chat template eval/SFT、EOS 处理、lazy e_t cache、warmup、gate override、contribution 诊断 |
| `scripts/run_phase1_matrix.sh` | 透传 chat / lazy / warmup / gate override / gate reg |
| `scripts/build_qa_standard_split.py` | 从 HF mirror 构建 BoolQ/TriviaQA/NQ-open 标准 split |
| `scripts/run_large_sft_queue.sh` | 标准 eval + 6000 题 SFT + 两阶段 + full eval + oracle/report |
| `scripts/run_chat_template_ablation.sh` | raw vs chat 对照（300 题） |
| `scripts/run_chat_sft_queue.sh` | chat-template SFT + chat eval + oracle/report |
| `src/qwen35_ple/eval/answers.py` | 清洗 `<|im_start|>` / `<|im_end|>` / `</think>` |
| `src/qwen35_ple/reader.py` | `last_gate` / `last_gate_raw` / contribution / hidden 诊断；gate override |
| `tests/test_qa_sft.py` | chat prompt / lazy cache 测试 |
| `tests/test_answers.py` | chat control token 清洗测试 |

### 4.3 数据

```text
data/qa-standard/train.jsonl      6000 题（BoolQ 2000 / TriviaQA 2000 / NQ 2000）
data/qa-standard/eval.jsonl       1500 题（每任务 500）
data/qa-standard/eval-600.jsonl    600 题（每任务 200）
data/qa-standard/eval-300.jsonl    300 题（每任务 100）
```

---

## 5. 踩过的坑与教训

### 5.1 AutoDL shutdown 误关机（最严重）

```text
为了测试关机命令，执行了：
  shutdown -h +100000

AutoDL 的 /usr/bin/shutdown wrapper 忽略了时间参数，
直接关闭了实例，导致 large-SFT 队列断电。

恢复：
  - 数据盘 /root/autodl-tmp 和 outputs 完好；
  - /dev/shm/qwen38-rows 丢失，用持久盘重新复制并校验；
  - large-SFT 队列重新启动并 resume；
  - 没有丢失已完成的结果。

教训：
  - 禁止使用容器 shutdown 的时间参数；
  - 定时关机只在 AutoDL 控制台设置；
  - 如果要 server-side watchdog，必须自己 sleep 到 deadline
    再调用 shutdown -h now，且要有 marker/保险。
```

### 5.2 scp / rsync 挂起

```text
本地 scp 到远程时挂起，超过 300s。
解决：
  - 小文件用 ssh 'cat > file' < local；
  - 多文件用 tar czf - files | ssh 'tar xzf -'。
```

### 5.3 `apply_chat_template` 返回类型

```text
transformers 5.x 的 apply_chat_template(tokenize=True)
返回 BatchEncoding/dict，不是 list；
直接 iter 得到 'input_ids' 字符串并报错。
修复：
  if isinstance(encoded, dict): encoded = encoded["input_ids"]
  elif hasattr(encoded, "input_ids"): encoded = encoded.input_ids
```

### 5.4 EOS 混淆

```text
tokenizer.eos_token_id = 248044  (<|endoftext|>)
chat 模式答案 EOS  = 248046  (<|im_end|>)

raw 评估用 248044；
chat 评估/SFT 必须把 248046 加入 stop/EOS，
否则 chat 生成可能不停止，或答案后继续输出。
```

### 5.5 custom 62 高估知识增益

```text
custom 62：TriviaQA 0.550 -> 0.917
标准 1500：TriviaQA 0.128 -> 0.146

教训：
  - 主结论必须用标准 held-out；
  - custom 集只能作为机制诊断；
  - 训练/评估分布太近会严重高估。
```

### 5.6 6000 题 SFT 的 e_t 内存

```text
6000 题 × 512 token × 2560 dim × 4 bytes ≈ 31.5GB
不能全部 materialize。
解决：
  - `_LazyQASFTCache`：按需 fetch PLE rows，LRU 只保留 128 项；
  - 训练速度可接受，内存可控。
```

### 5.7 gate 报告单臂为空

```text
check_phase2_gates.py 需要同一个文件里有 real/control/no-reader；
标准 eval 单臂运行时报告为空。
待修：
  - 支持跨文件拼三臂，或者为单臂输出简单的 per-task metrics。
```

### 5.8 生成循环 O(T²)，thinking eval 受阻

```text
当前 _qa_exact_match 每步重跑全序列并重新 fetch 整段 e_t；
没有 KV cache。
thinking 几百到几千 token 会非常慢。
待做：
  - KV-cache-aware 生成；
  - 只 fetch 新增 token 的 e_t；
  - 单独的 thinking ablation。
```

### 5.9 其他

```text
- 远程 git dubious ownership：用 `git -c safe.directory=...` 或全局 safe.directory。
- nohup 后台进程仍可能让 SSH 命令挂起：用 setsid + 重定向，并且用短命令轮询。
- /dev/shm 重启丢失：持久盘 + ensure_qwen38_rows.sh。
- 并发 GPU：large SFT + chat ablation 同时跑，显存到 ~16GB；
  可行但会互相拖慢，重要训练尽量串行。
```

---

## 6. 已完成内容

### 6.1 结果

```text
- layer2 全量诊断（3 corpora × 3 seeds × 3 modes）
- layer8 全量诊断（已有）
- gate ablation（layer2 / SFT mixed50）
- oracle routing upper bound（layer2 / layer8 / SFT configs）
- oracle context（gold answer）
- gate 统计（layer2 control/real，SFT mixed50/gate01）
- custom 62 SFT 矩阵（sft-only/mixed25/mixed50/mixed75/gate01/PURE_CODE）
- 标准 held-out 1500（no-reader / old SFT readers）
- 6000 题大规模 SFT（large mixed50，eval-600）
- chat template ablation（300 题，5 arms）
- 自动 morning report（OVERNIGHT_REPORT.md）
```

### 6.2 工程

```text
- qa-standard split builder + 数据
- lazy e_t cache
- chat template eval + chat template SFT
- EOS / stop token 修复
- chat control token 清洗
- two-stage warmup
- large-SFT / chat-SFT / chat-ablation / gate-ablation 队列脚本
- resume / partial / backup / manifest 继续可用
```

---

## 7. 未完成内容

### 7.1 正在运行

```text
- large mixed50-warmup250（两阶段）
- large mixed50 full 1500 eval（排队）
- oracle/report 更新（排队）
```

### 7.2 已挂载等待

```text
chat-SFT 队列（等待 LARGE_SFT_DONE）：
  - sft-chat-mixed50（chat 格式 6000 题 SFT，3 seeds）
  - sft-chat-mixed50-warmup250
  - full 1500 chat eval（no-reader + 3 seeds real）
  - oracle + report
  - 完成标记 CHAT_SFT_DONE
```

### 7.3 待做

```text
1. 2×2 factorial 的 D 条件：chat-trained reader 在 raw eval 下
2. chat-trained control 的完整比较
3. special-token PLE 注入消融
4. PLE coverage audit：
   标准 TriviaQA/NQ 的答案/entity 是否在 PLE 表中可检索
5. layer sweep（layer 2 / 4 / 8 / 12 / 16，chat 协议）
6. reader capacity sweep（更大 bridge / 部分解冻 value_proj）
7. PLE-native upper bound（Qwen3.8 native PLE on/off）
8. thinking on/off 作为独立 ablation（需要 KV-cache 生成）
9. check_phase2_gates.py 跨文件三臂支持
10. 更多标准任务、contamination audit、pass@k、CI/GPU smoke
```

---

## 8. 未来计划

### 8.1 立即（本轮队列完成后）

```text
1. 看 CHAT_SFT_DONE 和 OVERNIGHT_REPORT.md；
2. 计算 2×2 表：
   A = raw/raw
   B = raw/chat
   C = chat/chat
   D = chat/raw（待补）
3. 按 H1/H2/H3 判据决定下一步：
   - C 明显 > chat-no-reader 且 real > control => H1 为主，
     切 chat 协议，继续 scaling；
   - C ≈ chat-no-reader 且 real > control 微弱 => H2 为主，
     转向 coverage / layer / capacity；
   - C ≈ control ≈ no-reader => H3 为主，
     优先 PLE coverage audit 和 PLE-native upper bound。
```

### 8.2 中期（Phase A/B）

```text
Phase A（评估与复现加固）：
  - 冻结 canonical eval suite；
  - contamination audit；
  - 统一 result schema + CI + per-seed；
  - experiment registry + runner；
  - 资源 guardrail / 自动 artifact pull。

Phase B（配方/数据/机制锁定）：
  - SFT 10k-100k，多任务/多来源；
  - mixing ratio / warmup / answer-only / replay 小网格；
  - contribution norm / gate / patching / CKA；
  - 同预算 LoRA / RAG / zero-reader baseline；
  - 确定 recipe-v1。
```

### 8.3 长期（Phase C/D）

```text
Phase C：多任务/多语言/长上下文/代码/数学、pass@k、5M/20M rows scaling、
         reader capacity / injection layer scaling、KV cache / CPU 推理。
Phase D：PLE-native upper bound、可选 distillation、paper/artifact release。
```

---

## 9. 当前状态与检查点

```text
完成标记：
  /root/autodl-tmp/qwen35-ple/outputs/SFT_OVERNIGHT_DONE
  /root/autodl-tmp/qwen35-ple/outputs/GATE_ABLATION_DONE
  /root/autodl-tmp/qwen35-ple/outputs/chat-template-ablation/DONE

运行/等待：
  large-SFT queue  -> LARGE_SFT_DONE
  chat-SFT queue   -> CHAT_SFT_DONE

报告：
  /root/autodl-tmp/qwen35-ple/outputs/OVERNIGHT_REPORT.md
```

---

## 10. 一句话总结

> 本轮把评估从 custom 62 推到标准 1500，发现：
> **BoolQ 的 format 修复可以泛化，但知识增益没有泛化；**
> **chat template 显著改变 no-reader 基线，当前 raw-trained reader 的优势大部分消失。**
>
> 下一步的判决性实验是 **chat-template SFT + chat-template eval 的 2×2 factorial**；
> 如果 C 仍然没有超过 chat-no-reader，就必须转向 PLE coverage audit、layer sweep、
> reader capacity 和 PLE-native upper bound，而不是继续堆 SFT 数据。
