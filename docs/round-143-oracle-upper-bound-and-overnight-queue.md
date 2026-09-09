# Round 143：Oracle / PLE-native upper bound 与 overnight 自动队列

> 日期：2026-09-10  
> 主题：定义并计算 oracle routing upper bound；把 oracle context、answer-only SFT、gate 诊断串成无人值守队列  
> 状态：layer2 全量诊断运行中；overnight SFT 队列已启动并等待 layer2 完成

---

## 0. TL;DR

```text
1. 已实现 oracle routing upper bound 离线分析脚本。
2. layer8 全量 oracle：oracle(real/no)=0.566 vs no-reader=0.447；
   always-real=0.427。路由 headroom 存在，但主要来自“BoolQ 关掉 PLE”。
3. layer2 PURE_WIKI 3 seeds oracle：oracle(real/no)=0.587 vs no-reader=0.447；
   always-real=0.369。结论与 layer8 一致。
4. 已实现 answer-only QA SFT / gate regularization / oracle context / gate 统计。
5. 已启动 overnight 队列：
   - layer2 全量 corpus-only 诊断（先跑完）
   - held-out 62 QA baseline
   - oracle context upper bound
   - sft-only / mixed75 / mixed50 / mixed25 / mixed50+gate-reg
   - mixed50 在 PURE_CODE 上的泛化
   - gate 统计 + oracle 汇总 + 自动 morning report
```

---

## 1. Oracle upper bound 的定义

### 1.1 Oracle routing / oracle gate

对每个 QA item，固定同一 reader、backbone、PLE 表，只切换 PLE 是否注入：

```text
action a ∈ {real, control, no-reader}
U(a, x) = correctness(a, x) - λ * format_violation(a, x)
oracle(x) = max_a U(a, x)
```

它回答：

```text
在给定 reader/backbone/PLE 表下，完美路由能达到的任务级上界是多少？
```

注意：

- oracle 是 label-seeing 的乐观上界，不能部署；
- 它只说明“路由是否有用”，不说明“学出来的 gate 能否达到”；
- 我们用 held-out 62 题做 learned gate 的训练/评估，oracle 只做诊断。

### 1.2 判断规则

| 观察 | 解释 | 下一步 |
|---|---|---|
| `oracle ≈ no-reader` | PLE 内容对任务无正贡献 | 不要训练 router；先修 reader/PLE |
| `oracle > no-reader`，但 `always-real < no-reader` | 路由是瓶颈 | 训练现有 gate |
| `oracle > always-real` | 路由有额外 headroom | learned gate 值得做 |
| `oracle ≈ always-real` | 内容/reader 是瓶颈 | 改 reader/PLE，而不是 routing |
| `oracle_context >> no-reader` | 模型能用直接给的知识 | PLE 检索/注入是瓶颈 |
| `PLE-native >> oracle` | backbone 与 PLE 表不匹配 | 考虑 PLE-native 或蒸馏 |

### 1.3 当前 oracle 结果

layer8 全量（3 corpora × 3 seeds，150 QA）：

| Policy | Mean |
|---|---:|
| always-real | 0.4267 |
| always-control | 0.4252 |
| always-no-reader | 0.4467 |
| oracle(real/no) | 0.5659 |
| oracle(all) | 0.6074 |

layer2 PURE_WIKI（3 seeds，150 QA）：

| Policy | Mean |
|---|---:|
| always-real | 0.3689 |
| always-control | 0.3556 |
| always-no-reader | 0.4467 |
| oracle(real/no) | 0.5867 |
| oracle(all) | 0.5978 |

layer2 per-task：

| Task | real | control | no-reader | oracle(real/no) | oracle(all) |
|---|---:|---:|---:|---:|---:|
| boolq | 0.2000 | 0.2333 | 0.7600 | 0.7667 | 0.7733 |
| triviaqa | 0.8133 | 0.7533 | 0.4800 | 0.8533 | 0.8800 |
| nq | 0.0933 | 0.0800 | 0.1000 | 0.1400 | 0.1400 |

初步结论：

```text
BoolQ:
  real 0.20 << no-reader 0.76
  oracle ≈ no-reader 0.77
  => PLE 在 BoolQ 上没有正贡献；oracle 的收益主要是“关掉 PLE”。

TriviaQA:
  real 0.81，control 0.75，no-reader 0.48
  oracle(real/no)=0.85，oracle(all)=0.88
  => real 已经接近 oracle；routing headroom 约 2-3 分。

NQ:
  oracle 0.14 vs no-reader 0.10
  => 小 headroom。
```

因此当前最值得做的不是加复杂 router，而是：

```text
1. 让现有 gate 学会在 BoolQ 类输入上关闭/减弱；
2. 保持 TriviaQA 上的 real 优势；
3. 用 answer-only SFT / gate regularization 验证这两点。
```

---

## 2. Oracle context upper bound

对同一批 QA，把 gold answer 直接放进 prompt：

```text
Context: The answer to the question is <gold>.
Question: <question>
Answer:
```

对比：

```text
no-reader closed-book  vs  no-reader oracle-context
```

如果 `oracle-context >> closed-book`，说明模型本身能使用知识，瓶颈在 PLE 注入/检索；如果两者接近，说明瓶颈在 instruction-following / format。

队列会同时跑 62 held-out 和 150 两套。

---

## 3. PLE-native upper bound 的状态

当前远程只有：

```text
models/Qwen3.5-0.8B
models/Qwen3.8-Flash-Next-FP8-tokenizer
models/qwen38_ple  (PLE meta/index)
data/official_ple_reader.pt  (Qwen3.8 官方 PLE reader 权重)
```

没有完整的 Qwen3.8-Flash-Next 权重；24GB 4090 + 27GB 空闲磁盘也不足以放完整模型。

因此本轮能做的近似：

```text
OfficialSourceQwenReader = Qwen3.8 官方 key/value/norm/conv + 可训练 bridge/out_proj
```

它是“官方 PLE reader 语义 + Qwen3.5 backbone”的嫁接版本，但不是 PLE-native。
真正的 PLE-native on/off 仍待更大机器或 API。

---

## 4. Overnight 队列

脚本：

```text
scripts/run_layer2_full_overnight.sh
scripts/run_sft_matrix_overnight.sh
```

队列内容：

| Step | 内容 | 输出 |
|---|---|---|
| 1 | layer2 全量 corpus-only（PURE_WIKI/PURE_CODE/FW_STEM × 3 seeds × 3 modes） | `outputs/phase2-diagnostic-layer2/` |
| 2 | layer2 reader 在 held-out 62 QA 上的 QA-only rerun | `outputs/baseline-layer2-eval62/` |
| 3 | oracle context（62 + 150） | `outputs/oracle-context/` |
| 4 | answer-only SFT：sft-only / mixed75 / mixed50 / mixed25 | `outputs/sft-*/` |
| 5 | mixed50 + gate regularization (0.01) | `outputs/sft-mixed50-gate01/` |
| 6 | mixed50 在 PURE_CODE 上的泛化 | `outputs/sft-mixed50-purecode/` |
| 7 | gate 统计（BoolQ vs TriviaQA） | `outputs/gate-stats/` |
| 8 | oracle 汇总 + morning report | `outputs/oracle-analysis/`、`outputs/OVERNIGHT_REPORT.md` |

SFT 训练配置：

```text
data     : data/phase1/kb-wiki/qa.train.jsonl (88 items)
eval     : data/phase1/kb-wiki/qa.eval.jsonl (62 held-out items)
loss     : answer-only (prompt tokens masked)
steps    : 500
seeds    : 0 1 2
modes    : real / control / no-reader
reader   : OfficialSourceQwenReader + bridge/out MLP
layer    : 2
```

关键新增 flag：

```text
--qa-sft-file
--qa-sft-weight         0=corpus-only, 1=QA-only, 0.5=even mix
--qa-sft-max-len        默认 512，prompt 左截断
--qa-sft-full-loss      默认关闭（answer-only）
--gate-reg-weight       对 mean gate 加惩罚，推 gate 关闭
```

---

## 5. Morning checklist

```bash
# 1. 看队列是否完成
cat /root/autodl-tmp/qwen35-ple/outputs/SFT_OVERNIGHT_DONE

# 2. 看自动报告
cat /root/autodl-tmp/qwen35-ple/outputs/OVERNIGHT_REPORT.md

# 3. 看每个 config 的 gate
cat /root/autodl-tmp/qwen35-ple/outputs/gate-stats/*.md

# 4. 看 oracle 汇总
ls /root/autodl-tmp/qwen35-ple/outputs/oracle-analysis/

# 5. 拉回本地
bash scripts/pull_remote_results.sh
```

判定标准：

```text
A. 如果某个 SFT config 的 BoolQ real 明显高于 layer2 real（0.20），
   且 TriviaQA real 保持 >= 0.75：
   => answer-only SFT 有效，继续做两阶段 curriculum。

B. 如果 gate-reg config 的 BoolQ gate open-frac 明显下降，
   且 TriviaQA 不降：
   => gate regularization 是正确杠杆，扩大权重/步数。

C. 如果 oracle(real/no) 仍远高于所有 learned policy：
   => learned gate 还差得远，需要 conflict/sufficiency 数据或更强的 gate 训练。

D. 如果 oracle context 也很低：
   => 瓶颈在模型使用知识/format，而不是 PLE 注入。
```

---

## 6. 当前代码与 commit

```text
scripts/analyze_oracle_upper_bound.py    oracle 离线分析
scripts/analyze_reader_gate.py           gate 统计
scripts/make_oracle_context_qa.py        oracle context QA 构造
scripts/build_overnight_report.py        morning report
scripts/run_layer2_full_overnight.sh     layer2 全量
scripts/run_sft_matrix_overnight.sh      overnight 队列
tests/test_oracle_upper_bound.py
tests/test_qa_sft.py
tests/test_build_overnight_report.py
```

`src/qwen35_ple/reader.py`：

```text
EngramReader / OfficialSourceQwenReader 现在暴露
last_gate / last_gate_raw，用于诊断和 gate regularization。
```

`scripts/run_phase0.py`：

```text
新增 --qa-sft-*、--gate-reg-weight；
--qa-file 现在同时支持 JSON list 和 JSONL。
```
