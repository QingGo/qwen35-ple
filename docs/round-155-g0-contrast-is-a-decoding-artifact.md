# Round 155: G0 的"正结果"是解码假象 —— 拆解、工具与运维复盘

> 前置：`docs/round-152-lora-row-and-g0-4b.md` §2（G0 设计与原始数字）、
> `docs/round-153-goal-tech-debt-and-development-plan.md` §4.3（两条纪律）。
>
> 本文回答两件事：**机器为什么没关机**，以及**G0 那个 10 点提升到底是什么**。

---

## 0. 摘要

1. G0（frozen 4B + frozen PLE）在 `real` vs `control` 上给出本项目**第一个显著的"正"结果**：
   生成 EM +10.03±0.98 点，gold NLL +0.38±0.05 nat。
2. **该结果不成立。** 逐条拆解后，它是 control 臂的解码崩塌（TriviaQA 上 52.8% 的输出是空串）
   造成的假象，不是 PLE 内容效应。
3. 拆解分三步，每一步都排除一层解释：
   - **权重范数审计** → 排除"control 注入量更大"（两臂 adapter 范数比值 0.988–1.014×）；
   - **逐任务分解** → 整体 NLL 优势 100% 来自 BoolQ，另两个任务 control 反而更好；
   - **生成退化统计** → control 在 TriviaQA 上 264/500 输出空串、BoolQ 上系统性偏向 `Yes`。
4. 关键反证：在 control 吐空串的那 264 条上，**control 的 teacher-forced gold NLL（2.95）反而低于 real（3.10）**。
   即扰动小到不影响似然、却足以把贪心解码推入立即 EOS —— 这是 exposure bias，不是知识差异。
5. 机器没关机的原因是一个**代码拼接 bug + finisher 缺少存活检测**，已修复并入库。

---

## 1. 机器为什么一直开着（运维复盘）

G0 的两个实验臂 **08:55:27 就全部成功完成**，产物齐全。之后：

```text
=== [g0] finished g0-control 08:55:27 ===
wrote outputs/round152g0/arms-summary.md
wrote outputs/round152g0/arms-summary.json
baseline 'frozen-0.8b-no-reader' is not one of the runs
=== [g0] ERROR: missing artifact gold-nll-raw.md 08:55:27 ===
```

三连环失效：

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | `run_round152_g0_4b.sh` 把 baseline 追加进了 **死变量 `ARGS`**，而非真正传给 `summarize_gold_nll.py` 的 `GOLD_ARGS` | 摘要脚本 `SystemExit("baseline ... is not one of the runs")`，`gold-nll-raw.md` 永不生成 |
| 2 | 该脚本的产物校验把 `gold-nll-raw.md` 列为必需，缺失即 `exit 1` | **成功的实验被判失败**，`DONE` 标记永不写入 |
| 3 | finisher 只在 `sleep 60` 循环里等 `DONE`，**没有任何"生产者是否还活着"的检查**（预算 12 小时） | 在死掉的生产者旁边空转 ~3.5 小时，实例持续计费 |

修复（均已入库）：

* `scripts/run_round152_g0_4b.sh`：baseline 直接并入 `GOLD_ARGS`；`run_arm` 支持追加额外 flag；
  产物校验与实际生成的文件对齐。
* `scripts/round152_finish.sh`：**从远端脚本收进版本库**，并新增三处：
  * `producer_alive()` —— 生产者进程消失即停止等待（等待不能改变已经发生的事）；
  * 等待预算 12h → 90min；
  * `SHUTDOWN_ON_FAILURE`（默认 1）—— 产物不全且生产者已死时**仍然关机**并留下
    `outputs/chain/FINISH_BLOCKED`，而不是"拒绝关机"让机器整夜空转。
    原 v2 的"拒绝关机"在无人值守场景下等于持续计费。

> 教训（已并入纪律）：**给一个自动化流程加"拒绝继续"的保护，必须同时回答"那谁来推进/谁付费"。**
> 无人值守脚本的正确失败行为是"记录并停机"，不是"等待一个人类"。

---

## 2. G0 表面结果

4B backbone（hidden 2560 = PLE 源空间），table 冻结，仅训练 reader，raw prompt，standard 1500。

| Run | BoolQ EM | TriviaQA EM | NQ EM | Mean EM | Gold NLL |
|---|---:|---:|---:|---:|---:|
| g0-real | 0.874 | 0.390 | 0.112 | **0.4587** | 4.1624 |
| g0-control | 0.814 | 0.178 | 0.084 | **0.3587** | 4.5569 |

配对（n=1486/1500）：EM **+0.1003 ± 0.0098**（t≈10.2）；NLL **+0.3835 ± 0.0530**（t≈7.2）。

按既定预注册判据（≥2 点 或 ≥0.05 nat），G0 **表面上显著通过**。这正是它需要被拆解的原因。

---

## 3. 拆解第一步：权重范数审计（排除"注入量"混淆）

新工具 `scripts/audit_reader_checkpoints.py`（对应 round-153 纪律 #1）：

```bash
python scripts/audit_reader_checkpoints.py \
  --reference real=outputs/round152g0/reader-4b-real-seed0.pt \
  --arm control=outputs/round152g0/reader-4b-control-seed0.pt \
  --init data/official_ple_reader.pt
```

它检查三件事：冻结源张量是否跨臂**逐位一致**、可训练 adapter 是否**真的动了**、以及
**跨臂范数比**（某臂范数远大于另一臂时，对比就被注入规模混淆）。

4B 结果：

* 6 个冻结源张量（`key_proj`/`value_proj`/`norm_*`/`conv1d`）跨臂 **Δ=0**，`freeze_source` 未泄漏，两臂可比；
* adapter 范数比值 **0.988–1.014×**，注入规模匹配。

**结论：排除"control 靠更大注入量污染残差流"。** 那个 10 点差距只能来自注入的**内容**。

顺带对 0.8B LoRA 行的 reader 跑了同一审计：real 的 `out_proj.2` 范数 **1.315**，
control **1.031**（比值 **1.28×**，real 反而更大），而该行指标上**零内容效应**。
**注入幅度与指标效应之间没有单调关系**，进一步支持"幅度不是解释变量"。

---

## 4. 拆解第二步：逐任务分解（整体 NLL 优势只来自一个任务）

Δ = real − control，**负数 = real 更好**：

| task | n | meanΔ | medianΔ | real 更好 | control 更好 | ΣΔ |
|---|---:|---:|---:|---:|---:|---:|
| boolq | 500 | **−2.0443** | −1.6417 | 402 | 98 | −1022.1 |
| triviaqa | 500 | **+0.2943** | +0.2721 | 177 | 323 | +147.2 |
| nq | 500 | **+0.5665** | +0.4150 | 118 | 382 | +283.3 |
| **ALL** | 1500 | −0.3945 | **+0.0801** | 697 | 803 | −591.7 |

**整体 NLL 的"real 更好"完全由 BoolQ 撑起**；在 TriviaQA 和 NQ 上，control 的均值和中位数都更低。
全体中位数 `+0.0801` 已经指向"其实是平局"。

生成 EM 的不对称性（McNemar 风格）：

| task | EM real | EM control | real-only | control-only | 两者都对 | 两者都错 |
|---|---:|---:|---:|---:|---:|---:|
| boolq | 0.874 | 0.814 | 42 | 12 | 395 | 51 |
| triviaqa | 0.390 | 0.178 | **122** | **16** | 73 | 289 |
| nq | 0.112 | 0.084 | 25 | 11 | 31 | 433 |
| ALL | 0.4587 | 0.3587 | 189 | 39 | 499 | 773 |

TriviaQA 上 122:16 的不对称（7.6×）配上一个**更差**的 NLL，方向自相矛盾 —— 这是继续挖的信号。

---

## 5. 拆解第三步：control 臂的解码崩塌（真正的解释）

对生成文本做退化统计：

| 指标 | g0-real | g0-control |
|---|---:|---:|
| TriviaQA 输出**空串** | 2/500 | **264/500 (52.8%)** |
| TriviaQA distinct 答案数 | 476 | 224 |
| NQ 空串 | 0 | 29 |
| NQ 混入 BoolQ 式 `yes` | 0 | 11 |
| BoolQ gold=`no` 时误答 `Yes` | 22.9% | **39.8%** |
| BoolQ 预测 `Yes` 总数（gold=304） | 329 | 367 |

样本（real 对、control 错的 BoolQ 条目）：gold 全是 `no`，real 输出 ` no`，control 一律输出 ` Yes`。

**最关键的 dissociation**：在 control 吐空串的那 264 条上 ——

| | control | real |
|---|---:|---:|
| teacher-forced gold NLL | **2.9548** | 3.1033 |
| 生成 EM | **0/264** | 101/264 |

即 **control 在教师强制下给 gold 的似然更高，却在自由生成时立即 EOS**。
扰动小到不影响似然、大到能把贪心解码推出分布 —— 这是 **exposure bias / 解码崩塌**，
不是知识或内容差异。乱序行注入的是一种"分布外"记忆，real 行则让模型留在分布内。

---

## 6. 结论与影响

1. **G0 不构成"冻结跨模型 PLE 迁移携带内容"的证据。** 那个 +10 点 EM 是
   "control 崩了 / real 没崩"，而不是"real 带来了知识"。
2. **但 graft 不是 no-op。** 它能被乱序行推到整段生成崩塌，说明 reader/gate 通路对残差流
   有**强因果影响**；问题在于这个影响是**破坏性**的而非**信息性**的。
   这与 round-152 的结论（LoRA 共适应不再崩塌、但内容效应仍为零）方向一致。
3. **方法论修正（重要）**：当某一臂偏离分布时，**生成 EM 不是有效的内容探针**。
   它测的是解码稳定性。此后报告内容效应必须：
   * 同时给出 teacher-forced NLL 与生成 EM，**两者矛盾时以 NLL 为准并解释分歧**；
   * 报告**解码退化统计**（空串率、distinct 数、标签分布偏移），作为 arm 有效性的前提断言。
4. **对照臂设计修正**：`control`（乱序行）不是 `no-PLE` 的替代品。乱序行是一个**主动扰动**，
   它回答"内容是否匹配"，不回答"有 PLE 是否比没有好"。后者的唯一正确对照是 `--ple-off`。
   G0 脚本当时恰好删掉了这一臂，导致一个假阳性无法被内部证伪 —— 已补（见 §7）。

---

## 7. 状态：判决性对照仍在跑

`g0-nople`（同一 real reader + `--ple-off`，注入恒为零）是唯一能定性上述解释的臂，正在运行。
在展开之前先记录**预注册预测**（避免事后编故事）：

* 若 §5 的解释正确，则 g0-nople **不会出现空串崩塌**，其生成 EM 应≈ g0-real，
  且 gold NLL 应 ≤ g0-real（即 real PLE 不带来内容增益）；
* 若 g0-real 真的携带内容，则 g0-nople 应显著差于 g0-real。

无论哪个方向，都同时记录 teacher-forced NLL、生成 EM 与退化统计三项。

---

## 8. 本轮新增的工具与坑

**工具**

* `scripts/audit_reader_checkpoints.py` —— 无效臂审计（冻结一致性 / adapter 是否移动 / 跨臂范数比），
  已入 CI lint 列表。
* `scripts/round152_finish.sh` —— finisher 入库 + 存活检测 + 失败即关机。

**坑（新）**

* **`data/qa-standard/eval.jsonl` 是按任务顺序排列的**（500 boolq → 500 triviaqa → 500 nq）。
  因此 `--qa-max-items N`（前缀截断）在 N≤500 时**只覆盖 BoolQ**，不能用来做廉价的
  "多任务子集"。要子集必须做分层抽样。
* 日志里的 `QA {idx}/1500` 是**原始 item 下标**，而批次按 prompt 长度排序，
  所以进度数字**乱序且不可用于估算进度**（曾据此误判"跑完一半"）。
  真实进度只能数已打印的 QA 行数。
* 共享宿主机上 GPU 利用率会长期停在 ~31%、功耗 ~88W（CPU 侧瓶颈 + 邻居负载），
  同一 job 的吞吐可在 0.22–1.4 item/s 之间波动 6 倍。**不要用一次实测推算 ETA。**
