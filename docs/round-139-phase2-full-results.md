# Round 139：Phase 2 全量矩阵结果与门禁判定

> 日期：2026-09-09  
> 远程：AutoDL RTX 4090（`connect.nmb1.seetacloud.com:19236`）  
> 状态：全量矩阵已在 03:44 完成；实例已重启，`/dev/shm` 行表已丢失，但结果 JSON 完整保留

---

## 1. 完成性检查

检查项：

- 6 个 corpus 的 `phase1-*.json` 全部存在；
- 每个 corpus 含 `real` / `control` / `no-reader` 三线；
- 每条线 3 个 seed；
- 每个 seed 150 道 QA；
- 日志无 `Traceback`、CUDA OOM 或 `error:`；
- 进程 PID 36525 已退出；
- GPU 当前空闲（0% util, 1 MiB）。

结果：

| Corpus | real | control | no-reader |
|---|---:|---:|---:|
| PURE_WIKI | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |
| PURE_FINEWEB | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |
| PURE_STEM | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |
| PURE_CODE | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |
| FW_CODE | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |
| FW_STEM | 3 seeds / 450 QA | 3 seeds / 450 QA | 3 seeds / 450 QA |

产物已复制到本地：

```text
artifacts/phase2-full-4090/
  phase1-*.json.gz
  phase2-full.log.gz
  summary.json
  summary.md
  sha256.txt
```

---

## 2. PPL 结果

### 2.1 语料级均值

| Corpus | real PPL | control PPL | no-reader PPL | 判定 |
|---|---:|---:|---:|---|
| PURE_WIKI | 23.334 | 24.803 | 37.590 | real < control < no-reader |
| PURE_FINEWEB | 26.540 | 26.576 | 27.369 | real < control < no-reader（差距很小） |
| PURE_STEM | 6.402 | 6.735 | 8.418 | real < control < no-reader |
| PURE_CODE | 2.671 | 5.482 | 10.002 | real < control < no-reader |
| FW_CODE | 12.499 | 12.478 | 13.683 | real 略高于 control，但 real < no-reader |
| FW_STEM | 5.351 | 5.521 | 5.987 | real < control < no-reader |

### 2.2 Seed 级判定

```text
real < control      : 15 / 18
real < no-reader    : 18 / 18
```

结论：

- **real PLE 内容对语言建模有稳定增益**；
- 相对 control 的增益在 5/6 语料上成立，但在 PURE_FINEWEB、FW_CODE 上非常小；
- 相对 no-reader 的增益在所有 18 个 seed 上都成立。

按预注册 PPL 门禁（至少 5/6 语料、2/3 seeds 上 real < control，且 real < no-reader）：

```text
PPL gate: PASS（方向成立，但 PURE_FINEWEB / FW_CODE 需要重复验证）
```

---

## 3. QA 结果与问题

### 3.1 当前协议

```text
QA: 150 题（BoolQ / NQ / TriviaQA）
prompt: 原始问题文本，没有 instruction / answer marker
generation: greedy, 最多 96 new tokens
```

`run_phase0.py` 中 `_qa_exact_match` 的 `correct` 字段实际是：

```text
normalized gold answer 是否作为子串出现在 generated text 中
```

即 **contains**，不是严格 exact match。

### 3.2 汇总

按 `summarize_phase1_matrix.py`：

| Corpus | real contains | control contains | no-reader contains | real ext-exact | control ext-exact | no-reader ext-exact |
|---|---:|---:|---:|---:|---:|---:|
| PURE_WIKI | 0.351 | 0.347 | 0.647 | 0.000 | 0.004 | 0.013 |
| PURE_FINEWEB | 0.631 | 0.633 | 0.647 | 0.051 | 0.073 | 0.013 |
| PURE_STEM | 0.556 | 0.562 | 0.647 | 0.116 | 0.136 | 0.013 |
| PURE_CODE | 0.596 | 0.542 | 0.647 | 0.071 | 0.131 | 0.013 |
| FW_CODE | 0.589 | 0.598 | 0.647 | 0.113 | 0.116 | 0.013 |
| FW_STEM | 0.587 | 0.569 | 0.647 | 0.147 | 0.160 | 0.013 |

表面结论：

```text
contains:      no-reader 0.647 > real/control
extracted_exact: real > control 只在 FW_CODE / FW_STEM 的 BoolQ 上成立
```

### 3.3 为什么当前 QA 指标不可用

抽查生成文本后发现：

1. **`contains` 对 BoolQ 基本无效**
   - no-reader 经常生成 `A: Yes / B: No / Answer: <think>...`；
   - 生成文本同时包含 `yes` 和 `no`，gold 无论是什么都容易被 contains 命中；
   - no-reader 的 BoolQ contains 为 0.96，但多数并没有真正回答问题。

2. **`extract_answer` 的 last-sentence 规则不适用**
   - real/control 常输出 `assistant\nYes, ...` 或 `The answer is ...`；
   - 但当前 extractor 取最后一句话，经常拿到 reasoning 的尾句，而不是答案；
   - 因此 `extracted_exact` 接近 0，不能代表真实准确率。

3. **三线生成格式不同**
   - no-reader：较多原始 continuation、`<think>`、选项列表；
   - real/control：更多 `assistant` / 解释性回答；
   - 直接比较 contains 会把格式差异误判为知识差异。

4. **no-reader 是未训练 zero-shot baseline**
   - real/control 都训练了 reader/bridge；
   - no-reader 不训练，也不注入 PLE；
   - 因此 real/control 相对 no-reader 的差异同时包含：
     - reader 训练带来的格式/行为变化；
     - 真实 PLE 内容带来的信息增益。

5. **没有保存 reader checkpoint**
   - 本轮 matrix 没有传 `--save-reader`；
   - 训练后的 reader/bridge 已丢失；
   - 无法直接用同一组权重重跑“改进 prompt / 改进 scorer”的 QA 消融。

### 3.4 v2 评分（离线重评分，探索性）

新增 `score_answer_v2`：

- 优先解析显式 answer marker（`answer is` / `correct answer is` / `answer:`）；
- 无 marker 时取第一个有意义的句子；
- BoolQ 只接受显式 yes/no assertion；
- 同时出现 `A: Yes` / `B: No` 的选项列表视为 ambiguous，不自动判对。

用 `summarize_phase1_matrix.py --protocol v2` 对已有 generation 重评分：

#### BoolQ（extracted exact）

| Corpus | real | control | no-reader |
|---|---:|---:|---:|
| PURE_WIKI | 0.147 | 0.093 | 0.220 |
| PURE_FINEWEB | 0.373 | 0.487 | 0.220 |
| PURE_STEM | 0.727 | 0.667 | 0.220 |
| PURE_CODE | 0.613 | 0.533 | 0.220 |
| FW_CODE | 0.693 | 0.680 | 0.220 |
| FW_STEM | 0.640 | 0.647 | 0.220 |

#### TriviaQA（extracted contains）

| Corpus | real | control | no-reader |
|---|---:|---:|---:|
| PURE_WIKI | 0.767 | 0.813 | 0.540 |
| PURE_FINEWEB | 0.633 | 0.547 | 0.540 |
| PURE_STEM | 0.173 | 0.347 | 0.540 |
| PURE_CODE | 0.633 | 0.600 | 0.540 |
| FW_CODE | 0.653 | 0.593 | 0.540 |
| FW_STEM | 0.560 | 0.587 | 0.540 |

NQ 三线都接近 0，当前协议无区分度。

v2 观察：

- BoolQ 上 real > no-reader 在 5/6 语料成立；
- TriviaQA 上 real > no-reader 在 5/6 语料成立；
- 但 real vs control 仍然混合：
  - BoolQ：real > control 在 PURE_STEM / PURE_CODE / FW_CODE 成立；
  - TriviaQA：real > control 在 PURE_FINEWEB / PURE_CODE / FW_CODE 成立；
  - PURE_FINEWEB、FW_STEM 的 BoolQ 仍是 control 更好；
- **存在严重回退**：
  - PURE_WIKI BoolQ：real 0.147 < no-reader 0.220；
  - PURE_STEM TriviaQA：real 0.173 < no-reader 0.540；
- v2 是事后规则，尚未人工校验，不能直接当作正式门禁。

探索性结论：

- 任务级信号不是完全不存在；
- 但它目前不够稳定，且与 PPL 方向不完全一致；
- **不能排除“reader/bridge 训练带来的格式变化”是主要收益来源**；
- PURE_WIKI / PURE_STEM 的回退需要单独诊断。

---

## 4. 预注册门禁判定

| 门禁 | 结果 | 说明 |
|---|---|---|
| PPL：real < control，至少 5/6 语料、2/3 seeds | PASS | 15/18 seeds；差距在 PURE_FINEWEB / FW_CODE 很小 |
| PPL：real < no-reader | PASS | 18/18 seeds |
| 任务级：至少一个 QA 任务 real > no-reader | 部分成立，不可作为通过依据 | v2 下 BoolQ / TriviaQA 各 5/6 语料成立，但 PURE_WIKI / PURE_STEM 存在严重回退 |
| 任务级：real >= control | 混合 | v2 下 BoolQ 3/6、TriviaQA 3/6 成立；PURE_FINEWEB / FW_STEM 的 BoolQ 仍是 control 更好 |
| 审计：训练/QA 无污染 | 未完成 | 需补正式污染审计 |
| 机制：增益来自真实 PLE 内容而非 reader 训练 | 未证明 | control 与 real 的任务差异混合；无法排除格式效应 |

**总判定：不能进入 5M / 20M scaling。**

原因不是 PPL 失败，而是：

```text
PPL 增益成立
+ QA 协议本身需要修正
+ 任务级 real vs control 不稳定
+ 存在 PURE_WIKI / PURE_STEM 严重回退
+ 无法排除 reader 训练/格式效应
= 任务级门禁未通过
```

---

## 5. 下一步诊断计划

### P0-1：修复 QA 协议

- 改用 instruction-style prompt，例如：

```text
Question: <question>
Answer:
```

- 对 BoolQ 使用固定格式：

```text
Question: <passage + question>
Answer with one word, Yes or No:
```

- `extract_answer_v2` / `score_answer_v2` 已实现（`src/qwen35_ple/eval/answers.py`）：
  - 优先解析显式 answer marker；
  - 其次解析第一个有意义的句子；
  - BoolQ 只接受显式 yes/no assertion，列出两个选项视为 ambiguous；
  - `summarize_phase1_matrix.py --protocol v2` 已可用于离线重评分；
  - 保留旧 `contains` 作为辅助指标，不作为门禁。
- 仍需人工抽查 v2 的 20–30 条边界样本，确认不会把 reasoning 中的 yes/no 误当答案。
- 下一轮 QA 使用 instruction-style prompt 重新生成，而不是只靠离线 rescore。

### P0-2：保存 reader checkpoint

- `scripts/run_phase1_matrix.sh` 已新增 `--save-reader` / `SAVE_READER=1`；
- 保存路径：

```text
<output-dir>/reader-<corpus>-{mode}-seed{seed}.pt
```

- 下一轮诊断必须开启，避免再次出现“结果在、权重丢”；
- `run_phase0.py --load-reader` 已支持 `{mode}` / `{seed}` 占位符展开；
- matrix runner `--load-reader` 会自动读取
  `<output-dir>/reader-<corpus>-{mode}-seed{seed}.pt`，跳过训练、只跑 QA。

### P0-3：小规模诊断矩阵

先不跑全量，选：

```text
corpora: PURE_WIKI PURE_CODE FW_STEM
modes:   real control no-reader
seeds:   0 1 2
steps:   500
prompt:  instruction-style
```

重点回答：

1. 新 prompt 下 no-reader 是否输出干净答案；
2. real 是否 > control；
3. PURE_WIKI 回退是否仍然存在；
4. QA 指标是否与 PPL 方向一致。

### P0-4：PURE_WIKI 回退诊断

候选变量：

- layer 8 -> layer 2；
- reader 变体：official vs engram vs mlp；
- bridge/out MLP 开关；
- gate 初始化 / scale；
- 训练语料中 `assistant` / chat 格式比例。

### P1：基础设施

- 细粒度 resume（seed/mode 级 partial JSON + reader checkpoint）；
- 自动备份 JSON；
- 远程 bootstrap / manifest / sha256 校验；
- QA KV cache + reader 增量状态。

---

## 6. 当前远程状态

```text
/root/autodl-tmp/qwen35-ple/outputs/phase2-full  -> 6 个 JSON，完整
/dev/shm/qwen38-rows                             -> 已丢失（重启后未重新抽取）
GPU                                              -> 0% util, 1 MiB
```

实例目前空闲。若近期不跑诊断矩阵，可以关机节省费用；需要时重新抽取行表约 30 分钟。

---

## 7. 结论

> 本轮 Phase 2 的最大结论是：**真实 PLE 对 PPL 有稳定增益，但任务级增益尚未被证明，且当前 QA 协议不能作为门禁依据。**
>
> 因此不进入 5M/20M，而是先修 QA 协议、保存 reader checkpoint，并做小规模诊断矩阵，回答“任务增益来自真实 PLE 内容，还是仅仅来自 reader 训练”。
