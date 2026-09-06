# Round 110：PLE Projector v1 —— 3 seed paired、1k 数据、开放生成安全

> 日期：2026-09-06
> 状态：完成核心实验
> 目标：完成 Phase A 的后续验证：3 seed paired test、扩大局部续写数据、任务条件 one-hot、开放生成不退化验证。

---

## 1. 代码变更

- `PleProjector` 增加任务 one-hot 特征：
  - `task_code`
  - `task_name`
  - `task_number`
  - `task_general`
- 训练脚本 `scripts/train_ple_projector.py`：
  - 支持 `--code-corpus` 加载 `data/code-corpus.jsonl`；
  - 支持 `--dataset` 加载预构建的 projector 数据集；
  - 输出 `paired_eval`：每个 eval token 同时记录 base / fixed / projector 的 NLL 与 hit，用于配对检验；
  - 优化大样本运行：去掉冗余的重复评估。
- 新增：
  - `scripts/analyze_ple_projector_paired.py`：跨 seed 配对统计；
  - `scripts/build_ple_projector_dataset.py`：构建可复用的局部续写数据集；
  - `configs/ngram-fusion-router-projector-open.json`：开放生成压力测试配置；
  - `configs/ngram-fusion-router-projector-policy.json`：带 learned token policy 的开放生成安全配置。
- Router 现在会把任务 one-hot 传给 projector，保证 serving 与训练特征一致。

---

## 2. 3 seed paired 结果（100 samples / 100 steps / task one-hot）

| seed | base NLL | fixed NLL | projector NLL | fixed hit | projector hit | proj - fixed NLL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.617 | 1.977 | 1.758 | 0.590 | 0.650 | **+0.219** |
| 1 | 2.286 | 1.829 | 1.843 | 0.650 | 0.670 | -0.013 |
| 2 | 2.710 | 2.227 | 2.226 | 0.570 | 0.610 | +0.001 |

跨 seed 汇总：

```text
projector vs fixed NLL mean = +0.0688
projector vs fixed positive seeds = 2 / 3
fixed vs base NLL mean = +0.5266
projector vs base NLL mean = +0.5954
```

分任务：

- code：projector 在 seed0 上显著改进，seed1/2 与 fixed 基本持平，hit 三 seed 均不低于 fixed；
- name：projector 三 seed 均略差于 fixed，是当前主要短板；
- number：小样本，信号不稳定。

结论：

> 3 seed 下 projector 总体略优于 fixed，但没有达到统计显著；优势主要来自 code 类。

---

## 3. 1k 数据扩展

- 构建数据集：
  - `data/ple-projector-dataset-1k.jsonl`（1000 条）
  - `data/ple-projector-dataset-10k.jsonl`（10000 条）
- 使用 1k 训练 seed0：
  - `train=1000 / eval=780`
  - `steps=200 / batch=8`

结果：

| 条件 | NLL | hit |
|---|---:|---:|
| base | 1.987 | 0.619 |
| fixed calibration | 1.397 | 0.724 |
| learned projector | **1.189** | **0.782** |

| 任务 | fixed NLL | projector NLL | fixed hit | projector hit |
|---|---:|---:|---:|---:|
| code | 1.228 | **0.986** | 0.756 | **0.820** |
| number | 0.925 | 0.947 | 0.636 | 0.636 |
| name | **4.939** | 5.000 | 0.300 | 0.325 |

结论：

> 数据从 100 扩到 1000 后，projector 相对 fixed 的增益从约 +0.07 提高到 +0.21，主要来自 code 局部续写。  
> name/number 仍不是 projector 的强项，说明当前项目适合“code / 低熵局部记忆”，不应作为通用语义记忆。

---

## 4. 开放生成不退化验证

- 使用 5 个 NL 代码生成 prompt：
  - BM25-only；
  - BM25 + ngram retrieval；
  - BM25 + PLE projector fusion。
- 原始 projector 配置（无 learned token policy）：
  - 出现明显退化：重复代码、无关上下文片段、结构化输出被破坏。
- 增加 learned token policy 后：
  - 大多数 prompt 回到接近 BM25 的行为；
  - 仍有个别长尾代码片段进入生成，说明当前 policy 不是完美保险丝；
  - 但与 raw projector 相比，退化显著降低。

配置产物：

```text
configs/ngram-fusion-router-projector-open.json    # 压力测试（无 policy）
configs/ngram-fusion-router-projector-policy.json # 生产安全配置（带 policy）
```

结论：

> 开放生成不能默认开启 raw PLE projector；必须配 learned token policy / task routing 作为安全门。  
> 当前 policy 能大幅抑制退化，但尚未完全消除，后续应收集生成质量标签训练更强的 gate。

---

## 5. 下一步

1. 用 10k 数据集做 projector 训练（当前已构建 10k dataset，尚未训练）；
2. 为 name 任务增加专项校准或降低 projector 权重；
3. 用生成质量标签（重复率、结构正确性、judge score）训练 policy；
4. 将 policy + projector 加入正式 RAG serving，做端到端 A/B；
5. 进入 M2：1M tokens + 部分解冻/LoRA。

---

## 6. 主要产物

```text
src/qwen35_ple/projector.py
scripts/train_ple_projector.py
scripts/analyze_ple_projector_paired.py
scripts/build_ple_projector_dataset.py
configs/ngram-fusion-router-projector-open.json
configs/ngram-fusion-router-projector-policy.json
outputs/ple-projector-v1-seed{0,1,2}.json
outputs/ple-projector-v1-1k-seed0.json
outputs/ple-projector-paired-analysis.json
```
