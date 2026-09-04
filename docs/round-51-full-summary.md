# Round 51：本轮完整总结

> 日期：2026-09-04
> 范围：Phase A 实验、原论文调研、低资源方法研究、系统性复盘
> 状态：已完成当前阶段总结，下一阶段进入 P1 记忆接口原型

---

## 1. 本轮计划

1. 完成 Phase A：
   - 构建 rare-token 知识评测集；
   - 测量任务级 \(\Delta I(Y;E|H)\) / \(\Delta R^2\)；
   - 判断真实 PLE 是否在 rare 知识任务上 real > control。
2. 调研原始 Engram/PLE 论文：
   - 原论文到底在哪些下游任务上有收益；
   - 我们冻结嫁接的最好情况是否受限于这些任务。
3. 调研低资源复现路线：
   - GaLore / ReLoRA+sMuon / MoRA；
   - OPD / OPSD / Purified OPSD；
   - Memory Grafting / TokenMem / MLP Memory / MemSFT / PERK。
4. 形成系统性复盘和后续开发计划。

---

## 2. 核心发现

### 2.1 Phase A：当前 PLE 嫁接无下游收益

| 指标 | 结果 |
|---|---|
| 纯特征 \(\Delta R^2\) rare item | +0.000838 ± 0.000249 |
| 纯特征 \(\Delta R^2\) control | −0.000435 ± 0.000127 |
| 纯特征 rare token | +0.000294 ± 0.000106 |
| simple reader rare logprob real−control | ≈ −0.017 |
| MLP reader rare logprob real−control | ≈ +0.001 |
| 10 条生成式 EM | real/control/no-reader 全部 0.2 |

结论：

- PLE 有极弱因果信息；
- 当前 reader / backbone 没有把它转化为任务收益；
- loss 下降不代表智能提升。

### 2.2 原论文收益的机制

- DeepSeek Engram：
  - 知识、推理、代码、数学、长上下文都有增益；
  - 收益来自早期层静态重建卸载、注意力容量释放、有效深度增加。
- Qwen3.8-Flash-Next N-gram ablation：
  - 知识/中文任务收益较好；
  - 代码收益较小；
  - loss 与 downstream accuracy 不同步。

因此：

> 冻结 PLE + 小 reader 不能自动复现这些收益。

### 2.3 Qwen PLE 只有 2/3-gram

- `ngram_size = 3`；
- `heads_per_ngram = 8`；
- 总 head = 16；
- 无原生 4-gram。

如果需要 4-gram，必须自己构建外部 exact 4-gram bank。

### 2.4 低资源技术选型

- **MoRA**：同参数量更高秩，最适合 memory / continual pretraining；
- **GaLore**：全参数更新但只存低秩优化器状态；
- **ReLoRA + sMuon**：多轮低秩累积，极小资源；
- **OPD**：学生轨迹 + teacher 逐 token，可用于迁移 Qwen3.8 能力；
- **OPSD**：对共享规则有效，对长 CoT 推理需谨慎；
- **Purified OPSD**：通过 PMI 去除 reference shortcut，更适合推理。

---

## 3. 做的尝试

### 3.1 Phase A 实验

1. 构建 rare-kb v1：270 条 QA，rare 182 / common 88；
2. 纯特征任务级 \(\Delta R^2\) 探针；
3. 3 seeds 重复；
4. 5 条件 logit patch（no-reader / real / control / random / zero）；
5. rare/common/source 分层分析；
6. MLP(H,E⊥) reader 在 qa-expanded rare 上复测；
7. 10 条生成式 exact-match 冒烟。

### 3.2 调研

1. DeepSeek Engram 原始论文；
2. Qwen3.8-Flash-Next 技术报告；
3. Memory Grafting / XMemTransfer；
4. MLP Memory / MemSFT；
5. TokenMem；
6. PERK；
7. GaLore / MoRA / ReLoRA / sMuon / Riemannion；
8. OPD / OPSD / Purified OPSD / OmniOPSD。

### 3.3 代码与文档

1. 新增 `scripts/build_rare_kb.py`；
2. 新增 `scripts/mechanism_rare_task_r2.py`；
3. 新增 `scripts/analyze_rare_kb_logit.py`；
4. 修改 `scripts/mechanism_logit_patch.py`：
   - 支持 `--device cuda`；
   - `--limit` 默认改为全部。
5. CI 纳入新脚本；
6. 新增 `docs/round-49-phase-a.md`；
7. 新增 `docs/round-50-systematic-plan.md`；
8. 更新 README、roadmap、session-log。

---

## 4. 踩过的坑

| # | 坑 | 解决/状态 |
|---|---|---|
| 1 | Remote WSL 不能直接 scp 到 `/home/zeng/...` | 先 scp 到 `C:/Users/minam/`，再在 WSL 内 cp |
| 2 | `reader-mlp-residual-fixed.pt` 与当前 MLPValueReader 维度不匹配 | 改用 `reader-mlp-residual-concat.pt` |
| 3 | `mechanism_logit_patch.py` 默认 `--limit 12`，导致只跑 12 条 | 默认改为 `None` |
| 4 | ruff 报 TRY004 / RUF046 | 改 `TypeError`，去掉多余 `int()` |
| 5 | 生成式 exact-match 成本高 | 只用 10 条冒烟，主力改 logit patch |
| 6 | 远程后台任务/复杂命令容易挂起 | 用 setsid + nohup + 独立脚本 |
| 7 | Git 提交含复杂 Unicode 时工具异常 | 使用 ASCII commit message 规避 |

---

## 5. 完成的内容

### 5.1 代码

- `scripts/build_rare_kb.py`
- `scripts/mechanism_rare_task_r2.py`
- `scripts/analyze_rare_kb_logit.py`
- `scripts/mechanism_logit_patch.py` 增强

### 5.2 文档

- `docs/round-49-phase-a.md`
- `docs/round-50-systematic-plan.md`
- `docs/roadmap.md`
- `docs/session-log.md`
- `README.md`

### 5.3 已提交

```text
c61b365 docs(phase-a): rare-token benchmark and Phase A gate results
37f6f1d docs(round50): systematic review, low-resource memory interface plan and borrowing matrix
```

### 5.4 实验结果产物（WSL）

```text
outputs/mechanism-rare-task-r2.json
outputs/mechanism-rare-task-r2-3seed.json
outputs/mechanism-rare-logit-patch-full.json
outputs/mechanism-rare-mlp-concat-logit.json
data/rare-kb-v1.json
data/rare-kb-v1-items.json
```

---

## 6. 未完成的内容

| # | 未完成 |
|---|---|
| 1 | exact longest-match PLE bank |
| 2 | TokenMem 式独立 cross-attention 记忆通道 |
| 3 | MLP Memory 式 distribution-level memory + router fusion |
| 4 | MoRA / GaLore / ReLoRA 实验 |
| 5 | OPD / Purified OPSD 蒸馏流程 |
| 6 | 推理 / 代码 / 长上下文 real-vs-control 评测 |
| 7 | RAG / 教师蒸馏同口径 baseline |
| 8 | 4-gram 外部记忆 bank |
| 9 | 完整 3-seed 任务级显著性 |
| 10 | CPU 100 tok/s serving 闭环 |
| 11 | 大规模 5M–20M token 训练 |

---

## 7. 未来计划

### Phase P1：真正可用的 PLE 记忆接口

- 冻结 backbone；
- exact longest-match PLE bank；
- TokenMem 式 cross-attention；
- MLP Memory 式 distribution memory + router。

门禁：

```text
rare knowledge: real > control
通用能力不退化
```

### Phase P2：低资源 backbone adaptation

- MoRA 首选；
- GaLore 对照；
- ReLoRA + sMuon 作为极小资源方案。

数据：

- rare knowledge 1M–5M；
- GSM8K/MATH 1M–5M；
- HumanEval/MBPP 1M–5M；
- synthetic long-context 0.5M–2M。

门禁：

```text
rare / math / code / long-context: real > control
```

### Phase P3：OPD / Purified OPSD

- 先离线 teacher 数据蒸馏；
- 再 OPD；
- OPSD 只用于共享规则；
- 推理用 Purified OPSD。

### Phase P4：对照与决策

- PLE + MoRA；
- PLE + MoRA + OPD；
- RAG baseline；
- 纯蒸馏 student。

如果 PLE 始终无法 real > control：

```text
转 RAG / 蒸馏 / 更语义化记忆
```

### Phase E：产品化

- Store-P / access-order；
- vLLM / SGLang / CompileForge；
- CPU 100 tok/s；
- bundle e2e。

---

## 8. 借鉴矩阵

| 来源 | 借什么 | 不冲突原因 |
|---|---|---|
| Memory Grafting / XMemTransfer | exact longest-match、frozen memory、projection+gate+fallback | 只做 PLE 读取接口 |
| DeepSeek Engram / Qwen PLE | 早期层注入、contextual gating、prefetch | 不重训大表 |
| MLP Memory / MemSFT | distribution memory、router fusion、防遗忘 | 不替代 PLE 表 |
| TokenMem | 独立 cross-attention、conflict-aware gate | 不改 backbone self-attention |
| PERK | test-time LoRA 编码长上下文 | 补长上下文短板 |
| MoRA | 高秩更新 | 只用于 memory/continual pretraining |
| GaLore | 梯度低秩投影 | 作为 MoRA 替代 |
| ReLoRA + sMuon | 多轮低秩累积 | 极小资源方案 |
| OPD / Purified OPSD | 学生轨迹 + teacher、PMI 去 shortcut | 作为 post-training |
| RAG / ReAugKD | 同口径对照 | 判断 PLE 是否真的更好 |
| Hierarchical Memory | 长尾知识与 anchor 推理分离 | 支持小 anchor + external memory |
| MemLoRA | 任务专用 LoRA expert | 分阶段适配 |
