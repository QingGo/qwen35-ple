# Round 142：本轮 session 复盘、发现与后续计划

> 日期：2026-09-09  
> 主题：Phase 2 结果收尾、4090 重启与 86GB 数据盘、qwen38-rows 持久化、layer8/layer2 诊断、SFT/RL/数据混合/gate 的调研与方案收敛  
> 状态：layer8 全量诊断完成；layer2 冒烟完成；GPU 空闲；下一步做判决性诊断

---

## 0. TL;DR

本轮最重要的结论：

```text
1. Phase 2 全量矩阵已完成，真实 PLE 对 PPL 有稳定增益，
   但旧 QA 协议和 layer8 注入让任务级结论不可用。

2. 官方 PLE 注入层是 layer 2（Qwen3.8 config: ple_layer_ids=[2]），
   之前 Phase 2 使用的 layer 8 是错的。

3. layer2 冒烟（PURE_WIKI seed0）显示：
   - TriviaQA: real 0.860 > control 0.700 > no-reader 0.480
     => PLE 对知识型 QA 有真实增益
   - BoolQ:    real 0.240 < control 0.280 << no-reader 0.760
     => reader 注入仍然破坏格式/instruction-following

4. 当前 reader 训练是 corpus full next-token，不是 answer-only SFT；
   没有 prompt mask，没有 QA/format/replay 数据，没有 conflict/sufficiency 信号。

5. 问题更像 functional interference / representation drift，
   而不是参数级遗忘：backbone 冻结，原能力仍在，只是被 reader 注入压住。

6. 下一步不是加 router、加很多 loss、上 RL，
   而是先用 oracle / PLE-native upper bound 判断边界，
   再用 mixed SFT + 两阶段 curriculum 训练现有 gate。
```

---

## 1. 本轮目标与计划

### 1.1 原始计划

```text
1. Phase 2 全量矩阵收尾；
2. 4090 重启后扩容数据盘；
3. 持久化 qwen38-rows；
4. 跑小规模诊断矩阵；
5. 判断 pure PLE 嫁接是否值得 scaling。
```

### 1.2 实际执行

```text
1. Phase 2 全量矩阵已完成并归档；
2. 4090 重启，数据盘扩到 86GB；
3. qwen38-rows 已抽取到持久盘，并复制到 /dev/shm；
4. layer8 全量诊断完成（PURE_WIKI / PURE_CODE / FW_STEM）；
5. layer2 冒烟完成（PURE_WIKI seed0）；
6. 调研并收敛了 SFT/RL、数据混合、两阶段训练、gate/路由、loss 设计；
7. 新增 rows/manifest/bootstrap/resume/gate/tokenizer 等工程能力；
8. CI 绿色，关键 commit 已 push。
```

---

## 2. 完成的内容

### 2.1 工程与基础设施

| 内容 | 状态 |
|---|---|
| AutoDL 数据盘扩容到 86GB | ✅ |
| qwen38-rows 持久化到 `/root/autodl-tmp/qwen35-ple/qwen38-rows` | ✅ |
| qwen38-rows 复制到 `/dev/shm/qwen38-rows` | ✅ |
| 128 shard 校验 + manifest | ✅ |
| `remote-manifest.json` | ✅ |
| 一键 bootstrap | ✅ |
| 细粒度 resume | ✅ |
| 自动 backup | ✅ |
| QA v2 评分协议 | ✅ |
| tokenizer 对齐审计 | ✅ |
| 门禁检查脚本 | ✅ |
| CI 绿色 | ✅ |

### 2.2 实验

| 实验 | 状态 |
|---|---|
| Phase 2 全量矩阵（6 corpus × 3 seed × real/control/no-reader） | ✅ |
| layer8 诊断矩阵（PURE_WIKI / PURE_CODE / FW_STEM） | ✅ |
| layer2 冒烟（PURE_WIKI seed0） | ✅ |
| tokenizer 对齐审计 | ✅ |
| gate / routing / SFT / RL 文献调研 | ✅ |
| oracle upper bound | ❌ 未做 |
| PLE-native upper bound | ❌ 未做 |
| zero-reader / training-only 对照 | ❌ 未做 |
| mixed SFT + 两阶段 curriculum | ❌ 未做 |
| pass@k / 多采样评估 | ❌ 未做 |
| 5M / 20M scaling | ❌ 未做 |
| unseen-KB 多 split | ❌ 未做 |

### 2.3 新增/修改的脚本

| 脚本 | 作用 |
|---|---|
| `scripts/verify_qwen38_rows.py` | 校验 rows 128 shard、大小、sha256 manifest |
| `scripts/ensure_qwen38_rows.sh` | 优先 `/dev/shm`，其次持久盘，最后 ModelScope 抽取 |
| `scripts/remote_manifest.py` | 生成远程环境 manifest |
| `scripts/bootstrap_remote.sh` | 重启后一键检查/恢复/校验 |
| `scripts/pull_remote_results.sh` | 远程结果 rsync 回本地 |
| `scripts/run_phase2_diagnostic.sh` | 小规模诊断矩阵 wrapper |
| `scripts/check_phase2_gates.py` | PPL + 任务级门禁检查 |
| `scripts/compare_qwen_tokenizers.py` | Qwen3.5 vs Qwen3.8 tokenizer 审计 |
| `src/qwen35_ple/eval/answers.py` | v2 answer extraction / scoring |
| `src/qwen35_ple/eval/prompting.py` | instruction-style QA prompt |
| `src/qwen35_ple/eval/resume.py` | partial result resume/merge |
| `scripts/run_phase0.py` | prompt template、resume、backup、layer 支持 |
| `scripts/run_phase1_matrix.sh` | layer、save/load reader、prompt、resume、backup |
| `scripts/summarize_phase1_matrix.py` | v1/v2 protocol 汇总 |
| `scripts/evaluate_generated_answers.py` | v1/v2 protocol 重评分 |

### 2.4 文档与 artifact

```text
docs/round-139-phase2-full-results.md
docs/round-140-qwen35-qwen38-tokenizer-audit.md
docs/round-141-local-prep-and-reboot-checklist.md
artifacts/phase2-full-4090/
artifacts/tokenizer-audit/
artifacts/phase2-diagnostic-layer8/
artifacts/phase2-layer2-smoke/
```

### 2.5 关键 commit

```text
d945e36 phase2: record full matrix results and add v2 answer protocol
5b78e44 audit: align Qwen3.5 and Qwen3.8-Flash-Next tokenizers
0aa2a3e local-prep: rows bootstrap, resume, QA prompt, and gate checker
b79fc00 bootstrap: report engramdb-python package version
fff97da verify rows: accept metadata-only manifests
5ea78d4 gates: use task-specific v2 metrics
fc93572 diagnostic: expose PLE layer; default to official layer 2
c5d31ec docs: note PLE layer 2 and diagnostic output dirs
```

---

## 3. 关键发现

### 3.1 Phase 2 / layer8 结果

layer8 全量诊断：

| Corpus | real PPL | control PPL | no-reader PPL | real ext-exact | control ext-exact | no-reader ext-exact |
|---|---:|---:|---:|---:|---:|---:|
| PURE_WIKI | 23.442 | 24.659 | 37.590 | 0.131 | 0.062 | 0.287 |
| PURE_CODE | 2.662 | 5.571 | 10.002 | 0.302 | 0.309 | 0.287 |
| FW_STEM | 5.360 | 5.513 | 5.987 | 0.231 | 0.211 | 0.287 |

门禁：

```text
PPL gate: PASS
Task gate: FAIL
Overall: FAIL
```

主要问题：

- BoolQ 在 PURE_WIKI / FW_STEM 上严重低于 no-reader；
- PURE_CODE TriviaQA 低于 control；
- real/control 都出现格式回退，说明不只是 PLE 内容问题。

### 3.2 layer2 冒烟结果

```text
PURE_WIKI seed0, layer2, instruction prompt, 500 steps
```

| 指标 | real | control | no-reader |
|---|---:|---:|---:|
| PPL | 22.924 | 24.810 | 37.590 |
| BoolQ ext-exact | 0.240 | 0.280 | 0.760 |
| TriviaQA ext-contains | 0.860 | 0.700 | 0.480 |
| NQ ext-contains | 0.100 | 0.080 | 0.100 |

结论：

```text
layer2 修复了 TriviaQA 的 real > control > no-reader
但 BoolQ 仍然严重回退
```

也就是说：

```text
PLE 对知识型 QA 有真实增益
但 reader 注入仍然破坏 Yes/No 格式和 instruction-following
```

### 3.3 Tokenizer 对齐

Qwen3.5 vs Qwen3.8-Flash-Next：

```text
vocab.json      identical
merges.txt      identical
runtime vocab   248077 vs 248077
eos_token_id    248044 vs 248046
所有测试文本 token ids identical
```

结论：

```text
PLE rowid 所需的 raw-token 语义兼容
不需要因为 tokenizer 差异重新抽表
但生成 EOS（248046）和 PLE 分段 EOS（248044）是两个概念
```

### 3.4 当前 reader 训练事实

代码事实：

```python
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    ids[:, 1:].reshape(-1),
)
```

即：

```text
corpus 全 token next-token loss
没有 prompt mask
没有 QA / instruction 数据
没有 conflict / sufficiency 信号
```

所以格式退化是预期的，不是意外。

### 3.5 问题性质

更准确的描述：

```text
不是参数级遗忘（backbone 冻结）
而是 reader 注入导致的 functional interference / representation drift
原能力仍在，只是被错误/过强的注入压住
```

证据：

- no-reader 在 BoolQ 上 0.760；
- real/control 都退化；
- control 也退化，说明不是 PLE 内容本身，而是 reader 训练/注入方式。

---

## 4. 做过的尝试与结果

### 4.1 Phase 2 全量矩阵

- 6 corpus × 3 seed × 500 steps × real/control/no-reader × 150 QA；
- 03:44 完成；
- 结果已归档；
- 旧 QA contains 协议不可用，新增 v2 scoring；
- 没有保存 reader checkpoint，是当时的重大技术债。

### 4.2 v2 评分协议

新增：

```text
extract_answer_v2
score_answer_v2
summarize_phase1_matrix.py --protocol v2
evaluate_generated_answers.py --protocol v2
check_phase2_gates.py
```

作用：

- BoolQ 不再把“同时列出 Yes/No”判对；
- 优先显式 answer marker / 首句；
- 支持 per-task metric（BoolQ exact，open QA contains）。

### 4.3 Tokenizer 审计

- 下载 Qwen3.8-Flash-Next-FP8 tokenizer；
- 对比 vocab / merges / normalizer / pre_tokenizer / post_processor / decoder / added tokens；
- 结论：raw token id 兼容。

### 4.4 持久化 rows

- 86GB 数据盘；
- persistent rows 51.2GB；
- `/dev/shm` 副本；
- manifest valid；
- bootstrap 自动恢复。

### 4.5 layer8 全量诊断

- 完整跑完 3 corpus × 3 seed；
- 结果：PPL PASS，task FAIL；
- 发现 layer8 错误 + reader 注入干扰。

### 4.6 layer2 冒烟

- watcher 自动在 layer8 结束后启动；
- 发现 layer2 修复 TriviaQA，但 BoolQ 仍崩；
- 这是本轮最重要的正向 + 负向结果。

### 4.7 调研

调研并收敛了：

- SFT vs RL；
- pass@1 vs pass@k；
- 两阶段 SFT vs CPT → SFT；
- 数据混合 / replay / midtraining；
- prompt loss masking；
- 现有 gate vs 额外 router；
- conflict-aware / sufficiency-aware gating；
- loss 设计：不要 5–6 个 loss，先 curriculum + 2–3 个 loss；
- 科研品味：区分本质改进 vs trick。

---

## 5. 踩过的坑

### 5.1 `/dev/shm` 重启丢失

- `/dev/shm` 是 tmpfs；
- 重启后 rows 丢失；
- 解决：持久盘 + `/dev/shm` 副本 + ensure/bootstrap。

### 5.2 网络 / git / rsync

- GitHub HTTP/2 / TLS 不稳定；
- `git pull` 失败；
- rsync 在本机旧版本不支持 `--info=progress2`；
- tar 全仓同步因本地 `data/` / `outputs/` 过大超时；
- 解决：HTTP/1.1、显式文件 tar 同步、rsync `--progress`。

### 5.3 QA 协议

- 旧 `correct` 实际是 contains；
- BoolQ 同时列 Yes/No 会被误判；
- 旧 extractor 取最后一句，不适合；
- 解决：v2 protocol + task-aware metric。

### 5.4 manifest bug

- `--write-manifest` 写出的 metadata-only manifest 没有 sha256；
- 被误判为“缺少所有 hash”；
- 解决：metadata-only manifest 不再强制全量 hash。

### 5.5 layer8 错误

- 官方 PLE layer 是 2；
- Phase 2 用 layer8；
- 解决：新增 `--layer`，诊断默认 layer2。

### 5.6 数据混合 / loss 设计

- 曾提出 5–6 个 loss 同时训练；
- 风险：梯度冲突、单 loss 主导、难消融；
- 收敛为：
  ```text
  Phase 1: L_corpus + λ_qa L_qa
  Phase 2: L_qa + λ_gate L_gate_benefit
  必要时再加一个 KL anchor
  ```

### 5.7 关键词路由

- 曾考虑“BoolQ 关 PLE、TriviaQA 开 PLE”；
- 违反 Bitter Lesson；
- 收敛为：用现有 gate + counterfactual benefit / conflict / sufficiency 训练。

### 5.8 评测规模

- 150 QA、3 corpus、3 seed；
- 容易在 BoolQ/TriviaQA 上过拟合；
- 需要 held-out、pass@k、置信区间。

---

## 6. 未完成内容与技术债

### P0：决定方向生死

| 编号 | 债务 | 影响 |
|---|---|---|
| TD-1 | 没有 oracle upper bound | 不知道 PLE 理论上能带来多少任务增益 |
| TD-2 | 没有 PLE-native upper bound | 不知道 Qwen3.5 嫁接与 Qwen3.8 原生的差距 |
| TD-3 | 评测规模小 | 结论置信度不足 |
| TD-4 | reader 仍是 corpus-only 训练 | 格式崩坏的直接原因 |
| TD-5 | gate 没有 conflict/sufficiency 信号 | 不会在 BoolQ 上关闭 |
| TD-6 | 没有 zero-reader / training-only 对照 | 无法区分注入问题和训练问题 |

### P1：机制与效率

| 编号 | 债务 | 影响 |
|---|---|---|
| TD-7 | 没有 pass@k / 多采样 | 无法判断 coverage 是否被破坏 |
| TD-8 | 没有 gate 统计 | 无法解释注入行为 |
| TD-9 | 没有 reader output norm / hidden norm 对比 | 无法判断是否过强注入 |
| TD-10 | QA full recompute | 推理慢 3–4 倍 |
| TD-11 | 训练 GPU 利用率 20–60% | 成本/时间浪费 |
| TD-12 | mixed SFT / curriculum 未实现 | 数据混合方案未验证 |
| TD-13 | PLE-native / oracle 对照未做 | 无法判断方法边界 |

### P2：长期

| 编号 | 债务 | 影响 |
|---|---|---|
| TD-14 | 未进入 5M / 20M scaling | scaling 叙事缺失 |
| TD-15 | unseen-KB 多 split 未跑 | 泛化证据不足 |
| TD-16 | 机制分析不足 | 归因不足 |
| TD-17 | 最终论文贡献未确定 | 容易被看成工程 trick |

---

## 7. 未来计划

### Phase A：判决性诊断（最高优先级）

目标：

```text
回答：pure PLE 嫁接在 Qwen3.5-0.8B 上到底有没有任务级上界？
```

实验：

1. **Oracle gate upper bound**
   - PLE 有帮助就开，有害就关；
   - 看任务级上界。

2. **Zero-reader / training-only control**
   - reader 安装但不注入；
   - 判断格式崩坏是注入问题还是训练问题。

3. **Frozen official reader**
   - 不训练 reader，直接推理；
   - 判断官方 reader 与 Qwen3.5 是否兼容。

4. **Gate statistics**
   - BoolQ vs TriviaQA 的 gate 均值、方差、饱和率；
   - reader output norm / hidden norm。

5. **PLE-native upper bound**
   - Qwen3.8 / PLE-native 模型 vs 嫁接模型；
   - 判断差距是否来自 backbone 与 PLE 的预训练耦合。

判决规则：

```text
如果 oracle gate 都无法让 real 明显超过 no-reader：
  -> 纯 PLE 嫁接在当前 backbone 上可能不成立
  -> 记录负结果，转向 PLE-native 或 fallback

如果 oracle gate 有显著上界：
  -> 问题在 gate 训练，继续 Phase B
```

### Phase B：让现有 gate 学会 conflict / sufficiency

目标：

```text
不增加 router
只训练现有 gate
```

数据：

```text
corpus
+ knowledge QA
+ passage-grounded QA
+ context-sufficient / insufficient examples
+ context-PLE conflict examples
+ format-sensitive examples
```

训练：

```text
Phase 1：Knowledge Utilization
  L = L_corpus + λ_qa * L_qa

Phase 2：Faithful Compliance
  L = L_qa + λ_gate * L_gate_benefit
  （必要时再加一个 KL anchor）
```

原则：

```text
loss 不超过 2–3 个
大部分约束写进数据 / gate target / curriculum
```

评估：

```text
BoolQ
TriviaQA
NQ
gate 统计
format compliance
pass@1 / pass@k
```

成功标准：

```text
BoolQ 恢复到接近 no-reader
TriviaQA 保持 real > control > no-reader
gate 在两类任务上出现明显差异
```

### Phase C：机制与边界

如果 Phase B 成功：

```text
1. layer2 全量诊断
   PURE_WIKI / PURE_CODE / FW_STEM × 3 seeds

2. 机制分析
   - gate 统计
   - CKA
   - activation patching
   - reader output norm / hidden norm

3. 边界实验
   - layer 1 / 2 / 8
   - reader 变体
   - scale sweep
```

如果 Phase B 失败：

```text
不再加技巧
直接做 Phase A 的 oracle / PLE-native upper bound
判断是否应该 pivot
```

### Phase D：Scaling（只有门禁通过才做）

```text
5M scaling
  -> 1–2 个最优 corpus
  -> 5M 通过后
20M scaling
  -> unseen-KB 多 split
  -> layer / reader 变体
  -> 机制分析
  -> 论文 / artifact
```

### Phase E：基础设施与效率（并行）

```text
1. 细粒度 resume / auto-backup / bootstrap（已基本完成）
2. QA KV cache + reader 增量状态
3. 训练 micro-batch / grad accumulation / torch.compile
4. 持久盘 vs /dev/shm 吞吐 benchmark
5. 评测：pass@k、多采样、置信区间
```

---

## 8. 从类似项目借鉴什么

| 项目/方向 | 可以借 | 不能借 / 冲突边界 |
|---|---|---|
| Qwen3.8-Flash-Next PLE | 官方 layer2、gate/reader 语义、raw-token rowid、weight_scale | 不重训 Qwen3.8；不改冻结 PLE 表 |
| DeepSeek Engram | ContextAwareGating、ShortConv、multi-branch、EOS reset | 不引入第二套存储；不重写 reader 核心 |
| TokenMem | 两阶段 curriculum、thin gating adapter、conflict-aware injection | 不照搬 cross-attention channel；我们仍是 residual PLE |
| SHIFT | 冻结 backbone + lightweight gate、知识冲突缓解 | 不额外加独立 router；不变成 RAG 项目 |
| Memory-R1 | RL 做 memory policy / routing / filtering | 不用 RL 注入知识；不替代核心 PLE |
| Midtraining / Replay | 混合 specialized + general data、replay 防遗忘 | 不盲目调比例；比例是 supporting hyperparameter |
| TailSFT / InfoSFT / SFTMix | coverage-aware SFT、token weighting、confidence mixup | 不让这些变成核心创新；先验证 PLE 本身 |
| SFT vs RL 文献 | SFT 稳定格式、RL 提升 pass@1/组合 | 不用 RL 修格式/知识注入 |
| Data Mixing Laws / PiKE | 自适应数据混合、near-optimal region | 不把 ratio sweep 当核心贡献 |
| RAG / BM25 / kNN-LM | 外部知识上界、baseline、失败模式 | 不替代 PLE |
| PEFT / LoRA / MoRA | fallback 能力迁移 | 不进入纯 PLE 核心路径 |
| EngramDB | Store-I/View、ABI、rowid、bundle、checksum | 不让实验仓承担核心库职责 |
| AutoDL | 持久盘、定时关机、快照、成本控制 | 不为省钱牺牲结果完整性 |

---

## 9. 不冲突原则

1. 核心假设永远不变：

```text
real / control / no-reader
冻结 PLE 表
只训练 reader / bridge / gate
```

2. 一次只改一个变量。
3. 先做判决性实验，再做优化。
4. loss 不超过 2–3 个；约束写进数据和 target。
5. 不新增 router，除非现有 gate 被证明表达力不足。
6. RL / RAG / 蒸馏 / adapter 只在纯 PLE 被严格证伪后进入。
7. 比例、权重、步数都是 supporting engineering，不是核心贡献。
8. 负结果也是结果；提前定义 kill criteria。
9. 保持低资源、可复现、CI 绿色。

---

## 10. 当前服务器状态与下一步

```text
GPU：空闲
layer8 全量诊断：已完成
layer2 冒烟：已完成
rows：持久化 + /dev/shm 副本
reader checkpoints：已保存
```

下一步建议：

```text
1. 先做 Phase A 的 oracle / gate / zero-reader 诊断
2. 再决定是否进入 Phase B 的 mixed SFT + curriculum
3. 不要马上上 RL
4. 不要在 150 题上反复调 prompt
```

---

## 11. 关键路径索引

```text
代码仓库：
  /Users/zeng/code/qwen35-ple

远程：
  /root/autodl-tmp/qwen35-ple/repo
  /root/autodl-tmp/qwen35-ple/qwen38-rows
  /dev/shm/qwen38-rows

结果：
  outputs/phase2-diagnostic/        # layer8 全量
  outputs/phase2-layer2-smoke/      # layer2 冒烟
  artifacts/phase2-full-4090/
  artifacts/phase2-diagnostic-layer8/
  artifacts/phase2-layer2-smoke/

日志：
  logs/phase2-diagnostic.log
  logs/phase2-layer2-smoke.log
  logs/bootstrap.log
  logs/watcher.log
```

---

## 12. 结论

> **本轮最重要的科学发现是：**
>
> ```text
> layer2 下 PLE 对知识型 QA（TriviaQA）有真实增益
> 但现有 gate 在 BoolQ 类格式任务上不会关闭
> ```
>
> 问题不是参数级遗忘，而是 reader 注入导致的 functional interference。
>
> 下一步不是加 router、加很多 loss、上 RL，  
> 而是先用 oracle / PLE-native upper bound 判断边界，  
> 再用 mixed SFT + 两阶段 curriculum 训练现有 gate。
>
> 如果 oracle 也证明没有上界，  
> 那“纯 PLE 嫁接在非 PLE-native backbone 上的边界”  
> 本身就是一个值得记录的负结果。
