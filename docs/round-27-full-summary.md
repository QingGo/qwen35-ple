# Round 27 全量总结：暂停混比扫描，转向机制验证、流形对齐与案例研究

> 日期：2026-09-03
> 范围：M1–M5 数据构建与 M1 结果、control 机制、val loss 解读、流形/对齐调研、工程工具、后台任务管理
> 状态：M2–M5 已暂停；下一阶段以机制验证和 case 分析为主

---

## 1. 本轮目标

1. 继续推进 1M 混合语料实验，按计划跑 M1–M5。
2. 解释“混合语料 val loss 降低但性能没有明显提升”。
3. 分析 control 的 good/bad case。
4. 调研语义对齐分析方法和流形学习/数学对齐工具。
5. 为下一阶段机制验证和可解释性研究做准备。

---

## 2. 本轮计划

| 项目 | 原计划 |
|---|---|
| 语料 | ModelScope 下载 chat / wiki / CoT / tool 数据 |
| 构建 | 写 `build_mix.py`，构建 M1–M5 1M 混合语料 |
| 审计 | 写污染审计脚本，确保 QA 不在训练语料中 |
| 训练 | WSL 后台跑 M1–M5 三线 150 QA |
| 分析 | 比较 real / control / no-reader |
| 调研 | 文献调研：val loss 与能力关系、数据混合、机制解释、流形对齐 |
| 工程 | 增加逐步 val loss、CSV 导出、论文图工具 |

---

## 3. 本轮做了什么

### 3.1 数据与语料

- 从 ModelScope 下载：
  - `AI-ModelScope/alpaca-cleaned`
  - `Salesforce/wikitext`
  - `nohurry/Opus-4.6-Reasoning-3000x-filtered`
  - `iic/MSAgent-Bench` dev
- 新增：
  - `scripts/download_mix_sources.py`
  - `scripts/build_mix.py`
  - `scripts/audit_contamination.py`

### 3.2 M1–M5 混合语料

| Mix | general | chat | wiki | cot | tool | 实际 token |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 50% | 20% | 20% | 6% | 4% | 1,001,390 |
| M2 | 40% | 30% | 20% | 6% | 4% | 1,001,079 |
| M3 | 30% | 40% | 20% | 6% | 4% | 1,001,046 |
| M4 | 30% | 30% | 30% | 6% | 4% | 1,000,875 |
| M5 | 20% | 40% | 20% | 10% | 10% | 1,001,090 |

### 3.3 污染审计

- 使用 `--exclude-qa` 过滤 QA 答案短语、完整问题、QA 组合。
- M1–M5 审计结果：
  - 全部 150 题 low；
  - critical 0；
  - high 0。
- 注意：单字常见词如 `yes` / `no` / `six` / `water` 仍可能在自然语料中出现。

### 3.4 M1 三线结果

| 线 | val loss | PPL | QA EM | TriviaQA | NQ | BoolQ |
|---|---:|---:|---:|---:|---:|---:|
| no-reader | 2.4563 | 11.66 | 53.3% | 70% | 0% | 90% |
| real | 2.3949 | 10.97 | 50.7% | 76% | 0% | 76% |
| control | 2.4391 | 11.46 | 52.7% | 84% | 4% | 70% |

### 3.5 M2–M5

- 后台启动过 M2–M5。
- M2 已完成 real 训练、开始 real QA。
- 后根据判断暂停全部 M2–M5。
- 原因：
  - 当前问题不是混比微调；
  - M1 已显示 `real < no-reader`、control 也退化；
  - 继续跑 M2–M5 对科学判断帮助有限。

### 3.6 机制分析与语义对齐调研

完成两类调研：

1. **为什么 val loss 低但能力没有提升**
   - val loss 衡量的是对训练分布的拟合；
   - 不是下游任务能力；
   - vail 来自同一混合语料，跨语料不可比。

2. **流形 / 语义空间对齐**
   - PLE `e_t` 空间与 Qwen hidden 空间可视为两个语义流形；
   - 可借鉴：
     - Procrustes / CCA；
     - Manifold Alignment；
     - Gromov-Wasserstein；
     - Contrastive / InfoNCE；
     - MMD / HSIC；
     - CKA / kNN overlap / intrinsic dimension。

### 3.7 工程增强

- 新增：
  - `scripts/export_phase0_metrics.py`
  - `scripts/plot_phase0_metrics.py`
- 修改：
  - `scripts/run_phase0.py`
    - 增加英文数字归一化；
    - 增加逐题 QA 进度输出；
    - 增加可选 `--val-every`，用于后续逐步 val loss 曲线。
- 从 M1 JSON 已可导出：
  - `train_loss.csv`
  - `summary.csv`
  - `per_question.csv`
  - 图：QA EM、real task EM、val loss bar、train loss curve。

---

## 4. 关键发现

### 4.1 PLE 对语言建模仍有正信号

```text
val loss：real < control < no-reader
```

说明真实 PLE 确实降低了语言建模 loss。

### 4.2 任务级没有净收益

```text
QA EM：no-reader > control > real
```

### 4.3 control 也退化

- control = Qwen3.5 + 训练后 reader + 随机打乱 PLE e_t。
- control 不是原版模型。
- control BoolQ 70%，低于 no-reader 90%。
- 说明“注入任何向量 + 训练 reader”本身就会干扰简单判断题。

### 4.4 control 也有“知识型 good case”

- control 也能做对 Shakespeare、Newton、Rome、Poseidon。
- 说明这些不能单独作为“真实 PLE 语义对齐”的证据。

### 4.5 real 独有的非记忆新做对很少

- 较典型的例子：Leonardo da Vinci。
- 其他主要是 BoolQ 上的 yes/no 差异。
- 当前 PLE 语义对齐证据仍较弱。

### 4.6 BoolQ 退化特征

- 正确答案为 yes 时，real/control 更容易输出 no。
- real 错误 12 题中，9 题是 yes 答错；
- control 错误 15 题中，13 题是 yes 答错。
- 常见输出：
  - `Answer: No, ...`
  - 重复 `Answer:`
  - `[yes/no]` 格式混乱。

### 4.7 control good/bad case 结论

| 比较 | 数量 | 分布 |
|---|---:|---|
| control 新做对 vs no-reader | 14 | Trivia 9、BoolQ 3、NQ 2 |
| control 新做错 vs no-reader | 15 | BoolQ 13、Trivia 2 |

- control good case 主要是格式收益 + 基座已有知识；
- control bad case 说明随机 PLE 会破坏 passage 判断。

---

## 5. 尝试过但放弃的事情

| 尝试 | 结果 | 放弃原因 |
|---|---|---|
| M2–M5 全量后台扫描 | 仅 M2 跑到 real QA | 混比不是当前主要矛盾 |
| 通过各自语料 val loss 选择最佳 mix | 不可比 | 验证集来自不同分布 |
| 仅用“答案不在语料中”判断 PLE 有效 | 不充分 | control 也能做对类似题目 |
| 早期直接考虑 RL | 未执行 | 缺少机制证据，容易掩盖问题 |

---

## 6. 踩过的坑

| # | 坑 | 解决/状态 |
|---|---|---|
| 1 | 本地 `.venv` 缺少 numpy/transformers | 改用 `/Users/zeng/miniconda3/envs/qwen3-tts/bin/python` |
| 2 | 通过 ssh 写大段 heredoc 脚本会卡住 | 改用本地文件 + scp 到 WSL |
| 3 | `run_mix_batch.sh --mixes M2 M3 M4 M5` 只取第一个 mix | 修改 parser：`--mixes` 收集多个值直到下一个 option |
| 4 | 误用 `--force` 在测试 parser 时重跑了 M1 | 及时终止，未覆盖 M1 结果 |
| 5 | 污染过滤时把整个 general 单行语料当一条记录过滤掉 | 增加长文本按句子/字符分块后再过滤 |
| 6 | `summarize_mix_results.py` 名称提取错误 | 修复为正确提取 M1/M2 |
| 7 | 没有逐步 val loss | 为 run_phase0 增加 `--val-every` |
| 8 | 缺少论文图数据 | 增加 export/plot 工具 |
| 9 | M2–M5 后台任务管理复杂 | 使用 Windows Scheduled Task，结束后已删除 |

---

## 7. 完成 / 未完成

### 已完成

- [x] M1–M5 混合语料构建
- [x] 污染审计全部 low
- [x] M1 三线 150 QA
- [x] M1 结果分析与 control good/bad case
- [x] val loss 与能力关系调研
- [x] 流形对齐和数学工具调研
- [x] 新增机制/论文工具：
  - export_phase0_metrics.py
  - plot_phase0_metrics.py
  - `--val-every`
- [x] M2–M5 已暂停并清理任务

### 未完成

- [ ] reader 参数有效性分析
- [ ] CKA / Procrustes / kNN overlap 对齐诊断
- [ ] activation patching
- [ ] BoolQ logit lens
- [ ] PLE 检索忠实度分析
- [ ] layer/scale/gate 扫描
- [ ] 固定外部评测集
- [ ] manifold alignment / contrastive loss 实验
- [ ] 3 seeds
- [ ] 5M–20M
- [ ] SFT/RL
- [ ] serving 与 CPU 100 tok/s

---

## 8. 未来计划

### Phase A：机制验证（最高优先）

1. 测 PLE e_t 与 Qwen hidden 的：
   - CKA；
   - Procrustes residual；
   - kNN overlap；
   - intrinsic dimension。
2. 测 reader：
   - 训练前后参数变化；
   - 输出 norm；
   - gate 激活；
   - zero-init vs trained。
3. activation patching：
   - real / control / random / zero 替换关键 token e_t。
4. BoolQ：
   - 错误分类；
   - logit lens；
   - layer/scale/gate 扫描。

### Phase B：评测修正

1. 固定外部 LM probe。
2. 固定 QA/BoolQ/CoT/tool 评测。
3. train/val/test 分离。
4. 加入 `--val-every` 的逐步 val loss。

### Phase C：训练方法调整

根据机制结果决定：

- 语料：
  - 增加 BoolQ/QA 格式数据；
  - 增加真正需要外部记忆的知识题。
- 网络：
  - gate / scale；
  - top-k 记忆选择；
  - 更合适的注入层；
  - 短路/残差。
- Loss：
  - `L_lm + contrastive + neighbor + KL + task`。
- 优化器：
  - 分开学习率；
  - 大 warmup；
  - reader-only weight decay；
  - gradient clipping。

### Phase D：RL 决策门禁

满足以下条件才做 RL：

- real > no-reader；
- real > control；
- real 独有非记忆新做对；
- BoolQ 不显著退化；
- 3 seeds 稳定。

### Phase E：产品化

- vLLM / SGLang serving A/B；
- CPU 100 tok/s；
- 正式发布 checkpoint / bundle。

---

## 9. 借鉴矩阵

| 来源 | 借什么 | 不拿什么 | 为什么能共存 |
|---|---|---|---|
| 苏剑林 Scaling Law 解构 | loss 分解、数据混比独立、优化/架构/数据分开看 | 不照搬公式 | 帮助判断 val loss 本质 |
| XMemTransfer | 5M–20M 训练量级 | 不复制模型 | 规模参考 |
| Memory Grafting | 冻结外部记忆 + projection/gating | 不复制记忆提取 | 语义对齐而非背题 |
| DeepSeek Engram / engram-peft | 条件记忆、ShortConv、gate、训练基建 | 不引入第二套存储 | 直接复用 |
| EngramDB | manifest、证据库、位级一致 | 不改存储核心 | 可重建资产 |
| Manifold Learning | CKA、Procrustes、流形对齐、Gromov-Wasserstein | 不替代实验 | 提供数学对齐工具 |
| Optimal Transport | 用 pairwise metric 对齐两个空间 | 不引入外部检索 | 适合 PLE vs Qwen |
| Contrastive Learning | InfoNCE、MMD、邻域保持 | 不照搬视觉框架 | 可作为 loss |
| RAG / Memory Augmented | 检索忠实度、passage grounding | 不做外部检索系统 | 评估 PLE 使用 |
| Benchmark Contamination | held-out、n-gram、provenance | 不放弃知识评测 | 保证可信 |
| SFT/RL | SFT 先、RL 后 | 不提前 RL | 避免掩盖机制问题 |
| vLLM / SGLang / CompileForge | serving 与性能闭环 | 不复制引擎 | 产品化 |

---

## 10. 当前资产

### 结果

```text
outputs/phase0-M1-seed0.json
```

### 语料

```text
data/mixes/M1 ... M5
data/sources/
```

### 污染报告

```text
outputs/contamination-M1.json ... M5.json
```

### 工具

```text
scripts/build_mix.py
scripts/audit_contamination.py
scripts/download_mix_sources.py
scripts/export_phase0_metrics.py
scripts/plot_phase0_metrics.py
scripts/run_mix_batch.sh
scripts/summarize_mix_results.py
scripts/analyze_qa_lines.py
```

### 文档

```text
docs/round-26-systematic.md
docs/round-27-manifold-alignment.md
docs/round-27-full-summary.md
```

---

## 11. 当前纪律

1. 不再把“各自语料 val loss 更低”作为有效证据。
2. 不再把“答案不在语料中”单独作为 PLE 语义对齐证据。
3. 机制分析完成前，不进入 5M–20M。
4. 机制分析完成前，不做 RL。
5. 每个结论尽量：
   - 三线对照；
   - 固定外部评测；
   - 3 seeds；
   - 可复现命令；
   - 污染审计。
