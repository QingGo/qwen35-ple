# Round 49：Phase A — rare-token 知识评测与任务级因果信号

> 日期：2026-09-04
> 状态：Phase A 第一轮结果
> 目的：回答“真实 PLE 在 rare-token 知识任务上是否提供因果、任务相关的信号”。

---

## 1. 本轮做了什么

### 1.1 新增脚本

| 脚本 | 作用 |
|---|---|
| `scripts/build_rare_kb.py` | 构建 `rare-kb-v1` 评测集 |
| `scripts/mechanism_rare_task_r2.py` | 任务级 \(\Delta R^2(Y;E\mid H)\) 与 real/control 对照 |
| `scripts/analyze_rare_kb_logit.py` | 汇总 logit-patch 条件答案 logprob，按 rare/common/source 分层 |

### 1.2 评测集

- 来源：
  - `data/qa-expanded-150.json`：150 条 triviaqa / nq / boolq；
  - `alpaca_data_cleaned.json`：抽取 120 条短问答。
- 稀有度：
  - 以 `data/wet-1m-one.txt` 为参考语料；
  - 答案中任意内容词频 ≤ 5 记为 rare。
- 规模：270 条，rare 182 / common 88。

---

## 2. 实验 A：纯特征任务级 \(\Delta R^2\)

### 2.1 方法

- 对每条 QA 的答案 token 位置收集：
  - \(H\)：Qwen3.5 backbone 第 8 层 hidden；
  - \(E\)：真实 PLE feature；
  - \(Y\)：下一 token 的 input embedding。
- 用 ridge 回归计算：
  - \(\Delta R^2_{\text{real}} = R^2(Y;[H,E]) - R^2(Y;H)\)；
  - \(\Delta R^2_{\text{control}} = R^2(Y;[H,E_{\text{shuffled}}]) - R^2(Y;H)\)；
  - \(\Delta R^2_{\perp} = R^2(Y;[H,E_\perp]) - R^2(Y;H)\)。

### 2.2 结果

seed 0 / 1 次随机 split：

| 子集 | n | \(\Delta R^2_\text{real}\) | \(\Delta R^2_\text{control}\) | \(\Delta R^2_\perp\) |
|---|---:|---:|---:|---:|
| rare item | 1480 | **+0.000560** | −0.000260 | +0.000470 |
| common item | 291 | +0.000757 | −0.000749 | +0.000267 |
| rare token | 183 | +0.000422 | −0.000197 | +0.000294 |
| common token | 1097 | +0.001337 | −0.000589 | +0.000947 |

3 seeds（mean ± std）：

| 子集 | n | \(\Delta R^2_\text{real}\) | \(\Delta R^2_\text{control}\) | \(\Delta R^2_\perp\) |
|---|---:|---:|---:|---:|
| rare item | 1480 | **+0.000838 ± 0.000249** | −0.000435 ± 0.000127 | +0.000601 ± 0.000166 |
| common item | 291 | +0.000605 ± 0.000130 | −0.000907 ± 0.000293 | +0.000289 ± 0.000036 |
| rare token | 183 | +0.000294 ± 0.000106 | −0.000165 ± 0.000117 | +0.000197 ± 0.000082 |
| common token | 1097 | +0.001330 ± 0.000155 | −0.000574 ± 0.000177 | +0.000939 ± 0.000161 |

解读：

- **真实 E 在所有子集、全部 3 seeds 上都优于 shuffled control**，说明纯 PLE 行特征有因果增量信息；
- 数值仍然极小（~1e-4 到 1e-3）；
- rare item 的 \(\Delta R^2_\text{real}\) 略高于 common item，但 rare token 低于 common token；整体没有一致“rare 更强”证据；
- \(E_\perp\) 与全 \(E\) 相近，说明主要增量来自 E 中不能被 H 线性预测的部分；
- 3 seeds 的方向稳定：real 始终正、control 始终负或接近零。

---

## 3. 实验 B：reader 中介的任务答案 logprob

### 3.1 方法

- 使用已训练 simple reader `outputs/reader-real-seed0.pt`；
- 对 270 条 QA 做 5 条件 logit patch：no-reader / real / control / random / zero；
- 指标：答案 token 的平均 log-probability（越高越好）。

### 3.2 总体结果

| 条件 | mean answer logprob |
|---|---:|
| no-reader | −5.549 |
| real | **−5.058** |
| control | −5.186 |
| random | −5.516 |
| zero | −5.549 |

真实 PLE 总体优于 control 与 no-reader。

### 3.3 关键分层

| 层 | no-reader | real | control | real−control |
|---|---:|---:|---:|---:|
| rare（全部） | −4.248 | −4.210 | −4.207 | **−0.003** |
| common（全部） | −8.241 | −6.811 | −7.210 | **+0.399** |
| qa-expanded rare | −8.234 | −8.139 | −8.123 | **−0.017** |
| qa-expanded common | −9.402 | −7.742 | −8.200 | **+0.458** |
| alpaca rare | −1.517 | −1.518 | −1.525 | +0.007 |
| alpaca common | −0.882 | −0.916 | −0.941 | +0.025 |

关键结论：

- **在真正的 rare 知识子集（qa-expanded rare）上，real 与 control 几乎无差异，甚至略差**；
- real 相对 control 的明显优势集中在 common / boolq / qa-expanded common；
- Alpaca 语言小任务上 real 没有优势；
- 这再次验证：**loss/logprob 的改善可能来自 style 或局部模式，不一定等于 rare 知识智能提升**。

### 3.4 MLP(H,E⊥) reader 在 qa-expanded rare 上复测

使用 `outputs/reader-mlp-residual-concat.pt` 对 74 条 qa-expanded rare 做同样 5 条件 logit patch：

| 条件 | mean answer logprob |
|---|---:|
| no-reader | −8.2341 |
| real | **−8.2328** |
| control | −8.2340 |
| random | −8.2345 |
| zero | −8.2352 |

- real−control = **+0.0011**；
- real−no-reader = **+0.0013**；
- 改进后的 MLP reader 也没有把 rare 任务信号放大到可用水平。


---

## 4. 实验 C：10 条生成式 exact-match 冒烟

| Arm | EM |
|---|---:|
| real | 0.2 |
| control | 0.2 |
| no-reader | 0.2 |

虽然 real 的 val loss 低于 control/no-reader，但 10 条小样本上没有 EM 差异。生成式全量评测成本较高，暂未作为本轮的决策依据。

---

## 5. Phase A 结论

1. **纯 PLE 特征确有因果增量信息**：真实 E 的 \(\Delta R^2\) 在 rare/common 都高于 shuffled control；
2. **但增量极小**：线性回归视角下约 \(10^{-4}\) 量级；
3. **现有 simple reader 与 MLP(H,E⊥) reader 都无法在 rare 知识任务上把该信号转为可观测优势**：
   - simple reader：qa-expanded rare 上 real−control ≈ −0.017；
     - MLP reader：qa-expanded rare 上 real−control ≈ +0.001；
   - common 上 real−control 为正，主要来自 boolq/常见模式；
4. **没有证据支持“PLE 对 rare 知识任务有显著因果优势”**。

### 门禁判定

- 原计划 Phase A 门禁：“能回答 PLE 在真实知识任务上是否 > control”；
- 本轮答案：**纯特征层面是，任务/reader 层面在 rare 上不是**；
- 因此不进入大规模放大，但保留 Phase B 的窄口径可行性：需要先做一个**真正针对 rare knowledge 的 reader / gate**，而不是直接用当前 simple reader 下结论。

---

## 6. 下一轮

1. 用 3-seed 重复实验 A，确认 tiny positive 是否稳定；
2. 把 rare-kb 收敛为“qa-expanded rare + 新增长尾知识题”，去掉 Alpaca 语言题；
3. 在 Phase B 中实现 rare gate / fixed \(E_\perp\) + MLP(H,E⊥) + differential；
4. 如果在改进 reader 后 rare real−control 仍非正，则按停止条件将 PLE 定位为局部语言模式增强，转向 RAG / 蒸馏。

---

## 7. 产物路径（WSL）

- `data/rare-kb-v1.json`
- `data/rare-kb-v1-items.json`
- `outputs/mechanism-rare-task-r2.json`
- `outputs/mechanism-rare-task-r2-3seed.json`
- `outputs/mechanism-rare-logit-patch-full.json`
- `outputs/mechanism-rare-mlp-concat-logit.json`
- `/tmp/rare-kb-logit-summary.json`
- `outputs/reader-real-seed0.pt`（现有 simple reader）
- `outputs/reader-mlp-residual-concat.pt`（MLP reader）

---

## 8. 命令记录

```bash
# 构建
.venv/bin/python scripts/build_rare_kb.py --output data/rare-kb-v1.json

# 纯特征任务级 ΔR²
.venv/bin/python scripts/mechanism_rare_task_r2.py --device cuda --output outputs/mechanism-rare-task-r2.json

# 3 seeds 重复
.venv/bin/python scripts/mechanism_rare_task_r2.py --device cuda --seeds 0 1 2 --output outputs/mechanism-rare-task-r2-3seed.json

# 5 条件 logit patch（simple reader）
.venv/bin/python scripts/mechanism_logit_patch.py --limit 270 --conditions no-reader real control random zero --device cuda --qa-file data/rare-kb-v1-items.json --reader outputs/reader-real-seed0.pt --output outputs/mechanism-rare-logit-patch-full.json

# MLP reader 在 qa-expanded rare 上复测
.venv/bin/python scripts/mechanism_logit_patch.py --limit 74 --conditions no-reader real control random zero --device cuda --qa-file data/rare-kb-qa-rare.json --reader outputs/reader-mlp-residual-concat.pt --output outputs/mechanism-rare-mlp-concat-logit.json

# 汇总
.venv/bin/python scripts/analyze_rare_kb_logit.py --benchmark data/rare-kb-v1.json --logits outputs/mechanism-rare-logit-patch-full.json --output outputs/rare-kb-logit-summary.json
```
