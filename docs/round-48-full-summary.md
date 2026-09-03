# Round 48：本轮完整总结

> 日期：2026-09-03
> 范围：从“继续机制验证”到“MLP Value Reader 原型 + 战略复盘”的完整工作链
> 状态：已完成当前阶段总结，下一步进入 Phase A/B

---

## 1. 本轮计划

1. 继续机制验证，验证理论预测；
2. 用数学推导“什么是对齐的本质”；
3. 建立增量 R² / 梯度残差等理论指标；
4. 实现并测试 MLP Value Reader；
5. 比较 PCA / PLS / 稀有 token / Oracle MLP；
6. 修复 CI；
7. 形成系统性战略与开发计划。

---

## 2. 本轮完成的内容

### 2.1 工具与脚本

| 脚本 | 作用 |
|---|---|
| `mechanism_alignment.py` | CKA / Procrustes / kNN / intrinsic dimension |
| `mechanism_logit_patch.py` | real/control/random/zero logit patching |
| `mechanism_patching.py` | 逐 token 生成式 patching |
| `mechanism_incremental_r2.py` | next-token embedding 增量 R² |
| `mechanism_gradient_r2.py` | LM 梯度残差增量 R² |
| `mechanism_gradient_sweep.py` | PCA + 高/低梯度子集 |
| `mechanism_advanced_sweep.py` | PLS + 稀有/常见 token 子集 |
| `mechanism_oracle_mlp.py` | 非线性上界 |
| `train_mlp_value_reader.py` | 残差监督训练 MLP Value |
| `mechanism_mlp_dependence.py` | value 对 E 内容的依赖诊断 |

### 2.2 代码

- 新增 `MLPValueReader`；
- 注册 `mlp_value_v1`；
- `run_phase0.py` 支持 `--reader mlp`；
- 修复 CI ruff 0.16 问题；
- CI 纳入所有 mechanism 脚本。

### 2.3 文档

- round-28 到 round-47 共 20 份文档；
- 覆盖机制验证、数学证明、实验预注册、MLP prototype、战略规划。

---

## 3. 关键发现

### 3.1 对齐度量

| 指标 | 结果 |
|---|---|
| CKA | 0.15–0.22 |
| Procrustes alignment | 0.01–0.05 |
| kNN overlap | 0.068–0.084，随机基线 0.039 |
| PLE intrinsic dimension | ≈ 766 |
| Qwen hidden intrinsic dimension | 37–78 |

结论：PLE 与 Qwen hidden 的全局/局部几何对齐都弱。

### 3.2 增量 R²

- 线性 next-token embedding：ΔR²≈0.006–0.010；
- LM 梯度残差线性：ΔR²≈0.0058；
- MLP 梯度残差：ΔR²≈0.0206；
- Oracle MLP H+E⊥：ΔR²≈0.0228。

结论：非线性可以提取约 3–4 倍信息，但绝对量仍小。

### 3.3 PLS vs PCA

- PCA r=256：+0.002；
- PLS r=64：+0.0074；
- PLS 超过全维度线性基线。

结论：应该使用监督低秩方向，而不是无监督 PCA。

### 3.4 稀有 vs 常见 token

- 稀有 token：ΔR²≈+0.0152；
- 常见 token：ΔR²≈+0.0079。

结论：PLE 对稀有 token 更有用，支持 rare-token gate。

### 3.5 高梯度 vs 低梯度

- 高梯度：ΔR²≈-0.0048；
- 低梯度：ΔR²≈+0.0149。

结论：高梯度不等于需要记忆；gate 应该用 rarity 而不是 gradient norm。

### 3.6 MLP Reader 问题

- learned h_to_e 会退化（cos≈0.95+）；
- MLP(E_perp) 单独几乎无效（R²≈0）；
- MLP(H,E_perp) 恢复 Oracle 水平（R²≈0.275）；
- differential 注入信号太弱，random 反而最好；
- contrastive hinge 训练发散。

### 3.7 Loss 代理问题

- val loss real < control < no-reader；
- 但任务 EM 没有提升；
- 原因：Loss 下降主要来自 style/局部 n-gram/格式效应；
- control 也能降低 loss；
- 任务智能需要 \(\Delta I(Y;E|H)\)，不是 val loss。

---

## 4. 做的尝试

1. 完整 150 题 logit patching；
2. BoolQ/Trivia scale sweep；
3. next-token embedding 增量 R²；
4. LM 梯度残差增量 R²；
5. PCA 压缩；
6. PLS 监督压缩；
7. 高低梯度子集；
8. 稀有/常见 token 子集；
9. Oracle MLP；
10. MLPValueReader + LM loss；
11. MLPValueReader + 残差监督；
12. 固定 h_to_e；
13. MLP(H,E_perp)；
14. differential injection；
15. contrastive hinge；
16. value dependence 诊断。

---

## 5. 踩过的坑

| # | 坑 | 解决/状态 |
|---|---|---|
| 1 | next-token embedding 目标 R² 为负，代理不可靠 | 改用 LM 梯度残差 |
| 2 | learned h_to_e 退化 | 固定 ridge 投影 |
| 3 | MLP(E_perp) 单独无效 | 改为 MLP(H,E_perp) |
| 4 | contrastive hinge 发散 | 未解决，需 InfoNCE/triplet |
| 5 | CI ruff 0.16 8 个错误 | 已修复 |
| 6 | 新脚本未被 CI lint | 已加入 |
| 7 | 文档重复行/尾随空格 | 已清理 |
| 8 | WSL 默认 python 无 torch | 使用 `.venv/bin/python` |
| 9 | ssh 内联复杂命令卡住 | 改用脚本文件 + scp |
| 10 | 后台任务启动时 ssh 挂起 | 使用 setsid + 独立脚本 |
| 11 | `reader_config_from_args` 导入未用 | ruff 修复 |

---

## 6. 未完成的内容

| # | 未完成 |
|---|---|
| 1 | Rare-token 知识评测集 |
| 2 | 任务级 \(I(Y_{\text{task}};E\mid H)\) |
| 3 | Backbone adaptation（LoRA/部分解冻） |
| 4 | 稳定 contrastive value loss |
| 5 | RAG / 教师蒸馏同口径对照 |
| 6 | 3 seeds / 显著性 |
| 7 | 大规模训练（100k–1M） |
| 8 | CPU 100 tok/s serving |
| 9 | vLLM/SGLang/CompileForge 完整集成 |
| 10 | MLPValueReader registry roundtrip 测试 |
| 11 | 完整外部评测集（知识/推理/长上下文） |

---

## 7. 技术债清单

### 高优先级

- 没有 rare-task benchmark；
- 没有任务级条件互信息；
- h_to_e 退化只缓解未根治；
- 没有 backbone adaptation；
- 没有稳定 contrastive；
- 没有 RAG 对照；
- 没有 3 seeds。

### 中优先级

- 训练规模过小；
- 缺少 Memory Grafting 规模参考；
- 缺少 serving/perf；
- WSL/本地环境漂移。

### 低优先级

- 文档格式偶有问题；
- 新脚本测试覆盖不足；
- 缺少 RAG baseline 脚本。

---

## 8. 借鉴矩阵

| 来源 | 借鉴 |
|---|---|
| XMemTransfer / Memory Grafting | 5M–20M reader 训练 |
| DeepSeek Engram / engram-peft | gate、ShortConv、条件记忆 |
| NGM / MLP Memory | 训练无关/更语义化记忆 |
| RAG / ReAugKD | 外部检索与教师蒸馏 |
| Hierarchical Memory | 常见/长尾知识分离 |
| Selective Memory / MemPO | 小模型记忆编排 |
| SR-TTT | surprise-aware residual |
| Storage–Retrieval Gap | 诊断输出条件化假象 |
| Scaling Law 研究 | Loss 代理适用边界 |
| EngramDB / CompileForge | golden、契约、CPU serving |

---

## 9. 未来计划

### Phase A：任务与指标（1–2 周）

- rare-token 知识评测集；
- task-level ΔI；
- real/control CATE + 置信区间。

### Phase B：Reader 稳定（2–4 周）

- 固定 E_perp；
- MLP(H,E_perp) + differential；
- rare gate；
- 稳定 contrastive；
- 3 seeds。

### Phase C：Backbone 与规模（4–8 周）

- LoRA / 部分解冻；
- 100k–1M 数据；
- rare 过采样；
- SFT；
- RAG baseline。

### Phase D：RL / 混合记忆

- DPO/GRPO；
- 或 RAG / 蒸馏；
- 比较 PLE / PLE+RAG / RAG / 蒸馏。

### Phase E：产品化

- vLLM / SGLang / CompileForge；
- Store-P / access-order；
- CPU 100 tok/s；
- bundle e2e。

---

## 10. 停止条件

如果 Phase B/C 后：

```text
real 在 rare-task 上仍不显著 > control
```

则：

- 将 PLE 定位为“局部语言模式增强”；
- 转向 RAG / 蒸馏 / 更语义化记忆；
- 记录为可审计负面结果；
- 不进入大规模 RL。

---

## 11. 当前最优先行动

1. 建 rare-token 知识评测集；
2. 固定 E_perp + differential + rare gate；
3. 加 LoRA / backbone adaptation；
4. 加 RAG baseline；
5. 3 seeds 验证。
