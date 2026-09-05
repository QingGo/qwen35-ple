# Round 73：本轮完整总结

> 日期：2026-09-04
> 范围：PLE 重新定位、技术路线调研、数学推导、教师蒸馏准备、RAG/服务化完善
> 状态：当前阶段总结归档，下一阶段从 PLE-1 实验开始

---

## 1. 本轮计划

1. 重新思考 PLE 的定位，不再把它当“失败的语义知识记忆”；
2. 调研有限资源下所有可能提升 0.8B 的技术路线；
3. 搜索调研 PLE/Engram/n-gram 的多条使用路径；
4. 用数学推导“什么是最有效路径”；
5. 准备教师蒸馏资源与运行方式；
6. 继续完善 RAG 混合检索与 serving；
7. 把本轮计划、发现、尝试、踩坑、完成/未完成、未来计划整理成文档。

---

## 2. 核心发现

### 2.1 PLE 到底是什么

- Qwen3.8-Flash-Next PLE 原生只有 2-gram / 3-gram；
- 4-gram 需要外部构建；
- PLE 本质是：
  - 非参数
  - 离散键
  - 固定阶
  - 表面形式记忆

### 2.2 为什么 PLE 作为“语义知识记忆”失败

已有证据：

- Phase A：task-level ΔR² ≈ 1e-4 ~ 1e-3；
- P1 hidden injection：rare real−control ≈ +0.00013；
- B3 logit-space 3 seeds：rare real−control ≈ −0.00117 ± 0.00008；
- RAG rare Δ ≈ +0.851。

结论：

> PLE 的任务相关信息很低，不是读取方式问题，而是定位问题。

### 2.3 普通 LLM 的智能来自什么

普通 LLM 产生智能的关键机制：

- Attention 上下文动态路由
- 多层非线性组合
- Induction heads / in-context learning
- 压缩压力下被迫形成抽象
- 世界模型/推理模板

这些都不能靠 n-gram 查表获得。

### 2.4 PLE 的新定位

PLE 应该成为：

> **非参数、可寻址、长尾/局部/低熵外部记忆。**

具体新方向：

- 训练无关 n-gram LM；
- 非参数残差记忆；
- 语义可寻址 PLE；
- 只 gate 长尾/低熵；
- 与 RAG、teacher、base 组成多源系统。

---

## 3. 做的尝试

### 3.1 多轮搜索调研

完成多轮检索，覆盖：

- PLE/Engram 架构；
- n-gram LM 与神经 LM 融合；
- 训练无关记忆模块（NGM 等）；
- 冻结模型外部稀疏记忆（Ordo-M、Prometheus Mind、Memory Grafting）；
- RAG / ReAugKD / DRAG；
- OPD / Purified OPSD；
- QLoRA / LoRA / MoRA / GaLore / ReLoRA；
- PERK / test-time LoRA；
- 层次化记忆；
- 边缘/缓存/量化；
- Blackwell 信息序、信息瓶颈、资源分配等数学工具。

### 3.2 代码尝试

| 内容 | 结果 |
|---|---|
| `src/qwen35_ple/ngram_lm.py` | 完成 |
| `tests/test_ngram_lm.py` | 3 passed |
| `scripts/run_lora_distill.py` | 完成并 smoke 通过 |
| RAG hybrid/serving | 已完成并 smoke 通过 |
| 多任务评测 harness | 已加入 greedy exact-match |

### 3.3 数学推导

完成：

- 通道有效性排序：input ≥ logit ≥ hidden；
- 最优 logit 修正 = 条件对数似然比；
- Hidden 注入受 Jacobian 列空间限制；
- 多源 log-linear 融合是凸优化；
- 资源性价比选择 \(I/c\) 最高的源；
- N-gram 最优插值系数：
  \[
  \lambda^* = \frac{\mathrm{Cov}(L_t-L_b,\;L_n-L_b)}{\mathrm{Var}(L_n-L_b)}
  \]
- 自蒸馏收益受 teacher 噪声上界约束。

### 3.4 教师蒸馏准备

- 确认 Qwen3.8-Flash-Next 为 176B/6B MoE；
- 8GB GPU 可跑，但需约 48GB RAM；
- 当前 WSL 15GB RAM 不足；
- 推荐解耦：高 RAM/云导出 teacher，本地只训练 0.8B；
- 已跑通离线 teacher-text LoRA smoke：
  - 30 条 math/code；
  - trainable params = 540,672；
  - 10 步 loss ≈ 1.76。

---

## 4. 踩过的坑

| # | 坑 | 解决/状态 |
|---|---|---|
| 1 | PLE 作为语义记忆多次失败 | 重新定位为 n-gram/残差/长尾记忆 |
| 2 | 纯 logit head 初始 scale=0 导致梯度消失 | 改为 scale=1，head 零初始化 |
| 3 | HTTP smoke 生成 64 token 超时 | 降为 16 token 后通过 |
| 4 | 当前 WSL 无完整 Qwen3.8 teacher | 采用离线 teacher 解耦方案 |
| 5 | dense embedding 只是 token mean-pool | 记录为限制，后续换真正 sentence embedding |
| 6 | 本地无 torch/多个依赖 | 实验放在 WSL 远程环境 |
| 7 | RAG 检索质量受语料限制 | 后续用混合检索/rerank/高质量语料改进 |

---

## 5. 完成的内容

### 5.1 代码

- `src/qwen35_ple/ngram_lm.py`
- `tests/test_ngram_lm.py`
- `scripts/run_lora_distill.py`
- `scripts/run_rag_demo.py`（hybrid 版）
- `scripts/serve_rag_http.py`
- `scripts/smoke_rag_http.py`
- `src/qwen35_ple/rag.py`（Chunk/HybridRetriever/RRF）
- `src/qwen35_ple/serving/rag.py`（RAGServingAdapter）
- 混合检索 + 分块 + metadata + stop/prompt 控制

### 5.2 文档

- `round-64-end-to-end-routes-and-ple-usage.md`
- `round-65-teacher-distillation-with-current-resources.md`
- `round-66-running-qwen38-teacher.md`
- `round-67-research-routes-limited-resources.md`
- `round-68-ple-as-ngram-memory.md`
- `round-69-ple-paths-10plus-searches.md`
- `round-70-most-effective-path-math.md`
- `round-71-engram-vs-llm-intelligence.md`
- `round-72-systematic-rethink-v2.md`
- `round-73-full-summary.md`

### 5.3 实验/产物

- NgramLM 单元测试通过；
- LoRA teacher-text 蒸馏 smoke 通过；
- RAG hybrid demo 通过；
- HTTP serving `/health` 和 `/answer` 通过；
- 多任务评测 harness 加入 greedy exact-match。

---

## 6. 未完成的内容

| # | 未完成 |
|---|---|
| 1 | PLE 在低熵/代码/专名任务上的 real>control |
| 2 | N-gram λ* 实验 |
| 3 | \(I(Y;C\mid E_{\text{ngram}})\) 度量 |
| 4 | 非参数残差记忆 |
| 5 | 语义可寻址 PLE |
| 6 | 多源凸 router |
| 7 | RAG self-distillation |
| 8 | 真实 Qwen3.8 teacher logits 蒸馏 |
| 9 | GSM8K/MATH/HumanEval/MBPP 真实任务集 |
| 10 | 3-seed 全面覆盖 |
| 11 | CPU 100 tok/s serving |
| 12 | PLE 联合小规模预训练 |

---

## 7. 未来计划

### Phase PLE-1：证明 PLE 的真正价值

- 建低熵/代码/专名/数字评测；
- NgramLM real vs control；
- 估计 λ*；
- 测量 \(I(Y;C|E_{\text{ngram}})\)。

### Phase PLE-2：PLE 主创新架构

- 非参数残差记忆；
- 语义可寻址 PLE；
- 长尾 gate；
- 多源凸 router。

### Phase CAP-1：能力提升主线

- RAG self-distillation；
- 数据筛选 + QLoRA/MoRA；
- Qwen3.8 离线 teacher / OPD / Purified OPSD。

### Phase CAP-2：混合系统集成

- 多源消融；
- 3-seed；
- 污染审计。

### Phase PROD：产品化

- 量化/GGUF/ExecuTorch；
- CPU 100 tok/s；
- bundle/manifest/e2e。

---

## 8. 借鉴矩阵

| 项目 | 借什么 | 为什么不冲突 |
|---|---|---|
| XMemTransfer / Memory Grafting | 可寻址外部记忆 | PLE 作为外部记忆接口 |
| Ordo-M / Prometheus Mind | 冻结模型 + 稀疏记忆 | 支持 PLE 主创新 |
| NGM / 经典 n-gram LM | 训练无关 logit 插值 | 只作局部专家 |
| DeepSeek Engram / Qwen PLE | n-gram 查表、容量卸载 | 重新定义为互补记忆 |
| RAG / ReAugKD | 输入通道、teacher | PLE 可做词法 key |
| OPD / Purified OPSD | 学生轨迹 + teacher | 与 PLE 并行 |
| QLoRA / LoRA / MoRA | 低资源 adaptation | 用于能力提升 |
| PERK | test-time 适应 | 长上下文补充 |
| Hierarchical Memory | 长尾与推理分离 | 正合 PLE 定位 |
| 信息论/统计决策 | CMI、Blackwell 序、λ* | 用于门禁 |

---

## 9. 关键提交

本阶段相关提交包括：

```text
846d47d docs(strategy): end-to-end improvement routes and correct PLE role
bc1562f feat(distill): add LoRA teacher-text distillation runner and smoke
ac7cfac docs(teacher): how to run Qwen3.8-Flash-Next with limited GPU
082386a docs(routes): comprehensive low-resource technical route map
d16ff43 docs(ple): 10+ search survey of PLE usage paths
92d939e docs(math): derive most effective path with multi-view theory
1ad0e4e docs(theory): deep reflection on Engram vs LLM intelligence and PLE role
45ca330 docs(strategy): systematic rethink v2 with PLE as first-class innovation
```

---

## 10. 一句话总结

> 本轮完成了从“PLE 失败了”到“PLE 用错了”再到“PLE 应该作为可寻址残差/长尾外部记忆”的完整重构。
>
> 下一步不是放弃 PLE，而是先验证它真正擅长的局部/低熵/长尾任务，再把 PLE 做成主架构，同时用 RAG/蒸馏提升 0.8B 的实际能力。
