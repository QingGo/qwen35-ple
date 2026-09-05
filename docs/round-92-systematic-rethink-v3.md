# Round 92：系统性复盘 v3——终极目标、技术债、借鉴矩阵与更稳的开发计划

> 日期：2026-09-05  
> 状态：基于本 session 全部实验与调研形成  
> 定位：在 PLE-1/PLE-2/CAP-1 证据之后，重新校准终极目标、确认技术债、安排后续优先级，并明确可借鉴但互不冲突的外部工作。

---

## 1. 终极目标

### 1.1 北极星

> **以 PLE/外部稀疏可寻址记忆为核心创新，构建一个低资源、可复现、可审计、可部署的 0.8B 混合智能系统，并尽可能提升其端到端能力。**

### 1.2 三层量化目标

| 层次 | 目标 | 验收 |
|---|---|---|
| 科学 | 清楚 PLE/n-gram 能提供什么、不能提供什么 | real vs control、CMI、任务分域 |
| 创新 | 实现“可寻址外部记忆 + 多源任务 router” | PLE-2 完整架构、消融、3-seed |
| 产品 | 0.8B + RAG + PLE + adapter 可部署 | CPU 100 tok/s、bundle、e2e、量化 |

### 1.3 当前核心认知

1. PLE/n-gram 是**局部低熵/代码/专名/词法记忆**，不是语义知识记忆；
2. 语义/知识应由 RAG/Dense 承担；
3. 模型能力提升主要来自参数化 post-training：RAG self-distill、OPD/OPSD、MoRA/QLoRA；
4. 多源系统必须按任务 router，不能固定融合；
5. 我们的独特价值：把 Engram/PLE 从“失败语义记忆”改造成“可审计的可寻址外部记忆”。

---

## 2. 本轮 session 发现的技术债

### 2.1 科学/证据债

| # | 技术债 |
|---|---|
| S1 | PLE 在代码/专名/数字通过，但实体/语义 QA 明确失败，尚未形成完整“任务边界地图” |
| S2 | 真实 base logits 校准样本只有少数几个，且 control 出现伪信号 |
| S3 | 缺少正式 GSM8K/MATH/HumanEval/MBPP 评测 |
| S4 | PLE 3-seed 只覆盖 addressable/entity，未覆盖多源融合 |
| S5 | CAP-1 只有 logprob/held-out，exact match/pass@k 仍为 0 |
| S6 | 新 RAG self-distill 数据未做完整 contamination audit |

### 2.2 架构/方法债

| # | 技术债 |
|---|---|
| M1 | 校准后 n-gram 融合已接入 adapter，但未持久化最优参数到 serving |
| M2 | 没有任务条件 gate/router：语义任务应关闭 PLE，低熵/代码才激活 |
| M3 | RAG 第三通道已实现，但未与 PLE logit fusion 做联合消融 |
| M4 | 实体 value 测试为负，未设计“实体/知识走 RAG、词法走 PLE”的完整分工 |
| M5 | 尚未把 CAP-1 OPD/Purified OPSD 化：当前只是外部 RAG self-distill |
| M6 | MoRA/QLoRA 已跑通，但未做多 seed 和正式评测 |

### 2.3 工程/产品债

| # | 技术债 |
|---|---|
| E1 | 没有 CPU 100 tok/s serving 闭环 |
| E2 | 没有 bundle/manifest/e2e 部署验证 |
| E3 | 远程 WSL 训练产物未全部归档/可复现（outputs 未入 git） |
| E4 | CI 已修，但新实验脚本尚未统一纳入测试/评测协议 |
| E5 | RAG dense embedding 仍是弱 token mean-pool，不是 sota sentence embedding |

---

## 3. 开发计划

### Phase PLE-2c：任务边界与 router（P0）

1. 建立任务分类评测：
   - semantic/QA：Dense/RAG；
   - code/name/number/low-entropy：PLE/n-gram；
   - general：base。
2. 实现 **log-density ratio gate**：
   - 使用 \(E[\log(p_m/p_b)]>0\) 作为是否激活 PLE 的判据；
3. 将校准参数持久化：
   - `fusion_scale/bias/temperature` 写入配置/JSON；
   - `RAGServingAdapter` 加载并应用；
4. 多源 router 消融：
   - base / +RAG / +PLE / +MoRA / +all；
   - 3-seed。

**Gate**：语义任务不因 PLE 退化；代码/专名任务 PLE real>control；各源贡献可解释。

### Phase CAP-2：能力提升与正式评测（P1）

1. 扩大 RAG self-distill 数据到 500–1000；
2. 引入验证过滤 → Purified OPSD；
3. 保留通用/知识数据，避免 CAP-1 后 knowledge 下降；
4. 跑正式评测：
   - GSM8K / MATH / HumanEval / MBPP；
   - knowledge/entity 子集；
5. LoRA / QLoRA / MoRA 多 seed 对比。

**Gate**：正式评测上至少一个任务显著正提升；不牺牲通用知识；污染审计通过。

### Phase PROD：产品化（P2）

1. 量化/GGUF/ExecuTorch；
2. CPU 100 tok/s benchmark；
3. bundle + manifest；
4. e2e：RAG + PLE + adapter + serving；
5. 可复现脚本 + 配置 + 版本记录。

**Gate**：100 tok/s；bundle 可加载；e2e 回归通过。

---

## 4. 可借鉴但不冲突的项目矩阵

| 项目 | 借鉴什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| DeepSeek Engram / Qwen PLE | 稀疏查表、条件记忆、n-gram 地址 | 不把 n-gram 当语义预测器 | 我们是“外部可寻址记忆”，角色互补 |
| NGM / kNN-LM | 训练无关 logit 插值 | 不期待开放式生成全面提升 | 只用于低熵/局部任务 |
| Memory Grafting / Lngram | offline conditional memory、latent memory | 不复制大规模预训练 | 可作为未来 latent 化升级 |
| TF-Engram | SSD 大容量、预测预取 | 不追求本地 SSD 工程 | 未来容量扩展 |
| ReAugKD | 检索增强蒸馏 | 不把 RAG 当唯一能力来源 | CAP-1 的训练策略 |
| OPD / Purified OPSD | 自采样 + 验证过滤 + 教师蒸馏 | 不要求大规模 RL | 比 RL 更适合 0.8B |
| RAGRouter / L-RAG | 学习式/熵式查询路由 | 不直接照搬其训练成本 | 多源 router 参考 |
| RAG scaling laws | 检索规模/收益递减 | 不盲目增大 top-k | 决定 retriever 预算 |
| MoRA / QLoRA / DoRA | 高秩/量化参数适配 | 不把它们当记忆架构 | 与外部记忆互补 |
| ROME / MEMIT / INLAY | 直接权重/外部记忆编辑 | 不用于通用能力训练 | 只用于知识微调/清理 |
| TokenMem / MemSFT / RETRO | token 级记忆通道 | 不抢主模型推理职责 | 可用于未来记忆接口 |
| Sparse Delta Memory | 稀疏状态/门控记忆 | 不替换当前架构 | 未来长上下文/状态扩展 |
| Evolution Strategies | 无梯度直接优化 | 当前成本高，不优先 | 可作 CAP 备选 |
| DPO / ORPO / KTO | 无 RL 偏好优化 | 不与 PLE 记忆冲突 | 后期对齐备选 |

---

## 5. 如何更稳地前进

### 5.1 坚持证据门禁

- 每个新组件必须有 paired real/control；
- 3-seed；
- contamination audit；
- 正式评测，不只 val loss；
- 不把 PLE 当“万能记忆”。

### 5.2 保持可复现

- 每个实验脚本入库；
- 数据、配置、manifest 明确；
- 远程训练产物最好用 `outputs/` + `docs/` 记录关键指标；
- CI 覆盖测试与 lint。

### 5.3 先小后大

- 路由/门控用少量真实 logits 验证；
- 再扩展数据；
- 再产品化。

### 5.4 停止条件

1. 如果 PLE 在代码/专名/低熵任务上连 real>control 都不成立 → 不把 PLE 作为主创新点；
2. 如果 CAP-1 在正式评测无正提升且通用能力下降 → 停止继续烧算力，先做数据/验证；
3. 如果 PLE/RAG fusion 导致语义任务退化 → 关闭该通道，只保留低熵 gate。

---

## 6. 当前优先级

| 优先级 | 事项 |
|---|---|
| P0 | 任务条件 router + log-density gate |
| P0 | 持久化校准参数并接入 serving |
| P1 | 正式评测集（GSM8K/MATH/HumanEval/MBPP） |
| P1 | CAP-1 升级为 Purified OPSD |
| P1 | LoRA/QLoRA/MoRA 多 seed + 完整消融 |
| P2 | CPU 100 tok/s / bundle / e2e |
| P2 | 更强 dense embedding |
| P3 | latent n-gram / SSD 大容量记忆 |

---

## 7. 一句话

> 终极目标不是“让 PLE 成为万能记忆”，而是“用可审计的 PLE 外部记忆 + RAG + 参数化 post-training，构建一个低资源、按任务路由、可部署的 0.8B 混合智能系统”。当前最大技术债是“缺少任务 router 和正式评测”，下一步应先把 router、gate、校准持久化和正式评测做扎实，再进入产品化。
