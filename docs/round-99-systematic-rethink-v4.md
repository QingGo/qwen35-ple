# Round 99：系统性复盘 v4——终极目标、技术债、开发计划与借鉴矩阵

> 日期：2026-09-05  
> 状态：基于 P0 实测、多分支数学推导和形式化证明后的再次校准  
> 定位：在“PLE 当前无端到端正收益”的现实下，继续坚持 PLE 作为主创新的同时，把证据门槛、实现路径和产品化路线重新排序。

---

## 1. 终极目标

### 1.1 北极星

> 以 **可审计、可寻址的 PLE/外部稀疏记忆** 为核心创新，构建一个低资源、可复现、可部署的 0.8B 混合智能系统，并在真实局部任务上证明 PLE 的不可替代或至少可测增益，最终实现 CPU 100 tok/s 的端到端产品。

### 1.2 三层目标

| 层 | 目标 | 验收 |
|---|---|---|
| 科学层 | 明确 PLE 能做什么、不能做什么，并用数学证明 + 实验双重验证 | 每任务 Δ、real-control、正式评测 |
| 创新层 | PLE 作为“可寻址外部记忆 + 任务级 router”成为系统的一等公民 | 真正局部任务上 PLE real>control 且融合正收益 |
| 产品层 | 0.8B + RAG + PLE + adapter 可部署 | CPU 100 tok/s、bundle、e2e、可复现 |

### 1.3 当前核心认知

1. PLE/n-gram 是 **局部词法 / 低熵 / 可寻址外部记忆**，不是语义知识记忆；
2. 当前 P0 多源消融中 PLE 无正收益，且 arithmetic 为负；
3. 这不是“PLE 没有信息”，而是 **任务口径、记忆 bank、gate、校准都不匹配**；
4. RAG 对知识有效，MoRA 对 code-output 有效；
5. 最优方法 = 任务级验证筛选 + 凸融合校准 + 成本感知激活 + 在线源权重 + Purified OPSD。

---

## 2. 本轮 session 发现的技术债

### 2.1 科学/证据债

| # | 技术债 |
|---|---|
| S1 | 没有在真正的 PLE 强项任务上做端到端评测：当前只有代码题问答/算术计算 |
| S2 | PLE memory 不是同域 bank：用 CAP1 解题文本评测简单表达式/算术，支撑集严重不匹配 |
| S3 | 缺少 per-task \(\Delta_t\) 与 real/control paired 测量 |
| S4 | 没有正式 GSM8K / MATH / HumanEval / MBPP 评测 |
| S5 | 校准样本极小（4 样本 wiki），未按 code/name/number 分域校准 |
| S6 | 没有把“信息为正”和“受限融合可用”区分开，导致把 CMI 误当成充分条件 |

### 2.2 架构/方法债

| # | 技术债 |
|---|---|
| M1 | 当前 gate 使用 KL(p_m||p_b) 非负代理，不是真实 \(E[\log p_m-\log p_b]\) |
| M2 | 融合参数仍是全局单组，没有 per-task \((\lambda,\beta)\) |
| M3 | 没有实现支撑集质量校准（定理 3） |
| M4 | 没有成本感知激活规则（定理 4） |
| M5 | 只有规则 router，没有在线 Hedge/mirror descent 自适应 |
| M6 | CAP-1 仍是朴素 RAG self-distill，未升级 Purified OPSD |
| M7 | LoRA/QLoRA/MoRA 只有单次训练，无多 seed/正式对比 |
| M8 | RAG dense embedding 仍是 token mean-pool，不是语义级 dense |

### 2.3 工程/产品债

| # | 技术债 |
|---|---|
| E1 | 没有 CPU 100 tok/s serving 闭环 |
| E2 | 没有 bundle/manifest/e2e |
| E3 | 远程 WSL 训练产物未完全归档 |
| E4 | 新脚本和配置已入库，但尚未形成统一评测协议 |
| E5 | 多源融合结果未进入 serving 的 per-task 配置 |

---

## 3. 开发计划

### Phase PLE-2d：证据修复与窄口径验证（最高优先）

这是当前最关键的 Phase。

1. 构建真正的 PLE 强项评测：
   - HumanEval / MBPP next-token 或代码续写；
   - 专名/实体拼写；
   - 日期/数字格式；
   - 低熵局部模板。
2. 构建同域 memory bank：
   - 代码 bank：源码/函数体；
   - 专名 bank：实体语料；
   - 数字 bank：日期/号码/格式文本。
3. 测量每个任务：
   \[
   \Delta_t=E[\log p_m(Y)-\log p_b(Y)]
   \]
   以及：
   \[
   \Delta_{\text{real}}-\Delta_{\text{control}}
   \]
4. 只有同时满足：
   \[
   \Delta_t>0,\quad \Delta_{\text{real}}-\Delta_{\text{control}}>0
   \]
   才保留该任务的 PLE。
5. Per-task 校准：
   - 每个任务优化 \((\lambda,\beta)\)；
   - 用支撑集质量定理初始化/校准 \(\beta\)；
   - 保存 per-task 配置。
6. 多源消融 v2：
   - 包含真实局部任务；
   - 包含 real/control；
   - 3 seed；
   - 如果 PLE 在真局部任务上通过，则维持主创新定位；否则继续降级为“低优先级实验性组件”。

### Phase CAP-2：能力提升与正式评测

1. 正式评测集：
   - GSM8K / MATH / HumanEval / MBPP；
   - 知识/实体子集。
2. CAP-1 升级 Purified OPSD：
   - 自采样；
   - 验证/过滤；
   - 教师蒸馏；
   - 防止自增强坍缩。
3. LoRA / QLoRA / MoRA 多 seed 正式对比。
4. 保留通用/知识数据，防止 knowledge regression。
5. 引入在线 Hedge router 或至少 per-task 验证权重。

### Phase PROD：产品化

1. 量化 / GGUF / ExecuTorch；
2. CPU 100 tok/s benchmark；
3. bundle + manifest + e2e；
4. per-task fusion config 进入 serving；
5. 可复现脚本 + 数据 manifest + 版本记录。

---

## 4. 可借鉴但不冲突的项目

| 项目/方向 | 借鉴什么 | 不拿什么 | 为什么不冲突 |
|---|---|---|---|
| kNN-LM reliability | “何时依赖检索”的门控 | 不把 kNN 当语义记忆 | 告诉我们 PLE 不能无条件使用 |
| NGM | 免训练即插即用记忆 | 不复制其 embedding 方案 | 可作 PLE 未来接口 |
| MemSFT | 冻结 backbone + 只训 token-level router | 不引入外部参数化记忆替代 PLE | 支持“分布层融合 + router” |
| TokenMem | 独立记忆通道 + conflict gate | 不把长上下文全塞进 cross-attention | 与 PLE 的稀疏寻址互补 |
| Memory Grafting | offline conditional memory + target reader | 不追求大规模预训练 | 未来可做 PLE 的 latent 化升级 |
| RAG as Noisy ICL / Local Sufficiency | 理解何时检索有收益 | 不把 RAG 当唯一智能来源 | 为任务级 router 提供理论 |
| RAGRouter / L-RAG | 查询路由 / 不确定性触发 | 不照搬其训练成本 | 本地可用规则+验证过渡 |
| Log Opinion Pool / Bayesian fusion | 专家必须先校准再融合 | 不认为“多源必然更好” | 防止 PLE 负收益进入系统 |
| Rate-Distortion memory | 记忆大小/成本权衡 | 不无限扩容量 | 决定 PLE bank 规模 |
| Hedge / Prediction with Expert Advice | 轻量在线源权重 | 不替代正式评测 | 可作为 router 第二层 |
| Optimal Control / MDP | 把检索当控制输入 | 不做复杂在线规划 | 指导“何时激活” |
| Purified OPSD | 验证过滤 + 自蒸馏 | 不盲信自生成 | 防止 CAP 自增强坍缩 |
| MoRA / DoRA | 高秩/结构化参数化 | 不把它们当记忆架构 | 与 PLE 外部记忆互补 |
| ROME / MEMIT / AlphaEdit | 直接知识编辑 | 不用于通用能力训练 | 只在知识修复场景可用 |

---

## 5. 如何更稳地前进

### 5.1 证据门禁

每个 PLE 相关结论必须同时有：

- per-task Δ；
- real vs control；
- 3 seed；
- 正式指标（不只是 logprob）；
- 污染审计。

### 5.2 分阶段停止条件

1. 如果真实局部任务上 PLE real 仍不超过 control → 不再把 PLE 当作主创新；
2. 如果 CAP-1/Purified OPSD 在正式评测无正提升且通用能力下降 → 停止烧算力，先修数据；
3. 如果 PLE/RAG fusion 在语义任务导致退化 → 关闭该通道。

### 5.3 最小闭环优先

先做一个“窄口径 but 完整”的 PLE 证据闭环：

```text
同域 bank → 真局部任务 → per-task Δ → per-task calibration → 3-seed ablation → serving config
```

再扩展规模和产品化。

---

## 6. 当前优先级

| 优先级 | 事项 |
|---|---|
| P0 | 真实局部任务评测集 + 同域 PLE bank |
| P0 | per-task Δ / real-control / per-task \((\lambda,\beta)\) |
| P0 | 多源消融 v2 |
| P1 | 正式评测集 + Purified OPSD |
| P1 | MoRA/QLoRA 多 seed 对比 |
| P1 | 在线 Hedge router |
| P2 | CPU 100 tok/s / bundle / e2e |
| P2 | 更强 dense embedding / 语义记忆 |

---

## 7. 一句话

> 我们现在的核心任务不是继续堆方法，而是 **先把 PLE 在真正的局部任务上证明可用**；如果证明不了，就诚实降级，把 PLE 保留为可审计的实验性外部记忆，而不是产品必选项。  
> 只有完成这个证据闭环，后续的 RAG、MoRA、Purified OPSD 和产品化才有稳固基础。
