# Round 121：终极目标、技术债与下一阶段开发计划

> 日期：2026-09-07
> 背景：论文已完成 11 页、56 条参考文献、正式 Appendix、Diagram Design 图表；接下来需要从“论文形式完善”转向“证据与可复现性完善”。

---

## 1. 终极目标

> **证明“可审计 n-gram 外部记忆在低资源小模型上能做什么、不能做什么”，并把它做成可复现、可部署、可审计的系统。**

具体子目标：

1. 清晰说明 PLE / n-gram 记忆的适用边界；
2. 在真实 benchmark 上给出足够规模的证据；
3. 提供可复现的实验环境、脚本、权重、评测卡；
4. 展示低资源部署可行性（CPU/量化/内存）；
5. 把论文、代码、artifact 做成一个完整可审计 package。

---

## 2. 本轮 session 进展

已完成：

- ICML 风格论文稳定在 11 页；
- 56 条参考文献，全部在正文引用；
- 新增：
  - Problem Setting / 形式化；
  - Algorithm 1/2；
  - Error Analysis / Sensitivity 框架；
  - Acknowledgement / Data Availability；
  - Appendix 正式 A/B/C 编号；
  - Diagram Design 风格图表；
- 排版问题已修复：
  - Appendix 单独起页；
  - Algorithm 上下边框间距调整；
  - 列表缩进收紧。

---

## 3. 当前技术债

### 3.1 实验证据仍是最大短板

| 技术债 | 现状 | 目标 |
|---|---|---|
| HumanEval | 20 题 | 50 题 |
| TriviaQA | 100 条，exact=0 | 200 条 |
| 10k Projector | 3 seeds | 5 seeds |
| pass@k | 3×2 小样本 | 更大样本 |
| LLM judge | 小样本 | 100/200 全量 |
| sensitivity | 只有框架表 | 实际 memory size / n-gram order / data size 曲线 |
| Joint System | 3-seed 部分结果 | 完整联合系统表 |

### 3.2 可复现性不足

- 未发布 PLE Projector / MoRA 权重；
- 未验证容器内完整训练/评测；
- 无 CPU 吞吐、量化、内存占用数据；
- CI 没有自动构建 PDF；
- 没有 artifact checksum / release note。

### 3.3 论文内容还有提升空间

- 每个 benchmark 还没有完整“Task Setup → Baselines → Results → Case Study”结构；
- 缺少真实 error distribution / failure taxonomy；
- 缺少 sensitivity 图；
- 缺少成本/延迟对比；
- 没有 Acknowledgement / Data Availability 的实践经验支撑。

### 3.4 工具与工程债

- Diagram Design 已引入，但还没有自动化 PNG 导出/视觉回归；
- `fetch_arxiv_refs.py` 已加入，但搜索流程还不是完整 pipeline；
- 远程 GPU 环境不稳定，实验可重复执行性待验证。

---

## 4. 下一阶段开发计划

### Phase A：证据补强（最优先）

1. 跑 HumanEval 50；
2. 跑 TriviaQA 200；
3. 10k 补 5 seeds；
4. 扩大 pass@k；
5. 补全 LLM judge；
6. 增加 memory size / n-gram order / data size 曲线；
7. 增加 token policy on/off 消融。

### Phase B：可复现与 artifact

1. 发布 PLE Projector 权重；
2. 发布 Purified MoRA adapter 权重；
3. 统一 Dockerfile / 训练评测命令；
4. 增加 artifact checksum / release note；
5. CPU 吞吐与量化；
6. CI 增加 PDF 构建检查。

### Phase C：论文深化

1. 每个 benchmark 补完整实验闭环；
2. 增加真实 error distribution；
3. 增加 sensitivity / ablation 图；
4. 增加 cost / latency 对比；
5. 将 Appendix 作为正式技术附录。

### Phase D：投稿准备

1. 最终格式检查；
2. 统计显著性；
3. LLM/human judge 一致性；
4. 选择 venue：
   - TMLR
   - ACL Findings
   - 低资源 workshop

---

## 5. 可借鉴的类似项目（不冲突）

| 项目 | 借鉴什么 | 不冲突理由 |
|---|---|---|
| DeepSeek Engram / Qwen PLE | 可扩展 n-gram lookup、gating、disk offload | 作为底层记忆基础设施 |
| Memory Grafting | offline memory 构建、target-side reader | 作为外部记忆生成方法 |
| XMemTransfer | 跨模型 reader 适配 | 作为迁移/适配参考 |
| MemSFT / TokenMem | token router、distribution-level memory | 作为安全 gate 和融合策略参考 |
| kNN-LM 后续工作 | datastore 可靠性、open-ended 失败原因 | 作为非参数记忆的对照 |
| RAG / Self-RAG / FAIR-RAG | 自适应检索、可靠性判断 | 作为检索通道和 router 参考 |
| MemGPT / HippoRAG / From RAG to Memory | 长时记忆、非参数持续学习 | 作为长期记忆方向参考，不替代局部 n-gram |
| Lngram / Tensorizing Engram | latent n-gram、离散可解释表示 | 作为记忆表示优化方向 |
| PEFT（LoRA/QLoRA/MoRA） | 参数化能力增强 | 作为互补能力轴 |
| Diagram Design | 图表设计规范、anti-patterns | 用于论文表达，不改变科学方法 |
| TMLR / RepoEval | 可复现、评测卡、artifact | 用于发布标准 |

---

## 6. 核心战略

1. **证据优先**：论文形式已经足够，接下来必须补实验。
2. **边界优先**：写清楚“PLE 不是通用语义记忆”，比无限扩大方法更有价值。
3. **可复现优先**：权重、容器、脚本、评测卡必须齐全。
4. **少而深**：把 HumanEval 50 / TriviaQA 200 / 10k 5 seeds 做透，比堆更多弱实验更好。
5. **低资源优先**：每一步都考虑 CPU、量化、内存、单卡可运行。
6. **不冲突架构**：
   - PLE/Engram = 可审计局部记忆；
   - RAG = 文档级证据；
   - Adapter = 参数化能力；
   - Learned router/policy = 安全控制；
   - 四者互补，不是替代。

---

## 7. 当前最优先的一件事

> **在可用的 GPU 环境上跑完 HumanEval 50、TriviaQA 200、10k 5 seeds，并输出完整的 sensitivity/error analysis 数据。**
>
> 这是从“论文形式完善”走向“科学结论可信”的关键一步。
