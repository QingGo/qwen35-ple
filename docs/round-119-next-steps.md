# Round 119：当前状态与下一步计划

> 日期：2026-09-07
> 目标：在论文图表、标题、邮箱、公式、引文和 PDF 构建均已稳定之后，系统性规划后续工作。

---

## 1. 当前已完成

- 论文已使用 ICML Typst 模板，8 页；
- 标题精简为一行：
  - `Auditable N-Gram Memory for Small Language Models`
- 作者邮箱已改为实际邮箱；
- 公式已正常渲染；
- 正文引文统一为 `et al.`；
- System Architecture / Inference Pipeline 已使用 `diagram-design` 重新绘制；
  - 正交连线、分区标签、无遮挡、无文字溢出；
  - 保留了 HTML 源文件，可用 `make paper-diagrams` 重新导出 SVG；
- 删除了旧的 Fletcher/CeTZ vendor 依赖；
- 新增 `scripts/extract_dd_svgs.py`，并纳入 CI ruff 检查；
- PDF 根目录保持只有 `paper.pdf`。

---

## 2. 仍存在的问题

### 2.1 实验证据仍偏弱

| 项目 | 当前 | 目标 |
|---|---|---|
| HumanEval | 20 题 | 50 题 |
| TriviaQA | 100 条，exact match = 0 | 200 条 |
| 10k projector | 3 seeds | 5 seeds |
| pass@k | 3×2 小样本 | 更大样本 |
| LLM-as-judge | HumanEval 20、TriviaQA 20 | TriviaQA 100/200 |
| 敏感性曲线 | 无 | memory size / n-gram order / data size |
| 真实联合系统完整表 | 部分 | 完整 Joint System 表 |

### 2.2 论文内容深度仍不足

- 缺 Algorithm 1/2 伪代码；
- 缺真实 case study；
- 缺 sensitivity / ablation 图；
- 引文约 25 条，目标 40+；
- 缺 Threats to Validity；
- 缺 Broader Impact；
- 缺完整附录（逐题结果、完整 seed 表）。

### 2.3 可复现性不足

- 未发布 projector / adapter 权重；
- 未验证容器内完整训练/评测；
- 无 CPU/量化吞吐数据；
- CI 未构建 PDF；
- artifact 无 checksum / release note。

---

## 3. 优先级排序

| 优先级 | 事项 | 理由 |
|---|---|---|
| P0 | 补实验证据 | 论文当前核心边界结论基于小样本，reviewer 会质疑 |
| P1 | 补方法形式化与案例分析 | 不需要 GPU，能明显提升论文完整度 |
| P2 | 补可复现 artifact | 影响可信度和投稿要求 |
| P3 | 补系统部署数据 | 影响“低资源可部署”主张 |

---

## 4. 下一步具体任务

### Phase 0：证据优先（1–2 周）

1. HumanEval 扩到 50；
2. TriviaQA 扩到 200；
3. 10k Projector 补到 5 seeds；
4. pass@k 扩大到更多样本；
5. LLM judge 补全 TriviaQA 100/200；
6. 增加：
   - data size 曲线：100 / 1k / 10k；
   - n-gram order 消融；
   - memory size 消融；
   - token policy 开关消融。

### Phase 1：论文厚度（1 周）

1. 写 Algorithm 1：PLE Projector 推理；
2. 写 Algorithm 2：Token Policy + 安全 Gate；
3. 加一个真实 case study：
   - HumanEval 成功/失败；
   - TriviaQA 失败；
   - open-ended 退化案例；
4. 引文扩到 40+；
5. 增加 Threats to Validity；
6. 增加 Broader Impact；
7. 增加附录完整结果表。

### Phase 2：可复现与 artifact（1 周）

1. 发布 PLE Projector 权重；
2. 发布 Purified MoRA adapter 权重；
3. 统一容器训练/评测；
4. 增加 artifact checksum / release note；
5. 增加 CPU 吞吐与量化数据；
6. CI 增加 PDF 构建检查。

### Phase 3：投稿准备

1. 最终格式检查；
2. 统计显著性与 bootstrap CI；
3. 人工/LLM judge 一致性分析；
4. 附录完整；
5. 选定投稿 venue：
   - TMLR / ACL Findings / 低资源 workshop。

---

## 5. 当前最优先的一件事

> **先把 HumanEval 扩到 50、TriviaQA 扩到 200，并补 10k 5 seeds。**
>
> 这是所有后续论文写作和投稿的基础；没有这些证据，论文再美观也容易被审稿人质疑。

---

## 6. 检查标准

- [ ] 所有主要结果都有 3–5 seeds / 足够样本；
- [ ] NLL、pass@1、exact match、LLM judge 均有报告；
- [ ] 有 sensitivity / ablation 图；
- [ ] 有算法伪代码；
- [ ] 有至少 1 个详细 case study；
- [ ] 参考文献 ≥ 40；
- [ ] 公开权重、容器、评测卡、checksum；
- [ ] 论文可在 CI 中自动构建 PDF。
