# Round 116：与 Graph of States 论文的批判性对比

> 目标：对比 https://arxiv.org/pdf/2603.21250 与当前 `paper.pdf`，识别差距。

---

## 1. 总体规模

| 维度 | 参考论文 | 当前论文 |
|---|---|---|
| 页数 | 21 页 | 7 页 |
| 章节数 | 多，且每章子节很细 | 中等 |
| 参考文献 | 约 40+ 条 | 25 条 |
| 图 | 多张大图：架构图、流程图、场景图、敏感性分析图 | 4 张 matplot 图 |
| 表 | 多张详细实验表 | 5 张简洁表 |
| 案例研究 | 有：真实医疗诊断、分布式系统故障案例 | 无详细 case study |
| 算法/伪代码 | 有状态机、算法描述 | 无伪代码 |

---

## 2. 章节结构对比

### 参考论文结构（约）

```text
Abstract
1. Introduction
2. Related Work / Background
3. Method
   - formal definitions
   - state machine
   - algorithm
4. Experiments
   - setup
   - results
   - real-world case studies
5. Discussion / Conclusion
Acknowledgement
References
Appendix
```

### 当前论文结构

```text
Abstract
1. Introduction
2. Related Work
3. Background and Theory
4. Method
5. Experimental Setup
6. Results
7. Analysis
8. Limitations
9. Conclusion
Appendix: Project Development Timeline
References
```

结构框架相似，但每个章节的 **厚度** 差距很大。

---

## 3. 内容深度

### 参考论文更强的地方

1. **问题定义严谨**：
   - 明确定义 deductive / inductive / abductive 任务差异；
   - 列出四种失败模式；
   - 用图/示例解释。

2. **方法可复现**：
   - 给出 state machine、transition rules、算法伪代码；
   - 说明 multi-agent 协作流程；
   - 给出复杂度/成本讨论。

3. **实验丰富**：
   - 两个真实数据集；
   - 详细 baselines；
   - 每个数据集有 setup、result、case study；
   - 有 sensitivity analysis；
   - 有成本/延迟对比。

4. **案例详尽**：
   - 医疗诊断案例；
   - 分布式系统故障案例；
   - 每个案例都解释“为什么失败/成功”。

### 当前论文不足

1. **方法太“黑盒”**：
   - 只有 PLE Projector 的高层描述；
   - 没有算法伪代码；
   - 没有 state/transition 或 inference pipeline 图。

2. **实验像“报告”**：
   - 每个实验只有结果表；
   - 缺少实验任务的具体定义；
   - 缺少 case study；
   - 缺少 ablation / sensitivity analysis。

3. **正负结果解释不够深**：
   - “PLE 在 code 上有效”；
   - 但没有逐 token/错误模式分析。

---

## 4. 写作风格

参考论文：

- 更学术、更自信；
- 每段都有明确论点；
- 使用 “As shown in Figure X”、“Specifically”、“The core mechanism”；
- 行文是“提出问题 → 给出机制 → 证明/解释 → 实验验证”。

当前论文：

- 像是 extended abstract / technical report；
- 有结果，但在解释“为什么”上不够展开；
- 缺少对 failure mode 的详细讨论；
- 缺少与相关工作的深度对比。

---

## 5. 图表情况

### 当前缺的图

- PLE 系统架构图；
- PLE Projector 流程图；
- token policy / router 决策图；
- 数据规模-性能曲线（100 / 1k / 10k）；
- n-gram order 消融图；
- memory size 消融图；
- 失败/成功案例图；
- HumanEval 通过/失败对照图；
- 开放生成退化案例图。

### 当前已有

- 10k per-seed improvement 柱状图；
- HumanEval pass@1 / repetition 柱状图；
- Joint system 柱状图；
- LLM judge 柱状图。

---

## 6. 引文数量

当前：25 条。

参考论文：明显更多，且分布在正文每个段落。

需要提升：

- 在 Method 中引用 memory / PLE / Engram / XMemTransfer / NGM / MemSFT；
- 在 Analysis 中引用 kNN-LM negative result / RAG / log-opinion-pool；
- 在 Limitations 中引用 relevant work；
- 目标：40–50 条。

---

## 7. 排版

当前已经使用 ICML Typst 模板，双栏、标题、作者、摘要、页码、running header 和参考样式都接近参考论文。

主要差距不是排版，而是 **内容密度**。

---

## 8. 提升优先级

| 优先级 | 改进 |
|---|---|
| P0 | 增加系统架构图 + PLE Projector 流程图 |
| P0 | 增加算法/伪代码或 inference pipeline |
| P0 | 增加 case study（HumanEval 成功/失败、TriviaQA 失败、开放生成退化） |
| P1 | 增加数据规模曲线、n-gram order、memory size 消融 |
| P1 | 将引用扩充到 40+ |
| P1 | 增加 Threats to Validity / Broader Impact |
| P2 | 每个实验补充详细 setup、baseline 说明、指标定义 |
| P2 | 增加 appendix 中的完整结果表 |

---

## 9. 结论

> 当前论文的“骨架”已经接近会议论文，但“肉”还不够厚。
> 主要缺：
> 1. 体系结构和算法图；
> 2. 真实案例与分析；
> 3. 消融/敏感性实验；
> 4. 更多引用和更深入的写作。
