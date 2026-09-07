# Round 120：与参考论文的章节级对比与改善清单

> 对比对象：https://arxiv.org/pdf/2603.21250（Graph of States）
> 当前论文：`paper.pdf`（8 页，25 条参考文献）

---

## 1. 章节组织对比

| 维度 | 参考论文 | 当前论文 | 差距 |
|---|---|---|---|
| 篇幅 | 21 页 | 8 页 | 内容深度不足 |
| 参考文献 | 约 40+ 条 | 25 条 | 需扩充 |
| 图 | 14 处引用 | 6 张图 | 缺敏感性/案例图 |
| 表 | 12 处引用 | 4 张表 | 缺完整结果表 |
| 算法 | 2 个正式 Algorithm | 2 个 Algorithm | 已补上 |
| 案例研究 | 主文/附录都有详细 case | 仅附录简短 case | 需加强 |
| 数据可用性 | 有数据/隐私声明 | 无 | 需补 |

---

## 2. 参考论文章节结构

```text
Abstract
1. Introduction
2. Related Work
   - 推理框架
   - 神经符号推理
   - 多智能体
3. Methodology
   - 问题定义与形式化
   - 双层系统定义
   - 双向神经符号交互
   - 状态转换
4. Experiments
   - 4.1 医疗诊断
     - Task Setup
     - Baselines
     - Results
     - Sensitivity
     - Case Study
   - 4.2 分布式系统故障诊断
     - Task Setup
     - Baselines
     - Results
     - Sensitivity
     - Case Study
5. Conclusion
Acknowledgement
References
Appendix
  A. Algorithm
  B. Limitations
  C. Error Analysis / Case Details
```

---

## 3. 当前论文章节结构

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
10. Appendix
    A. Algorithm 1/2
    B. Real-vs-Control Protocol
    C. HumanEval passes
    D. TriviaQA case
    E. Open-ended case
    F. Limitations and Future Work
References
```

---

## 4. 每个章节的差距与改进建议

### 4.1 Introduction

- 参考论文：清楚列出四种失败模式，并用图说明动机。
- 我们：研究问题和贡献清楚，但缺少“问题动机图/失败模式图”。
- 建议：
  - 增加一张“朴素 PLE 融合会出什么问题”的动机图；
  - 强化“为什么现有多数记忆方法在大模型上，而我们要研究小模型”的转折；
  - 增加一段“本文与现有系统的差异”论述。

### 4.2 Related Work

- 参考论文：每个相关工作都有批判性对比。
- 我们：目前较简洁。
- 建议：
  - 扩到 40+ 条引用；
  - 把 5 条未使用的文献用上：
    - `longbench2024`
    - `gsm8k2021`
    - `math2021`
    - `ragrouter2026`
    - `memtrap2026`
  - 增加对 Memory Layers、TokenMem、XMemTransfer、MemSFT 的“为什么不够/为什么不同”对比。

### 4.3 Background and Theory

- 参考论文：有严格形式化定义、符号、状态机。
- 我们：有背景和理论，但缺少统一“问题设定”小节。
- 建议：
  - 增加 `Problem Setting`：
    - 冻结 LM；
    - n-gram memory；
    - logit fusion；
    - 可审计性定义；
    - real/control 定义。
  - 增加更多形式化符号表。

### 4.4 Method

- 参考论文：每一节都有“核心机制 + 为什么”。
- 我们：已有 Method，但有些像模块介绍。
- 建议：
  - 增加对 Algorithm 1/2 的正文引用；
  - 给每个模块加“设计动机/为什么不用替代方案”；
  - 增加数据流描述：“As shown in Figure 1…”。

### 4.5 Experiments

- 参考论文：每个数据集都有完整章节。
- 我们：按指标组织，缺少“每个 benchmark 一个完整实验故事”。
- 建议重新组织：
  - 6.1 Local Continuation
  - 6.2 HumanEval
  - 6.3 TriviaQA
  - 6.4 Joint System
  - 6.5 Open-Ended / Safety
  - 6.6 Sensitivity / Ablation
- 每个实验都包含：
  - Task Setup
  - Baselines
  - Results
  - Case / Analysis

### 4.6 Analysis

- 参考论文：有错误类型统计、定量分布、失败机制表。
- 我们：高层分析，缺图表。
- 建议：
  - 增加 error type / failure mode 表；
  - 增加 LLM judge 一致性分析；
  - 增加 token policy 开关消融；
  - 增加 HumanEval 成功/失败对照。

### 4.7 Limitations / Conclusion

- 参考论文：Limitations 在附录，结论更自信并有未来方向。
- 我们：主文有 Limitations，已经较诚实。
- 建议：
  - 增加数据可用性 / 复现声明；
  - 增加 Acknowledgement；
  - 结论增加“未来 scaling”方向。

### 4.8 Appendix

- 已从 Phase 流水账改成技术 Appendix。
- 可继续学习：
  - 参考论文 Algorithm 后有详细案例和错误分布；
  - 我们应增加：
    - 完整 seed 表；
    - HumanEval 逐题结果；
    - TriviaQA 示例；
    - LLM judge 重复一致性；
    - memory-size / n-gram-order 敏感性图。

---

## 5. 引文差距

当前：

```text
论文引用：57 次
唯一引用：23 个
参考文献：25 条
未使用：5 条
```

目标：

```text
参考文献：40+ 条
```

需要新增的主题：

- 小模型/低资源 LLM；
- 记忆增强 / 外部记忆；
- 检索增强；
- 非参数化 LM/kNN；
- 参数高效微调；
- 模型审计 / 可解释性；
- 长上下文与记忆注入失败；
- LLM-as-judge；
- 代码生成 benchmark。

---

## 6. 优先级

| 优先级 | 项目 |
|---|---|
| P0 | 将 Results 重组为按 benchmark 的完整实验章节 |
| P0 | 引文扩到 40+，使用未引用文献 |
| P1 | 增加 Problem Setting / 形式化 |
| P1 | 增加 error/failure 分析和 LLM judge 一致性 |
| P1 | 增加 sensitivity/ablation 图 |
| P2 | 增加 Acknowledgement / Data Availability |
| P2 | 将 Appendix 改为字母编号，模仿参考论文 |
