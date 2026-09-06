# Round 113：完成 P0/P1 后再评估：离论文还有多远？

> 日期：2026-09-06
> 状态：基于最新 P0/P1 结果的第二轮系统评估
> 方法：结合已有证据 + 多轮 web 调研，重新判断论文完整度。

---

## 1. 我们刚刚补上了什么

### 1.1 真实基准

- 真实 `openai/humaneval` 前 5 题；
- `BM25 + PLE` 得到 1 个 pass；
- 这是第一个真实公开基准上的正面信号。

### 1.2 外部记忆基线

- 实现小型 hidden-state kNN-LM；
- 基线结果：hit 提升、NLL 退化；
- 说明当前 kNN-LM 不是强基线。

### 1.3 统计

- 5 seed；
- paired rows；
- bootstrap 95% CI；
- projector vs fixed CI `[0.025, 0.187]`，排除零。

### 1.4 联合系统表

- base / RAG / PLE / MoRA / RAG+MoRA / All；
- 3 seed 聚合，已形成系统级证据。

---

## 2. 现在的“完成度”

如果按一篇 **边界/系统论文** 的标准：

| 项目 | 状态 |
|---|---|
| 提出清晰问题 | ✅ |
| 方法可运行 | ✅ |
| 真实基准最小子集 | ⚠️ 只有 5 题 HumanEval |
| kNN-LM 基线 | ✅ 小型版 |
| 5 seed + CI | ✅ |
| 联合系统表 | ✅ |
| 错误分析 | ⚠️ 有部分，不足 |
| 生成质量指标 | ⚠️ pass@1 雏形，缺 NQ/TriviaQA/LLM judge |
| 可复现 artifact | ⚠️ 有脚本，缺公开权重/数据/容器 |
| 效率数据 | ❌ |
| 理论-实验闭环 | ⚠️ 有理论，缺实验映射 |
| 更大规模记忆 | ❌ 10k 未训练 |

---

## 3. 距离发表的剩余关键缺口

### 3.1 真实基准规模不够

当前 HumanEval 只有 5 题，审稿人不接受。

最低要求：

```text
HumanEval: 至少 20–50 题
或 NQ / TriviaQA: 100–200 条，带 exact match
或两者都做
```

### 3.2 外部记忆强基线不足

我们只做了一个“小型 kNN-LM”，而且它不强。

论文要比较：

- NGM（训练无关 memory module）
- MemSFT / TokenMem（如果有资源）
- Memory Grafting target-side reader（如果可运行）
- 至少一个当代外部记忆方法

如果这些基线全部效果不佳，可以写成：

> “我们证明在 0.8B 低资源场景下，简单/主流外部记忆方法也未能超越 RAG 或 adapter。”

这也是有价值的负结果。

### 3.3 更大数据规模

当前：

- 1k 训练仅 seed0；
- 10k dataset 已构建但未训练；
- 没有容量曲线。

需要：

```text
100 / 1k / 10k 训练规模  × 3–5 seed
记忆规模 100 / 1k / 10k / 100k
```

至少需要一条“数据规模 → 收益”曲线。

### 3.4 生成质量指标

目前大量使用 teacher-forced NLL。

论文至少需要：

- HumanEval / MBPP pass@k；
- NQ / TriviaQA exact match；
- 重复率 / 结构性；
- LLM-as-judge + 与人类一致性；
- 真实生成文本案例。

### 3.5 完整系统在真实任务上的联合表

当前联合系统表来自内部 rare-kb / 代码输出任务。

如果把论文定位为“系统论文”，需要把：

```text
base / RAG / PLE / MoRA / all
```

放到：

```text
真实 HumanEval 子集
真实 NQ / TriviaQA 子集
```

上，至少各 100 条。

### 3.6 可公开复现

需要：

- 公开 adapter 权重；
- 公开 HumanEval / NQ 处理脚本；
- 公开记忆表 / 数据 manifest；
- 容器或 lockfile；
- artifact 评估 checklist；
- 运行时间。

### 3.7 效率

如果论文保留“低资源可部署”主张，需要：

```text
CPU tok/s
内存占用
检索延迟
量化损失
```

当前 CPU 只有约 2 tok/s，尚且不满足。

### 3.8 理论验证

仓库有很多理论，但对论文而言，可选：

- Blackwell 序 / rate-distortion 的实验验证；
- log-linear fusion 是否近优；
- 记忆规模-性能曲线。

TMLR/ML 会议加分；ACL 类不一定必需。

---

## 4. 最小可行论文还有多远？

### 如果投 TMLR / ACL Findings / 低资源 workshop

还差：

1. HumanEval 20–50 题 + NQ 或 TriviaQA 100 条；
2. NGM 或 MemSFT 基线（任选一个可运行）；
3. 10k 数据 3–5 seed；
4. 一个生成质量表（exact match / pass@1 / judge）；
5. 公开 artifact 清单 + 容器；
6. 坦诚的边界章节。

预计工作量：**3–6 周**。

### 如果投 NeurIPS / ICML / ICLR

还差：

1. 5M–20M token 规模记忆实验；
2. 强外部记忆 baseline 的直接对比；
3. 真实完整基准上的稳定正收益；
4. 模型规模扩展；
5. 理论-实验闭环；
6. 完整可复现。

预计工作量：**数个月**，且现有资源风险很高。

---

## 5. 建议

当前最现实的目标不是“证明 PLE 是通用记忆增强”，而是：

> **“在低资源 0.8B 模型中，可审计 n-gram 外部记忆到底能提供什么、不能提供什么；以及什么时候它值得与 RAG、adapter 联合使用。”**

建议下一步：

1. 扩大 HumanEval 到 20–50 题；
2. 下载并跑 NQ 或 TriviaQA 100 条；
3. 跑 10k 数据 3 seed；
4. 实现或接入 NGM 基线；
5. 补齐结构化 artifact/评测卡；
6. 如果 PLE 在真实任务上仍只是“局部小收益”，就把它写成边界/系统论文，而不是主创新论文。

---

## 6. 参考

- [TMLR Acceptance Criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [ReproEvalCard](https://aclanthology.org/2026.acl-short.22/)
- [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [MINTEval](https://huggingface.co/papers/2605.18565)
- [MemSFT](https://github.com/LUMIA-Group/MemSFT)
- [XMemTransfer](https://github.com/OLAResearch/XMemTransfer)
- [kNN-LM](https://papers.lunadong.com/paper/4555)
- [Assessing small language models for code generation](https://acm-stag.literatumonline.com/doi/10.1016/j.jss.2026.112815)
