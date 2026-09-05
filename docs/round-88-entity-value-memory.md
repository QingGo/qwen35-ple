# Round 88：实体 value（QA snippet）3-seed 记忆评测

> 日期：2026-09-05  
> 状态：完成，结果偏弱  
> 结论：用 QA snippet 作为实体 value 时，n-gram 精确寻址对“从问题预测答案首 token”几乎无信号，再次说明 PLE 适合局部词法，而非语义知识。

---

## 1. 实验

新增 `scripts/run_ple2_entity_memory_eval.py`

- 实体 value：`question + answer` 组成的 QA snippet；
- 数据集：`data/qa-expanded-150.json`；
- 80/20 train/eval；
- 3 seeds（0/1/2）；
- 指标：答案首 token 在 n-gram top-1/3/5 的命中率；
- 对照：同 value 内部打乱 token 顺序。

## 2. 结果

| Seed | N | real@1 | control@1 | real@3 | control@3 | real@5 | control@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.033 | 0.000 |
| 2 | 30 | 0.033 | 0.000 | 0.033 | 0.000 | 0.033 | 0.000 |
| mean | - | 0.011 | 0.000 | 0.011 | 0.000 | 0.022 | 0.000 |

## 3. 解读

- real 略高于 control，但绝对水平极低（≤2.2%）；
- 这符合 PLE/n-gram 的定位：
  - 擅长代码、专名拼写、局部低熵 token；
  - 不擅长“语义 QA 知识检索/推理”；
- 实体/知识记忆应继续走 **Dense/RAG + 参数化模型**，而不是 n-gram 精确 value。

## 4. 对 PLE-2 的完成度

- 已完成 value 类型：
  - 语义 chunk（Wiki 段落）✅
  - 函数/类块（Code AST）✅
  - 实体条目（QA snippet）✅
- 3-seed：✅
- 结果差异：code/function 最强，entity 最弱，符合预期。

---

## 5. 下一步

1. 实体 value 不应作为 n-gram 主通道；
2. 应把实体/知识交给 RAG/Dense；
3. PLE n-gram 继续用于代码/专名/低熵局部任务；
4. 最终系统按任务 router 融合。
