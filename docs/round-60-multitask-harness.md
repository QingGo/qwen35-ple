# Round 60：多任务评测 harness 初步搭建

> 日期：2026-09-04
> 状态：已建立最小多任务评测入口，结果仅为烟测
> 目标：后续 RAG / 蒸馏 / 记忆方法都用同一评测入口比较。

---

## 1. 为什么需要

现在只有 rare-kb 知识问答。  
后续主路径是 RAG + 教师蒸馏，必须知道：

- 是否只提升知识问答；
- 是否提升推理/简单计算；
- 是否提升代码/表达式理解；
- 是否在多个任务上稳定。

## 2. 已实现

新增：

```text
scripts/run_multi_task_eval.py
```

当前任务：

| 任务 | 数据 |
|---|---|
| knowledge | rare-kb QA |
| arithmetic | 生成式四则运算 |
| code-output | 简单 Python 表达式求值 |

指标：

- answer-token average logprob；
- first-token hit；
- 可选：knowledge 的 RAG 条件。

## 3. 烟测结果

运行 50 knowledge + 10 arithmetic + 10 code：

| 任务 | n | no-context logprob | first-hit |
|---|---:|---:|---:|
| knowledge | 50 | -1.982 | 0.0 |
| arithmetic | 10 | -7.287 | 0.0 |
| code-output | 10 | -14.250 | 0.0 |

烟测说明：

- 当前 frozen 0.8B 在生成式短答案上 first-token 命中很低；
- 这更强调后续需要使用生成式 exact-match / 解码评测，而不是只依赖 logprob；
- 也说明“小模型 + 无检索/蒸馏”在数学/代码上确实弱。

## 4. 后续完善

1. 接入真实 GSM8K / MATH / HumanEval / MBPP 格式；
2. 使用 greedy/beam decoding 统计 exact-match；
3. 每个方法跑 3 seeds；
4. 加入 real/control / RAG / distilled 四条件；
5. 污染审计。

## 5. 产物

- `scripts/run_multi_task_eval.py`
- WSL output：`outputs/multi-task-eval-smoke.json`
