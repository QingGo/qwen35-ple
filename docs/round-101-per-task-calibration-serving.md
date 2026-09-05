# Round 101：Per-task 校准参数持久化并接入 Serving

> 日期：2026-09-05  
> 状态：完成  
> 结论：把 round-100 得到的 per-task \((\lambda,\beta)\) 写入 `configs/ngram-fusion-router.json`，并让 `TaskConditionedNgramLogitProcessor` 按任务加载不同融合参数。

---

## 1. 修改内容

### 1.1 `src/qwen35_ple/router.py`

`TaskConditionedNgramLogitProcessor` 新增：

```python
per_task_fusion: dict[str, dict[str, float]]
```

在生成时，如果当前任务命中：

```python
per_task_fusion[task]
```

则使用该任务的：

- `scale`
- `bias`
- `temperature`

否则回退到全局参数和 `task_scale`。

`build_task_conditioned_processor` 自动从配置读取：

```python
router.fusion_per_task
```

### 1.2 配置

`configs/ngram-fusion-router.json` 新增：

```json
"fusion_per_task": {
  "code":   { "scale": 0.5, "bias": 3.0, "temperature": 0.5 },
  "name":   { "scale": 1.0, "bias": 2.4, "temperature": 0.5 },
  "number": { "scale": 0.0, "bias": 0.0, "temperature": 1.0 }
}
```

参数来源：

- `outputs/ple-evidence-base-s0/s1/s2.json`
- 每个任务 3-seed 校准参数的均值。

---

## 2. 为什么这样改

### 2.1 上一版问题

全局参数：

```json
scale = 1.0
bias = -1.0
```

来自 4 样本 wiki 校准，不能代表 code/name/number 的真实最优融合。

### 2.2 Round-100 实测

| 任务 | 平均最优 scale | 平均最优 bias |
|---|---:|---:|
| code | 0.5 | 3.0 |
| name | 1.0 | 2.4 |
| number | -0.25 | 0.96 |

其中：

- code/name：真实融合正收益；
- number：真实融合接近零，所以配置为 `scale=0`，即等效关闭。

---

## 3. 测试

新增：

```python
test_task_conditioned_processor_uses_per_task_fusion
```

验证：

- per-task 参数会被使用；
- state_dict 包含 `per_task_fusion`。

当前测试：

```text
13 passed (test_router.py)
```

---

## 4. 后续

1. 如果有新的 per-task 校准，直接更新 `fusion_per_task`；
2. Serving 中 `TaskConditionedNgramLogitProcessor` 会自动加载；
3. future：可把 per-task 校准扩展到 temperature，甚至按代码语言/任务细分。
