# Round 93：任务条件 Router + Log-Density Gate + 校准配置持久化

> 日期：2026-09-05  
> 状态：P0 关键组件落地  
> 结论：实现可审计的查询任务分类、检索通道权重路由、log-density-ratio 门控，并把真实 base-logit 校准参数持久化为 serving JSON。

---

## 1. 本轮完成

### 1.1 src/qwen35_ple/router.py

新增/扩展：

| 组件 | 说明 |
|---|---|
| TaskClassifier | 规则式查询任务分类：semantic / code / name / number / general，可审计、可替换为学习式分类器 |
| TaskRouter | 在任务分类基础上返回每类任务的检索通道权重（BM25/Dense/N-gram） |
| LogDensityRatioGate | 实现 round-89 的 E[log(p_m/p_b)]>0 门控；支持 expected_kl、pseudo_label、memory_top1、hybrid 四种模式 |
| TaskConditionedNgramLogitProcessor | 生成期融合处理器：语义任务直接关闭 PLE，局部/低熵任务先过 density gate，再应用校准后的 scale/bias/temperature |
| save/load_fusion_router_config | JSON 配置持久化与加载 |
| build_task_conditioned_processor | 从配置构建生成期处理器 |
| build_task_router_from_config | 从配置构建查询级 router |

### 1.2 检索路由

HybridRetriever 新增 set_channel_weights(...)；RAGServingAdapter 在每次 answer() 前按查询任务动态设置：

- semantic：Dense 权重最高，N-gram 检索权重为 0；
- code / name / number / low_entropy：N-gram/PLE 权重提升；
- general：保持默认三通道均等。

### 1.3 校准配置持久化

新建：

```text
configs/ngram-fusion-router.json
```

当前取自 outputs/fusion-calibration.json 的 real temp_scale_bias：

```json
{
  "fusion": {
    "scale": 1.0,
    "bias": -1.0,
    "temperature": 0.5,
    "enabled": true
  },
  "router": {
    "mode": "expected_kl",
    "min_log_density_ratio": 0.0,
    "semantic_tasks": ["semantic", "knowledge", "qa"],
    "ple_tasks": ["code", "name", "number", "low_entropy"],
    "task_scale": {},
    "channel_weights": {},
    "classifier": {}
  }
}
```

scripts/run_fusion_calibration.py 新增 --router-config，每次校准后自动把最优 real 参数写入该 JSON。

### 1.4 Serving 接入

RAGServingAdapter 支持：

```python
RAGServingAdapter(
    model,
    tokenizer,
    retriever,
    ngram_memory=mem,
    fusion_config="configs/ngram-fusion-router.json",
)
```

- 自动构建 TaskRouter 和 TaskConditionedNgramLogitProcessor；
- 自动按查询任务设置检索通道权重和 PLE 生成门控；
- 返回结果中附带 task 字段。

scripts/run_rag_demo.py 与 scripts/serve_rag_http.py 均已接入该配置路径。

---

## 2. 数学依据

沿用 round-89 的判别条件：

    E_p_t[log p_m(Y|H) - log p_b(Y|H)] > 0

在 serving 中：

1. 无法直接获得真实 p_t，因此用任务级先验替代：语义任务关闭 PLE，局部/低熵任务才允许；
2. 用 E_p_m[log p_m - log p_b] 作为运行时“记忆相对 base 是否更尖锐/更偏向局部”的代理；
3. 用 pseudo_label_log_density_ratio 作为保守的“是否和当前 base 最佳 token 一致”信号，防止语义任务意外激活。

---

## 3. 测试

- 新增 TaskClassifier / TaskRouter / LogDensityRatioGate 单测；
- 新增任务条件处理器语义关闭、代码激活测试；
- 新增配置保存/加载、构建处理器和 router 测试；
- 新增 HybridRetriever.set_channel_weights 测试；
- 新增 RAGServingAdapter 自动构建测试。

当前：

```text
89 passed, 9 skipped, 1 xfailed
```

---

## 4. 仍待完成（P0 后半）

1. 多源完整消融：base / +RAG / +PLE / +MoRA / +all；
2. 3-seed 任务 router 评测；
3. 正式评测集：GSM8K / MATH / HumanEval / MBPP；
4. 扩大校准样本，把 control 伪信号压住；
5. 把 channel_weights 与 min_log_density_ratio 从“规则默认值”升级为“证据校准值”。
