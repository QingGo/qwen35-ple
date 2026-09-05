# Round 81：校准后的 n-gram 融合接入 Serving/Router

> 日期：2026-09-05  
> 状态：完成  
> 内容：独立 logit processor + `RAGServingAdapter` 可选集成 + demo/http 参数。

---

## 1. 新增

### `src/qwen35_ple/router.py`

```python
CalibratedNgramLogitProcessor(
    memory,
    scale=...,
    bias=...,
    temperature=...,
)
```

- 接收当前生成上下文，查 `AddressableNgramMemory.continuation_distribution`；
- 用 `fusion.fuse_ngram_logits` 对 base logits 做校准注入；
- 支持 torch tensor / numpy；
- 可 `enabled=False` 关闭。

### `RAGServingAdapter` 可选 `logit_processor`

```python
adapter = RAGServingAdapter(
    ...,
    logit_processor=calibrated_ngram_router,
)
```

在每一步生成时：

```python
logits = model(...).logits[0, -1]
if logit_processor is not None:
    logits = logit_processor(logits, generated)
nxt = argmax(logits)
```

---

## 2. Demo / HTTP 参数

- `scripts/run_rag_demo.py`
- `scripts/serve_rag_http.py`

新增：

```bash
--use-ngram-fusion
--fusion-scale 1.0
--fusion-bias 0.0
--fusion-temperature 1.0
```

当 `--use-ngram --use-ngram-fusion` 同时开启时，PLE n-gram 同时参与：

1. 混合检索（NgramKeyRetriever）；
2. 生成阶段的 logit 校准融合（CalibratedNgramLogitProcessor）。

---

## 3. 测试

新增 `tests/test_router.py`：

- disabled 时原样返回；
- 校准参数能改变 argmax；
- state_dict 可导出/持久化。

测试结果：14 passed（router + rag + fusion）。

---

## 4. 意义

这是 PLE-2 首次进入实际 serving 路径：

```text
检索：BM25 + Dense + N-gram/PLE
生成：base logits + calibrated n-gram log-prior
```

为后续“任务 router / 长尾 gate / 多源凸融合”提供了可直接替换的接口。

---

## 5. 下一步

1. 扩大校准样本并保存最优参数；
2. 在 code/name/number 任务上对比：
   - 无 fusion；
   - single λ；
   - scale+bias；
   - temperature+scale+bias；
3. 加入任务条件 gate（长尾/低熵时激活，语义任务关闭）；
4. 跑真实 RAG 问答生成消融。
