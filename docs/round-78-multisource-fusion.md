# Round 78：多源 log-linear 融合与 n-gram 校准工具

> 日期：2026-09-05  
> 状态：基础工具落地，单元测试通过  
> 目标：解决 round-74 发现的“raw log P_ngram 直接加到 base logits 需要校准”的问题，并为 PLE-2 多源 router 提供可复用组件。

---

## 1. 新增模块

`src/qwen35_ple/fusion.py`

- `fuse_ngram_logits(base_logits, ngram_probs, scale, bias, temperature)`
  - 支持 n-gram 温度、缩放、偏置；
  - 明确处理“只调整 n-gram 命中的 token”这一稀疏 prior 形式。
- `calibrate_ngram_fusion(...)`
  - 在小样本上网格搜索最优 `(scale, bias)`；
  - 返回 NLL 下降（nats / bits）。
- `weight_logit_sources(base_logits, source_logits, weights)`
  - 多源 log-linear 融合。
- `mixture_distribution(distributions, weights)`
  - 稀疏类别分布混合。
- `softmax`
  - 稳定 softmax。

---

## 2. 为什么需要这些工具

Round-74 小样本 base fusion 显示：

- real n-gram 相对 control 有明显信息（0.43 bits vs 0.005 bits）；
- 但最优单 λ 为负且贴边界，说明：
  1. n-gram 概率尺度与 base raw logits 不匹配；
  2. 缺少对“n-gram 候选 token 族”的整体偏置；
  3. 需要可学习的 temperature / scale / bias / gate。

`fusion.py` 就是为后续 PLE-2 router 提供的最小可实现基元。

---

## 3. 测试

新增 `tests/test_fusion.py`：

- n-gram 插值会保留高概率 token 的排序；
- 校准能在合成数据上降低 NLL；
- 多源 logit 融合正确；
- sparse mixture 归一正确；
- softmax 正常。

```text
15 passed（含 fusion + rag + addressable）
```

---

## 4. 下一步

1. 用真实 base logits 小样本跑 `calibrate_ngram_fusion`，对比：
   - 单 λ；
   - scale + bias；
   - scale + bias + temperature；
   - hidden-dependent gate。
2. 把校准结果接入 `RAGServingAdapter` / 多源 router；
3. 在 RAG + PLE 三通道上做消融。

---

## 5. 一句话

> 从“负 λ 异常”到“可校准的多源融合工具”，PLE-2 的工程基础又补齐一块。
