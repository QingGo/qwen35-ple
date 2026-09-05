# Round 75：PLE-2 第一步——可寻址 n-gram 外部记忆原型

> 日期：2026-09-05  
> 状态：PLE-2 架构代码落地，单元测试通过  
> 目标：在 PLE-1 证明 n-gram 局部词法记忆有效后，把 PLE 做成“可寻址非参数外部记忆”，而不是仅仅作为一个 log-prob prior。

---

## 1. 新增模块

`src/qwen35_ple/addressable_memory.py`

### 1.1 核心类

`AddressableNgramMemory`

- **key**：离散、精确、稀疏的 token n-gram；
- **value**：外部值（文档 id / 语料块 / 实体 id）；
- 同时维护：
  - `next_counts`：n-gram → 下一个 token 的经验分布；
  - `value_index`：n-gram → 外部 value 的命中计数。

### 1.2 主要 API

```python
mem = AddressableNgramMemory(min_order=2, max_order=4)

# 建立索引，value_id 指向外部文档/知识块
mem.add_document(tokens, value_id=0)

# 获取局部 continuation（top-k + 命中阶数）
mem.topk(context, k=5)

# 按 n-gram 地址检索外部值
mem.retrieve(context, top_k=5)
```

### 1.3 特性

- 训练无关；
- 非参数；
- 可审计（整数 key + 计数）；
- 可直接作为 RAG 的“词法 key 检索通道”；
- 可扩展到 PLE 真实表 value 向量 / 文档向量。

---

## 2. 测试

新增 `tests/test_addressable_memory.py`：

1. continuation 分布正确；
2. 最长 n-gram 匹配能检索到正确 value；
3. miss 返回空；
4. 同一文档的所有 n-gram 都可被检索到。

```text
7 passed (含 ngram_lm)
```

---

## 3. 为什么这是 PLE 主创新的下一步

PLE-1 已经证明：

> 真实有序 n-gram 在 code / name / number 上明显超过打乱 control。

但 PLE 如果只是 `log P_ngram` 插值，仍容易被 base model 的 logits 尺度问题淹没（round-74 小样本 λ 贴边界）。

PLE-2 的做法是：

1. 用 n-gram 做 **离散地址**；
2. value 不是“下一个 token 概率”，而是 **外部知识/文档/代码片段/实体条目**；
3. 在推理时：
   - 先用 n-gram 精确寻址；
   - 再通过 router/gate 决定是否从外部 value 取残差；
   - 形成“非参数残差记忆”。

---

## 4. 下一步

1. 写 `ple2_addressable_eval`：
   - real vs control 的 top-k continuation recall；
   - retrieved value 中是否包含真实 continuation 的命中率；
2. 把 `AddressableNgramMemory` 接入 RAG：
   - `HybridRetriever` 增加 n-gram key 路由；
3. 实现多源凸 router：
   - base + RAG + n-gram + teacher；
   - 学习式温度/门控，解决 round-74 的 λ 校准问题；
4. 在代码/专名/数字任务上跑完整 PLE-2 消融。

---

## 5. 一句话

> PLE 从“记忆概率”升级为“可寻址外部记忆仓库”：离散 n-gram 是地址，外部文档/实体是值。
