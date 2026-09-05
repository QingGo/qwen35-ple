# Round 77：PLE-2 接入 RAG——N-gram 词法寻址作为第三检索通道

> 日期：2026-09-05  
> 状态：代码集成完成，单元测试通过  
> 目标：把 PLE-2 的可寻址记忆真正接入现有 RAG 产品栈，而不是停留在独立实验。

---

## 1. 新增内容

### 1.1 `NgramKeyRetriever`

`src/qwen35_ple/rag.py`

把 `AddressableNgramMemory` 包装成与 `BM25Index` 同接口的检索器：

```python
ngram = NgramKeyRetriever(
    memory,
    tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
)
ngram.search(query, top_k=5)
# -> [corpus_doc_index, ...]
```

### 1.2 `HybridRetriever` 三通道融合

现在 `HybridRetriever` 支持：

1. BM25 词法；
2. Dense 向量；
3. **N-gram 精确寻址（PLE）**。

三者统一走 RRF：

```python
retriever = HybridRetriever(
    bm25,
    dense_vectors,
    ngram_retriever=ngram,
    ngram_weight=1.0,
)
```

### 1.3 Demo / HTTP serving 参数

- `scripts/run_rag_demo.py`
- `scripts/serve_rag_http.py`

新增：

```bash
--use-ngram              # 启用 PLE n-gram 寻址
--ngram-weight 1.0       # RRF 中 n-gram 通道的权重
```

---

## 2. 为什么这是主创新落点

PLE-2 的定位是：

> 用稀疏精确 n-gram 作为离散地址，去访问外部知识/文档。

现在它不再只是评测原型，而是：

```text
查询
 ├─ BM25（词频）
 ├─ Dense（语义）
 └─ N-gram 精确地址（PLE）
      ↓
RRF 融合
      ↓
RAG 生成
```

这保留了 PLE 的“可寻址外部记忆”创新，同时不抢主模型推理职责。

---

## 3. 测试

新增 `test_ngram_key_retriever_in_hybrid`：

- 构造小 vocab + 两篇文档；
- `NgramKeyRetriever` 能按 n-gram 地址检索到正确文档；
- `HybridRetriever` 在加入 n-gram 后仍返回正确 top-1。

```text
14 passed（含 rag / addressable / ngram tests）
```

---

## 4. 局限与下一步

1. `NgramKeyRetriever` 当前直接使用整篇 chunk 作为 value，尚未做“函数块/实体条目”级别；
2. 尚未在真实 RAG 评测上对比 BM25 / Dense / Ngram / Hybrid 四线；
3. 尚未做 n-gram 通道的 3-seed 消融；
4. 下一步应测：
   - 加入 n-gram 后，rare/entity/code 类查询的检索命中是否上升；
   - 多源 router 的温度/权重校准。

---

## 5. 一句话

> PLE 已经从“独立 n-gram 实验”变成“RAG 混合检索的第三通道”，向产品化混合记忆系统又前进了一步。
