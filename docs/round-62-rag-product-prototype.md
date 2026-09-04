# Round 62：RAG 产品化原型

> 日期：2026-09-04
> 状态：已建立最小可复用 RAG 模块和单条查询 demo
> 目标：从“评测脚本”走向“可复用的 RAG 产品路径”。

---

## 1. 完成内容

### 1.1 可复用 RAG 模块

新增：

```text
src/qwen35_ple/rag.py
```

包含：

- `tokenize`
- `load_corpus`：支持 txt / jsonl
- `BM25Index`：纯 Python/NumPy BM25
- `build_rag_prompt`

### 1.2 单条查询 demo

新增：

```text
scripts/run_rag_demo.py
```

使用：

```bash
python scripts/run_rag_demo.py \
  --model data/models/Qwen3.5-0.8B \
  --corpus data/sources/wikitext.jsonl \
  --question "Who is Nikola Tesla?"
```

流程：

1. 加载 frozen 0.8B；
2. BM25 检索 top-k；
3. 拼接 Context + Question + Answer；
4. greedy 生成回答。

## 2. 烟测结果

- Demo 可运行；
- 当前 BM25 检索质量一般：对 “Nikola Tesla” 检索到的片段并不理想；
- 模型生成了较长的 thinking-style 文本，说明 **0.8B 本身在生成格式上还需要蒸馏/SFT**；
- 这验证了后续 D2 的必要性。

## 3. 下一步产品化工作

1. 混合检索：
   - BM25 + embedding/rerank；
2. 语料质量与分块：
   - 更高质量的 knowledge/code/math corpus；
   - 统一 chunk 与 metadata；
3. prompt 与生成控制：
   - 加 stopping / 简洁格式；
   - 针对知识问答使用短答案 prompt；
4. serving：
   - 把 RAG 路径接入 vLLM/SGLang/CompileForge；
5. 遥测：
   - retrieval hit rate；
   - answer exact-match；
   - latency。

## 4. 产物

- `src/qwen35_ple/rag.py`
- `scripts/run_rag_demo.py`
- `tests/test_rag.py`
