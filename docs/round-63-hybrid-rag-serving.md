# Round 63：混合检索、语料分块、rerank 与 serving 接入

> 日期：2026-09-04
> 状态：混合 RAG 产品化路径已完成第一版
> 目标：把 RAG 从“BM25 baseline”升级为可复用、可 serving 的混合检索路径。

---

## 1. 完成内容

### 1.1 语料分块与 metadata

新增：

- `Chunk` dataclass：
  - `text`
  - `doc_id`
  - `chunk_index`
  - `source`
- `chunk_text`
- `chunk_corpus`

支持：

- 按字符/近似 sentence 切分；
- overlap；
- 保留文档级 provenance。

### 1.2 混合检索

新增：

- `HybridRetriever`
- `reciprocal_rank_fusion`

混合方式：

- BM25 词法检索；
- Dense embedding：
  - 当前使用 backbone token embedding mean-pool 作为轻量 dense 向量；
  - 未来可替换为 sentence-transformer / contextual encoder；
- RRF 融合两个排名；
- 支持 `candidate_pool`，先取候选再 rerank。

### 1.3 Prompt / stopping 控制

- `RAGServingAdapter.build_prompt`
  - 可切换 `concise`；
  - 统一生成 `Context + Question + Answer`；
- `_generate`
  - `max_new_tokens`
  - EOS 停止
  - 可配置 `stop_sequences`

### 1.4 Serving 接入

新增：

- `src/qwen35_ple/serving/rag.py`
  - `RAGServingAdapter`
  - 统一 `retrieve` / `build_prompt` / `answer`
- `scripts/serve_rag_http.py`
  - 标准库 HTTP server；
  - `/health`
  - `/answer?q=...`

这为后续 vLLM/SGLang/CompileForge 替换 transport 提供了统一接口。

## 2. 烟测结果

- `run_rag_demo.py --mode hybrid` 可运行；
- 对 50 docs 小语料可完成分块、dense embedding、RRF、生成；
- HTTP server 代码已提供，可在 WSL 启动后通过 `/health` 和 `/answer` 访问。

## 3. 已知限制

1. Dense embedding 目前是静态 token embedding mean-pool，不是 sota sentence embedding；
2. 检索质量受限于当前 wikitext 语料和 BM25；
3. 0.8B 生成格式仍需蒸馏/SFT；
4. HTTP server 只是演示 transport，不是生产级并发/权限/鉴权。

## 4. 下一步

1. 接入真实 sentence-transformer 或模型 hidden embedding；
2. 增加 reranker（cross-encoder）；
3. 更高质量语料 + 分块策略调优；
4. 将 HTTP transport 替换为 vLLM/SGLang/CompileForge；
5. 继续 D2 教师蒸馏。

## 5. 产物

- `src/qwen35_ple/rag.py`
- `src/qwen35_ple/serving/rag.py`
- `scripts/run_rag_demo.py`
- `scripts/serve_rag_http.py`
- `tests/test_rag.py`
