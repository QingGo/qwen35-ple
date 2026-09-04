"""Tests for the reusable lightweight RAG module."""

from __future__ import annotations

from qwen35_ple.rag import BM25Index, build_rag_prompt, tokenize


def test_tokenize() -> None:
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_bm25_retrieves_relevant_doc() -> None:
    docs = [
        "The capital of France is Paris.",
        "The largest planet is Jupiter.",
        "Python is a programming language.",
    ]
    index = BM25Index(docs)
    hits = index.retrieve("capital of France", top_k=1)
    assert hits == ["The capital of France is Paris."]


def test_rag_prompt() -> None:
    prompt = build_rag_prompt("context text", "What is X?")
    assert "Context:" in prompt
    assert "What is X?" in prompt
    assert "Answer:" in prompt


def test_chunk_text_produces_metadata() -> None:
    from qwen35_ple.rag import chunk_text

    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, doc_id=7, chunk_size=50, overlap=10, source="test")
    assert len(chunks) > 1
    assert all(c.doc_id == 7 for c in chunks)
    assert all(c.source == "test" for c in chunks)
    assert chunks[0].chunk_index == 0


def test_rrf_reranks_hybrid() -> None:
    from qwen35_ple.rag import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]])
    # 2 appears first in one list and third in the other; should be top or near top.
    assert fused[0] in (0, 2)
    assert set(fused[:3]) == {0, 1, 2}


def test_hybrid_retriever_combines_bm25_and_dense() -> None:
    import numpy as np

    from qwen35_ple.rag import BM25Index, HybridRetriever

    docs = [
        "The capital of France is Paris.",
        "The largest planet is Jupiter.",
        "Python is a programming language.",
    ]
    bm25 = BM25Index(docs)
    dense = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    retriever = HybridRetriever(bm25, dense)
    hits = retriever.retrieve("capital France", top_k=1, query_vector=np.array([1.0, 0.0, 0.0]))
    assert hits == ["The capital of France is Paris."]
