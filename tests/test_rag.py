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
