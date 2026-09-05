"""Reusable lightweight RAG components for the 0.8B product prototype.

This module is intentionally dependency-light: BM25 is implemented in pure
Python/NumPy, so it can be reused across eval scripts, serving adapters, and
future product integrations without pulling in a search framework.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    """Simple word-level tokenizer for BM25."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def load_corpus(path: str | Path, max_docs: int | None = None) -> list[str]:
    """Load a plain-text file or JSONL corpus."""
    path = Path(path)
    docs: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text") or obj.get("solution") or "")
                if text.strip():
                    docs.append(text.strip())
    else:
        docs = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    if max_docs is not None:
        docs = docs[:max_docs]
    return docs


class BM25Index:
    """Tiny, dependency-free BM25 index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(d) for d in docs]
        self.doc_len = np.asarray([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        for tokens in self.doc_tokens:
            c = Counter(tokens)
            self.tf.append(c)
            for term in c:
                self.df[term] += 1
        self.n = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[int]:
        q_tokens = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float64)
        for term in q_tokens:
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for i, c in enumerate(self.tf):
                tf = c.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * self.doc_len[i] / self.avgdl
                    )
                    scores[i] += idf * tf * (self.k1 + 1.0) / denom
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        return np.argsort(scores)[-k:][::-1].tolist()

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return [self.docs[i] for i in self.search(query, top_k)]


def build_rag_prompt(context: str, question: str) -> str:
    """Format retrieved context and a question for a frozen causal LM."""
    return (
        f"Use the following information to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


@dataclass(frozen=True)
class Chunk:
    """A chunked retrieval unit with lightweight provenance metadata."""

    text: str
    doc_id: int
    chunk_index: int
    source: str | None = None


def chunk_text(
    text: str,
    *,
    doc_id: int = 0,
    chunk_size: int = 800,
    overlap: int = 100,
    source: str | None = None,
) -> list[Chunk]:
    """Split a document into character-based chunks with metadata."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(text=text, doc_id=doc_id, chunk_index=0, source=source)]
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a newline/space near the boundary.
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + chunk_size // 2:
                end = cut + 1
        chunks.append(
            Chunk(
                text=text[start:end].strip(),
                doc_id=doc_id,
                chunk_index=idx,
                source=source,
            )
        )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        idx += 1
    return chunks


def chunk_corpus(
    docs: list[str],
    *,
    chunk_size: int = 800,
    overlap: int = 100,
    sources: list[str] | None = None,
) -> list[Chunk]:
    """Chunk all documents, preserving document provenance."""
    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        src = sources[i] if sources is not None else None
        chunks.extend(
            chunk_text(doc, doc_id=i, chunk_size=chunk_size, overlap=overlap, source=src)
        )
    return chunks


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[int]:
    """Combine multiple ranked lists by Reciprocal Rank Fusion."""
    if not rankings:
        return []
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class NgramKeyRetriever:
    """Use PLE-style exact n-gram addresses as an external lexical retrieval channel.

    The query is tokenized to subword ids; the last n tokens form an exact
    n-gram key; ``AddressableNgramMemory.retrieve`` returns external value ids
    (by default document indices) that contain those exact n-gram keys.
    """

    def __init__(
        self,
        memory,
        *,
        value_to_index=None,
        tokenizer=None,
        top_k: int = 5,
    ) -> None:
        self.memory = memory
        self.value_to_index = value_to_index or (lambda x: x)
        self.tokenizer = tokenizer or (lambda text: [])
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[int]:
        ids = self.tokenizer(query)
        if not ids:
            return []
        k = top_k or self.top_k
        matches = self.memory.retrieve(ids, top_k=k)
        return [self.value_to_index(m.value_id) for m in matches]



class HybridRetriever:
    """Combine BM25 lexical scores with dense embedding cosine scores.

    The dense vectors are provided externally (for example computed from the
    backbone's token embeddings).  This class only handles fusion and rerank.
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense_vectors: np.ndarray | None = None,
        *,
        dense_weight: float = 1.0,
        ngram_retriever=None,
        ngram_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.dense_vectors = dense_vectors
        self.dense_weight = dense_weight
        self.ngram_retriever = ngram_retriever
        self.ngram_weight = ngram_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        query_vector: np.ndarray | None = None,
        candidate_pool: int = 50,
    ) -> list[int]:
        bm25_ranking = self.bm25.search(query, top_k=candidate_pool)
        rankings = [bm25_ranking]
        weights = [1.0]
        if self.ngram_retriever is not None:
            ngram_ranking = self.ngram_retriever.search(query, top_k=candidate_pool)
            if ngram_ranking:
                rankings.append(ngram_ranking)
                weights.append(self.ngram_weight)
        if self.dense_vectors is not None and query_vector is not None:
            dense_scores = self.dense_vectors @ query_vector
            # Normalize each vector to unit length for cosine.
            dense_norms = np.linalg.norm(self.dense_vectors, axis=1)
            q_norm = np.linalg.norm(query_vector)
            if dense_norms.size and q_norm > 0:
                dense_scores = dense_scores / (dense_norms * q_norm)
            dense_ranking = np.argsort(dense_scores)[::-1][:candidate_pool].tolist()
            rankings.append(dense_ranking)
            weights.append(self.dense_weight)
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
        return fused[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        query_vector: np.ndarray | None = None,
        candidate_pool: int = 50,
    ) -> list[Chunk | str]:
        results = self.search(
            query,
            top_k=top_k,
            query_vector=query_vector,
            candidate_pool=candidate_pool,
        )
        return [self.bm25.docs[i] for i in results]


def mean_pool_embeddings(
    token_ids: list[list[int]],
    embedding_matrix: np.ndarray,
) -> np.ndarray:
    """Mean-pool token embeddings into document/query vectors.

    This is a lightweight static dense embedding baseline.  For production,
    replace it with a sentence-transformer or a contextual encoder.
    """
    vectors: list[np.ndarray] = []
    for ids in token_ids:
        if not ids:
            vectors.append(np.zeros(embedding_matrix.shape[1], dtype=np.float32))
            continue
        vectors.append(embedding_matrix[ids].mean(axis=0))
    return np.asarray(vectors, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


__all__ = [
    "BM25Index",
    "Chunk",
    "HybridRetriever",
    "NgramKeyRetriever",
    "build_rag_prompt",
    "chunk_corpus",
    "chunk_text",
    "load_corpus",
    "mean_pool_embeddings",
    "normalize_vectors",
    "reciprocal_rank_fusion",
    "tokenize",
]
