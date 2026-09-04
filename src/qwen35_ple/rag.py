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


__all__ = [
    "BM25Index",
    "build_rag_prompt",
    "load_corpus",
    "tokenize",
]
