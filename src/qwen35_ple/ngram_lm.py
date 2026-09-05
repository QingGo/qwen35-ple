"""Training-free n-gram language model built from a token corpus.

This is the key reframe for PLE: instead of using the frozen PLE table as a
semantic knowledge memory, use the *exact n-gram structure* as a sparse lexical
memory / local low-entropy prior.  It can be combined with a neural base model at
the logit level:

    final_logits = base_logits + scale * log P_ngram(next | context)

Advantages:

* no training;
* auditable;
* exact-match local memory;
* complementary to RAG (semantic) and to the base model (general reasoning).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np


class NgramLM:
    """A small exact-match backoff n-gram language model."""

    def __init__(self, max_order: int = 4, min_count: int = 1) -> None:
        self.max_order = int(max_order)
        self.min_count = int(min_count)
        # counts[n][context] -> Counter(next_token)
        self.counts: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(self.max_order + 1)
        ]
        self.total_tokens = 0
        self.unigram: Counter[int] = Counter()

    def add_sequence(self, tokens: Iterable[int]) -> None:
        tokens = [int(t) for t in tokens]
        for i, tok in enumerate(tokens):
            self.total_tokens += 1
            self.unigram[tok] += 1
            for n in range(1, self.max_order + 1):
                if i >= n:
                    ctx = tuple(tokens[i - n : i])
                    self.counts[n][ctx][tok] += 1

    @classmethod
    def from_tokens(cls, tokens: Iterable[int], max_order: int = 4) -> NgramLM:
        lm = cls(max_order=max_order)
        lm.add_sequence(tokens)
        return lm

    def _context_counts(self, context: tuple[int, ...]) -> Counter[int] | None:
        n = min(self.max_order, len(context))
        while n >= 1:
            key = tuple(context[-n:])
            c = self.counts[n].get(key)
            if c and sum(c.values()) >= self.min_count:
                return c
            n -= 1
        return None

    def distribution(self, context: Iterable[int]) -> dict[int, float] | None:
        ctx = tuple(int(x) for x in context)
        c = self._context_counts(ctx)
        if c is None:
            return None
        total = sum(c.values())
        return {tok: cnt / total for tok, cnt in c.items()}

    def logprob(self, token: int, context: Iterable[int]) -> float:
        dist = self.distribution(context)
        if dist is not None:
            p = dist.get(int(token), 0.0)
            if p > 0:
                return math.log(p)
        # Backoff to unigram with a tiny floor, or uniform if unknown.
        if self.unigram:
            p = self.unigram.get(int(token), 0) / max(1, self.total_tokens)
            if p > 0:
                return math.log(p)
        return math.log(1e-12)

    def score_sequence(self, tokens: list[int]) -> float:
        """Average log-probability of a sequence."""
        if not tokens:
            return 0.0
        total = 0.0
        for i in range(1, len(tokens)):
            total += self.logprob(tokens[i], tokens[:i])
        return total / max(1, len(tokens) - 1)

    def topk(self, context: Iterable[int], k: int = 5) -> list[tuple[int, float]]:
        dist = self.distribution(context)
        if dist is None:
            return []
        return sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]

    def stats(self) -> dict[str, int]:
        return {
            "max_order": self.max_order,
            "total_tokens": self.total_tokens,
            "unique_unigrams": len(self.unigram),
            **{
                f"order_{n}_contexts": len(self.counts[n])
                for n in range(1, self.max_order + 1)
            },
        }


def interpolate_logits(
    base_logits: np.ndarray,
    ngram_probs: dict[int, float] | None,
    vocab_index: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Add an n-gram log-prior to base logits for a single position.

    ``base_logits`` is a 1-D logits vector; ``ngram_probs`` maps token id to
    probability.  Only known tokens are adjusted.
    """
    out = base_logits.copy()
    if not ngram_probs:
        return out
    for tok, p in ngram_probs.items():
        if 0 <= tok < len(out) and p > 0:
            out[tok] += scale * math.log(p)
    return out


__all__ = ["NgramLM", "interpolate_logits"]
