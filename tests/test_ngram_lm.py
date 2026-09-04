"""Tests for the training-free n-gram LM."""

from __future__ import annotations

import numpy as np

from qwen35_ple.ngram_lm import NgramLM, interpolate_logits


def test_ngram_basic() -> None:
    tokens = [1, 2, 3, 1, 2, 3, 4]
    lm = NgramLM.from_tokens(tokens, max_order=3)
    assert lm.logprob(3, [1, 2]) > -10
    assert lm.topk([1, 2], k=2)[0][0] in (3,)
    assert lm.logprob(999, [1, 2]) < 0


def test_interpolate_logits() -> None:
    base = np.zeros(10, dtype=np.float32)
    out = interpolate_logits(base, {5: 0.5, 7: 0.5}, 5, scale=1.0)
    assert out[5] < 0
    assert out[7] < 0
    assert out[0] == 0.0


def test_backoff() -> None:
    tokens = [1, 2, 3, 4, 5, 6]
    lm = NgramLM.from_tokens(tokens, max_order=4)
    # unknown context should back off to unigram/uniform without error
    p = lm.logprob(1, [99, 98, 97])
    assert np.isfinite(p)
