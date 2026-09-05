"""Tests for multi-source log-linear fusion utilities."""

from __future__ import annotations

import numpy as np

from qwen35_ple.fusion import (
    calibrate_ngram_fusion,
    fuse_ngram_logits,
    mixture_distribution,
    softmax,
    weight_logit_sources,
)


def test_fuse_ngram_logits_prefers_high_ngram_prob() -> None:
    base = np.zeros(10, dtype=np.float32)
    dist = {5: 0.9, 7: 0.1}
    # Without bias, ngram candidates are shifted down relative to raw logits;
    # with a positive bias the ngram candidate family can be boosted.
    out_no_bias = fuse_ngram_logits(base, dist, scale=2.0)
    assert out_no_bias[5] > out_no_bias[7]
    out = fuse_ngram_logits(base, dist, scale=2.0, bias=3.0)
    assert np.argmax(out) == 5


def test_calibrate_ngram_fusion_improves_nll() -> None:
    base = np.zeros(10, dtype=np.float32)
    result = calibrate_ngram_fusion(
        [base],
        [5],
        [{5: 0.9, 7: 0.1}],
        scale_grid=np.linspace(0.0, 5.0, 11),
        bias_grid=np.linspace(-2.0, 2.0, 9),
    )
    assert result["delta_nll_nats"] > 0
    assert result["best_mean_nll"] < result["base_mean_nll"]


def test_weight_logit_sources_and_mixture() -> None:
    base = np.zeros(4, dtype=np.float32)
    src_a = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)
    src_b = np.array([0.0, 0.0, 3.0, 0.0], dtype=np.float32)
    out = weight_logit_sources(base, [src_a, src_b], [1.0, 1.0])
    assert out[2] == 3.0
    assert out[1] == 2.0

    mix = mixture_distribution([{1: 0.5, 2: 0.5}, {2: 0.25, 3: 0.75}], [0.5, 0.5])
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert mix[2] > 0


def test_softmax_is_normalized() -> None:
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(probs.sum() - 1.0) < 1e-9
