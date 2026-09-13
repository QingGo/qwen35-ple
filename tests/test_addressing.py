"""Tests for :mod:`qwen35_ple.addressing` (Stage 1.5f, torch-free).

The load-bearing tests are the two-way ones: the real spec must measure as
lossless **and** the deliberately crushed spec must measure as lossy, on the
same corpus.  A statistic that cannot see the crushed case proves nothing about
the real one.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.addressing import (
    build_examples,
    crushed_spec,
    evaluate_addressing,
    make_spec,
    most_frequent_continuations,
    resolution_report,
    top1_accuracy,
)
from qwen35_ple.ple_hash import real_spec

SPEC = real_spec()


def _corpus(n: int = 4000, *, vocab: int = 300, seed: int = 0) -> np.ndarray:
    """A stream with realistic repetition: a Zipf-ish token distribution."""
    rng = np.random.default_rng(seed)
    ids = np.minimum(
        rng.zipf(1.5, size=n) - 1, vocab - 1,
    ).astype(np.int64)
    return ids


def _structured_corpus(*, n_phrases: int = 3000, length: int = 6,
                       vocab: int = 5000, seed: int = 0) -> np.ndarray:
    """A stream made of repeated fixed phrases.

    This is the regime where an exact n-gram memory is supposed to win -- and,
    for the same reason, the only regime in which an *addressing* control can be
    seen.  On a Zipf stream the modal continuation is the same token for almost
    every context, so top-1 is blind to collisions; on repeated phrases the
    continuation is genuinely context-determined.
    """
    rng = np.random.default_rng(seed)
    phrases = rng.integers(1, vocab, size=(n_phrases, length))
    picked = rng.integers(0, n_phrases, size=1200)
    return phrases[picked].reshape(-1).astype(np.int64)


# --------------------------------------------------------------------------
# spec construction
# --------------------------------------------------------------------------
def test_make_spec_derives_offsets_and_total():
    spec = make_spec([3, 5, 7] + [11] * 13)
    assert spec.head_offsets[0] == 0
    assert spec.head_offsets[1] == 3
    assert spec.head_offsets[2] == 8
    assert spec.total == 3 + 5 + 7 + 11 * 13


def test_make_spec_requires_the_full_head_count():
    with pytest.raises(ValueError, match="expected 16 head sizes"):
        make_spec([7, 11])


def test_make_spec_rejects_degenerate_heads():
    with pytest.raises(ValueError, match="at least two rows"):
        make_spec([1] + [11] * 15)


def test_crushed_spec_shrinks_every_head():
    crushed = crushed_spec(2000)
    assert crushed.total < SPEC.total
    assert all(c < r for c, r in zip(crushed.prime_sizes, SPEC.prime_sizes))


def test_crush_factor_must_be_at_least_two():
    with pytest.raises(ValueError, match="crush factor"):
        crushed_spec(1)


# --------------------------------------------------------------------------
# build_examples alignment
# --------------------------------------------------------------------------
def test_build_examples_aligns_key_tuple_and_target():
    tokens = np.arange(20, dtype=np.int64)
    keys, rows, targets = build_examples(tokens, SPEC)
    assert keys.shape[0] == tokens.shape[0] - 3
    # first example is position 2, predicting token 3
    assert targets[0] == 3
    assert keys[0] == (2 * 1_000_003 + 1) * 1_000_003 + 0
    # the row tuple for position 2 must be the tuple rowids_for_seq returned
    full = np.asarray(SPEC.rowids_for_seq(tuple(int(t) for t in tokens)))
    np.testing.assert_array_equal(rows[0], full[2])


def test_build_examples_rejects_short_streams():
    with pytest.raises(ValueError, match="at least 5 tokens"):
        build_examples(np.arange(4), SPEC)


def test_build_examples_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        build_examples(np.zeros((4, 4), dtype=np.int64), SPEC)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------
def test_resolution_is_injective_on_a_distinct_corpus():
    keys = np.arange(500, dtype=np.int64)
    rows = np.stack([keys, keys + 1, keys * 7], axis=1)
    rep = resolution_report(keys, rows)
    assert rep.n_distinct_trigrams == 500
    assert rep.n_distinct_tuples == 500
    assert rep.injective
    assert rep.lost_tuples == 0


def test_resolution_detects_a_lossy_tuple():
    keys = np.arange(500, dtype=np.int64)
    rows = np.stack([keys % 10, keys % 10, keys % 10], axis=1)
    rep = resolution_report(keys, rows)
    assert rep.n_distinct_tuples == 10
    assert not rep.injective
    assert rep.lost_tuples == 490


def test_resolution_enforces_tuples_never_exceed_trigrams():
    # a tuple is a function of the key, so distinct tuples <= distinct keys
    keys = np.array([1, 1, 2, 2, 2], dtype=np.int64)
    rows = np.stack([keys, keys], axis=1)
    rep = resolution_report(keys, rows)
    assert rep.n_distinct_tuples <= rep.n_distinct_trigrams


def test_per_head_excess_is_reported_even_when_the_tuple_survives():
    keys = np.arange(200, dtype=np.int64)
    # head 0 collides badly, head 1 is injective -> the tuple is still injective
    rows = np.stack([keys % 4, keys], axis=1)
    rep = resolution_report(keys, rows)
    assert rep.per_head_excess[0] == 200 - 4
    assert rep.per_head_excess[1] == 0
    assert rep.injective


def test_resolution_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="aligned"):
        resolution_report(np.zeros(5, dtype=np.int64), np.zeros((4, 3), dtype=np.int64))


# --------------------------------------------------------------------------
# continuation tables
# --------------------------------------------------------------------------
def test_most_frequent_continuations_picks_the_mode():
    keys = np.array([1, 1, 1, 2], dtype=np.int64)
    targets = np.array([7, 7, 9, 3], dtype=np.int64)
    table = most_frequent_continuations(keys, targets)
    assert table == {1: 7, 2: 3}


def test_top1_accuracy_reports_coverage_for_unseen_keys():
    table = {1: 7}
    got = top1_accuracy(np.array([1, 2]), np.array([7, 5]), table)
    assert got["accuracy"] == pytest.approx(0.5)
    assert got["coverage"] == pytest.approx(0.5)


def test_top1_accuracy_on_empty_input_is_zero_not_nan():
    got = top1_accuracy(np.array([]), np.array([]), {})
    assert got["accuracy"] == 0.0
    assert got["n"] == 0.0


def test_most_frequent_continuations_rejects_misalignment():
    with pytest.raises(ValueError, match="align"):
        most_frequent_continuations(np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64))


# --------------------------------------------------------------------------
# the two-way validation
# --------------------------------------------------------------------------
def test_real_spec_is_lossless_on_a_realistic_stream():
    rep = evaluate_addressing(_corpus(4000), SPEC)
    assert rep["resolution"]["injective"]
    assert rep["top1_gap"] == pytest.approx(0.0)


def test_top1_is_blind_to_collisions_on_a_zipf_stream():
    """Why the control needs structure, not just repetition.

    A Zipf stream collides just as badly, but every context's modal
    continuation is the same high-frequency token, so the crushed spec keeps its
    top-1 accuracy.  Recorded here so the choice of control corpus is not read
    as cherry-picking later.
    """
    tokens = _corpus(4000)
    crushed = evaluate_addressing(tokens, crushed_spec(2_000_000))
    assert not crushed["resolution"]["injective"]
    assert crushed["top1_gap"] < 0.05


def test_crushed_spec_is_lossy_on_a_structured_stream():
    tokens = _structured_corpus()
    real = evaluate_addressing(tokens, SPEC)
    crushed = evaluate_addressing(tokens, crushed_spec(2_000_000))
    assert not crushed["resolution"]["injective"]
    assert crushed["resolution"]["lost_tuples"] > 0
    # the loss must be visible in held-out prediction, not only in counting
    assert crushed["top1_gap"] > 0.05
    assert real["top1_gap"] < crushed["top1_gap"]


def test_real_spec_keeps_its_full_resolution_on_the_structured_stream():
    rep = evaluate_addressing(_structured_corpus(), SPEC)
    assert rep["resolution"]["injective"]
    assert rep["top1_gap"] == pytest.approx(0.0)


def test_evaluate_rejects_a_degenerate_split_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        evaluate_addressing(_corpus(100), SPEC, train_fraction=1.0)


def test_evaluate_reports_the_spec_geometry():
    rep = evaluate_addressing(_corpus(600), SPEC)
    assert rep["spec"]["total_rows"] == SPEC.total
    assert len(rep["spec"]["prime_sizes"]) == 16
