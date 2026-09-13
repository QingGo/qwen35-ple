"""Tests for :mod:`qwen35_ple.margin` (Stage 1.5c analytics, torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.margin import (
    COUNT_EDGES,
    MARGIN_QUANTILES,
    MIN_CONTEXT_COUNT,
    GainEntry,
    bucket_labels,
    bucket_summary,
    cross_tab,
    gain_curve,
    margin_from_nll,
    oracle_gain,
    positive_share,
    top_fraction_mask,
    verdict_from_curve,
)


# --------------------------------------------------------------------------
# margin_from_nll
# --------------------------------------------------------------------------
def test_margin_is_backbone_minus_count():
    bb = np.array([3.0, 2.0, 5.0])
    cnt = np.array([2.0, 2.5, 6.0])
    np.testing.assert_allclose(margin_from_nll(bb, cnt), [1.0, -0.5, -1.0])


def test_margin_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        margin_from_nll(np.zeros(3), np.zeros(4))


def test_margin_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        margin_from_nll(np.zeros((2, 2)), np.zeros((2, 2)))


def test_margin_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        margin_from_nll(np.array([1.0, np.inf]), np.array([1.0, 1.0]))


# --------------------------------------------------------------------------
# positive_share
# --------------------------------------------------------------------------
def test_positive_share_counts_strictly_positive():
    m = np.array([1.0, 0.0, -1.0, 2.0])
    assert positive_share(m) == pytest.approx(0.5)


def test_positive_share_respects_mask():
    m = np.array([1.0, 0.0, -1.0, 2.0])
    mask = np.array([True, True, True, False])
    assert positive_share(m, mask) == pytest.approx(1.0 / 3.0)


def test_positive_share_empty_selection_is_zero():
    assert positive_share(np.array([1.0]), np.array([False])) == 0.0


# --------------------------------------------------------------------------
# top_fraction_mask
# --------------------------------------------------------------------------
def test_top_fraction_mask_takes_the_most_positive():
    m = np.array([0.1, 5.0, -2.0, 3.0])
    mask = top_fraction_mask(m, 50.0)
    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_top_fraction_mask_q_one_percent_still_selects_one_position():
    m = np.arange(1000, dtype=np.float64)
    mask = top_fraction_mask(m, 1.0)
    assert int(mask.sum()) == 10


def test_top_fraction_mask_breaks_ties_by_position_order():
    m = np.zeros(4)
    mask = top_fraction_mask(m, 50.0)
    np.testing.assert_array_equal(mask, [True, True, False, False])


@pytest.mark.parametrize("q", [0.0, -1.0, 101.0])
def test_top_fraction_mask_rejects_out_of_range_q(q):
    with pytest.raises(ValueError, match="q must be"):
        top_fraction_mask(np.zeros(4), q)


def test_top_fraction_mask_realizable_excludes_rare_contexts():
    m = np.array([10.0, 9.0, 8.0, 7.0])
    ctx = np.array([100, 3, 50, 1])
    mask = top_fraction_mask(m, 100.0, context_counts=ctx, min_context_count=5)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_top_fraction_mask_realizable_q_is_fraction_of_eligible_pool():
    m = np.array([10.0, 9.0, 8.0, 7.0])
    ctx = np.array([100, 3, 50, 1])
    # 2 eligible positions, 50% -> 1
    mask = top_fraction_mask(m, 50.0, context_counts=ctx, min_context_count=5)
    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_top_fraction_mask_requires_context_counts_when_threshold_given():
    with pytest.raises(ValueError, match="requires context_counts"):
        top_fraction_mask(np.zeros(4), 50.0, min_context_count=5)


def test_top_fraction_mask_no_eligible_positions_returns_all_false():
    m = np.array([1.0, 2.0])
    ctx = np.array([0, 1])
    mask = top_fraction_mask(m, 50.0, context_counts=ctx, min_context_count=5)
    assert not mask.any()


# --------------------------------------------------------------------------
# oracle_gain
# --------------------------------------------------------------------------
def test_oracle_gain_normalises_by_all_positions():
    m = np.array([4.0, 4.0, -1.0, -1.0])
    mask = np.array([True, True, False, False])
    e = oracle_gain(m, mask)
    # sum of positives over the selection = 8, divided by 4 positions
    assert e.corpus_nll_reduction == pytest.approx(2.0)
    assert e.mean_margin_selected == pytest.approx(4.0)
    assert e.share_positive_selected == pytest.approx(1.0)
    assert e.share_of_positions == pytest.approx(0.5)


def test_oracle_gain_clips_negative_margins():
    m = np.array([-3.0, -3.0])
    e = oracle_gain(m, np.array([True, True]))
    assert e.corpus_nll_reduction == 0.0
    assert e.share_positive_selected == 0.0


def test_oracle_gain_empty_selection_is_zero_not_nan():
    e = oracle_gain(np.array([1.0, 2.0]), np.array([False, False]))
    assert e.corpus_nll_reduction == 0.0
    assert e.n_selected == 0


def test_oracle_gain_records_eligible_pool():
    e = oracle_gain(np.array([1.0, 2.0]), np.array([True, False]), n_eligible=1)
    assert e.n_eligible == 1
    assert e.q_percent == pytest.approx(100.0)


# --------------------------------------------------------------------------
# gain_curve
# --------------------------------------------------------------------------
def test_gain_curve_reduction_is_monotone_in_q():
    rng = np.random.default_rng(0)
    m = rng.normal(size=500)
    curve = gain_curve(m, quantiles=(1.0, 5.0, 20.0, 100.0))
    reds = [e.corpus_nll_reduction for e in curve]
    assert reds == sorted(reds)


def test_gain_curve_returns_both_pools_when_context_counts_given():
    m = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -1.0, -2.0])
    ctx = np.array([10, 10, 10, 10, 1, 1, 1, 1])
    curve = gain_curve(m, quantiles=(50.0,), context_counts=ctx, min_context_count=5)
    labels = {e.label for e in curve}
    assert labels == {"all", "ctx_count>=5"}
    real = next(e for e in curve if e.realizable)
    assert real.n_eligible == 4
    assert real.n_selected == 2


def test_gain_curve_default_quantiles_are_the_preregistered_grid():
    curve = gain_curve(np.linspace(-1, 1, 10))
    assert tuple(e.q_percent for e in curve) == MARGIN_QUANTILES


# --------------------------------------------------------------------------
# bucket_summary / cross_tab
# --------------------------------------------------------------------------
def test_bucket_summary_partitions_every_position():
    m = np.array([1.0, 2.0, 3.0, 4.0])
    counts = np.array([0, 1, 10, 1000])
    rows = bucket_summary(m, counts)
    assert sum(r["n"] for r in rows) == 4
    assert all(r["bucket"] in {b for b in [
        "0", "1", "2", "3-4", "5-9", "10-49", "50-199", "200-999", "1000"
    ]} for r in rows)


def test_bucket_summary_reports_mean_nll_when_supplied():
    m = np.array([1.0, 1.0])
    counts = np.array([0, 100])
    bb = np.array([3.0, 6.0])
    cnt = np.array([2.0, 5.0])
    rows = {r["bucket"]: r for r in bucket_summary(m, counts, bb_nll=bb, cnt_nll=cnt)}
    assert rows["0"]["mean_bb_nll"] == pytest.approx(3.0)
    assert rows["50-199"]["mean_cnt_nll"] == pytest.approx(5.0)


def test_bucket_summary_requires_matching_shapes():
    with pytest.raises(ValueError, match="counts shape"):
        bucket_summary(np.zeros(3), np.zeros(4))


def test_cross_tab_cell_counts_sum_to_positions():
    rng = np.random.default_rng(1)
    m = rng.normal(size=64)
    tc = rng.integers(0, 100, size=64)
    cc = rng.integers(0, 100, size=64)
    tab = cross_tab(m, tc, cc)
    assert sum(c["n"] for c in tab["cells"]) == 64
    assert len(tab["cells"]) == len(COUNT_EDGES) ** 2


def test_cross_tab_empty_cells_are_marked_not_dropped():
    m = np.zeros(1)
    tab = cross_tab(m, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
    occupied = [c for c in tab["cells"] if c["n"] > 0]
    assert len(occupied) == 1
    assert len(tab["cells"]) > 1


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def _entry(q: float, red: float, realizable: bool = True) -> GainEntry:
    return GainEntry(
        q_percent=q, n_selected=10, n_eligible=100, share_of_positions=0.1,
        mean_margin_selected=1.0, share_positive_selected=0.9,
        corpus_nll_reduction=red, realizable=realizable, label="x",
    )


def test_verdict_empty_when_no_position_is_positive():
    v = verdict_from_curve(
        [_entry(5.0, 0.9)], positive_share_all=0.0, n_positions=100,
    )
    assert v.label == "NO_POSITIVE_POSITION"


def test_verdict_empty_when_best_gain_below_floor():
    v = verdict_from_curve(
        [_entry(5.0, 0.049), _entry(10.0, 0.01)],
        positive_share_all=0.30, n_positions=100,
    )
    assert v.label == "MEMORY_CEILING_EMPTY"


def test_verdict_thin_between_floor_and_room():
    v = verdict_from_curve(
        [_entry(5.0, 0.19)], positive_share_all=0.30, n_positions=100,
    )
    assert v.label == "MEMORY_CEILING_THIN"


def test_verdict_room_at_threshold():
    v = verdict_from_curve(
        [_entry(5.0, 0.20)], positive_share_all=0.30, n_positions=100,
    )
    assert v.label == "MEMORY_CEILING_ROOM"


def test_verdict_ignores_selections_above_the_q_ceiling():
    # a huge gain at q=50% must not rescue a dead q<=20% region
    v = verdict_from_curve(
        [_entry(5.0, 0.01), _entry(50.0, 5.0)],
        positive_share_all=0.30, n_positions=100,
    )
    assert v.label == "MEMORY_CEILING_EMPTY"
    assert v.best_q == pytest.approx(5.0)


def test_verdict_prefers_realizable_pool_over_all_positions():
    v = verdict_from_curve(
        [_entry(5.0, 5.0, realizable=False), _entry(5.0, 0.30, realizable=True)],
        positive_share_all=0.30, n_positions=100,
    )
    assert v.best_reduction == pytest.approx(0.30)
    assert v.details["used_realizable_pool"] is True


def test_verdict_falls_back_to_unfiltered_pool_when_no_context_counts():
    v = verdict_from_curve(
        [_entry(5.0, 0.30, realizable=False)],
        positive_share_all=0.30, n_positions=100,
    )
    assert v.details["used_realizable_pool"] is False
    assert v.label == "MEMORY_CEILING_ROOM"


def test_verdict_raises_without_any_entry_in_range():
    with pytest.raises(ValueError, match="q ceiling"):
        verdict_from_curve([_entry(50.0, 1.0)], positive_share_all=0.3, n_positions=10)


def test_verdict_serialises_to_dict():
    v = verdict_from_curve(
        [_entry(5.0, 0.30)], positive_share_all=0.3, n_positions=10,
    )
    d = v.to_dict()
    assert d["label"] == "MEMORY_CEILING_ROOM"
    assert isinstance(d["details"], dict)


def test_min_context_count_sits_inside_the_bucket_grid():
    # The realization filter must be expressible on the frequency axis, else the
    # cross-tab and the realizable curve would describe different populations.
    assert MIN_CONTEXT_COUNT in COUNT_EDGES


# --------------------------------------------------------------------------
# bucket_labels (shared with the Stage 1.5e grouped fusion probe)
# --------------------------------------------------------------------------
def test_bucket_labels_agree_with_bucket_summary_counts():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 2000, size=500)
    idx, labels = bucket_labels(counts)
    rows = bucket_summary(np.zeros(500), counts)
    for i, label in enumerate(labels):
        assert label == rows[i]["bucket"]
        assert int(np.count_nonzero(idx == i)) == rows[i]["n"]


def test_bucket_labels_put_an_unseen_context_in_the_first_bucket():
    idx, labels = bucket_labels(np.array([0, 10 ** 9]))
    assert labels[idx[0]] == "0"
    assert labels[idx[1]] == "1000"


def test_bucket_labels_reject_edges_not_starting_at_zero():
    with pytest.raises(ValueError, match="must start at 0"):
        bucket_labels(np.array([1]), edges=(1, 2))
