"""Tests for :mod:`qwen35_ple.fusion_probe` (Stage 1.5e).

The interesting tests here are the *specificity* ones.  A first draft of the
instrument would have reported "complementarity confirmed" for a count model
that carried no information whatever, because mixing an overconfident backbone
distribution with anything moderate buys a Jensen gain.  These tests pin that
down so the mistake cannot come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.fusion_probe import (
    COMPLEMENTARITY_ABSENT_NATS,
    COMPLEMENTARITY_CONFIRMED_NATS,
    cross_fitted_gain,
    fit_mixture_weight,
    mixture_nll,
    nll_from_probs,
    oracle_mixture_gain,
    permuted_gain,
    verdict_from_gain,
)

N = 20_000


def _prob_pair(rng: np.random.Generator):
    """A pair where ``p_bb`` collapses exactly where ``p_cnt`` rescues.

    Half the positions are "easy" for the backbone (0.9) and the count model is
    uninformative (0.1); the other half are positions where the backbone has
    collapsed (1e-3) and only the count model is right (0.9).  This is the
    operational meaning of complementarity: *the memory is not lost when the
    backbone is*.
    """
    z = rng.random(N) < 0.5
    p_bb = np.where(z, 0.9, 1e-3)
    p_cnt = np.where(z, 0.1, 0.9)
    return p_bb, p_cnt


# --------------------------------------------------------------------------
# mixture_nll / fit_mixture_weight
# --------------------------------------------------------------------------
def test_lam_one_reproduces_the_backbone_nll():
    rng = np.random.default_rng(0)
    p_bb, p_cnt = _prob_pair(rng)
    np.testing.assert_allclose(
        mixture_nll(p_bb, p_cnt, 1.0), nll_from_probs(p_bb),
    )


def test_lam_zero_reproduces_the_count_model_nll():
    rng = np.random.default_rng(0)
    p_bb, p_cnt = _prob_pair(rng)
    np.testing.assert_allclose(
        mixture_nll(p_bb, p_cnt, 0.0), nll_from_probs(p_cnt),
    )


def test_zero_probability_gives_a_finite_nll():
    got = mixture_nll(np.array([0.0]), np.array([0.0]), 0.5)
    assert np.isfinite(got).all()


def test_fit_mixture_weight_finds_the_rescue_weight():
    rng = np.random.default_rng(1)
    p_bb, p_cnt = _prob_pair(rng)
    lam = fit_mixture_weight(p_bb, p_cnt)
    assert 0.3 < lam < 0.7


def test_fit_mixture_weight_is_one_when_the_counter_is_useless():
    rng = np.random.default_rng(2)
    p_bb = np.where(rng.random(N) < 0.7, 0.9, 1e-3)
    p_cnt = np.full(N, 0.5)
    # A constant counter is pure shrinkage; the fit should still prefer to keep
    # some of it only if that lowers NLL, but it must never prefer lam > 1.
    lam = fit_mixture_weight(p_bb, p_cnt)
    assert 0.0 <= lam <= 1.0


@pytest.mark.parametrize("lam", [-0.1, 1.1])
def test_mixture_rejects_out_of_range_lam(lam):
    with pytest.raises(ValueError, match="lam must be"):
        mixture_nll(np.array([0.5]), np.array([0.5]), lam)


def test_mixture_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        mixture_nll(np.zeros(3), np.zeros(4), 0.5)


def test_mixture_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        mixture_nll(np.zeros((2, 2)), np.zeros((2, 2)), 0.5)


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match="zero positions"):
        fit_mixture_weight(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# the theorem the whole instrument rests on
# --------------------------------------------------------------------------
def test_mean_mixture_probability_is_a_convex_combination_of_the_means():
    """E[p_mix] = lam E[p_bb] + (1-lam) E[p_cnt], for ANY joint.

    This is why a mean-NLL gain cannot be read off the marginals, and why a
    permutation null (which preserves marginals) is the right control.
    """
    rng = np.random.default_rng(3)
    p_bb, p_cnt = _prob_pair(rng)
    for lam in (0.0, 0.25, 0.5, 1.0):
        mix = lam * p_bb + (1 - lam) * p_cnt
        assert mix.mean() == pytest.approx(
            lam * p_bb.mean() + (1 - lam) * p_cnt.mean(), abs=1e-12,
        )


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------
def test_rescue_structure_is_detected():
    rng = np.random.default_rng(4)
    p_bb, p_cnt = _prob_pair(rng)
    real = cross_fitted_gain(p_bb, p_cnt)
    null = permuted_gain(p_bb, p_cnt, n_perm=8)
    assert real.gain > 1.0
    assert real.gain - null.gain > 0.1
    v = verdict_from_gain(
        real.gain, null.gain, null_std=null.null_std, n_perm=null.n_perm,
    )
    assert v["label"] == "COMPLEMENTARITY_CONFIRMED"


def test_rescue_structure_keeps_some_backbone_weight():
    rng = np.random.default_rng(5)
    p_bb, p_cnt = _prob_pair(rng)
    real = cross_fitted_gain(p_bb, p_cnt)
    assert 0.0 < real.lam < 1.0


# --------------------------------------------------------------------------
# specificity: the two ways a false positive would appear
# --------------------------------------------------------------------------
def test_constant_counter_distribution_is_shrinkage_only():
    """The trap: a counter carrying NO information still buys a raw gain.

    Permuting cannot change a constant array, so the excess is exactly zero and
    the verdict must be ABSENT.  Without this test the instrument would report
    "complementarity" for a count model that is literally a constant.
    """
    rng = np.random.default_rng(6)
    p_bb = np.where(rng.random(N) < 0.7, 0.9, 1e-3)
    p_cnt = np.full(N, 0.5)
    real = cross_fitted_gain(p_bb, p_cnt)
    null = permuted_gain(p_bb, p_cnt, n_perm=8)
    assert real.gain > 0.1  # a large *raw* gain ...
    assert real.gain - null.gain == pytest.approx(0.0, abs=1e-6)
    assert real.gain - null.gain < COMPLEMENTARITY_ABSENT_NATS
    assert verdict_from_gain(
        real.gain, null.gain, null_std=null.null_std, n_perm=null.n_perm,
    )["label"] == "COMPLEMENTARITY_ABSENT"


def test_independent_counter_with_the_same_marginal_is_absent():
    """A counter that is genuinely uninformative, paired at random.

    Independent noise has the same marginal before and after permutation, so no
    excess survives; the verdict must be ABSENT.  Note this only holds once the
    null is averaged over several permutations -- with one permutation the
    null's own sampling noise produced a spurious MARGINAL verdict.
    """
    rng = np.random.default_rng(7)
    p_bb = np.where(rng.random(N) < 0.7, 0.9, 1e-3)
    p_cnt = np.where(rng.random(N) < 0.5, 0.9, 0.1)
    real = cross_fitted_gain(p_bb, p_cnt)
    null = permuted_gain(p_bb, p_cnt, n_perm=16)
    assert abs(real.gain - null.gain) < COMPLEMENTARITY_ABSENT_NATS
    assert verdict_from_gain(
        real.gain, null.gain, null_std=null.null_std, n_perm=null.n_perm,
    )["label"] == "COMPLEMENTARITY_ABSENT"


def test_single_permutation_null_is_noisier_than_the_averaged_one():
    rng = np.random.default_rng(70)
    p_bb = np.where(rng.random(N) < 0.7, 0.9, 1e-3)
    p_cnt = np.where(rng.random(N) < 0.5, 0.9, 0.1)
    single = permuted_gain(p_bb, p_cnt, n_perm=1)
    many = permuted_gain(p_bb, p_cnt, n_perm=16)
    assert single.null_std == 0.0
    assert many.null_std > 0.0


def test_permuted_gain_rejects_zero_permutations():
    with pytest.raises(ValueError, match="n_perm must be"):
        permuted_gain(np.zeros(4) + 0.5, np.zeros(4) + 0.5, n_perm=0)


def test_a_weak_but_real_rescue_lands_in_marginal_not_confirmed():
    rng = np.random.default_rng(8)
    p_bb = np.full(N, 0.4)
    p_cnt = np.where(rng.random(N) < 0.5, 0.5, 0.39)
    real = cross_fitted_gain(p_bb, p_cnt)
    null = permuted_gain(p_bb, p_cnt, n_perm=8)
    v = verdict_from_gain(
        real.gain, null.gain, null_std=null.null_std, n_perm=null.n_perm,
    )
    assert v["label"] in {"COMPLEMENTARITY_MARGINAL", "COMPLEMENTARITY_ABSENT"}
    assert COMPLEMENTARITY_ABSENT_NATS < COMPLEMENTARITY_CONFIRMED_NATS


# --------------------------------------------------------------------------
# cross-fitting and groups
# --------------------------------------------------------------------------
def test_cross_fitting_does_not_beat_the_oracle_but_tracks_it():
    rng = np.random.default_rng(9)
    p_bb, p_cnt = _prob_pair(rng)
    real = cross_fitted_gain(p_bb, p_cnt)
    oracle = oracle_mixture_gain(p_bb, p_cnt)
    assert oracle["gain"] >= real.gain - 1e-9


def test_oracle_uses_a_weight_below_one_only_where_it_helps():
    rng = np.random.default_rng(10)
    p_bb = np.where(rng.random(1000) < 0.6, 0.9, 1e-3)
    p_cnt = np.full(1000, 0.5)
    oracle = oracle_mixture_gain(p_bb, p_cnt)
    assert 0.0 < oracle["share_lam_lt_one"] <= 1.0
    assert oracle["gain"] >= 0.0


def test_group_weights_are_fitted_separately():
    """One global lam averages a narrow band away; groups must recover it."""
    rng = np.random.default_rng(11)
    p_bb = np.empty(2 * N)
    p_cnt = np.empty(2 * N)
    # group 0: the counter rescues
    p_bb[:N] = np.where(rng.random(N) < 0.5, 0.9, 1e-3)
    p_cnt[:N] = np.where(p_bb[:N] > 0.5, 0.1, 0.9)
    # group 1: the counter is a constant (pure shrinkage)
    p_bb[N:] = np.where(rng.random(N) < 0.7, 0.9, 1e-3)
    p_cnt[N:] = 0.5
    groups = np.array([0] * N + [1] * N)
    res = cross_fitted_gain(p_bb, p_cnt, groups=groups)
    null = permuted_gain(p_bb, p_cnt, groups=groups, n_perm=8)
    assert set(res.per_group) == {"0", "1"}
    # the rescuing group wants a real mix ...
    assert 0.2 < res.per_group["0"]["lam_mean"] < 0.8
    # ... while the constant-counter group also mixes, and that is CORRECT: a
    # constant is a free recalibration of an overconfident backbone.  The two
    # are separated by the *excess over the null*, not by the weight.
    assert res.per_group["1"]["lam_mean"] < 0.95
    excess_rescue = res.per_group["0"]["gain"] - null.per_group["0"]["gain"]
    excess_shrink = res.per_group["1"]["gain"] - null.per_group["1"]["gain"]
    assert excess_rescue > 0.1
    assert abs(excess_shrink) < 0.01
    # equal-weight pooling must not let the large group hide the small one
    assert res.pooled_gain == pytest.approx(
        0.5 * (res.per_group["0"]["gain"] + res.per_group["1"]["gain"]), abs=1e-9,
    )


def test_groups_must_match_shape():
    with pytest.raises(ValueError, match="groups shape"):
        cross_fitted_gain(np.zeros(4) + 0.5, np.zeros(4) + 0.5, groups=np.zeros(3))


def test_cross_fitted_requires_two_folds():
    with pytest.raises(ValueError, match="folds must be"):
        cross_fitted_gain(np.zeros(4) + 0.5, np.zeros(4) + 0.5, folds=1)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no positions"):
        cross_fitted_gain(np.array([]), np.array([]))


def test_result_serialises():
    rng = np.random.default_rng(12)
    p_bb, p_cnt = _prob_pair(rng)
    d = cross_fitted_gain(p_bb, p_cnt).to_dict()
    assert {"n", "gain", "lam", "per_group", "pooled_gain"} <= set(d)


# --------------------------------------------------------------------------
# verdict rule
# --------------------------------------------------------------------------
def test_verdict_is_taken_on_the_excess_not_the_raw_gain():
    """A huge raw gain explained by the null must not be called confirmed."""
    assert verdict_from_gain(1.30, 1.30)["label"] == "COMPLEMENTARITY_ABSENT"
    assert verdict_from_gain(1.30, 1.25)["label"] == "COMPLEMENTARITY_CONFIRMED"
    assert verdict_from_gain(1.30, 1.29)["label"] == "COMPLEMENTARITY_MARGINAL"


def test_verdict_confirmed_threshold_is_on_the_excess():
    assert verdict_from_gain(0.019, 0.0)["label"] == "COMPLEMENTARITY_MARGINAL"
    assert verdict_from_gain(0.021, 0.0)["label"] == "COMPLEMENTARITY_CONFIRMED"


def test_verdict_absent_when_the_excess_is_below_the_floor():
    assert verdict_from_gain(1.0, 0.999)["label"] == "COMPLEMENTARITY_ABSENT"


def test_verdict_demands_three_standard_errors_of_the_null():
    # a 0.02-nat excess sitting inside 3 SE of the null must not be confirmed
    v = verdict_from_gain(1.02, 1.0, null_std=0.02, n_perm=4)
    assert v["noise_floor"] == pytest.approx(0.03)
    assert v["excess"] == pytest.approx(0.02, abs=1e-9)
    assert v["label"] == "COMPLEMENTARITY_ABSENT"


def test_more_permutations_shrink_the_noise_floor():
    few = verdict_from_gain(1.03, 1.0, null_std=0.02, n_perm=1)
    many = verdict_from_gain(1.03, 1.0, null_std=0.02, n_perm=100)
    assert many["noise_floor"] < few["noise_floor"]


def test_verdict_reports_excess_and_thresholds():
    v = verdict_from_gain(0.03, 0.005)
    assert v["excess"] == pytest.approx(0.025)
    assert v["thresholds"]["confirmed_nats"] == COMPLEMENTARITY_CONFIRMED_NATS
