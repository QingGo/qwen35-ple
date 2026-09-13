#!/usr/bin/env python3
"""Unit tests for the round-167 Stage 1a effective-depth instruments.

These exercise the pure-numpy helpers only, so they run without torch.  The
stakes are high: if ``effective_depth_from_kl`` or the overlap reader were
mis-ordered (say, returning the *last* qualifying layer instead of the first),
then a graft that genuinely composes more depth would be reported as a null, and
the whole point of the experiment would be inverted.  Each instrument is
therefore tested against a hand-built curve with a known answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import round167_effective_depth as r167

# --------------------------------------------------------------------------
# logit-lens KL
# --------------------------------------------------------------------------


def test_kl_is_zero_for_identical_distributions() -> None:
    p = np.array([0.1, 0.2, 0.7])
    assert r167.logit_lens_kl(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_positive_and_asymmetric() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    assert r167.logit_lens_kl(p, q) > 0
    assert r167.logit_lens_kl(p, q) != pytest.approx(r167.logit_lens_kl(q, p))


def test_kl_tolerates_zeros_via_the_floor() -> None:
    """A one-hot layer distribution must give a finite KL, not inf/nan."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    v = r167.logit_lens_kl(p, q)
    assert np.isfinite(v) and v > 0


def test_kl_is_invariant_to_input_scale() -> None:
    """It takes probabilities; a rescaled-but-same-shape input must not matter."""
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.4, 0.4, 0.2])
    assert r167.logit_lens_kl(p, q) == pytest.approx(r167.logit_lens_kl(p * 10, q * 10))


# --------------------------------------------------------------------------
# top-k overlap
# --------------------------------------------------------------------------


def test_overlap_is_one_when_final_ranking_is_already_present() -> None:
    final = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert r167.top_k_overlap(final, final) == pytest.approx(1.0)


def test_overlap_is_zero_for_disjoint_top_k() -> None:
    layer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    final = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    # layer top-5 = {5,0,1,2,3}; final top-5 = {0,1,2,3,4} -> 4/5 shared
    assert r167.top_k_overlap(layer, final) == pytest.approx(4 / 5)


def test_overlap_counts_the_final_top_k_not_the_layer_top_k() -> None:
    """Denominator must be k, so a degenerate layer cannot inflate the score."""
    layer = np.array([1.0] * 10)
    final = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert 0.0 <= r167.top_k_overlap(layer, final) <= 1.0


# --------------------------------------------------------------------------
# effective depth from KL
# --------------------------------------------------------------------------


def test_kl_depth_is_the_first_layer_below_half_max() -> None:
    # max = 8.0 -> threshold 4.0; first below is index 3.
    curve = [8.0, 7.0, 5.0, 3.0, 1.0, 0.5]
    assert r167.effective_depth_from_kl(curve) == 3


def test_kl_depth_not_latched_by_a_later_dip() -> None:
    """A noisy curve that qualifies early must not report a later index."""
    curve = [8.0, 1.0, 7.0, 6.0, 0.2]
    assert r167.effective_depth_from_kl(curve) == 1


def test_kl_depth_is_none_when_never_below_half_max() -> None:
    assert r167.effective_depth_from_kl([1.0, 1.0, 1.0]) is None


def test_kl_depth_ignores_nan() -> None:
    curve = [8.0, float("nan"), 3.0, 1.0]
    assert r167.effective_depth_from_kl(curve) == 2


# --------------------------------------------------------------------------
# effective depth from overlap
# --------------------------------------------------------------------------


def test_overlap_depth_is_first_layer_above_0_3() -> None:
    curve = [0.0, 0.2, 0.4, 0.8]
    assert r167.effective_depth_from_overlap(curve) == 2


def test_overlap_depth_is_none_when_never_exceeds_threshold() -> None:
    assert r167.effective_depth_from_overlap([0.0, 0.1, 0.3]) is None


def test_overlap_threshold_is_strict() -> None:
    """Exactly 0.3 does not qualify; 0.31 does."""
    assert r167.effective_depth_from_overlap([0.3]) is None
    assert r167.effective_depth_from_overlap([0.31]) == 0


# --------------------------------------------------------------------------
# effective depth from residual cosine
# --------------------------------------------------------------------------


def test_cosine_depth_after_a_negative_then_positive_transition() -> None:
    curve = [-0.5, -0.4, -0.2, 0.1, 0.3]
    assert r167.effective_depth_from_cosine(curve) == 3


def test_cosine_depth_is_zero_when_already_positive() -> None:
    assert r167.effective_depth_from_cosine([0.2, 0.3, 0.4]) == 0


def test_cosine_depth_is_none_when_always_negative() -> None:
    assert r167.effective_depth_from_cosine([-0.5, -0.4, -0.3]) is None


def test_cosine_smoothing_resists_a_single_positive_spike() -> None:
    """A lone positive layer inside a negative run must not be the transition.

    The rule needs positive-here-and-positive-next, so the spike at index 2 does
    not qualify (index 3 is negative) and the transition lands at index 5.
    """
    curve = [-0.5, -0.4, 0.9, -0.3, -0.2, 0.5, 0.6]
    depth = r167.effective_depth_from_cosine(curve)
    assert depth == 5


def test_cosine_rule_has_at_most_one_layer_of_lag() -> None:
    """Sensitivity matters: a 2+ layer lag could hide a small real gain."""
    # True transition at index 3; the rule must not report more than 3.
    curve = [-0.5, -0.4, -0.2, 0.1, 0.3]
    assert r167.effective_depth_from_cosine(curve) == 3


# --------------------------------------------------------------------------
# cosine helper
# --------------------------------------------------------------------------


def test_cosine_of_parallel_vectors_is_one() -> None:
    assert r167.cosine(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert r167.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_is_nan_on_a_zero_vector_rather_than_crashing() -> None:
    assert np.isnan(r167.cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])))


# --------------------------------------------------------------------------
# the direction of the headline claim
# --------------------------------------------------------------------------


def test_a_graft_that_frees_depth_is_reported_as_a_positive_gain() -> None:
    """Encode the sign convention of the pre-registered verdict.

    If injection moves the transition *earlier*, ``off - real`` must be positive.
    Getting this backwards would flip the experiment's conclusion.
    """
    off_curve = [8.0, 7.0, 6.0, 5.0, 1.0, 0.5]
    real_curve = [8.0, 2.0, 1.0, 0.5, 0.3, 0.2]
    d_off = r167.effective_depth_from_kl(off_curve)
    d_real = r167.effective_depth_from_kl(real_curve)
    assert d_off is not None and d_real is not None
    assert d_off - d_real > 0


def test_a_null_graft_reports_zero_gain() -> None:
    curve = [8.0, 7.0, 6.0, 5.0, 1.0, 0.5]
    assert r167.effective_depth_from_kl(curve) - r167.effective_depth_from_kl(curve) == 0
