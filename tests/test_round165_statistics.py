#!/usr/bin/env python3
"""Unit tests for the round-165 read-out-repair statistics.

These exercise only the pure-numpy helpers, so they run without torch.  The
frozen decision rule is tested directly: a mis-ordered verdict would let a
collapsing read-out be reported as a repaired one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import round165_readout_repair as r165

# --------------------------------------------------------------------------
# participation ratio
# --------------------------------------------------------------------------


def test_pr_is_one_for_collinear_rows():
    """All rows on one line through the origin -> variance is rank one."""
    base = np.array([1.0, 2.0, 3.0, 4.0])
    mat = np.stack([base * s for s in (1.0, 2.0, 3.0, 5.0)])
    assert r165.participation_ratio(mat) == pytest.approx(1.0, abs=1e-9)


def test_pr_is_one_for_a_scaled_constant_direction():
    """The round-164 collapse: one direction, varying magnitude."""
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(64)
    direction /= np.linalg.norm(direction)
    scales = rng.standard_normal(200) * 0.33 + 1.0
    mat = np.outer(scales, direction)
    assert r165.participation_ratio(mat) == pytest.approx(1.0, abs=1e-9)


def test_pr_grows_with_isotropic_noise():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((400, 64))
    # An isotropic cloud has PR close to the ambient dimension.
    assert r165.participation_ratio(mat) > 40.0


def test_pr_rejects_a_constant_readout():
    mat = np.ones((10, 4))
    with pytest.raises(ValueError, match="zero variance"):
        r165.participation_ratio(mat)


def test_pr_needs_two_rows():
    with pytest.raises(ValueError, match="at least two rows"):
        r165.participation_ratio(np.ones((1, 4)))


# --------------------------------------------------------------------------
# bootstrap interval
# --------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 1.0, size=500)
    lo, hi = r165.bootstrap_ci(x, n=400, seed=0)
    assert lo < x.mean() < hi
    assert hi - lo < 1.0


def test_bootstrap_ci_is_deterministic_given_a_seed():
    rng = np.random.default_rng(3)
    x = rng.normal(size=100)
    assert r165.bootstrap_ci(x, n=200, seed=7) == r165.bootstrap_ci(x, n=200, seed=7)


def test_bootstrap_ci_is_tight_for_a_constant_sample():
    x = np.full(50, 2.5)
    lo, hi = r165.bootstrap_ci(x, n=100, seed=0)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(2.5)


# --------------------------------------------------------------------------
# total variation
# --------------------------------------------------------------------------


def test_tv_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert r165.total_variation(p, p) == pytest.approx(0.0)


def test_tv_is_one_for_disjoint_distributions():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    assert r165.total_variation(p, q) == pytest.approx(1.0)


def test_tv_ignores_a_permutation_of_mass():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.5, 0.0, 0.5, 0.0])
    assert r165.total_variation(p, q) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# mean pairwise cosine
# --------------------------------------------------------------------------


def test_mean_pairwise_cosine_of_identical_rows_is_one():
    row = np.array([1.0, 0.0, 0.0])
    mat = np.stack([row, row, row])
    assert r165.mean_pairwise_cosine(mat) == pytest.approx(1.0)


def test_mean_pairwise_cosine_of_orthogonal_rows_is_zero():
    mat = np.eye(3)
    assert r165.mean_pairwise_cosine(mat) == pytest.approx(0.0)


def test_mean_pairwise_cosine_rejects_zero_norm_rows():
    with pytest.raises(ValueError, match="zero-norm"):
        r165.mean_pairwise_cosine(np.array([[1.0, 0.0], [0.0, 0.0]]))


# --------------------------------------------------------------------------
# variant construction and norm matching
# --------------------------------------------------------------------------


def _variant_kwargs(**over):
    base = {
        "c_prod": np.array([[3.0, 0.0], [0.0, 4.0]]),
        "c_bar": np.array([0.5, 0.5]),
        "value_proj_e": np.array([[1.0, 1.0], [2.0, -1.0]]),
        "oracle_out": np.array([[1.0, 0.0], [0.0, 2.0]]),
        "rand_w": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "gold_emb": np.array([0.25, 0.75]),
        "target_norm": 0.5,
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "variant", ["centered", "random_proj", "oracle_ungated", "positive_control"]
)
def test_variant_is_norm_matched_at_the_answer_position(variant):
    vec = r165.build_variant_vectors(variant, **_variant_kwargs())
    assert vec.shape == (2, 2)
    # The anchor is the answer position; one scalar is applied to every row.
    assert np.linalg.norm(vec[-1]) == pytest.approx(0.5)


def test_production_is_not_rescaled():
    """Production is the reference, so it must keep its own norm."""
    kwargs = _variant_kwargs(target_norm=999.0)
    vec = r165.build_variant_vectors("production", **kwargs, match=False)
    assert np.array_equal(vec, kwargs["c_prod"])


def test_norm_matching_uses_the_requested_position():
    kwargs = _variant_kwargs(pos=0, target_norm=0.5)
    vec = r165.build_variant_vectors("oracle_ungated", **kwargs)
    assert np.linalg.norm(vec[0]) == pytest.approx(0.5)


def test_norm_matching_rejects_a_dead_zero_anchor():
    kwargs = _variant_kwargs(oracle_out=np.zeros((2, 2)), target_norm=0.5)
    with pytest.raises(ValueError, match="zero vector at position"):
        r165.build_variant_vectors("oracle_ungated", **kwargs)


def test_centered_variant_removes_the_shared_offset():
    kwargs = _variant_kwargs()
    vec = r165.build_variant_vectors("centered", **kwargs)
    raw = kwargs["c_prod"] - kwargs["c_bar"]
    # Scale factor is target/(raw mean norm); direction must be unchanged.
    cos = np.sum(vec * raw, axis=1) / (
        np.linalg.norm(vec, axis=1) * np.linalg.norm(raw, axis=1)
    )
    assert np.allclose(cos, 1.0)


def test_positive_control_is_constant_across_positions():
    vec = r165.build_variant_vectors("positive_control", **_variant_kwargs())
    assert np.allclose(vec[0], vec[1])


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        r165.build_variant_vectors("nope", **_variant_kwargs())


def test_random_proj_needs_its_matrix():
    with pytest.raises(ValueError, match="random_proj needs"):
        r165.build_variant_vectors("random_proj", **_variant_kwargs(rand_w=None))


def test_oracle_ungated_needs_its_projection():
    with pytest.raises(ValueError, match="oracle_ungated needs"):
        r165.build_variant_vectors("oracle_ungated", **_variant_kwargs(oracle_out=None))


def test_a_zero_variant_vector_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        r165.build_variant_vectors(
            "random_proj", **_variant_kwargs(value_proj_e=np.zeros((2, 2)))
        )


# --------------------------------------------------------------------------
# the frozen verdict rule
# --------------------------------------------------------------------------


def _report(**over):
    """A minimal report where every non-collapsing variant is null."""
    report = {
        "end_to_end": {
            "production": {
                "dNLL_gold_ci95": [-0.01, 0.01],
                "tv_real_vs_shuf_ci95": [0.0, 0.001],
            }
        }
    }
    for variant in r165.NON_COLLAPSING:
        report["end_to_end"][variant] = {
            "dNLL_gold_ci95": [-0.02, 0.02],
            "tv_real_vs_shuf_ci95": [0.0, 0.002],
        }
    for variant, block in over.items():
        report["end_to_end"][variant].update(block)
    return report


def test_underpowered_wins_over_everything():
    """A dead injection path must not be read as support for the bound."""
    report = _report(
        centered={"dNLL_gold_ci95": [0.5, 1.0], "tv_real_vs_shuf_ci95": [0.5, 0.6]}
    )
    assert r165._verdict(report, [-0.001, 0.001]) == "UNDERPOWERED"


def test_null_non_collapsing_variants_support_the_bound():
    assert r165._verdict(_report(), [0.05, 0.2]) == "BOUND_SURVIVES_REPAIRED_READOUT"


def test_a_positive_variant_with_more_tv_is_the_constraint():
    report = _report(
        centered={"dNLL_gold_ci95": [0.01, 0.30], "tv_real_vs_shuf_ci95": [0.02, 0.05]}
    )
    assert r165._verdict(report, [0.05, 0.2]) == "READOUT_IS_THE_CONSTRAINT"


def test_a_positive_nll_without_extra_tv_is_not_enough():
    """Both conditions of the frozen rule must hold, not just one."""
    report = _report(
        centered={"dNLL_gold_ci95": [0.01, 0.30], "tv_real_vs_shuf_ci95": [0.0, 0.001]}
    )
    assert r165._verdict(report, [0.05, 0.2]) == "BOUND_SURVIVES_REPAIRED_READOUT"


def test_ci_touching_zero_is_not_significant():
    report = _report(centered={"dNLL_gold_ci95": [0.0, 0.30]})
    assert r165._verdict(report, [0.05, 0.2]) == "BOUND_SURVIVES_REPAIRED_READOUT"
