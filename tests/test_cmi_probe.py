#!/usr/bin/env python3
"""Tests for the CMI probe: the instrument must be falsified before it is used.

Two design mistakes were found while building this, and both are pinned here as
tests rather than left as comments, because each would have produced a confident
wrong answer on real data.

1. A **linear** probe cannot express complementary information. If ``h`` reveals
   one latent and ``e`` another and the label depends on the combination, no
   linear function of the concatenation can compute it, so a linear probe reports
   a near-zero CMI exactly where this instrument is supposed to look. Hence the
   MLP, and ``test_linear_probe_cannot_see_complementary_information`` exists to
   stop that from being silently reverted.

2. The wider ``(h, e)`` probe **overfits**, so an uninformative row yields a
   *negative* gap (about -0.7 nats), not a zero one. Reporting the raw gap would
   make an empty channel look worse than empty and a weak channel look empty.
   Hence the excess over a shuffled-row null, with the size of the artefact
   pinned by ``test_width_bias_is_negative_so_the_raw_gap_is_unusable``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.cmi_probe import (
    cmi_gap,
    cmi_with_null,
    fit_linear_probe,
    fit_mlp_probe,
    held_out_nll,
    mlp_scores,
    synth_benchmark,
    synth_features_for_labels,
)

# Small and fast: the assertions are about direction and ordering, not precision.
FAST = {"n": 1200, "folds": 4, "n_bootstrap": 150, "mlp_hidden": 64, "mlp_epochs": 80}


def _verdict(dose: float, seed: int, **over):
    kw = {
        "seed": seed,
        "folds": FAST["folds"],
        "n_bootstrap": FAST["n_bootstrap"],
        "mlp_hidden": FAST["mlp_hidden"],
        "mlp_epochs": FAST["mlp_epochs"],
    }
    kw.update(over)
    return cmi_with_null(*synth_benchmark(n=FAST["n"], dose=dose, seed=seed), **kw)


# --------------------------------------------------------------------------
# The null: an uninformative row must produce no gain.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_null_row_produces_no_excess_gain(seed: int) -> None:
    v = _verdict(0.0, seed)
    assert abs(v.delta_excess) < 0.2, f"null leaked a gain: {v.delta_excess:+.3f}"
    assert not v.positive


def test_width_bias_is_negative_so_the_raw_gap_is_unusable() -> None:
    """Document the artefact the excess corrects for."""
    v = _verdict(0.0, 5)
    assert v.width_bias < -0.1, f"expected a real width bias, got {v.width_bias:+.3f}"
    assert v.observed.delta < 0.0


# --------------------------------------------------------------------------
# The positive control: planted complementary information must be recovered.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_complementary_signal_is_recovered(seed: int) -> None:
    v = _verdict(1.5, seed)
    assert v.delta_excess > 0.3, f"missed complementary signal: {v.delta_excess:+.3f}"
    assert v.positive


def test_excess_is_monotone_in_the_planted_dose() -> None:
    """The strongest check available: the statistic must ORDER known levels."""
    vals = [_verdict(d, 3).delta_excess for d in (0.0, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals), f"not monotone: {vals}"
    assert abs(vals[0]) < 0.2 and vals[-1] > 0.5


def test_linear_probe_cannot_see_complementary_information() -> None:
    """Why the MLP exists. This check must not be reverted.

    With the same planted signal the linear probe's excess stays well below the
    MLP's. If the default probe were swapped back to linear, every real
    measurement would be biased toward "the channel is empty" precisely in the
    complementary case that motivates the experiment.
    """
    X_h, X_e, y = synth_benchmark(n=FAST["n"], dose=2.0, seed=7)
    common = {"folds": FAST["folds"], "n_bootstrap": FAST["n_bootstrap"], "seed": 7}
    lin = cmi_with_null(X_h, X_e, y, probe="linear", **common)
    mlp = cmi_with_null(
        X_h, X_e, y, probe="mlp", mlp_hidden=FAST["mlp_hidden"],
        mlp_epochs=FAST["mlp_epochs"], **common
    )
    assert mlp.delta_excess > lin.delta_excess + 0.3, (
        f"linear={lin.delta_excess:+.3f} should be far below mlp={mlp.delta_excess:+.3f}"
    )


# --------------------------------------------------------------------------
# The interpretation trap
# --------------------------------------------------------------------------


def test_a_strong_h_does_not_shrink_the_complementary_gap() -> None:
    """Why the benchmark is built as a composition rather than a duplicate.

    With ``y = (a + b) mod K``, knowing ``a`` perfectly still leaves ``y`` uniform
    over ``b`` -- so ``nll_h`` is bounded below by ``log K`` no matter how strong
    ``h`` is, and the measured CMI stays large. Had ``e`` been made to carry the
    *same* latent as ``h`` (the first draft), a strong ``h`` would have driven the
    gap to zero and the instrument would have looked blind for a reason that has
    nothing to do with the data.
    """
    strong = cmi_with_null(
        *synth_benchmark(n=FAST["n"], dose=2.0, h_signal=6.0, seed=9),
        seed=9, folds=FAST["folds"], n_bootstrap=FAST["n_bootstrap"],
        mlp_hidden=FAST["mlp_hidden"], mlp_epochs=FAST["mlp_epochs"],
    )
    assert strong.observed.nll_h > 2.0, "log K floors nll_h in this construction"
    assert strong.delta_excess > 0.5, (
        f"a strong h must not hide complementary information: {strong.delta_excess:+.3f}"
    )


# --------------------------------------------------------------------------
# Direction, mechanics and degenerate input
# --------------------------------------------------------------------------


def test_gap_sign_convention_is_e_helps() -> None:
    X_h, X_e, y = synth_benchmark(n=FAST["n"], dose=2.0, seed=1)
    r = cmi_gap(X_h, X_e, y, folds=FAST["folds"], seed=1,
                n_bootstrap=FAST["n_bootstrap"],
                mlp_hidden=FAST["mlp_hidden"], mlp_epochs=FAST["mlp_epochs"])
    assert r.delta == pytest.approx(r.nll_h - r.nll_he)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="first dimension"):
        cmi_gap(np.zeros((10, 2)), np.zeros((9, 2)), np.zeros(10, dtype=int))


def test_too_few_positions_raises() -> None:
    with pytest.raises(ValueError, match="at least"):
        held_out_nll(np.zeros((4, 3)), np.zeros(4, dtype=int), 2, folds=5)


def test_bad_probe_name_raises() -> None:
    with pytest.raises(ValueError, match="probe must be"):
        held_out_nll(np.zeros((40, 3)), np.zeros(40, dtype=int), 2, folds=5, probe="svm")


def test_mlp_fits_a_nonlinear_boundary() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64) + (X[:, 2] > 0).astype(np.int64)
    params = fit_mlp_probe(X, y, 3, hidden=64, epochs=300, seed=0)
    assert (mlp_scores(params, X).argmax(axis=1) == y).mean() > 0.8


def test_linear_probe_recovers_a_linear_boundary() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    W, mu, sigma = fit_linear_probe(X, y, 2, ridge=1e-6)
    assert ((((X - mu) / sigma) @ W).argmax(axis=1) == y).mean() > 0.9


def test_cmi_gap_accepts_both_probes() -> None:
    X_h, X_e, y = synth_benchmark(n=600, dose=1.0, seed=0)
    for probe in ("linear", "mlp"):
        r = cmi_gap(X_h, X_e, y, folds=4, seed=0, n_bootstrap=50, probe=probe,
                    mlp_hidden=32, mlp_epochs=40)
        assert r.n == 600


def test_result_serialises_without_the_raw_array() -> None:
    X_h, X_e, y = synth_benchmark(n=600, dose=1.0, seed=0)
    r = cmi_gap(X_h, X_e, y, folds=4, seed=0, n_bootstrap=50, probe="linear")
    assert "delta_per_position" not in r.as_dict()
    v = cmi_with_null(X_h, X_e, y, folds=4, seed=0, n_bootstrap=50, probe="linear")
    assert "delta_per_position" not in v.as_dict()["observed"]
    assert "delta_excess" in v.as_dict()


def test_synth_features_at_zero_dose_are_uncorrelated_with_the_label() -> None:
    y = np.arange(32) % 4
    X = synth_features_for_labels(y, 8, signal=0.0, seed=0)
    assert abs(float(np.corrcoef(X[:, 0], y)[0, 1])) < 0.35
