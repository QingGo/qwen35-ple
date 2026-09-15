"""Tests for the gate study.

The study's entire conclusion is a ratio, and the one mistake that would make
that ratio meaningless is a feature that leaks the label.  `nll_none` predicts
`nll_none - nll_frozen > 0` almost perfectly and is also, by construction,
unavailable at inference -- a gate built on it would be the best-looking result in
the project and completely unrunnable.  So the refusal is enforced in code and
tested first.

Second is the fold structure.  The coupling audit showed a changed row perturbs
every later position in its 1024-token chunk with |delta| decaying in token
distance, so neighbouring positions share features AND labels.  A random split
would train on a position and test on its neighbour, so contiguity is a
correctness requirement and is pinned as one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load("scripts/round169_gate_study.py", "round169_gate_study")


def _rec(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "score": np.arange(n, dtype=np.int64),
        "nll_none": rng.normal(2.5, 1.0, n),
        "nll_frozen": rng.normal(2.4, 1.0, n),
        "has_train_row": (rng.random(n) < 0.4).astype(np.uint8),
        "ent_none": rng.normal(1.0, 0.2, n),
        "ent_frozen": rng.normal(0.9, 0.2, n),
        "maxp_none": rng.random(n),
        "maxp_frozen": rng.random(n),
        "logmargin_none": rng.random(n),
        "logmargin_frozen": rng.random(n),
        "top1_none": rng.integers(0, 50, n),
        "top1_frozen": rng.integers(0, 50, n),
        "context_count": rng.integers(0, 100, n),
        "hit_none": (rng.random(n) < 0.5).astype(np.uint8),
    }


# --------------------------------------------------------------------------
# the leak guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["nll_none", "nll_frozen", "nll_trained", "delta",
                                   "hit_none", "hit_frozen", "hit_trained"])
def test_oracle_features_are_refused_by_name(field):
    with pytest.raises(ValueError, match="oracle"):
        GS.observable_features(_rec(), ["ent_none", field])


def test_the_refusal_names_every_offending_feature():
    with pytest.raises(ValueError) as e:
        GS.observable_features(_rec(), ["nll_none", "delta", "ent_none"])
    msg = str(e.value)
    assert "nll_none" in msg and "delta" in msg


def test_the_default_features_contain_no_oracle():
    assert not (set(GS.DEFAULT_FEATURES) & GS.ORACLE_FIELDS)


def test_a_missing_feature_is_named():
    with pytest.raises(ValueError, match="no field"):
        GS.observable_features({"score": np.zeros(3)}, ["ent_none"])


def test_non_finite_features_are_rejected():
    r = _rec()
    r["ent_none"] = r["ent_none"].copy()
    r["ent_none"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GS.observable_features(r, ["ent_none"])


def test_misaligned_features_are_rejected():
    r = _rec(n=10)
    r["ent_frozen"] = r["ent_frozen"][:5]
    with pytest.raises(ValueError, match="expected 10"):
        GS.observable_features(r, ["ent_none", "ent_frozen"])


# --------------------------------------------------------------------------
# derived features
# --------------------------------------------------------------------------


def test_entropy_shift_is_none_minus_frozen():
    r = _rec(n=6)
    r["ent_none"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r["ent_frozen"] = np.array([0.5, 2.5, 2.0, 3.0, 6.0, 1.0])
    X, names = GS.observable_features(r, ["ent_shift"])
    assert names == ["ent_shift"]
    assert X[:, 0].tolist() == [0.5, -0.5, 1.0, 1.0, -1.0, 5.0]


def test_agreement_is_one_when_the_two_arms_pick_the_same_argmax():
    r = _rec(n=4)
    r["top1_none"] = np.array([7, 8, 9, 10])
    r["top1_frozen"] = np.array([7, 0, 9, 1])
    X, _ = GS.observable_features(r, ["agree"])
    assert X[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_ent_shift_needs_both_entropies():
    with pytest.raises(ValueError, match="ent_none and ent_frozen"):
        GS.observable_features({"score": np.zeros(2), "ent_none": np.zeros(2)}, ["ent_shift"])


def test_feature_matrix_is_column_stacked_in_the_order_asked():
    r = _rec(n=5)
    r["ent_none"] = np.arange(5.0)
    r["maxp_none"] = np.arange(5.0) * 10
    X, names = GS.observable_features(r, ["ent_none", "maxp_none"])
    assert names == ["ent_none", "maxp_none"]
    assert X[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert X[:, 1].tolist() == [0, 10, 20, 30, 40]


# --------------------------------------------------------------------------
# fold structure
# --------------------------------------------------------------------------


def test_folds_are_contiguous_blocks_of_the_stream():
    # A random split would put a position and its neighbour in different folds,
    # and the coupling makes those two nearly the same example.
    score = np.arange(100, dtype=np.int64)
    folds = GS.stream_folds(score, 5)
    for k in range(5):
        idx = np.flatnonzero(folds == k)
        assert idx.size == 20
        assert idx.min() == k * 20 and idx.max() == k * 20 + 19


def test_folds_follow_stream_order_not_array_order():
    score = np.array([50, 10, 90, 20, 80, 30, 70, 40, 60, 0], dtype=np.int64)
    folds = GS.stream_folds(score, 2)
    # The two smallest stream positions must land in fold 0, not the first two slots.
    assert folds[np.argmin(score)] == 0
    assert folds[np.argmax(score)] == 1


def test_fold_count_must_be_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        GS.stream_folds(np.arange(10, dtype=np.int64), 1)


def test_every_position_lands_in_exactly_one_fold():
    rng = np.random.default_rng(0)
    score = np.sort(rng.integers(0, 10_000, 997)).astype(np.int64)
    folds = GS.stream_folds(score, 7)
    assert folds.min() == 0 and folds.max() == 6
    assert np.bincount(folds).sum() == score.size


# --------------------------------------------------------------------------
# standardisation
# --------------------------------------------------------------------------


def test_standardisation_uses_train_statistics_only():
    train = np.arange(100, dtype=np.float64).reshape(-1, 1)
    test = np.full((10, 1), 1000.0)
    _, te = GS.standardize(train, test)
    # Exactly what the TRAIN statistics say it should be.  Had the test block been
    # standardised with its own mean and spread, every value would be 0.
    assert te.mean() == pytest.approx((1000.0 - train.mean()) / train.std())


def test_standardisation_is_identity_when_train_is_already_standard():
    train = np.array([[0.0], [1.0], [2.0]])
    out, _ = GS.standardize(train, train)
    assert out.mean() == pytest.approx(0.0)
    assert out.std() == pytest.approx(1.0, abs=0.2)


def test_a_constant_feature_does_not_divide_by_zero():
    train = np.full((5, 1), 3.0)
    out, _ = GS.standardize(train, train)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# gate value
# --------------------------------------------------------------------------


def test_gate_value_is_the_sum_of_gain_where_the_gate_says_yes():
    gain = np.array([2.0, -1.0, 3.0, -4.0])
    dec = np.array([True, False, True, False])
    assert GS.gate_value(gain, dec) == pytest.approx(5.0 / 4)
    assert GS.gate_value(gain, np.ones(4, dtype=bool)) == pytest.approx(0.0)
    assert GS.gate_value(gain, np.zeros(4, dtype=bool)) == pytest.approx(0.0)


def test_gate_value_uses_the_given_total_as_denominator():
    gain = np.array([2.0, 2.0])
    assert GS.gate_value(gain, np.array([True, False]), total=8) == pytest.approx(0.25)


def test_gate_value_rejects_a_mismatched_decision():
    with pytest.raises(ValueError, match="same length"):
        GS.gate_value(np.ones(3), np.ones(2, dtype=bool))


def test_gate_value_rejects_a_nonpositive_total():
    with pytest.raises(ValueError, match="positive"):
        GS.gate_value(np.ones(2), np.ones(2, dtype=bool), total=0)


# --------------------------------------------------------------------------
# thresholds and the learned gate
# --------------------------------------------------------------------------


def test_best_threshold_finds_a_separating_cut():
    f = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 5.3])
    gain = np.array([-1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0])
    t, val = GS.best_threshold(f, gain, np.ones(8, dtype=bool))
    assert 0.3 < t <= 5.0
    assert val == pytest.approx(8.0 / 8)


def test_best_threshold_lower_is_better_direction():
    f = np.array([0.0, 0.1, 5.0, 5.1])
    gain = np.array([2.0, 2.0, -1.0, -1.0])
    _, val = GS.best_threshold(f, gain, np.ones(4, dtype=bool), higher_is_better=False)
    assert val == pytest.approx(4.0 / 4)


def test_best_threshold_on_an_empty_fold_raises():
    with pytest.raises(ValueError, match="empty"):
        GS.best_threshold(np.ones(4), np.ones(4), np.zeros(4, dtype=bool))


def test_logistic_regression_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    w = GS.logistic_fit(X, y)
    pred = GS.logistic_apply(w, X) >= 0.5
    assert (pred == (y > 0.5)).mean() > 0.95


def test_logistic_apply_is_monotone_in_the_score():
    w = np.array([0.0, 1.0])
    X = np.array([[0.0], [1.0], [2.0]])
    p = GS.logistic_apply(w, X)
    assert p[0] < p[1] < p[2]


def test_logistic_fit_rejects_misaligned_labels():
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        GS.logistic_fit(np.zeros((4, 2)), np.zeros(3))


# --------------------------------------------------------------------------
# the two regimes
# --------------------------------------------------------------------------


def test_two_regime_report_splits_on_has_train_row():
    gain = np.array([2.0, 2.0, -1.0, -1.0])
    row = np.array([True, True, False, False])
    got = GS.two_regime_report(gain, row)
    assert got["seen"]["n"] == 2 and got["seen"]["gain"] == pytest.approx(2.0)
    assert got["unseen"]["n"] == 2 and got["unseen"]["gain"] == pytest.approx(-1.0)
    assert got["seen"]["share"] == pytest.approx(0.5)


def test_two_regime_report_counts_gated_value_separately():
    gain = np.array([2.0, -1.0, 2.0, -1.0])
    row = np.array([True, True, False, False])
    dec = np.array([True, False, False, False])
    got = GS.two_regime_report(gain, row, dec)
    assert got["seen"]["total"] == pytest.approx(1.0)
    assert got["seen"]["gated_total"] == pytest.approx(2.0)
    assert got["unseen"]["gated_total"] == pytest.approx(0.0)


def test_two_regime_report_rejects_a_mismatch():
    with pytest.raises(ValueError, match="same length"):
        GS.two_regime_report(np.ones(3), np.ones(2, dtype=bool))


# --------------------------------------------------------------------------
# render and end to end
# --------------------------------------------------------------------------


def _study(**over):
    base = {
        "record": "r.npz", "n": 100, "features": ["ent_none"], "k_folds": 10,
        "unconditional": 0.175931, "oracle_sign": 0.534162, "oracle_surprisal": 0.207040,
        "oracle_tau": 2.0,
        "gates": [{"name": "threshold: ent_none", "keep": 0.5, "test_value": 0.19,
                   "train_multiple": 1.15, "test_multiple": 1.08,
                   "kind": "single-feature threshold"}],
        "regimes": {
            "seen": {"n": 40, "share": 0.4, "gain": 0.3, "total": 12.0,
                     "gated_total": 12.0, "frac_positive": 0.6},
            "unseen": {"n": 60, "share": 0.6, "gain": 0.05, "total": 3.0,
                       "gated_total": 3.0, "frac_positive": 0.52},
        },
        "logistic_coef_mean": {},
        "note": "note text",
    }
    base.update(over)
    return base


def test_render_shows_both_oracles_so_the_scale_is_visible():
    text = GS.render(_study())
    assert "oracle sign" in text and "unrunnable" in text


def test_render_prints_train_and_test_multiple_side_by_side():
    # The gap between them is the first thing a leak looks like, so they travel
    # together on every row.
    text = GS.render(_study())
    assert "1.08x" in text and "1.15x" in text


def test_end_to_end_on_a_synthetic_record(tmp_path, monkeypatch):
    n = 4000
    rng = np.random.default_rng(1)
    row = (rng.random(n) < 0.4)
    ent = rng.normal(1.0, 0.5, n)
    # The injection helps exactly where the backbone entropy is high, so a gate on
    # ent_none should find real value, and the coupling-free synthetic stream means
    # a held-out fold is honest.
    gain = np.where(ent > 1.0, 0.6, -0.2) + rng.normal(0, 0.05, n)
    rec = {
        "score": np.arange(n, dtype=np.int64),
        "nll_none": (2.5 + rng.normal(0, 0.1, n)).astype(np.float32),
        "has_train_row": row.astype(np.uint8),
        "ent_none": ent.astype(np.float32),
        "ent_frozen": (ent - 0.1).astype(np.float32),
    }
    rec["nll_frozen"] = (rec["nll_none"].astype(np.float64) - gain).astype(np.float32)
    path = tmp_path / "gate.npz"
    np.savez_compressed(path, **rec)
    out = tmp_path / "study.json"
    monkeypatch.setattr(sys, "argv", [
        "gs", "--gate-record", str(path), "--features", "ent_none,ent_frozen",
        "--k-folds", "5", "--out", str(out),
    ])
    assert GS.main() == 0
    got = json.loads(out.read_text())
    best = got["gates"][0]
    assert best["test_multiple"] > 1.1, got["gates"]
    assert got["regimes"]["seen"]["n"] == int(row.sum())


def test_end_to_end_refuses_an_oracle_feature(tmp_path, monkeypatch):
    rec = _rec(n=100)
    path = tmp_path / "gate.npz"
    np.savez_compressed(path, **rec)
    monkeypatch.setattr(sys, "argv", [
        "gs", "--gate-record", str(path), "--features", "nll_none",
    ])
    with pytest.raises(ValueError, match="oracle"):
        GS.main()
