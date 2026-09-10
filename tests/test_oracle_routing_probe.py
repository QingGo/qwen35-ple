"""Tests for the oracle-routing probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_oracle_routing_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_oracle_routing_probe", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_handles_perfect_reversed_and_ties():
    module = _load_module()
    y = np.asarray([0, 0, 1, 1])
    assert module._auc(y, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert module._auc(y, np.asarray([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert module._auc(y, np.asarray([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_stratified_folds_cover_all_items():
    module = _load_module()
    labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 0])
    groups = np.asarray(["boolq"] * 4 + ["triviaqa"] * 4)
    folds = module._stratified_folds(labels, groups, n_folds=4)
    seen = np.concatenate([test for _, test in folds])
    assert sorted(seen.tolist()) == list(range(len(labels)))
    # Each (group, label) stratum here holds two items, so at most two folds can
    # be non-empty: folds with an empty test or train side are dropped rather
    # than returned unusable.  Assert the invariant that matters (full coverage,
    # every fold usable) instead of demanding more folds than the strata fill.
    assert len(folds) >= 1
    for train, test in folds:
        assert len(train) > 0 and len(test) > 0
        assert not set(train.tolist()) & set(test.tolist())


def test_stratified_folds_fills_requested_folds_when_strata_allow():
    module = _load_module()
    labels = np.asarray([0, 1] * 8)
    groups = np.asarray(["boolq", "triviaqa"] * 8)
    folds = module._stratified_folds(labels, groups, n_folds=4)
    seen = np.concatenate([test for _, test in folds])
    assert sorted(seen.tolist()) == list(range(len(labels)))
    assert len(folds) == 4
    for train, test in folds:
        assert not set(train.tolist()) & set(test.tolist())


def test_cross_val_learns_separable_signal():
    module = _load_module()
    rng = np.random.default_rng(0)
    n = 60
    y = np.asarray([0, 1] * (n // 2))
    X = rng.normal(size=(n, 2)).astype(np.float32)
    X[y == 1, 0] += 3.0
    groups = np.asarray(["boolq"] * n)
    result = module._cross_val(X, y, groups, n_folds=3)
    assert result["auc"] > 0.9
    assert result["n"] == n
