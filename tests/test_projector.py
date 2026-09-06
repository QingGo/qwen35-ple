"""Tests for the Phase A PLE projector."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qwen35_ple.projector import (
    FEATURE_NAMES,
    PleProjector,
    add_task_features,
    compute_memory_features,
    load_projector,
    save_projector,
)


def test_compute_memory_features_basic() -> None:
    logits = np.zeros(10, dtype=np.float32)
    dist = {5: 0.9, 7: 0.1}
    feats = compute_memory_features(logits, dist, matched_order=3)
    assert feats["matched_order"] == 3.0
    assert feats["density_ratio"] > 0.0
    assert feats["memory_top1_prob"] == 0.9
    assert feats["memory_top1_agree_base"] == 0.0


def test_add_task_features_one_hot() -> None:
    feats = add_task_features({"matched_order": 3.0}, "code")
    assert feats["task_code"] == 1.0
    assert feats["task_name"] == 0.0
    assert feats["task_number"] == 0.0
    assert feats["task_general"] == 0.0


def test_projector_zero_init_is_inert() -> None:
    proj = PleProjector(hidden_size=4, feature_names=FEATURE_NAMES, hidden_dim=8)
    hidden = torch.randn(1, 4)
    features = torch.zeros(1, len(FEATURE_NAMES))
    scale, bias = proj(hidden, features)
    assert float(scale[0]) == 0.0
    assert float(bias[0]) == 0.0


def test_projector_predict_np() -> None:
    proj = PleProjector(hidden_size=4, feature_names=FEATURE_NAMES, hidden_dim=8)
    hidden = np.zeros((4,), dtype=np.float32)
    feats = {name: 1.0 for name in FEATURE_NAMES}
    scale, bias = proj.predict_np(hidden, feats)
    assert float(scale) == 0.0
    assert float(bias) == 0.0


def test_projector_save_load_roundtrip(tmp_path) -> None:
    proj = PleProjector(
        hidden_size=4,
        feature_names=FEATURE_NAMES,
        hidden_dim=8,
        num_layers=2,
    )
    proj.set_feature_stats(
        np.arange(len(FEATURE_NAMES), dtype=np.float32),
        np.ones(len(FEATURE_NAMES), dtype=np.float32),
    )
    path = tmp_path / "projector.json"
    save_projector(path, proj)
    loaded = load_projector(path)
    hidden = torch.randn(2, 4)
    features = torch.randn(2, len(FEATURE_NAMES))
    s1, b1 = proj(hidden, features)
    s2, b2 = loaded(hidden, features)
    assert torch.allclose(s1, s2, atol=1e-6)
    assert torch.allclose(b1, b2, atol=1e-6)
