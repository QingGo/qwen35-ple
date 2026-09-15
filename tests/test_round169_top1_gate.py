"""Tests for the top-1 gate: the project's first non-log currency.

The gate exists to stop a line of work, so the properties worth pinning are the
ones that would let it fail in the permissive direction: a delta that is zero, a
delta that is negative, a delta that is positive but indistinguishable from
noise, and a record that does not carry the hits at all.  All four must not
produce GAIN.
"""

from __future__ import annotations

import importlib.util
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


GATE = _load("scripts/round169_top1_verdict.py", "round169_top1_verdict")


def _record(n=4000, seed=0, frozen_lift=0.0, trained_lift=0.0):
    rng = np.random.default_rng(seed)
    surprisal = rng.exponential(scale=2.0, size=n)
    base = rng.random(n) < 0.30
    hit_none = base.astype(np.uint8)
    return {
        "nll_none": surprisal,
        "hit_none": hit_none,
        "hit_frozen": (rng.random(n) < 0.30 + frozen_lift).astype(np.uint8),
        "hit_trained": (rng.random(n) < 0.30 + trained_lift).astype(np.uint8),
    }


def test_the_tail_is_selected_by_backbone_surprisal():
    rec = _record()
    g = GATE.gate(rec, tau=2.0)
    expected = int((rec["nll_none"] >= 2.0).sum())
    assert g["n_tail"] == expected
    assert 0 < g["n_tail"] < g["n_total"]


def test_a_positive_separated_delta_is_a_gain():
    rng = np.random.default_rng(1)
    n = 20_000
    rec = _record(n=n, seed=1)
    # force a real, large lift: 20 percentage points
    rec["hit_frozen"] = (rng.random(n) < 0.10).astype(np.uint8)
    rec["hit_none"] = (rng.random(n) < 0.02).astype(np.uint8)
    g = GATE.gate(rec, tau=0.0)
    assert g["headline"]["delta"] > 0
    assert g["headline"]["t"] > 3
    assert g["verdict"] == "GAIN"


@pytest.mark.parametrize("lift", [0.0, -0.02])
def test_no_lift_or_a_negative_lift_is_flat(lift):
    g = GATE.gate(_record(seed=2, frozen_lift=lift), tau=0.0)
    assert g["verdict"] == "FLAT"


def test_a_positive_but_noisy_delta_is_flat():
    # n small and the two arms independent -> the delta is within noise.
    g = GATE.gate(_record(n=200, seed=3, frozen_lift=0.03), tau=0.0)
    if g["headline"]["delta"] > 0 and (g["headline"]["t"] or 0) < 3:
        assert g["verdict"] == "FLAT"


def test_the_verdict_function_is_harsh_by_construction():
    assert GATE.verdict(-0.01, -50) == "FLAT"
    assert GATE.verdict(0.0, 99) == "FLAT"
    assert GATE.verdict(0.01, 2.9) == "FLAT"
    assert GATE.verdict(0.01, 3.1) == "GAIN"
    assert GATE.verdict(None, None) == "FLAT"
    assert GATE.verdict(float("nan"), 9.0) == "FLAT"


def test_a_record_without_hits_is_refused_rather_than_defaulted():
    rec = _record()
    del rec["hit_frozen"]
    with pytest.raises(ValueError, match="hit_none / hit_frozen"):
        GATE.gate(rec, tau=0.0)


def test_a_tau_above_every_position_is_refused():
    with pytest.raises(ValueError, match="no positions"):
        GATE.gate(_record(), tau=1e9)


def test_paired_top1_matches_a_hand_computation():
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 0, 0], dtype=np.uint8)
    c = GATE.paired_top1(a, b)
    assert c["n"] == 4
    assert c["acc_a"] == pytest.approx(0.5)
    assert c["acc_b"] == pytest.approx(0.5)
    assert c["delta"] == pytest.approx(0.0)


def test_paired_top1_refuses_misaligned_arms():
    with pytest.raises(ValueError, match="not parallel"):
        GATE.paired_top1(np.array([0, 1]), np.array([0, 1, 1]))
