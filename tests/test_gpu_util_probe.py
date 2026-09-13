#!/usr/bin/env python3
"""Tests for the GPU utilisation probe and its diagnosis rule.

The classification drives which of four very different fixes gets applied, and
applying the wrong one wastes a GPU-hour rather than saving one, so each case is
pinned with a hand-built sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gpu_util_probe as p


def _stats(util: float, used: float, total: float = 24564.0) -> dict:
    return {
        "available": True,
        "n_samples": 12,
        "util_median": util,
        "util_mean": util,
        "util_p10": util,
        "memory_used_mib_median": used,
        "memory_total_mib": total,
        "memory_fraction_median": used / total,
        "samples": [util] * 12,
    }


def test_saturated_above_target() -> None:
    d = p.classify(_stats(80.0, 20000.0))
    assert d["verdict"] == p.SATURATED
    assert d["fixes"] == []


def test_exactly_at_target_is_saturated() -> None:
    assert p.classify(_stats(50.0, 20000.0))["verdict"] == p.SATURATED


def test_occupancy_bound_is_the_round167_case() -> None:
    """23% utilisation at 21% memory -- the measured Stage 1b condition."""
    d = p.classify(_stats(23.0, 5153.0))
    assert d["verdict"] == p.OCCUPANCY_BOUND
    assert any("concurrently" in f for f in d["fixes"])


def test_memory_bound_when_the_card_is_nearly_full() -> None:
    d = p.classify(_stats(30.0, 23000.0))
    assert d["verdict"] == p.MEMORY_BOUND
    assert any("checkpointing" in f for f in d["fixes"])


def test_under_batched_in_the_middle_band() -> None:
    """Between the low and high memory fractions there is some, not all, headroom."""
    d = p.classify(_stats(30.0, 0.60 * 24564.0))
    assert d["verdict"] == p.UNDER_BATCHED


def test_unavailable_gpu_is_unknown_not_saturated() -> None:
    d = p.classify({"available": False})
    assert d["verdict"] == "UNKNOWN"


def test_target_is_configurable() -> None:
    assert p.classify(_stats(60.0, 20000.0), target=70.0)["verdict"] != p.SATURATED
    assert p.classify(_stats(60.0, 20000.0), target=50.0)["verdict"] == p.SATURATED


def test_unavailable_gpu_exits_zero_from_main(monkeypatch, capsys) -> None:
    """A machine with no GPU must not fail the gate; it just cannot be judged."""
    monkeypatch.setattr(sys, "argv", ["probe", "--seconds", "0", "--require", "50"])
    monkeypatch.setattr(p, "sample", lambda *a, **k: {"available": False})
    assert p.main() == 0


def test_gate_returns_one_below_target(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["probe", "--seconds", "0", "--require", "50"])
    monkeypatch.setattr(p, "sample", lambda *a, **k: _stats(23.0, 5153.0))
    assert p.main() == 1


def test_gate_returns_zero_at_or_above_target(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["probe", "--seconds", "0", "--require", "50"])
    monkeypatch.setattr(p, "sample", lambda *a, **k: _stats(77.0, 12000.0))
    assert p.main() == 0
