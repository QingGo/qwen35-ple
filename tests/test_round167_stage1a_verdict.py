#!/usr/bin/env python3
"""Tests for the round-167 Stage 1a pre-registered verdict rule.

The rule is frozen in the pre-registration before any number existed, so these
tests pin the *rule*, not the outcome: each of the five verdicts is produced from
a hand-built report with a known answer.  If a later edit inverted the sign of
`gain` (which would report a depth-freeing graft as a null), the DEPTH_FREED and
DEPTH_NULL tests would catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import round167_stage1a_verdict as v

INSTRS = v.INSTRUMENTS


def _report(off, real, shuf, tv=0.5):
    """Build a minimal report dict shaped like the Stage 1a output."""
    return {
        "config": {"max_items": 200, "n_layers": 24, "layer": 2, "model": "/m/Qwen3.5-0.8B"},
        "sanity": {
            "mean_total_variation_off_vs_real": tv,
            "injection_is_live": tv > 1e-6,
        },
        "verdict_inputs": {
            "effective_depth_by_condition": {
                "off": dict(zip(INSTRS, off)),
                "real": dict(zip(INSTRS, real)),
                "shuf": dict(zip(INSTRS, shuf)),
            }
        },
    }


def test_underpowered_when_injection_is_not_live():
    """A dead injection path must never produce a scientific verdict."""
    r = _report([10, 10, 10], [10, 10, 10], [10, 10, 10], tv=1e-9)
    assert v.decide(r)["verdict"] == "UNDERPOWERED"


def test_underpowered_at_the_exact_threshold():
    r = _report([10, 10, 10], [5, 5, 5], [10, 10, 10], tv=1e-6)
    assert v.decide(r)["verdict"] == "UNDERPOWERED"


def test_depth_freed_needs_two_instruments_and_a_shuf_contrast():
    """Real moves depth earlier; shuf does not."""
    r = _report([12, 12, 12], [5, 5, 12], [12, 12, 12])
    out = v.decide(r)
    assert out["verdict"] == "DEPTH_FREED"
    assert set(out["freed_instruments"]) == set(INSTRS[:2])


def test_one_instrument_is_not_enough_for_depth_freed():
    """A single instrument moving must not carry the verdict."""
    r = _report([12, 12, 12], [5, 12, 12], [12, 12, 12])
    assert v.decide(r)["verdict"] == "MIXED"


def test_perturbation_only_when_shuf_matches_real():
    """Depth moves, but the shuffled rows move it just as much."""
    r = _report([12, 12, 12], [5, 5, 5], [5, 5, 5])
    assert v.decide(r)["verdict"] == "PERTURBATION_ONLY"


def test_depth_null_when_nothing_clears_the_floor():
    r = _report([10, 10, 10], [9, 11, 10], [10, 9, 11])
    assert v.decide(r)["verdict"] == "DEPTH_NULL"


def test_a_one_layer_shift_is_below_the_floor():
    """1 layer on a 24-layer model is explicitly not a valid shift."""
    r = _report([10, 10, 10], [9, 9, 9], [10, 10, 10])
    assert v.decide(r)["verdict"] == "DEPTH_NULL"


def test_exactly_two_layers_qualifies():
    r = _report([10, 10, 10], [8, 8, 8], [10, 10, 10])
    assert v.decide(r)["verdict"] == "DEPTH_FREED"


def test_mixed_when_instruments_disagree():
    """One instrument clears the floor, another moves the other way."""
    r = _report([12, 12, 12], [5, 20, 12], [12, 12, 12])
    assert v.decide(r)["verdict"] == "MIXED"


def test_sign_convention_positive_gain_means_earlier_transition():
    """off - real > 0 must be reported as a gain, not as a regression."""
    r = _report([12, 12, 12], [5, 5, 12], [12, 12, 12])
    g = v.decide(r)["gains"][INSTRS[0]]
    assert g["gain_real"] == 7
    assert g["off"] == 12 and g["real"] == 5


def test_none_effective_depths_do_not_crash_and_do_not_free():
    """An instrument that never stabilises returns None; it cannot vote."""
    report = _report([10, 10, 10], [10, 10, 10], [10, 10, 10])
    ed = report["verdict_inputs"]["effective_depth_by_condition"]
    ed["real"][INSTRS[0]] = None
    out = v.decide(report)
    assert out["gains"][INSTRS[0]]["gain_real"] is None
    assert out["verdict"] in v.VERDICTS


def test_rule_constants_match_the_pre_registration():
    """The thresholds are quoted from the frozen document; keep them pinned."""
    assert v.DEPTH_SHIFT_LAYERS == 2
    assert v.UNDERPOWERED_TV == 1e-6
    assert v.INSTRUMENTS == (
        "from_kl_half_max",
        "from_top5_overlap_0.3",
        "from_residual_cosine",
    )
