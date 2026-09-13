#!/usr/bin/env python3
"""Tests for the round-167 Stage 1b frozen decision rule.

The rule decides whether the read-out collapse is a training-recipe artifact or
an architectural limit, and that distinction changes what the paper may claim.
So each of the five verdicts is pinned against a hand-built input, including the
sign of the comparison: a variant that *raises* PR(c_t) must be read as the
collapse being relieved, not as missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import round167_stage1b_analysis as a


def _stages(pr_e_t=136.7, pr_vp=14.4, pr_bs=2.7, pr_ct=1.02, dim=2560):
    return {
        "e_t": {"PR": pr_e_t, "cos_real_vs_shuf": 0.99, "dim": dim, "duplicate_row_fraction": 0.0},
        "value_proj": {"PR": pr_vp, "cos_real_vs_shuf": 0.5, "dim": 2560, "duplicate_row_fraction": 0.0},
        "branch_sum": {"PR": pr_bs, "cos_real_vs_shuf": 0.3, "dim": 2560, "duplicate_row_fraction": 0.0},
        "c_t": {"PR": pr_ct, "cos_real_vs_shuf": 0.1, "dim": 1024, "duplicate_row_fraction": 0.9},
    }


def test_baseline_reproduced_with_historical_numbers() -> None:
    """The production configuration should reproduce 1.020 and stay architectural."""
    out = a.decide({"prod": _stages()})
    assert out["verdict"] == "COLLAPSE_IS_ARCHITECTURAL"


def test_recipe_verdict_when_a_variant_lifts_pr_fivefold() -> None:
    out = a.decide({"prod": _stages(), "linear-nozero": _stages(pr_bs=40.0, pr_ct=30.0)})
    assert out["verdict"] == "COLLAPSE_IS_RECIPE"
    assert out["relieved_variants"] == ["linear-nozero"]


def test_fivefold_relative_is_not_enough_without_absolute_floor() -> None:
    """5x of 1.02 is 5.1, which clears both; 5x of 0.5 would not."""
    out = a.decide({"prod": _stages(pr_ct=0.5), "v": _stages(pr_ct=2.6)})
    assert out["verdict"] != "COLLAPSE_IS_RECIPE"


def test_partial_when_lifted_but_below_the_recipe_bar() -> None:
    out = a.decide({"prod": _stages(), "nozero": _stages(pr_bs=6.0, pr_ct=3.0)})
    assert out["verdict"] == "PARTIAL"


def test_baseline_not_reproduced_guard() -> None:
    out = a.decide({"prod": _stages(pr_ct=1.4)})
    assert out["verdict"] == "BASELINE_NOT_REPRODUCED"


def test_baseline_tolerance_boundary() -> None:
    """1.07 is exactly 1.020 + 0.05, so it is still accepted."""
    assert a.decide({"prod": _stages(pr_ct=1.07)})["verdict"] != "BASELINE_NOT_REPRODUCED"
    assert a.decide({"prod": _stages(pr_ct=1.08)})["verdict"] == "BASELINE_NOT_REPRODUCED"


def test_addressing_sanity_failure_invalidates() -> None:
    """If PR(e_t) collapses the addressing is not what we think it is."""
    out = a.decide({"prod": _stages(pr_e_t=10.0)})
    assert out["verdict"] == "INVALID"
    assert "prod" in out["invalid_variants"]


def test_no_prod_variant_is_invalid() -> None:
    assert a.decide({"nozero": _stages()})["verdict"] == "INVALID"


def test_rule_constants_match_the_pre_registration() -> None:
    assert a.OFFSET == "12"
    assert a.HISTORICAL_BASELINE_PR_C_T == 1.020
    assert a.BASELINE_TOLERANCE == 0.05
    assert a.MIN_E_T_PR == 50.0
    assert a.RECIPE_MULTIPLE == 5.0
    assert a.RECIPE_ABSOLUTE == 5.0
    assert a.ARCHITECTURAL_CEILING == 2.0


def test_extract_reads_the_provenance_shape() -> None:
    prov = {
        "by_offset": {
            "12": {
                "overall": {
                    "e_t": {"PR": 136.71, "cos_real_vs_shuf_mean": 0.9, "dim": 2560, "duplicate_row_fraction": 0.0},
                    "value_proj": {"PR": 14.4, "cos_real_vs_shuf_mean": 0.5, "dim": 2560, "duplicate_row_fraction": 0.0},
                    "branch_sum": {"PR": 2.676, "cos_real_vs_shuf_mean": 0.3, "dim": 2560, "duplicate_row_fraction": 0.0},
                    "c_t": {"PR": 1.020, "cos_real_vs_shuf_mean": 0.1, "dim": 1024, "duplicate_row_fraction": 0.9},
                }
            }
        }
    }
    out = a.extract(prov)
    assert out["e_t"]["PR"] == pytest.approx(136.71)
    assert out["c_t"]["PR"] == pytest.approx(1.020)


def test_attenuation_reports_the_per_stage_drop() -> None:
    s = _stages(pr_e_t=100.0, pr_vp=10.0, pr_bs=2.0, pr_ct=1.0)
    assert a._ratio(s, "e_t", "value_proj") == pytest.approx(10.0)
    assert a._ratio(s, "branch_sum", "c_t") == pytest.approx(2.0)
    assert a._ratio(s, "e_t", "c_t") == pytest.approx(100.0)
