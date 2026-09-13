"""Tests for :mod:`qwen35_ple.stage2_gate_verdict` (frozen Stage 2.2 rule).

Every branch of the pre-registered rule gets a test, including the branch
*order* -- a rule whose branches are checked in the wrong order silently changes
its answer, and nothing else in the pipeline would catch that.
"""

from __future__ import annotations

import pytest

from qwen35_ple.stage2_gate_verdict import (
    EFFECT_MARGIN,
    OPEN_FRAC_DROP,
    SATURATED_FRAC,
    VerdictInputs,
    apply_rule,
    collect_inputs,
    render_markdown,
)

MODES = ["scalar", "per_dim"]
SEEDS = [0, 1, 2]


def _eval_report(triviaqa: float, nq: float) -> dict:
    return {
        "summary": {
            "real": {"details": [{"qa_exact": {"metrics": {
                "qa_triviaqa_em": triviaqa, "qa_nq_em": nq,
            }}}]},
        }
    }


def _gate_report(open_frac: float, *, injected: int = 100) -> dict:
    return {
        "aggregate": {
            "triviaqa": {"gate_open_frac_all_entries": open_frac},
            "nq": {"gate_open_frac_all_entries": open_frac},
        },
        "injected": injected,
    }


def _inputs(**over) -> VerdictInputs:
    """A healthy baseline: saturated scalar, selective per_dim, real gain."""
    base = VerdictInputs(
        modes=list(MODES),
        open_frac={"scalar": 0.95, "per_dim": 0.20},
        real_mean={"scalar": 0.10, "per_dim": 0.20},
        control_mean={"scalar": 0.10, "per_dim": 0.05},
        n_seeds={"scalar": 3, "per_dim": 3},
        injected={"scalar": 300, "per_dim": 300},
    )
    for key, value in over.items():
        setattr(base, key, value)
    return base


# --------------------------------------------------------------------------
# collect_inputs
# --------------------------------------------------------------------------
def test_rule_mean_averages_triviaqa_and_nq():
    evals = {
        ("scalar", "real", s): _eval_report(0.10, 0.20) for s in SEEDS
    }
    gates = {("scalar", s): _gate_report(0.95) for s in SEEDS}
    got = collect_inputs(evals, gates, modes=["scalar"], seeds=SEEDS)
    assert got.real_mean["scalar"] == pytest.approx(0.15)
    assert got.n_seeds["scalar"] == 3


def test_collect_inputs_averages_the_gate_fraction_over_tasks_and_seeds():
    evals = {("scalar", "real", s): _eval_report(0.1, 0.2) for s in SEEDS}
    gates = {("scalar", s): _gate_report(0.90 + 0.01 * s) for s in SEEDS}
    got = collect_inputs(evals, gates, modes=["scalar"], seeds=SEEDS)
    assert got.open_frac["scalar"] == pytest.approx((0.90 + 0.91 + 0.92) / 3)


def test_collect_inputs_records_missing_artifacts():
    got = collect_inputs({}, {}, modes=MODES, seeds=SEEDS)
    assert "gate:scalar:seed0" in got.missing
    assert "eval:per_dim:control:seed2" in got.missing


def test_collect_inputs_reports_per_seed_rows():
    evals = {("scalar", "real", s): _eval_report(0.1 * s, 0.2) for s in SEEDS}
    gates = {("scalar", s): _gate_report(0.9) for s in SEEDS}
    got = collect_inputs(evals, gates, modes=["scalar"], seeds=SEEDS)
    rows = got.per_seed["scalar:real"]
    assert [r["seed"] for r in rows] == SEEDS
    assert rows[1]["triviaqa_em"] == pytest.approx(0.1)


# --------------------------------------------------------------------------
# branch 0: the run is void
# --------------------------------------------------------------------------
def test_missing_inputs_is_incomplete_not_a_null():
    got = _inputs(missing=["gate:per_dim:seed1"])
    assert apply_rule(got)["label"] == "INCOMPLETE"


def test_zero_injection_is_underpowered():
    got = _inputs(injected={"scalar": 0, "per_dim": 300})
    out = apply_rule(got)
    assert out["label"] == "UNDERPOWERED"
    assert "no-op" in out["reason"]


def test_a_non_saturated_scalar_arm_means_the_disease_was_not_reproduced():
    got = _inputs(open_frac={"scalar": 0.40, "per_dim": 0.10})
    out = apply_rule(got)
    assert out["label"] == "DISEASE_NOT_REPRODUCED"
    assert "< 0.5" in out["reason"]
    # the disease itself is saturation (round-146: mean 0.77-0.98)
    assert "0.77-0.98" in out["reason"]


# --------------------------------------------------------------------------
# branch order
# --------------------------------------------------------------------------
def test_saturation_is_checked_before_selectivity():
    """O(per_dim) >= 0.90 must win even when the metric moved a lot.

    The rule lists GATE_STILL_SATURATED first, so a per_dim gate that is just as
    open disproves the *mechanism* regardless of what the metric did.
    """
    got = _inputs(
        open_frac={"scalar": 0.99, "per_dim": 0.95},
        real_mean={"scalar": 0.10, "per_dim": 0.30},
        control_mean={"scalar": 0.10, "per_dim": 0.05},
    )
    assert apply_rule(got)["label"] == "GATE_STILL_SATURATED"


def test_anti_no_op_is_checked_before_saturation():
    got = _inputs(
        open_frac={"scalar": 0.99, "per_dim": 0.95},
        injected={"scalar": 300, "per_dim": 0},
    )
    assert apply_rule(got)["label"] == "UNDERPOWERED"


# --------------------------------------------------------------------------
# the three real branches
# --------------------------------------------------------------------------
def test_selectivity_helps_when_all_three_conditions_hold():
    # Margins are kept clear of the thresholds: 0.13 - 0.10 is 0.019999... in
    # binary floating point, so sitting exactly on the boundary is a test of
    # IEEE754, not of the rule.
    got = _inputs(
        open_frac={"scalar": 0.95, "per_dim": 0.70},
        real_mean={"scalar": 0.10, "per_dim": 0.13},
        control_mean={"scalar": 0.10, "per_dim": 0.05},
    )
    out = apply_rule(got)
    assert out["label"] == "GATE_SELECTIVITY_HELPS"
    assert out["selectivity"]["open_frac_drop"] == pytest.approx(0.25)


def test_selectivity_no_effect_when_the_metric_does_not_move():
    got = _inputs(
        open_frac={"scalar": 0.95, "per_dim": 0.70},
        real_mean={"scalar": 0.100, "per_dim": 0.105},
        control_mean={"scalar": 0.10, "per_dim": 0.05},
    )
    assert apply_rule(got)["label"] == "GATE_SELECTIVITY_NO_EFFECT"


def test_partial_when_the_gate_did_not_become_selective():
    got = _inputs(open_frac={"scalar": 0.95, "per_dim": 0.80})
    out = apply_rule(got)
    assert out["label"] == "PARTIAL"
    assert out["selectivity"]["is_selective"] is False


def test_partial_when_the_gain_is_not_content_related():
    """Selective, moves the metric, but no better than its own control."""
    got = _inputs(
        open_frac={"scalar": 0.95, "per_dim": 0.70},
        real_mean={"scalar": 0.10, "per_dim": 0.20},
        control_mean={"scalar": 0.10, "per_dim": 0.19},
    )
    out = apply_rule(got)
    assert out["label"] == "PARTIAL"
    assert out["selectivity"]["gain_vs_control"] == pytest.approx(0.01)


def test_a_small_positive_gain_is_no_effect_not_partial():
    """The frozen ELSE-IF chain puts |gain| < 0.02 in NO_EFFECT, even when the
    arm beats its own control: a gate that is selective and moves the metric by
    a hair is exactly the "selectivity does not matter" finding."""
    got = _inputs(
        open_frac={"scalar": 0.95, "per_dim": 0.70},
        real_mean={"scalar": 0.20, "per_dim": 0.21},
        control_mean={"scalar": 0.10, "per_dim": 0.01},
    )
    assert apply_rule(got)["label"] == "GATE_SELECTIVITY_NO_EFFECT"


def test_a_selective_but_harmful_gate_falls_through_to_partial():
    """Recorded because the frozen rule has NO branch for it.

    A drop of 0.05 is not NO_EFFECT (|gain| >= 0.02) and not HELPS, so the
    catch-all PARTIAL fires.  The rule's author did not enumerate this case; it
    is reported as PARTIAL with the arm-by-arm numbers, and this test exists so
    that the omission is visible rather than silently reinterpreted.
    """
    got = _inputs(
        open_frac={"scalar": 0.95, "per_dim": 0.70},
        real_mean={"scalar": 0.20, "per_dim": 0.15},
        control_mean={"scalar": 0.10, "per_dim": 0.05},
    )
    out = apply_rule(got)
    assert out["label"] == "PARTIAL"
    assert out["selectivity"]["gain_vs_scalar"] == pytest.approx(-0.05)


def test_thresholds_are_the_pre_registered_ones():
    out = apply_rule(_inputs())
    assert out["thresholds"] == {
        "open_frac_drop": OPEN_FRAC_DROP,
        "effect_margin": EFFECT_MARGIN,
        "saturated_frac": SATURATED_FRAC,
    }


def test_rule_requires_both_arms():
    with pytest.raises(ValueError, match="scalar and per_dim"):
        apply_rule(_inputs(modes=["scalar"]))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def test_markdown_reports_every_quantity_the_rule_read():
    out = apply_rule(_inputs())
    md = render_markdown(out)
    assert out["label"] in md
    assert "gate_open_frac_all_entries" in md
    assert "gain vs the arm's own control" in md
    assert "frozen in the pre-registration" in md
