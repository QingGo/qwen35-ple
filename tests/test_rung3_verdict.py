"""Tests for the frozen Stage 1.5g rung-3 decision rule.

The rule itself is the deliverable here: it was written down in
``docs/round-168-stage1.5g-preregistration.md`` section 3 before any aligned
number existed, so what these tests pin is that the implementation still says
what the document says -- including the two amendments (4.1 conservative-across-K,
4.2 dead-needs-both-probes) that are easy to lose.
"""

from __future__ import annotations

import pytest

from qwen35_ple.rung3_verdict import (
    CARRY_FLOOR,
    DEAD_MARGIN,
    DEGRADED_FLOOR,
    PRIMARY_K,
    SENSITIVITY_K,
    VerdictInputs,
    apply_rule,
    collect_inputs,
    recovery,
    render_markdown,
    verdict_for_k,
)


def make_report(
    *,
    probe: float,
    count: float,
    shuffled: float | None = None,
    majority: float = 0.05,
    mlp: float | None = None,
    mlp_shuffled: float | None = None,
    shuffled_eval: float | None = None,
) -> dict:
    """A probe JSON with just the keys the rule reads."""
    results: dict[str, dict] = {
        "probe_raw_rows": {"eval": {"top1": probe}},
        "count_trigram": {"eval": {"top1": count}},
        "majority_train_prior": {"eval": {"top1": majority}},
    }
    if shuffled is not None:
        results["control_shuffled_train_rows"] = {"eval": {"top1": shuffled}}
    if shuffled_eval is not None:
        results["control_shuffled_eval_rows"] = {"eval": {"top1": shuffled_eval}}
    if mlp is not None:
        results["probe_mlp_raw_rows"] = {"eval": {"top1": mlp}}
        results["control_shuffled_train_rows_mlp"] = {
            "eval": {"top1": mlp_shuffled if mlp_shuffled is not None else majority}
        }
    return {"results": results}


# --------------------------------------------------------------------------- #
# the four frozen branches
# --------------------------------------------------------------------------- #
def test_recovery_is_the_ratio_of_the_two_named_quantities():
    assert recovery(make_report(probe=0.30, count=0.50)) == pytest.approx(0.60)


def test_high_r_carries_the_trigram():
    v = verdict_for_k(make_report(probe=0.30, count=0.50, shuffled=0.06))
    assert v["label"] == "ROWS_CARRY_THE_TRIGRAM"
    assert v["recovery_R"] == pytest.approx(0.60)


def test_middling_r_is_degraded():
    # R = 0.30, comfortably between the two frozen floors.
    v = verdict_for_k(make_report(probe=0.15, count=0.50, shuffled=0.06))
    assert v["label"] == "ROWS_DEGRADED"


def test_low_r_but_above_the_floor_is_nearly_dead():
    # R = 0.16: below the degraded floor, and the probe is 0.08 vs floor 0.06,
    # which clears the dead margin so ROWS_DEAD must NOT fire.
    v = verdict_for_k(make_report(probe=0.08, count=0.50, shuffled=0.06))
    assert v["label"] == "ROWS_NEARLY_DEAD"


def test_below_the_shuffled_floor_is_dead_when_both_probes_agree():
    v = verdict_for_k(
        make_report(probe=0.055, count=0.50, shuffled=0.06, mlp=0.055, mlp_shuffled=0.06)
    )
    assert v["label"] == "ROWS_DEAD"
    assert v["linear_at_floor"] is True
    assert v["mlp_at_floor"] is True


# --------------------------------------------------------------------------- #
# section 4.2: one weak probe is not evidence the rows are empty
# --------------------------------------------------------------------------- #
def test_a_live_mlp_blocks_the_dead_verdict():
    """The amendment that matters: a linear probe's R is a lower bound (TD-9)."""
    v = verdict_for_k(
        make_report(probe=0.055, count=0.50, shuffled=0.06, mlp=0.25, mlp_shuffled=0.06)
    )
    assert v["label"] != "ROWS_DEAD"
    assert v["label"] == "ROWS_NEARLY_DEAD"
    assert "NOT ROWS_DEAD" in v["reason"]


def test_a_skipped_mlp_also_blocks_the_dead_verdict():
    v = verdict_for_k(make_report(probe=0.055, count=0.50, shuffled=0.06))
    assert v["label"] != "ROWS_DEAD"
    assert v["mlp_top1"] is None
    assert "was not run" in v["reason"]


def test_the_floor_uses_the_shuffled_row_control_not_the_label_permutation():
    """Section 2 names the shuffled-ROW control as the mandatory one."""
    v = verdict_for_k(
        make_report(probe=0.055, count=0.50, shuffled=0.06, shuffled_eval=0.5)
    )
    assert v["floor"] == pytest.approx(0.06)
    assert v["shuffled_eval_top1"] == pytest.approx(0.5)


def test_majority_prior_can_be_the_larger_floor():
    v = verdict_for_k(make_report(probe=0.20, count=0.50, shuffled=0.02, majority=0.30))
    assert v["floor"] == pytest.approx(0.30)
    # 0.20 <= 0.30 + 0.005, so the probe really is at the floor.
    assert v["linear_at_floor"] is True


# --------------------------------------------------------------------------- #
# incomplete inputs must not silently become a verdict
# --------------------------------------------------------------------------- #
def test_zero_count_trigram_makes_r_undefined():
    v = verdict_for_k(make_report(probe=0.30, count=0.0, shuffled=0.06))
    assert v["label"] == "INCOMPLETE"
    assert "undefined" in v["reason"]


def test_missing_shuffled_control_is_incomplete_not_a_pass():
    v = verdict_for_k(make_report(probe=0.30, count=0.50))
    assert v["label"] == "INCOMPLETE"
    assert "shuffled" in v["reason"]


def test_a_high_probe_with_no_control_cannot_claim_success():
    """The round-161 lesson: without the control, 0.30/0.50 is uninterpretable."""
    v = verdict_for_k(make_report(probe=0.48, count=0.50))
    assert v["label"] == "INCOMPLETE"


# --------------------------------------------------------------------------- #
# section 4.1: both K values, and disagreement resolves conservatively
# --------------------------------------------------------------------------- #
def test_agreement_across_k_is_reported():
    inputs = collect_inputs(
        {
            PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06),
            SENSITIVITY_K: make_report(probe=0.29, count=0.50, shuffled=0.06),
        }
    )
    out = apply_rule(inputs)
    assert out["verdict"] == "ROWS_CARRY_THE_TRIGRAM"
    assert out["agreement"] is True


def test_disagreement_takes_the_more_conservative_verdict():
    inputs = collect_inputs(
        {
            PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06),  # CARRY
            SENSITIVITY_K: make_report(probe=0.15, count=0.50, shuffled=0.06),  # DEGRADED
        }
    )
    out = apply_rule(inputs)
    assert out["verdict"] == "ROWS_DEGRADED"
    assert out["agreement"] is False


def test_conservatism_orders_dead_worst_even_when_k_1000_looks_better():
    inputs = collect_inputs(
        {
            PRIMARY_K: make_report(probe=0.35, count=0.50, shuffled=0.06),  # CARRY
            SENSITIVITY_K: make_report(
                probe=0.055, count=0.50, shuffled=0.06, mlp=0.055, mlp_shuffled=0.06
            ),  # DEAD
        }
    )
    out = apply_rule(inputs)
    assert out["verdict"] == "ROWS_DEAD"


def test_a_missing_k_is_recorded_as_a_problem():
    inputs = collect_inputs({PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06)})
    assert any(str(SENSITIVITY_K) in p for p in inputs.problems)


def test_one_incomplete_k_blocks_the_whole_verdict():
    """Section 4.1 compares the K values; a missing one cannot be compared."""
    inputs = collect_inputs(
        {
            PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06),
            SENSITIVITY_K: make_report(probe=0.30, count=0.50),  # no control
        }
    )
    out = apply_rule(inputs)
    assert out["verdict"] == "INCOMPLETE"
    assert out["per_k"][str(PRIMARY_K)]["label"] == "ROWS_CARRY_THE_TRIGRAM"
    assert out["per_k"][str(SENSITIVITY_K)]["label"] == "INCOMPLETE"


def test_an_empty_panel_is_incomplete():
    out = apply_rule(VerdictInputs())
    assert out["verdict"] == "INCOMPLETE"


# --------------------------------------------------------------------------- #
# the frozen constants and the scope note must survive edits
# --------------------------------------------------------------------------- #
def test_frozen_constants_match_the_preregistration():
    assert (CARRY_FLOOR, DEGRADED_FLOOR, DEAD_MARGIN) == (0.50, 0.20, 0.005)
    assert (PRIMARY_K, SENSITIVITY_K) == (5000, 1000)


def test_result_carries_the_frozen_rule_and_the_scope_note():
    out = apply_rule(
        collect_inputs({PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06)})
    )
    assert out["frozen_rule"]["carry_floor"] == CARRY_FLOOR
    assert "nothing about usefulness" in out["scope_note"]


def test_render_markdown_states_the_verdict_and_the_action():
    out = apply_rule(
        collect_inputs(
            {
                PRIMARY_K: make_report(probe=0.30, count=0.50, shuffled=0.06),
                SENSITIVITY_K: make_report(probe=0.30, count=0.50, shuffled=0.06),
            }
        )
    )
    out["tag"] = "wiki"
    md = render_markdown(out)
    assert "ROWS_CARRY_THE_TRIGRAM" in md
    assert "localised in rung 4" in md
    assert "| 5000 |" in md
