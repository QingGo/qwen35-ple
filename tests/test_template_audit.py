"""Tests for :mod:`qwen35_ple.template_audit` (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from qwen35_ple.template_audit import (
    TEMPLATE_DETERMINED_SHARE,
    AddressDiversity,
    address_diversity,
    per_task_diversity,
    verdict_from_diversity,
    verdict_from_tasks,
)


def _collapsed(n: int = 10, heads: int = 4) -> np.ndarray:
    return np.tile(np.arange(heads, dtype=np.int64), (n, 1))


def _diverse(n: int = 10, heads: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 10 ** 6, size=(n, heads))


# --------------------------------------------------------------------------
# address_diversity
# --------------------------------------------------------------------------
def test_identical_rows_collapse_completely():
    got = address_diversity(_collapsed(10))
    assert got.n == 10
    assert got.n_distinct == 1
    assert got.modal_share == 1.0
    assert got.entropy_bits == 0.0
    assert got.entropy_bits_normalised == 0.0


def test_distinct_rows_have_no_modal_mass():
    got = address_diversity(_diverse(10))
    assert got.n_distinct == 10
    assert got.modal_share == pytest.approx(0.1)
    assert got.entropy_bits == pytest.approx(np.log2(10))


def test_modal_share_is_the_largest_group():
    rows = np.array([[1, 1], [1, 1], [1, 1], [2, 2]])
    got = address_diversity(rows)
    assert got.modal_share == pytest.approx(0.75)
    assert got.modal_tuple == (1, 1)


def test_single_item_is_trivially_modal_but_not_informative():
    got = address_diversity(np.array([[3, 4]]))
    assert got.modal_share == 1.0
    # normalised entropy is defined as 0 for n == 1 rather than dividing by zero
    assert got.entropy_bits_normalised == 0.0


def test_diversity_rejects_wrong_rank():
    with pytest.raises(ValueError, match="n_items, n_heads"):
        address_diversity(np.zeros(5, dtype=np.int64))


def test_diversity_rejects_empty_input():
    with pytest.raises(ValueError, match="no items"):
        address_diversity(np.zeros((0, 4), dtype=np.int64))


def test_counts_are_json_safe():
    import json
    got = address_diversity(_collapsed(3))
    json.dumps(got.to_dict())  # must not raise


# --------------------------------------------------------------------------
# per_task_diversity
# --------------------------------------------------------------------------
def test_per_task_diversity_splits_by_task():
    rows = np.vstack([_collapsed(4), _diverse(4)])
    tasks = ["triviaqa"] * 4 + ["nq"] * 4
    got = per_task_diversity(rows, tasks)
    assert set(got) == {"nq", "triviaqa"}
    assert got["triviaqa"].modal_share == 1.0
    assert got["nq"].modal_share == pytest.approx(0.25)


def test_per_task_diversity_requires_aligned_tasks():
    with pytest.raises(ValueError, match="!= items"):
        per_task_diversity(_collapsed(4), ["a", "b"])


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def test_verdict_flags_a_template_determined_null():
    gen = address_diversity(_collapsed(10))
    content = address_diversity(_diverse(10))
    out = verdict_from_diversity(gen, content)
    assert out["label"] == "TEMPLATE_DETERMINES_ADDRESSING"
    assert "forced by the template" in out["reason"]


def test_verdict_reports_when_the_counterfactual_fails():
    gen = address_diversity(_collapsed(10))
    content = address_diversity(np.vstack([_collapsed(9), _diverse(1)]))
    out = verdict_from_diversity(gen, content)
    assert out["label"] == "TEMPLATE_DETERMINES_ADDRESSING"
    assert "does not restore diversity" in out["reason"]


def test_verdict_clears_a_template_that_is_already_item_specific():
    gen = address_diversity(_diverse(10))
    content = address_diversity(_diverse(10))
    out = verdict_from_diversity(gen, content)
    assert out["label"] == "ADDRESSING_IS_ITEM_SPECIFIC"


def test_verdict_threshold_is_the_frozen_constant():
    assert TEMPLATE_DETERMINED_SHARE == 0.90
    out = verdict_from_diversity(
        AddressDiversity(10, 2, 0.89, 0.5, 0.5),
        AddressDiversity(10, 10, 0.1, 3.0, 1.0),
    )
    assert out["label"] == "ADDRESSING_IS_ITEM_SPECIFIC"
    assert out["threshold"] == TEMPLATE_DETERMINED_SHARE


# --------------------------------------------------------------------------
# verdict_from_tasks: the per-task statistic
# --------------------------------------------------------------------------
def test_per_task_verdict_sees_a_collapse_a_global_share_hides():
    """Two templates dilute the global share to 0.667 while every task is 1.000.

    This is not hypothetical: it is what the first run of the audit reported on
    our own evaluation file, and the global-threshold rule called it
    item-specific.
    """
    per_task_gen = {
        "triviaqa": address_diversity(_collapsed(200)),
        "nq": address_diversity(_collapsed(200)),
        "boolq": address_diversity(_collapsed(200) + 1000),
    }
    per_task_content = {
        "triviaqa": address_diversity(_diverse(200)),
        "nq": address_diversity(_diverse(200)),
        "boolq": address_diversity(_diverse(200)),
    }
    out = verdict_from_tasks(
        per_task_gen, per_task_content, n_template_variants=2,
    )
    assert out["label"] == "TEMPLATE_DETERMINES_ADDRESSING"
    assert out["worst_task_modal_share"] == 1.0
    assert "set by the template, not by the item" in out["reason"]


def test_per_task_verdict_clears_when_a_task_is_item_specific():
    gen = {
        "a": address_diversity(_collapsed(50)),
        "b": address_diversity(_diverse(50)),
    }
    content = {k: address_diversity(_diverse(50)) for k in gen}
    out = verdict_from_tasks(gen, content)
    assert out["label"] == "ADDRESSING_IS_ITEM_SPECIFIC"
    assert out["worst_task"] == "b"


def test_per_task_verdict_reports_when_the_counterfactual_fails():
    gen = {"a": address_diversity(_collapsed(50))}
    content = {"a": address_diversity(_collapsed(50) + 7)}
    out = verdict_from_tasks(gen, content)
    assert out["label"] == "TEMPLATE_DETERMINES_ADDRESSING"
    assert "does not restore diversity" in out["reason"]


def test_per_task_verdict_requires_matching_task_sets():
    with pytest.raises(ValueError, match="same tasks"):
        verdict_from_tasks(
            {"a": address_diversity(_collapsed(4))},
            {"b": address_diversity(_diverse(4))},
        )


def test_per_task_verdict_rejects_empty_input():
    with pytest.raises(ValueError, match="no generation tasks"):
        verdict_from_tasks({}, {})
