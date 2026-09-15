"""Tests for the coupling audit.

This tool exists to turn "the rows are coupled through the autoregressive state"
from an inference into a measurement, so the properties worth pinning are the
ones that would let it manufacture that conclusion:

  * ``last_change_before`` must EXCLUDE the position itself.  A position whose own
    row moved would otherwise be at distance 0 from "a change", and the
    dose-response -- the entire result -- would be measuring the wrong thing;
  * a position with no earlier change at all belongs to NEITHER side of the
    boundary split.  Folding it into "boundary between" would inflate the very
    count the conclusion rests on;
  * the rule is one-sided: a boundary is necessary for an exact zero, not
    sufficient, so the reported failure count is "zeros NOT boundary-separated";
  * empty bins report zeros, not NaN.
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


COUP = _load("scripts/round169_coupling_audit.py", "round169_coupling_audit")


# --------------------------------------------------------------------------
# stream_order
# --------------------------------------------------------------------------


def test_stream_order_sorts_by_stream_position():
    score = np.array([50, 10, 30])
    assert COUP.stream_order(score).tolist() == [1, 2, 0]


def test_stream_order_is_stable_on_ties():
    # Ties cannot happen in a real record, but an unstable sort would make the
    # whole analysis depend on the platform's sort implementation.
    score = np.array([5, 5, 5, 1])
    assert COUP.stream_order(score).tolist() == [3, 0, 1, 2]


def test_stream_order_rejects_a_matrix():
    with pytest.raises(ValueError, match="1-D"):
        COUP.stream_order(np.zeros((4, 2), dtype=np.int64))


# --------------------------------------------------------------------------
# last_change_before
# --------------------------------------------------------------------------


def test_last_change_excludes_the_position_itself():
    # THE property.  Position 2's own row changed; that says nothing about where
    # its delta came from, so the answer must still point back to position 0.
    score = np.array([0, 1, 2, 3])
    changed = np.array([True, False, True, False])
    # Position 0 has no earlier change at all, even though its own row moved.
    assert COUP.last_change_before(score, changed).tolist() == [-1, 0, 0, 2]


def test_last_change_is_minus_one_before_anything_changed():
    score = np.array([0, 1, 2])
    changed = np.array([False, False, True])
    assert COUP.last_change_before(score, changed).tolist() == [-1, -1, -1]


def test_last_change_uses_tokens_not_array_slots():
    # The scored positions are sparse, so the reachable distance is the stream
    # gap, not one per element.
    score = np.array([0, 10, 500, 501])
    changed = np.array([True, False, False, False])
    # The change is at token 0; the distances are 10, 500, 501 -- not 1, 2, 3.
    assert COUP.last_change_before(score, changed).tolist() == [-1, 0, 0, 0]


def test_last_change_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        COUP.last_change_before(np.zeros(3, dtype=np.int64), np.zeros(4, dtype=bool))


# --------------------------------------------------------------------------
# boundary_split
# --------------------------------------------------------------------------


def test_boundary_split_separates_same_block_from_across_a_boundary():
    score = np.array([10, 20, 1030, 1040])
    last = np.array([0, 0, 20, 1030])
    same, other = COUP.boundary_split(score, last, chunk=1024)
    # 10 and 20 both sit in block 0 with a change at 0.
    assert same.tolist() == [True, True, False, True]
    # 1030's last VISIBLE change (20) is in block 0; 1040's (1030) is its own.
    assert other.tolist() == [False, False, True, False]


def test_boundary_split_excludes_positions_with_no_earlier_change():
    # Folding these into either group would put never-perturbed positions into a
    # comparison about perturbation.
    score = np.array([10, 20])
    last = np.array([-1, -1])
    same, other = COUP.boundary_split(score, last, chunk=1024)
    assert not same.any() and not other.any()


def test_boundary_split_rejects_a_nonpositive_chunk():
    with pytest.raises(ValueError, match="positive"):
        COUP.boundary_split(np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), 0)


# --------------------------------------------------------------------------
# dose_response
# --------------------------------------------------------------------------


def test_dose_response_bins_by_distance():
    delta = np.array([1.0, 1.0, 1.0, 1.0])
    distance = np.array([1, 1, 100, 100])
    elig = np.ones(4, dtype=bool)
    got = COUP.dose_response(delta, distance, elig, edges=((1, 1), (100, 100)))
    assert got[0]["n"] == 2 and got[1]["n"] == 2


def test_dose_response_reports_abs_mean_not_signed_mean():
    # A decaying oscillation would cancel to zero in the signed mean while the
    # perturbation is plainly there.
    delta = np.array([1.0, -1.0, 1.0, -1.0])
    distance = np.ones(4, dtype=int)
    got = COUP.dose_response(delta, distance, np.ones(4, dtype=bool), edges=((1, 1),))[0]
    assert got["mean"] == pytest.approx(0.0)
    assert got["mean_abs"] == pytest.approx(1.0)


def test_dose_response_empty_bin_is_zero_not_nan():
    got = COUP.dose_response(np.ones(2), np.ones(2, dtype=int), np.zeros(2, dtype=bool),
                             edges=((1, 1),))[0]
    assert got["n"] == 0 and got["mean_abs"] == 0.0 and got["exact_zero"] == 1.0


def test_dose_response_respects_eligibility():
    delta = np.array([5.0, 5.0])
    distance = np.array([1, 1])
    got = COUP.dose_response(delta, distance, np.array([True, False]),
                             edges=((1, 1),))[0]
    assert got["n"] == 1


def test_dose_response_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        COUP.dose_response(np.ones(3), np.ones(2, dtype=int), np.ones(3, dtype=bool))


# --------------------------------------------------------------------------
# zero_rule
# --------------------------------------------------------------------------


def test_zero_rule_counts_an_exact_zero_across_a_boundary():
    n = 4100
    score = np.arange(n, dtype=np.int64)
    changed = np.zeros(n, dtype=bool)
    changed[0] = True          # one change, at the very start of block 0
    delta = np.full(n, 0.5)
    delta[2000] = 0.0          # block 1, no visible change in its own block
    last = COUP.last_change_before(score, changed)
    got = COUP.zero_rule(delta, changed, score, last, chunk=1024)
    assert got["n_zero"] == 1
    assert got["boundary_zero"] == 1
    assert got["zeros_not_boundary"] == 0
    assert got["zeros_explained_frac"] == 1.0


def test_zero_rule_counts_a_zero_that_the_boundary_rule_cannot_explain():
    # Same block as the change and still exactly zero: this is the case the
    # mechanism must be judged on, so it has to be counted separately.
    n = 100
    score = np.arange(n, dtype=np.int64)
    changed = np.zeros(n, dtype=bool)
    changed[10] = True
    delta = np.full(n, 0.5)
    delta[50] = 0.0
    last = COUP.last_change_before(score, changed)
    got = COUP.zero_rule(delta, changed, score, last, chunk=1024)
    assert got["n_zero"] == 1
    assert got["zeros_not_boundary"] == 1
    assert got["boundary_zero"] == 0
    assert got["zeros_explained_frac"] == 0.0


def test_zero_rule_separates_zeros_on_changed_rows():
    n = 40
    score = np.arange(n, dtype=np.int64)
    changed = np.zeros(n, dtype=bool)
    changed[5] = True
    delta = np.full(n, 0.5)
    delta[5] = 0.0             # training changed this row and nothing moved
    last = COUP.last_change_before(score, changed)
    got = COUP.zero_rule(delta, changed, score, last, chunk=1024)
    assert got["n_zero"] == 1
    assert got["n_zero_on_changed_row"] == 1
    assert got["n_zero_untouched"] == 0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def _coup(**over):
    base = {
        "record": "r.npz",
        "n": 100,
        "n_changed": 40,
        "n_untouched": 60,
        "chunk": 1024,
        "threshold": 10,
        "zero_rule": {
            "n": 100, "n_zero": 3, "n_zero_untouched": 2, "n_zero_on_changed_row": 1,
            "boundary_n": 10, "boundary_zero": 2, "boundary_zero_frac": 0.2,
            "zeros_not_boundary": 1, "zeros_explained_frac": 2 / 3,
        },
        "groups": {
            "untouched, change in same chunk": {
                "n": 50, "mean_abs": 0.05, "frac_gt_1e3": 0.9, "exact_zero": 0.0},
            "untouched, boundary between": {
                "n": 10, "mean_abs": 0.04, "frac_gt_1e3": 0.6, "exact_zero": 0.2},
        },
        "dose": [
            {"low": 1, "high": 1, "n": 10, "mean": 0.0, "mean_abs": 0.07,
             "frac_gt_1e3": 0.9, "exact_zero": 0.0},
            {"low": 2, "high": 4, "n": 10, "mean": 0.0, "mean_abs": 0.05,
             "frac_gt_1e3": 0.9, "exact_zero": 0.0},
            {"low": 5, "high": 16, "n": 0, "mean": 0.0, "mean_abs": 0.0,
             "frac_gt_1e3": 0.0, "exact_zero": 1.0},
        ],
        "dose_monotone": True,
        "verdict": "VERDICT: the coupling is the autoregressive state.",
    }
    base.update(over)
    return base


def test_render_marks_a_decaying_curve_ok():
    text = COUP.render(_coup())
    assert "ok" in text


def test_render_flags_a_rising_step_as_up():
    a = _coup()
    a["dose"][1]["mean_abs"] = 0.99
    assert "UP" in COUP.render(a)


def test_render_skips_empty_bins():
    assert "5-16" not in COUP.render(_coup())


def test_render_prints_the_failure_count():
    assert "zeros NOT boundary-separated 1" in COUP.render(_coup())


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def _write(tmp: Path, *, chunk_break: bool):
    """A stream with one change, and one position that is (or is not) protected.

    ``chunk_break`` puts the zero in the block AFTER the change, where a state
    reset makes it explicable; otherwise it sits in the same block, where the
    mechanism says it should not be zero.
    """
    n = 4100
    score = np.arange(n, dtype=np.int64)
    counts = np.zeros(700000, dtype=np.int64)
    si = np.zeros(n, dtype=np.int64)
    counts[0] = 100                     # row 0 is trained
    si[0] = 0
    si[1:] = 600000                     # every other position reads an untrained row
    delta = np.full(n, 0.5, dtype=np.float32)
    zpos = 2000 if chunk_break else 50
    delta[zpos] = 0.0
    np.save(tmp / "trigram-train-count.npy", counts)
    np.savez(tmp / "rec.deltas.npz", score=score, delta=delta, snapshot_index=si)
    return zpos


def _run(monkeypatch, tmp: Path, out: Path) -> int:
    monkeypatch.setattr(sys, "argv", [
        "coup", "--snapshot-dir", str(tmp), "--record", str(tmp / "rec.deltas.npz"),
        "--threshold", "10", "--chunk", "1024", "--out", str(out),
    ])
    return COUP.main()


def test_end_to_end_a_reset_boundary_explains_the_zero(tmp_path, monkeypatch):
    _write(tmp_path, chunk_break=True)
    out = tmp_path / "c.json"
    assert _run(monkeypatch, tmp_path, out) == 0
    got = json.loads(out.read_text())
    assert got["zero_rule"]["n_zero"] == 1
    assert got["zero_rule"]["zeros_explained_frac"] == 1.0


def test_end_to_end_a_zero_inside_the_block_is_counted_against_the_rule(tmp_path, monkeypatch):
    _write(tmp_path, chunk_break=False)
    out = tmp_path / "c.json"
    _run(monkeypatch, tmp_path, out)
    got = json.loads(out.read_text())
    assert got["zero_rule"]["n_zero"] == 1
    assert got["zero_rule"]["zeros_not_boundary"] == 1
    assert got["zero_rule"]["zeros_explained_frac"] == 0.0


def test_end_to_end_refuses_a_subset_count_array(tmp_path, monkeypatch):
    _write(tmp_path, chunk_break=True)
    np.save(tmp_path / "trigram-train-count.npy", np.zeros(10, dtype=np.int64))
    with pytest.raises(SystemExit, match="full-table count array"):
        _run(monkeypatch, tmp_path, tmp_path / "c.json")
