"""Tests for the row-gating audit.

The audit decides whether phase 5's falsification is an implementation bug or a
real result, so the properties worth pinning are the ones that would let it
answer "the mask is fine" when it is not:

  * the bank on disk is a keep-SUBSET of the full table, while the record's
    ``snapshot_index`` is a full-table id -- translating with the wrong one
    produces confidently wrong answers rather than an error, and that is the exact
    mistake this audit was written to catch (it was made twice by hand first);
  * a position whose row did not move must be reported as unmoved even if its
    delta is non-zero, because that gap is the finding;
  * NaN must not masquerade as a change;
  * an empty selection must report zeros, not NaN.
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


AUDIT = _load("scripts/round169_audit_row_gating.py", "round169_audit_row_gating")


# --------------------------------------------------------------------------
# keep_inverse
# --------------------------------------------------------------------------


def test_keep_inverse_round_trips():
    keep = np.array([2, 5, 7, 11])
    inv = AUDIT.keep_inverse(keep)
    assert inv[keep].tolist() == [0, 1, 2, 3]


def test_keep_inverse_marks_rows_outside_the_keep_set():
    # This is the whole point: a full-table id that the eval stream never reads
    # must be detectable, not silently mapped onto some other row.
    inv = AUDIT.keep_inverse(np.array([2, 5, 7]))
    assert inv[0] == -1
    assert inv[6] == -1
    assert inv[3] == -1


def test_keep_inverse_negative_id_is_rejected():
    # A negative id would index from the end and quietly return a real row.
    with pytest.raises(ValueError, match="negative"):
        AUDIT.keep_inverse(np.array([-1, 4]))


def test_keep_inverse_empty():
    assert AUDIT.keep_inverse(np.zeros(0, dtype=np.int64)).size == 0


# --------------------------------------------------------------------------
# rows_changed
# --------------------------------------------------------------------------


def test_rows_changed_identical_is_all_false():
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert not AUDIT.rows_changed(a, a.copy()).any()


def test_rows_changed_flags_only_the_row_that_moved():
    a = np.zeros((4, 3), dtype=np.float32)
    b = a.copy()
    b[2, 1] = 1e-9  # one element is enough
    got = AUDIT.rows_changed(a, b)
    assert got.tolist() == [False, False, True, False]


def test_rows_changed_treats_nan_as_unchanged():
    a = np.full((2, 2), np.nan, dtype=np.float32)
    b = a.copy()
    assert not AUDIT.rows_changed(a, b).any()


def test_rows_changed_nan_against_a_number_is_a_change():
    a = np.full((2, 2), np.nan, dtype=np.float32)
    b = np.zeros((2, 2), dtype=np.float32)
    assert AUDIT.rows_changed(a, b).tolist() == [True, True]


def test_rows_changed_shape_mismatch_is_an_error_not_a_broadcast():
    # E0 is the FULL table and the bank is the keep subset; letting numpy
    # broadcast them would compare the wrong rows and still return an answer.
    with pytest.raises(ValueError, match="must match"):
        AUDIT.rows_changed(np.zeros((667404, 2560), dtype=np.float32),
                           np.zeros((121488, 2560), dtype=np.float32))


# --------------------------------------------------------------------------
# crosstab
# --------------------------------------------------------------------------


def test_crosstab_identical_masks_agree_completely():
    m = np.array([True, False, True, True])
    got = AUDIT.crosstab(m, m.copy())
    assert got["agreement"] == 1.0
    assert got["both"] == 3 and got["neither"] == 1
    assert got["selected_only"] == 0 and got["changed_only"] == 0


def test_crosstab_disjoint_masks_agree_not_at_all():
    got = AUDIT.crosstab(np.array([True, True, False, False]),
                         np.array([False, False, True, True]))
    assert got["agreement"] == 0.0
    assert got["both"] == 0
    assert got["selected_only"] == 2 and got["changed_only"] == 2


def test_crosstab_partitions_the_positions():
    rng = np.random.default_rng(0)
    sel = rng.random(500) < 0.4
    chg = rng.random(500) < 0.6
    got = AUDIT.crosstab(sel, chg)
    assert got["both"] + got["selected_only"] + got["changed_only"] + got["neither"] == 500


def test_crosstab_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        AUDIT.crosstab(np.zeros(4, dtype=bool), np.zeros(5, dtype=bool))


# --------------------------------------------------------------------------
# band_sums
# --------------------------------------------------------------------------


def test_band_sums_reproduce_the_frozen_table_arithmetic():
    delta = np.array([1.0, 1.0, 1.0, 1.0])
    counts = np.array([0, 2, 10, 50])
    got = {b["threshold"]: b for b in AUDIT.band_sums(delta, counts, 4, (1, 10, 50))}
    assert got[1]["sum_over_total"] == pytest.approx(3 / 4)
    assert got[1]["n"] == 3
    assert got[10]["sum_over_total"] == pytest.approx(2 / 4)
    assert got[50]["sum_over_total"] == pytest.approx(1 / 4)


def test_band_sums_counts_shrink_as_the_threshold_rises():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 100, size=1000)
    bands = AUDIT.band_sums(np.ones(1000), counts, 1000)
    ns = [b["n"] for b in bands]
    assert ns == sorted(ns, reverse=True)


def test_band_sums_total_is_the_denominator_not_delta_size():
    # The pre-registration divided by all scored positions while summing over a
    # subset; decoupling the two is what makes that expressible.
    delta = np.array([1.0, 1.0])
    counts = np.array([10, 10])
    got = AUDIT.band_sums(delta, counts, 8, (10,))[0]
    assert got["sum_over_total"] == pytest.approx(2 / 8)
    assert got["n"] == 2


def test_band_sums_empty_selection_is_zero_not_nan():
    got = AUDIT.band_sums(np.ones(4), np.zeros(4, dtype=int), 4, (10,))[0]
    assert got["n"] == 0
    assert got["sum_over_total"] == 0.0


def test_band_sums_rejects_a_nonpositive_total():
    with pytest.raises(ValueError, match="positive"):
        AUDIT.band_sums(np.ones(2), np.ones(2, dtype=int), 0, (1,))


def test_band_sums_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        AUDIT.band_sums(np.ones(3), np.ones(4, dtype=int), 3, (1,))


# --------------------------------------------------------------------------
# region / noise_floor
# --------------------------------------------------------------------------


def test_region_reports_both_arms_and_the_difference():
    a = np.array([1.0, 3.0, 5.0, 7.0])
    b = np.array([0.0, 0.0, 4.0, 6.0])
    sel = np.array([False, False, True, True])
    got = AUDIT.region(a, b, sel)
    assert got["n"] == 2
    assert got["mean_a"] == pytest.approx(6.0)
    assert got["mean_b"] == pytest.approx(5.0)
    assert got["mean_difference"] == pytest.approx(-1.0)


def test_region_on_an_empty_selection_is_zero_not_nan():
    got = AUDIT.region(np.ones(3), np.ones(3), np.zeros(3, dtype=bool))
    assert got["n"] == 0 and got["mean_a"] == 0.0 and got["mean_b"] == 0.0


def test_noise_floor_is_one_when_nothing_leaks():
    delta = np.array([0.0, 0.0, 9.0, 0.0])  # moved rows are excluded by the mask
    untouched = np.array([True, True, False, True])
    got = AUDIT.noise_floor(delta, untouched)
    assert got["n"] == 3
    assert got["exact_zero_frac"] == 1.0
    assert got["absmax"] == 0.0


def test_noise_floor_reports_the_leak():
    delta = np.array([0.0, 0.01, 0.0, -0.02])
    got = AUDIT.noise_floor(delta, np.ones(4, dtype=bool))
    assert got["exact_zero_frac"] == pytest.approx(0.5)
    assert got["absmax"] == pytest.approx(0.02)
    assert got["mean"] == pytest.approx(-0.0025)


def test_noise_floor_on_an_empty_selection_is_clean():
    got = AUDIT.noise_floor(np.ones(3), np.zeros(3, dtype=bool))
    assert got["n"] == 0 and got["exact_zero_frac"] == 1.0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def _audit(**over):
    base = {
        "n_positions": 6,
        "n_keep": 7,
        "n_unmapped": 0,
        "train_count": {"shape": "(12,)", "max_count": 11, "ge10": 2, "ge50": 0},
        "bank_rows": 7,
        "bank_rows_changed": 2,
        "positions_changed": 2,
        "threshold": 10,
        "crosstab": [
            {"threshold": 10, "n": 2, "share": 2 / 6, "agreement": 1.0,
             "both": 2, "selected_only": 0, "changed_only": 0, "neither": 4}
        ],
        "bands": [
            {"threshold": 10, "n": 2, "share": 2 / 6, "sum_over_total": 1 / 3,
             "predicted": 1 / 3}
        ],
        "regions": [
            {"name": "ALL", "n": 6, "share": 1.0, "mean_a": 0.5,
             "mean_b": 0.25, "mean_difference": -0.25}
        ],
        "noise_floor": {"n": 4, "mean": 0.0025, "absmax": 0.01, "exact_zero_frac": 0.75},
        "predicted_total": 1 / 3,
        "measured_total": 0.25,
        "shortfall": -1 / 12,
        "tolerance": 2e-4,
    }
    base.update(over)
    return base


def test_render_says_so_when_the_frozen_number_did_not_reproduce():
    text = AUDIT.render(_audit())
    assert "NO" in text


def test_render_marks_a_reproduced_number_yes():
    a = _audit()
    a["bands"][0]["predicted"] = a["bands"][0]["sum_over_total"]
    assert "yes" in AUDIT.render(a)


def test_render_calls_out_a_nonzero_delta_on_unmoved_rows():
    text = AUDIT.render(_audit())
    assert "NOT zero" in text


def test_render_is_quiet_when_the_leak_is_absent():
    text = AUDIT.render(_audit(noise_floor={"n": 4, "mean": 0.0, "absmax": 0.0,
                                            "exact_zero_frac": 1.0}))
    assert "NOT zero" not in text


def test_render_reports_the_tolerance_multiple():
    # The whole verdict is "how many tolerances off", so the multiple has to be
    # printed rather than left for the reader to divide.
    text = AUDIT.render(_audit())
    assert f"{abs(-1 / 12) / 2e-4:.2f}x" in text


# --------------------------------------------------------------------------
# end to end on a synthetic snapshot
# --------------------------------------------------------------------------


def _write_snapshot(root: Path, *, leak: float = 0.0):
    """A miniature table where the answer is known by hand.

    Rows 10 and 11 have training count >= 10 and are in the keep set; every other
    keep row is frozen.  Six scored positions index rows [0, 2, 10, 11, 4, 6], so
    exactly two of them sit on a row that moves.
    """
    rows, dim = 12, 3
    e0 = np.arange(rows, dtype=np.float32)[:, None] * np.ones((1, dim), dtype=np.float32)
    keep = np.array([0, 2, 4, 6, 8, 10, 11], dtype=np.int64)
    bank = e0[keep].copy()
    bank[5] += 100.0
    bank[6] += 100.0
    counts = np.arange(rows, dtype=np.int64)
    si = np.array([0, 2, 10, 11, 4, 6], dtype=np.int64)
    code = np.arange(si.size, dtype=np.int64)
    delta_a = np.array([0.1, 0.2, 1.0, 1.0, 0.3, 0.4], dtype=np.float32)
    delta_b = np.array([leak, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    np.save(root / "E0.npy", e0)
    np.save(root / "keep-wiki-eval.npy", keep)
    np.save(root / "trigram-train-count.npy", counts)
    np.save(root / "masked.npy", bank)
    np.savez(root / "masked.deltas.npz", delta=delta_b, snapshot_index=si, trigram_code=code)
    np.savez(root / "base.deltas.npz", delta=delta_a, snapshot_index=si, trigram_code=code)
    return {"delta_a": delta_a, "delta_b": delta_b, "si": si, "counts": counts}


def _run_main(monkeypatch, root: Path, out: Path, *extra: str) -> int:
    argv = [
        "audit", "--snapshot-dir", str(root),
        "--masked-bank", str(root / "masked.npy"),
        "--masked-deltas", str(root / "masked.deltas.npz"),
        "--baseline-deltas", str(root / "base.deltas.npz"),
        "--threshold", "10", "--predicted", repr(2.0 / 6), "--tolerance", "2e-4",
        "--out", str(out), *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return AUDIT.main()


def test_end_to_end_identifies_a_clean_mask(tmp_path, monkeypatch):
    _write_snapshot(tmp_path)
    out = tmp_path / "audit.json"
    assert _run_main(monkeypatch, tmp_path, out) == 0
    got = json.loads(out.read_text())
    assert got["n_positions"] == 6
    assert got["bank_rows_changed"] == 2
    assert got["positions_changed"] == 2  # rows 10 and 11, exactly
    row = next(c for c in got["crosstab"] if c["threshold"] == 10)
    assert row["agreement"] == 1.0
    assert row["both"] == 2 and row["selected_only"] == 0 and row["changed_only"] == 0


def test_end_to_end_recomputes_the_band_sum_correctly(tmp_path, monkeypatch):
    _write_snapshot(tmp_path)
    out = tmp_path / "audit.json"
    _run_main(monkeypatch, tmp_path, out)
    got = json.loads(out.read_text())
    band = next(b for b in got["bands"] if b["threshold"] == 10)
    assert band["n"] == 2
    assert band["sum_over_total"] == pytest.approx(2.0 / 6)
    assert got["measured_total"] == pytest.approx(1.0 / 6)  # 0.5 + 0.5, over six


def test_end_to_end_detects_a_leak_on_an_unmoved_row(tmp_path, monkeypatch):
    # No row moved at position 0, so its delta must be zero; a non-zero value
    # there is coupling outside the row table.
    _write_snapshot(tmp_path, leak=0.01)
    out = tmp_path / "audit.json"
    _run_main(monkeypatch, tmp_path, out)
    nf = json.loads(out.read_text())["noise_floor"]
    assert nf["n"] == 4
    assert nf["exact_zero_frac"] == pytest.approx(0.75)
    assert nf["absmax"] == pytest.approx(0.01)


def test_end_to_end_sees_no_leak_when_there_is_none(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, leak=0.0)
    out = tmp_path / "audit.json"
    _run_main(monkeypatch, tmp_path, out)
    assert json.loads(out.read_text())["noise_floor"]["exact_zero_frac"] == 1.0


def test_end_to_end_refuses_a_record_from_different_positions(tmp_path, monkeypatch):
    _write_snapshot(tmp_path)
    # Same shape, different trigrams: comparing these would answer about the wrong
    # positions while looking perfectly healthy.
    si = np.array([0, 2, 10, 11, 4, 6], dtype=np.int64)
    np.savez(tmp_path / "base.deltas.npz", delta=np.zeros(6, dtype=np.float32),
             snapshot_index=si, trigram_code=np.arange(6, dtype=np.int64) + 99)
    out = tmp_path / "audit.json"
    with pytest.raises(SystemExit, match="same positions"):
        _run_main(monkeypatch, tmp_path, out)


def test_end_to_end_refuses_a_position_outside_the_keep_set(tmp_path, monkeypatch):
    _write_snapshot(tmp_path)
    si = np.array([0, 2, 10, 11, 4, 9], dtype=np.int64)  # row 9 is not in keep
    code = np.arange(6, dtype=np.int64)
    for name in ("masked", "base"):
        np.savez(tmp_path / f"{name}.deltas.npz",
                 delta=np.zeros(6, dtype=np.float32), snapshot_index=si, trigram_code=code)
    with pytest.raises(SystemExit, match="keep index does not contain"):
        _run_main(monkeypatch, tmp_path, tmp_path / "audit.json")
