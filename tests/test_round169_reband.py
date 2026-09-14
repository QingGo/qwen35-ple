"""Tests for the B1 re-banding tool (CPU, no torch, no row table).

The tool exists because the published band table stratifies by the wrong
variable, so the properties worth pinning are the ones that would let the
correction go quietly wrong:

* the band edges are the published ones, so the two tables can be compared;
* a position is assigned by its OWN trigram's count, taken from the snapshot row
  it was actually injected from -- not by anything derived from the eval stream;
* there are no empty or double-counted bands, and the band n's sum to the total;
* an out-of-range snapshot index is refused rather than silently wrapping.
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


REBAND = _load("scripts/round169_reband.py", "round169_reband")


def test_band_edges_match_the_published_table():
    assert REBAND.BANDS == [(1, 1), (2, 2), (3, 4), (5, 9), (10, 49), (50, 199), (200, 10**9)]
    assert REBAND.band_label(200, 10**9) == "[200,+]"
    assert REBAND.band_label(3, 4) == "[3,4]"


def test_a_position_is_banded_by_its_own_count():
    counts = np.array([1, 1, 2, 4, 5, 10, 50, 200, 10_000], dtype=np.int64)
    delta = np.array([-1.0, -1.0, -0.5, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3])
    tab = REBAND.band_table(counts, delta)
    assert tab["[1,1]"]["n"] == 2 and tab["[1,1]"]["mean"] == pytest.approx(-1.0)
    assert tab["[2,2]"]["n"] == 1
    assert tab["[3,4]"]["n"] == 1
    assert tab["[5,9]"]["n"] == 1
    assert tab["[10,49]"]["n"] == 1
    assert tab["[50,199]"]["n"] == 1
    assert tab["[200,+]"] == {**tab["[200,+]"], "n": 2}


def test_bands_partition_every_position_exactly_once():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 5000, size=10_000).astype(np.int64)
    delta = rng.normal(size=10_000)
    tab = REBAND.band_table(counts, delta)
    assert sum(v["n"] for v in tab.values()) == counts.size


def test_the_band_mean_is_the_mean_of_its_members():
    counts = np.array([1, 1, 1, 200], dtype=np.int64)
    delta = np.array([1.0, 2.0, 3.0, -1.0])
    tab = REBAND.band_table(counts, delta)
    assert tab["[1,1]"]["mean"] == pytest.approx(2.0)
    assert tab["[200,+]"] == {**tab["[200,+]"], "n": 1}
    assert tab["[200,+]"]["mean"] == pytest.approx(-1.0)


def test_own_counts_reads_through_the_snapshot_index_not_the_stream():
    record = {"snapshot_index": np.array([3, 0, 2], dtype=np.int64)}
    train_count = np.array([11, 22, 33, 44], dtype=np.int64)
    assert REBAND.own_counts(record, train_count).tolist() == [44, 11, 33]


def test_own_counts_refuses_an_index_outside_the_count_table():
    # An out-of-range index would silently wrap under numpy's negative indexing
    # and band a position by some unrelated trigram's frequency.
    record = {"snapshot_index": np.array([0, 7], dtype=np.int64)}
    train_count = np.array([11, 22, 33], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        REBAND.own_counts(record, train_count)


def test_band_table_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="parallel"):
        REBAND.band_table(np.array([1, 2, 3]), np.array([1.0, 2.0]))


def test_paired_stats_matches_the_projects_definition():
    delta = np.array([1.0, 2.0, 3.0, 4.0])
    st = REBAND.paired_stats(delta)
    assert st["n"] == 4
    assert st["mean"] == pytest.approx(2.5)
    assert st["se"] == pytest.approx(delta.std(ddof=1) / 2.0)
    assert st["t"] == pytest.approx(2.5 / st["se"])


def test_end_to_end_writes_both_tables_and_names_the_artifact(tmp_path, capsys):
    n = 40
    rng = np.random.default_rng(1)
    snap = rng.integers(0, 10, size=n).astype(np.int64)
    train_count = np.array([1, 2, 3, 5, 10, 50, 200, 900, 1500, 4000], dtype=np.int64)
    own = train_count[snap]
    ctx = np.where(own > 3, own, 0)  # mimic the published variable's extra zeros
    rec = tmp_path / "eval-real-lr3.162e-4.deltas.npz"
    np.savez_compressed(
        rec,
        score=np.arange(n, dtype=np.int64),
        rowid=np.arange(n, dtype=np.int64),
        snapshot_index=snap,
        trigram_code=np.arange(n, dtype=np.int64),
        delta=rng.normal(scale=0.01, size=n).astype(np.float32),
        nll_frozen=rng.normal(size=n).astype(np.float32),
        nll_trained=rng.normal(size=n).astype(np.float32),
        nll_none=rng.normal(size=n).astype(np.float32),
        context_count=ctx.astype(np.int64),
    )
    snapdir = tmp_path / "snap"
    snapdir.mkdir()
    np.save(snapdir / "trigram-train-count.npy", train_count)

    out_json = tmp_path / "reband.json"
    out_md = tmp_path / "reband.md"
    sys.argv = [
        "round169_reband.py",
        "--record", str(rec),
        "--snapshot-dir", str(snapdir),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]
    assert REBAND.main() == 0
    out = capsys.readouterr().out
    assert "by OWN trigram count (corrected)" in out
    assert "by context_counts (as published)" in out
    assert out_json.exists() and out_md.exists()
    blob = json.loads(out_json.read_text())
    assert blob["tag"] == "eval-real-lr3.162e-4"
    assert blob["own_min"] == int(own.min())
    assert blob["ctx_zero"] == int((ctx == 0).sum())
    assert "by_own_count" in blob and "by_context_count" in blob
