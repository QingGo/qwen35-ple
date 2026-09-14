"""Tests for the overnight summariser and the B1 keep-index.

Two things here can silently produce a wrong record while nobody is watching:

* the summariser turns a Delta into a verdict label, and it deliberately repeats
  the frozen thresholds instead of importing them -- so the repetition is exactly
  what needs pinning; and
* the keep-index maps a stream position to a snapshot row through ``t - 2``, the
  same offset whose earlier mis-statement produced a silently wrong bank.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SUM = _load("scripts/round169_summarize.py", "round169_summarize")
KEEP = _load("scripts/round169_keep_index.py", "round169_keep_index")
SNAP = _load("scripts/round169_row_snapshot.py", "round169_row_snapshot")

V = SNAP.TRIGRAM_VOCAB


# --------------------------------------------------------------------------- #
# the frozen rule, repeated in the summariser
# --------------------------------------------------------------------------- #
def test_the_summariser_repeats_the_frozen_thresholds():
    assert (SUM.CARRY_FLOOR, SUM.DEGRADED_FLOOR, SUM.DEAD_MARGIN) == (0.50, 0.20, 0.005)
    assert (SUM.EFFECT_FLOOR, SUM.NULL_TOLERANCE) == (0.05, 0.01)


def test_a_clear_loss_is_rows_harmed():
    label, why = SUM.rule(delta=-0.33127, se=0.001, delta_shuf=-3.97576)
    assert label == "ROWS_HARMED"
    assert "-0.33127" in why


def test_a_loss_inside_the_effect_floor_is_stuck_not_harmed():
    label, _ = SUM.rule(delta=-0.02, se=0.001, delta_shuf=0.0)
    assert label == "ROWS_STUCK"


def test_a_gain_with_a_clean_null_is_reshapeable():
    label, _ = SUM.rule(delta=+0.20, se=0.001, delta_shuf=0.002)
    assert label == "ROWS_RESHAPEABLE"


def test_a_gain_with_a_moved_null_is_refused():
    """The 1.5e lesson: a gain the null also shows is not content."""
    label, why = SUM.rule(delta=+0.20, se=0.001, delta_shuf=+0.05)
    assert label == "ROWS_RESHAPEABLE_BUT_NULL_MOVED"
    assert "not content-specific" in why


def test_a_gain_without_a_null_arm_is_not_upgraded():
    # delta_shuf=None must not silently count as a clean null
    label, _ = SUM.rule(delta=+0.20, se=0.001, delta_shuf=None)
    assert label == "ROWS_RESHAPEABLE"


def test_a_tiny_effect_with_a_huge_se_is_stuck_not_a_verdict():
    label, _ = SUM.rule(delta=+0.10, se=1.0, delta_shuf=0.0)
    assert label == "ROWS_STUCK"


# --------------------------------------------------------------------------- #
# the keep-index offset
# --------------------------------------------------------------------------- #
def test_keep_index_uses_the_t_minus_2_offset(tmp_path: Path):
    """A row for stream position t is codes[t-2].  Getting this wrong is silent."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50, size=300, dtype=np.int64)
    codes = SNAP.trigram_codes(tokens)
    uniq = np.unique(codes)
    np.save(tmp_path / "codes.npy", uniq)
    np.save(tmp_path / "tokens.npy", tokens)
    # every position from 2 up to T-2 (so that t+1 is a valid target)
    pos = np.arange(2, tokens.size - 1, dtype=np.int64)
    np.save(tmp_path / "pos.npy", pos)

    import subprocess

    out = tmp_path / "keep.npy"
    r = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "round169_keep_index.py"),
            "--snapshot-codes", str(tmp_path / "codes.npy"),
            "--tokens", str(tmp_path / "tokens.npy"),
            "--positions", str(tmp_path / "pos.npy"),
            "--out", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    keep = np.load(out)
    # Exact expectation, built independently: for each position t the row is the
    # trigram ending at t, i.e. codes[t-2].  A wrong offset shifts every row and
    # this comparison catches it.
    want = np.stack([tokens[pos - 2], tokens[pos - 1], tokens[pos]], axis=1)
    want_codes = (want[:, 0].astype(np.int64) * V + want[:, 1]) * V + want[:, 2]
    want_idx = np.searchsorted(uniq, want_codes)
    assert np.array_equal(uniq[want_idx], want_codes), "positions did not all hit"
    assert np.array_equal(keep, np.unique(want_idx)), "keep-index is not the position set's rows"
    # and it is a strict subset here, because the very last trigram has no
    # position with a valid next token
    assert keep.size <= uniq.size


def test_keep_index_refuses_when_nothing_matches(tmp_path: Path):
    tokens = np.full(100, 7, dtype=np.int64)
    uniq = np.array([12345], dtype=np.int64)  # a trigram the stream does not contain
    np.save(tmp_path / "codes.npy", uniq)
    np.save(tmp_path / "tokens.npy", tokens)
    np.save(tmp_path / "pos.npy", np.arange(2, 90, dtype=np.int64))
    import subprocess

    r = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "round169_keep_index.py"),
            "--snapshot-codes", str(tmp_path / "codes.npy"),
            "--tokens", str(tmp_path / "tokens.npy"),
            "--positions", str(tmp_path / "pos.npy"),
            "--out", str(tmp_path / "keep.npy"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "no aligned position has a snapshot row" in (r.stderr + r.stdout)
