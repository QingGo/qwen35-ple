"""Tests for the Stage B1 row-retraining plumbing (CPU, no torch, no row table).

The thing worth testing here is *index arithmetic*, because every previous bug in
this project that survived to a result was an off-by-something in a mapping, and
a wrong row-to-position mapping produces a plausible-looking number rather than a
crash.  Specifically:

* ``codes[i]`` must describe the trigram ENDING at token ``i + 2``;
* the injection at stream position ``t`` must use ``codes[t - 2]``;
* a training window starting at ``s`` must request exactly the rows for positions
  ``s .. s + batch_tokens``, and no others.

The bound check in the snapshot script caught a real instance of the first kind
of error on its first run, so these are not hypothetical.
"""

from __future__ import annotations

import importlib.util
import itertools
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


SNAP = _load("scripts/round169_row_snapshot.py", "round169_row_snapshot")
TRAIN = _load("scripts/round169_train_rows.py", "round169_train_rows")

V = SNAP.TRIGRAM_VOCAB


# --------------------------------------------------------------------------- #
# the trigram encoding
# --------------------------------------------------------------------------- #
def test_codes_have_one_fewer_pair_than_tokens():
    t = np.arange(10, dtype=np.int64)
    assert SNAP.trigram_codes(t).size == 8


def test_code_i_describes_the_trigram_ending_at_token_i_plus_2():
    """This is the single most load-bearing index convention in B1."""
    rng = np.random.default_rng(0)
    t = rng.integers(0, V, size=500, dtype=np.int64)
    codes = SNAP.trigram_codes(t)
    for i in (0, 1, 7, 250, codes.size - 1):
        end = i + 2  # the token index this trigram ends at
        want = (int(t[end - 2]) * V + int(t[end - 1])) * V + int(t[end])
        assert int(codes[i]) == want, f"code {i} should end at token {end}"


def test_encoding_is_injective_over_a_small_vocabulary():
    import itertools

    tri = np.array(list(itertools.product(range(7), repeat=3)), dtype=np.int64)
    codes = (tri[:, 0] * V + tri[:, 1]) * V + tri[:, 2]
    assert np.unique(codes).size == codes.size


def test_a_stream_shorter_than_three_tokens_encodes_to_nothing():
    assert SNAP.trigram_codes(np.array([1, 2], dtype=np.int64)).size == 0


def test_the_frozen_radix_is_large_enough_and_fits_int64():
    # The largest token id observed anywhere in this project is 248069.
    assert V > 248_069
    assert V**3 < 2**63


# --------------------------------------------------------------------------- #
# the stream-position -> snapshot-row mapping used by the trainer
# --------------------------------------------------------------------------- #
def _tiny_stream(n: int = 400, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 40, size=n, dtype=np.int64)


def test_every_stream_position_maps_to_the_row_of_its_own_trigram():
    t = _tiny_stream()
    codes = SNAP.trigram_codes(t)
    uniq, inv = np.unique(codes, return_inverse=True)
    pos = inv.astype(np.int64)  # codes_uniq[pos] == codes
    assert np.array_equal(uniq[pos], codes)
    # stream position t uses pos[t-2]; verify against a brute-force trigram
    for tt in (2, 3, 100, t.size - 1):
        want = (int(t[tt - 2]) * V + int(t[tt - 1])) * V + int(t[tt])
        assert int(uniq[pos[tt - 2]]) == want


def test_window_requests_exactly_the_positions_it_will_score():
    """A window of B+1 tokens scores B predictions and needs B+1 rows."""
    t = _tiny_stream()
    codes = SNAP.trigram_codes(t)
    uniq, inv = np.unique(codes, return_inverse=True)
    pos = inv.astype(np.int64)
    B = 16
    s = 2
    idx = TRAIN.window_row_indices(pos, s, B)
    assert idx.size == B + 1, "the hook requires e_t to match the hidden-state length"
    # idx[j] must be the row for stream position s + j
    for j in range(B + 1):
        tt = s + j
        want = (int(t[tt - 2]) * V + int(t[tt - 1])) * V + int(t[tt])
        assert int(uniq[idx[j]]) == want, f"row {j} is not the trigram ending at token {tt}"


def test_consecutive_windows_tile_the_stream_without_gaps():
    t = _tiny_stream()
    codes = SNAP.trigram_codes(t)
    _, inv = np.unique(codes, return_inverse=True)
    pos = inv.astype(np.int64)
    B = 16
    starts = list(range(2, t.size - B - 2, B))
    assert len(starts) > 2
    for a, b in itertools.pairwise(starts):
        assert b - a == B
        wa = TRAIN.window_row_indices(pos, a, B)
        wb = TRAIN.window_row_indices(pos, b, B)
        # window ``a`` covers stream positions a .. a+B, so its LAST row is the
        # same trigram as window ``b``'s FIRST row (b == a+B).
        assert wa[-1] == wb[0]
        # ... and the positions each window SCORES are disjoint: window a scores
        # a .. a+B-1, window b scores a+B .. a+2B-1.
        assert wa[:-1][-1] != wb[0]


def test_the_start_offset_never_indexes_before_the_stream():
    t = _tiny_stream(n=64)
    codes = SNAP.trigram_codes(t)
    _, inv = np.unique(codes, return_inverse=True)
    pos = inv.astype(np.int64)
    B = 8
    for s in range(2, t.size - B - 2, B):
        idx = TRAIN.window_row_indices(pos, s, B)
        assert idx.size == B + 1
        assert idx.min() >= 0 and idx.max() < codes.size


def test_every_start_keeps_the_window_inside_the_stream():
    t = _tiny_stream(n=200)
    B = 16
    starts = list(range(2, t.size - B - 2, B))
    assert starts, "no windows would be produced"
    for s in starts:
        assert s + B + 1 <= t.size, "window runs past the end of the stream"
        assert s - 2 >= 0, "row window starts before the stream"
