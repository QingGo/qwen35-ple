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
import pytest

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

# --------------------------------------------------------------------------- #
# The per-position record: the artifact the band table cannot replace.
# --------------------------------------------------------------------------- #
EVAL = _load("scripts/round169_eval_rows.py", "round169_eval_rows")


def _record_inputs(n: int = 6):
    """(score, delta, lf, lt, ln, ctx) for a scored set of ``n`` positions."""
    score = np.arange(100, 100 + n, dtype=np.int64)
    delta = np.linspace(-0.5, 0.5, n).astype(np.float32)
    lf = np.full(n, 2.5, dtype=np.float32)
    lt = lf - delta
    ln = np.full(n, 2.7, dtype=np.float32)
    ctx = np.arange(1, n + 1, dtype=np.int64)
    return score, delta, lf, lt, ln, ctx


def test_the_record_is_parallel_to_the_scored_positions():
    score, delta, lf, lt, ln, ctx = _record_inputs()
    rec = EVAL.per_position_record(score, delta, lf, lt, ln, ctx)
    assert set(rec) == {
        "score", "delta", "nll_frozen", "nll_trained", "nll_none", "context_count",
    }
    for key, arr in rec.items():
        assert arr.shape == (score.size,), key
    assert np.array_equal(rec["score"], score)


def test_the_record_keeps_the_eval_sign_convention():
    # delta = frozen - trained everywhere in this project, and the B1 verdict is
    # read in that convention.  A record that silently flipped it would reverse
    # every conclusion drawn from it.
    score, _, lf, lt, ln, ctx = _record_inputs()
    delta = lf - lt
    rec = EVAL.per_position_record(score, delta, lf, lt, ln, ctx)
    assert np.allclose(rec["delta"], rec["nll_frozen"] - rec["nll_trained"], atol=1e-6)


def test_context_count_is_optional_but_checked_when_present():
    score, delta, lf, lt, ln, _ = _record_inputs()
    assert "context_count" not in EVAL.per_position_record(score, delta, lf, lt, ln)
    with_ctx = EVAL.per_position_record(score, delta, lf, lt, ln, np.ones(score.size))
    assert "context_count" in with_ctx


def test_the_record_carries_every_key_the_correction_needs():
    # snapshot_index joins to trigram-train-count.npy and gives the EXACT count of
    # the injected trigram; trigram_code identifies that trigram.  Without both,
    # the published band table cannot be recomputed after the box is gone.
    score, delta, lf, lt, ln, ctx = _record_inputs()
    snap = np.arange(1000, 1000 + score.size, dtype=np.int64)
    code = np.arange(500, 500 + score.size, dtype=np.int64)
    rec = EVAL.per_position_record(score, delta, lf, lt, ln, ctx, snap=snap, code=code)
    assert set(rec) == {
        "score", "snapshot_index", "trigram_code",
        "delta", "nll_frozen", "nll_trained", "nll_none", "context_count",
    }
    assert np.array_equal(rec["snapshot_index"], snap)
    assert np.array_equal(rec["trigram_code"], code)


def test_rowid_is_not_in_the_record():
    # rowids_from_tokens returns [T, 16] -- the graft addresses 16 heads and
    # fetch_e_t concatenates them -- so "the row id" has no scalar meaning, and
    # storing it would add ~57 MB per arm.  This failed once as a shape error
    # ("expected 665, got 10640"); the key is gone on purpose, not by accident.
    score, delta, lf, lt, ln, ctx = _record_inputs()
    rec = EVAL.per_position_record(score, delta, lf, lt, ln, ctx)
    assert "rowid" not in rec


def test_the_extra_keys_are_optional():
    score, delta, lf, lt, ln, _ = _record_inputs()
    bare = EVAL.per_position_record(score, delta, lf, lt, ln)
    assert "snapshot_index" not in bare and "trigram_code" not in bare


def test_a_misaligned_field_is_rejected_rather_than_written():
    # The failure mode this guards: a record whose arrays disagree in length
    # still saves, still loads, and silently joins the wrong delta to the wrong
    # row -- which is exactly the species of bug that produced this project's
    # earlier "19,996/20,000 bound disagreements".
    score, delta, lf, lt, ln, ctx = _record_inputs(n=6)
    with pytest.raises(ValueError, match=r"expected \(6,\)"):
        EVAL.per_position_record(score, delta, lf, lt, ln, ctx[:5])


def test_a_two_dimensional_field_is_rejected_not_merely_size_checked():
    # This is the shape that actually shipped by mistake: all_rowids[score] has
    # the right leading dimension and 16x the elements.  A leading-axis check
    # ACCEPTS it; only requiring one value per position names the problem.
    score, delta, lf, lt, ln, ctx = _record_inputs(n=4)
    wide = np.zeros((4, 16), dtype=np.int64)
    with pytest.raises(ValueError, match=r"shape \(4, 16\)"):
        EVAL.per_position_record(score, delta, lf, lt, ln, ctx, snap=wide)


def test_a_misaligned_optional_field_is_rejected_too():
    score, delta, lf, lt, ln, ctx = _record_inputs(n=6)
    with pytest.raises(ValueError, match="snapshot_index"):
        EVAL.per_position_record(
            score, delta, lf, lt, ln, ctx, snap=np.arange(3, dtype=np.int64)
        )


def test_extra_fields_are_merged_into_the_record():
    score, delta, lf, lt, ln, ctx = _record_inputs(n=5)
    rec = EVAL.per_position_record(
        score, delta, lf, lt, ln, ctx,
        extra={"ent_none": np.full(5, 1.5), "top1_frozen": np.arange(5)},
    )
    assert rec["ent_none"].tolist() == [1.5] * 5
    assert rec["top1_frozen"].tolist() == [0, 1, 2, 3, 4]


def test_an_extra_field_with_the_wrong_shape_is_rejected_by_name():
    # The `extra` path exists so a SECOND artifact can carry more per-position
    # quantities through the SAME check.  Merging before the check is the point.
    score, delta, lf, lt, ln, ctx = _record_inputs(n=5)
    with pytest.raises(ValueError, match="ent_none"):
        EVAL.per_position_record(
            score, delta, lf, lt, ln, ctx, extra={"ent_none": np.zeros((5, 4))}
        )


def test_an_extra_field_may_not_shadow_a_core_field():
    # A right-shaped overwrite of `score` passes every shape check and corrupts
    # the record.  Only the name catches it.
    score, delta, lf, lt, ln, ctx = _record_inputs(n=5)
    with pytest.raises(ValueError, match="overwrite a core record field"):
        EVAL.per_position_record(
            score, delta, lf, lt, ln, ctx, extra={"score": np.zeros(5, dtype=np.int64)}
        )


def test_extra_defaults_to_none_and_adds_nothing():
    score, delta, lf, lt, ln, ctx = _record_inputs(n=5)
    bare = EVAL.per_position_record(score, delta, lf, lt, ln, ctx)
    explicit = EVAL.per_position_record(score, delta, lf, lt, ln, ctx, extra=None)
    assert sorted(bare) == sorted(explicit)


def test_the_record_round_trips_through_npz(tmp_path):
    score, delta, lf, lt, ln, ctx = _record_inputs()
    rec = EVAL.per_position_record(score, delta, lf, lt, ln, ctx)
    path = tmp_path / "eval-real-lr3.162e-4.deltas.npz"
    np.savez_compressed(path, **rec)
    back = np.load(path)
    assert sorted(back.files) == sorted(rec)
    for key in rec:
        assert np.array_equal(back[key], rec[key]), key


# --------------------------------------------------------------------------- #
# The t-2 offset: per-trigram arrays versus per-position arrays.
# --------------------------------------------------------------------------- #
def test_the_per_trigram_arrays_take_the_t_minus_2_offset():
    # codes[i] describes the trigram ENDING at i+2, so a scored stream position t
    # is described by codes[t-2].  Indexing by t directly raises IndexError only
    # for the last few positions, which is how the first per-position record
    # failed on every arm while still writing a valid primary JSON.
    score = np.array([2, 3, 100, 999], dtype=np.int64)
    idx = EVAL.trigram_index(score, n_trigrams=1000)
    assert idx.tolist() == [0, 1, 98, 997]
    assert (score - idx == 2).all()


def test_the_offset_helper_rejects_positions_that_would_run_off_the_end():
    # The exact failure: a stream of 1,152,891 tokens has 1,152,889 trigrams, and
    # the position 1,152,889 (== n_trigrams) is the first one out of bounds.
    with pytest.raises(ValueError, match="t-2 offset"):
        EVAL.trigram_index(np.array([0, 1, 1_152_889], dtype=np.int64), n_trigrams=1_152_889)
    assert EVAL.trigram_index(np.array([2, 3], dtype=np.int64), 10).tolist() == [0, 1]


def test_the_offset_helper_accepts_the_last_representable_position():
    n = 1_152_889
    assert EVAL.trigram_index(np.array([n + 1], dtype=np.int64), n).tolist() == [n - 1]


def test_the_offset_helper_handles_an_empty_score():
    assert EVAL.trigram_index(np.empty(0, dtype=np.int64), 10).size == 0


# --------------------------------------------------------------------------- #
# Row gating: per-row step weights from each row's training count.
# --------------------------------------------------------------------------- #
def test_the_default_gating_reproduces_the_existing_arms_exactly():
    # threshold 0 must be a no-op, or every pre-registered B1 arm stops being a
    # baseline for the gated ones.
    c = np.array([0, 1, 2, 50, 10_000], dtype=np.int64)
    w = TRAIN.row_weights(c, 0)
    assert w.tolist() == [1.0] * 5
    assert w.dtype == np.float32


def test_the_hard_mask_holds_every_row_below_the_threshold():
    c = np.array([0, 1, 9, 10, 11], dtype=np.int64)
    assert TRAIN.row_weights(c, 10).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    # a held row contributes exactly zero, which is what makes a masked arm's
    # aggregate predictable in advance from the frozen band table
    assert TRAIN.row_weights(c, 10)[:3].sum() == 0.0


def test_shrinkage_is_monotone_and_bounded():
    c = np.array([0, 1, 10, 100, 10_000], dtype=np.int64)
    w = TRAIN.row_weights(c, 10, 1.0)
    assert w[0] == 0.0
    assert w[2] == pytest.approx(0.5)          # c == k
    assert np.all(np.diff(w) > 0)              # monotone in evidence
    assert np.all(w >= 0) and np.all(w < 1)    # never amplifies


def test_shrinkage_approaches_one_for_heavily_seen_rows():
    # The point of the weight: a row with overwhelming evidence is left alone.
    assert TRAIN.row_weights(np.array([10**6]), 10, 1.0)[0] > 0.999
    assert TRAIN.row_weights(np.array([10**6]), 50, 1.0)[0] > 0.999


def test_a_larger_threshold_shrinks_every_row_more():
    c = np.array([1, 10, 100, 1000], dtype=np.int64)
    w10 = TRAIN.row_weights(c, 10, 1.0)
    w50 = TRAIN.row_weights(c, 50, 1.0)
    assert np.all(w50 <= w10)


def test_the_scale_power_controls_how_sharp_the_gate_is():
    c = np.array([5, 10, 20], dtype=np.int64)
    soft = TRAIN.row_weights(c, 10, 1.0)
    sharp = TRAIN.row_weights(c, 10, 4.0)
    # a higher power drives the weight toward the hard mask at c != k
    assert sharp[0] < soft[0] and sharp[2] > soft[2]
    assert np.all((sharp >= 0) & (sharp <= 1))


def test_the_count_table_need_not_be_sorted_or_dense():
    # row_weights indexes by ROW, so the count vector must not be reordered.
    c = np.array([100, 1, 50, 2], dtype=np.int64)
    assert TRAIN.row_weights(c, 10).tolist() == [1.0, 0.0, 1.0, 0.0]
