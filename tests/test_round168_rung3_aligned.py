"""Tests for the Stage 1.5g alignment layer (CPU only, no torch, no row table).

Two things can go silently wrong here, and both would be invisible in the final
number:

* the rebuilt position set could stop being the set Stage 1.5c actually scored,
  which is why the exporter refuses to write when its count disagrees with the
  recorded ``n_scored_positions``; and
* the probe could be handed positions it cannot legally score (no next token,
  target outside the candidate set), which is why ``split_aligned_positions``
  validates instead of trusting its input.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALIGNED = _load("scripts/round168_rung3_aligned.py", "round168_rung3_aligned")
PROBE = _load("scripts/probe_table_next_token.py", "probe_table_next_token")


# --------------------------------------------------------------------------- #
# rebuilding the Stage 1.5c position set
# --------------------------------------------------------------------------- #
def _write_stage15c(tmp_path: Path, *, n: int = 300, n_scored_recorded: int | None = None):
    """The two npz files Stage 1.5c leaves behind, plus its margin json."""
    positions = np.arange(2, n, dtype=np.int64)  # the count model scored these
    cnt = np.full(n, 5.0)
    final = np.full(n, 3.0)
    lens = np.full(n, 12.0)
    # a chunk boundary leaves NaN in the backbone alone
    for i in (10, 11, 200):
        final[i] = np.nan
    # a stream tail leaves NaN in the count model alone
    for i in (50, 51):
        cnt[i] = np.nan
    np.savez(
        tmp_path / "counts.npz",
        cnt_nll=cnt,
        positions=positions,
        eval_npy="data/eval.npy",
        train_npy="data/train.npy",
    )
    np.savez(tmp_path / "backbone.npz", bb_final=final, bb_lens=lens)

    valid = np.isfinite(cnt) & np.isfinite(final) & np.isfinite(lens)
    mask = np.zeros(n, dtype=bool)
    mask[positions] = True
    valid &= mask
    n_scored = int(valid.sum()) if n_scored_recorded is None else n_scored_recorded
    (tmp_path / "margin.json").write_text(json.dumps({"n_scored_positions": n_scored}))
    return int(valid.sum())


def test_aligned_positions_intersects_all_three_finiteness_masks(tmp_path: Path):
    expected = _write_stage15c(tmp_path)
    pos, meta = ALIGNED.aligned_positions(tmp_path / "counts.npz", tmp_path / "backbone.npz")
    assert pos.size == expected
    assert meta["n_eval_tokens"] == 300
    # ascending, so the contiguous split in the probe means what it says
    assert np.all(np.diff(pos) > 0)
    # the three NaN holes are gone: 2 in the count stream, 3 in the backbone
    # (one of which, index 200, is only reachable through the positions mask)
    assert 10 not in pos and 50 not in pos and 200 not in pos
    assert pos[0] == 2


def test_positions_outside_the_count_mask_are_excluded(tmp_path: Path):
    _write_stage15c(tmp_path)
    pos, _ = ALIGNED.aligned_positions(tmp_path / "counts.npz", tmp_path / "backbone.npz")
    # index 0 and 1 are finite everywhere but were never scored by the counts
    assert 0 not in pos and 1 not in pos


def test_export_refuses_to_write_when_the_count_drifts_from_stage_1_5c(tmp_path: Path):
    _write_stage15c(tmp_path, n_scored_recorded=999)
    args = argparse.Namespace(
        counts=str(tmp_path / "counts.npz"),
        backbone=str(tmp_path / "backbone.npz"),
        margin_json=str(tmp_path / "margin.json"),
        out=str(tmp_path / "out.npy"),
    )
    with pytest.raises(SystemExit, match="ALIGNMENT DRIFT"):
        ALIGNED.cmd_export(args)
    assert not (tmp_path / "out.npy").exists()


def test_export_writes_the_usable_set_and_records_the_tail_drop(tmp_path: Path):
    _write_stage15c(tmp_path)
    args = argparse.Namespace(
        counts=str(tmp_path / "counts.npz"),
        backbone=str(tmp_path / "backbone.npz"),
        margin_json=str(tmp_path / "margin.json"),
        out=str(tmp_path / "out.npy"),
    )
    assert ALIGNED.cmd_export(args) == 0
    saved = np.load(tmp_path / "out.npy")
    meta = json.loads((tmp_path / "out.meta.json").read_text())
    # index 299 is the last scored position and has no next token in a 300-token
    # stream, so it is dropped from the file while staying in the drift count
    assert meta["n_aligned"] == saved.size + 1
    assert meta["dropped_no_next_token"] == 1
    assert saved.max() + 1 < meta["n_eval_tokens"]


# --------------------------------------------------------------------------- #
# the probe's aligned split
# --------------------------------------------------------------------------- #
def _stream(n: int = 1000, vocab_token: int = 7):
    toks = np.full(n, vocab_token, dtype=np.int64)
    return toks


def test_split_is_contiguous_and_the_temperature_set_sits_inside_the_fit_set():
    toks = _stream(1000)
    pos = np.arange(2, 998, dtype=np.int64)
    train, temp, ev, meta = PROBE.split_aligned_positions(
        pos, toks, keep_targets=np.array([7]), train_frac=0.60, temp_frac=0.10,
        min_kept=100,
    )
    assert len(train) + len(ev) == len(pos)
    assert meta["cut_train"] == pytest.approx(round(0.60 * len(pos)), abs=1)
    assert meta["contiguous_split"] is True
    # the temperature block is the tail of the fit block, and eval starts after
    assert temp[0] == train[meta["cut_temp"]]
    assert train[-1] < ev[0]
    assert set(temp.tolist()) <= set(train.tolist())
    assert set(temp.tolist()).isdisjoint(ev.tolist())


def test_targets_outside_the_candidate_set_are_dropped_rather_than_mislabelled():
    """``cls`` is zero-initialised, so an unkept target would score as vocab[0]."""
    toks = np.full(1000, 7, dtype=np.int64)
    toks[100:200] = 9  # these targets are not in the candidate set
    pos = np.arange(2, 998, dtype=np.int64)
    train, temp, ev, meta = PROBE.split_aligned_positions(
        pos, toks, keep_targets=np.array([7]), train_frac=0.60, temp_frac=0.10,
        min_kept=100,
    )
    kept = np.concatenate([train, temp, ev])
    assert not np.isin(toks[kept + 1], [9]).any()
    assert meta["n_dropped_target_outside_topk"] > 0
    assert meta["n_positions_kept"] == len(pos) - meta["n_dropped_target_outside_topk"]


def test_a_position_without_a_next_token_is_rejected_not_silently_wrapped():
    toks = _stream(1000)
    pos = np.array([2, 500, 999], dtype=np.int64)  # 999 + 1 == T
    with pytest.raises(ValueError, match="do not fit the eval stream"):
        PROBE.split_aligned_positions(
            pos, toks, keep_targets=np.array([7]), train_frac=0.60, temp_frac=0.10
        )


def test_an_empty_position_set_is_rejected():
    with pytest.raises(ValueError, match="no aligned positions"):
        PROBE.split_aligned_positions(
            np.array([], dtype=np.int64),
            _stream(),
            keep_targets=np.array([7]),
            train_frac=0.60,
            temp_frac=0.10,
        )


def test_a_temperature_fraction_at_or_above_the_train_fraction_is_rejected():
    with pytest.raises(ValueError, match="temp_frac"):
        PROBE.split_aligned_positions(
            np.arange(2, 998, dtype=np.int64),
            _stream(),
            keep_targets=np.array([7]),
            train_frac=0.60,
            temp_frac=0.60,
        )


def test_a_too_small_kept_set_is_rejected():
    with pytest.raises(ValueError, match="top-K filter"):
        PROBE.split_aligned_positions(
            np.arange(2, 998, dtype=np.int64),
            _stream(),
            keep_targets=np.array([12345]),  # never matches
            train_frac=0.60,
            temp_frac=0.10,
        )


# --------------------------------------------------------------------------- #
# aggregation end to end
# --------------------------------------------------------------------------- #
def _probe_json(probe: float, count: float, shuffled: float, mlp: float, mlp_shuf: float):
    def frame(v):
        return {"eval": {"top1": v}}

    return {
        "results": {
            "probe_raw_rows": frame(probe),
            "count_trigram": frame(count),
            "control_shuffled_train_rows": frame(shuffled),
            "majority_train_prior": frame(0.04),
            "probe_mlp_raw_rows": frame(mlp),
            "control_shuffled_train_rows_mlp": frame(mlp_shuf),
        }
    }


def test_aggregate_reads_both_k_and_writes_json_and_markdown(tmp_path: Path):
    big = tmp_path / "k5000.json"
    small = tmp_path / "k1000.json"
    big.write_text(json.dumps(_probe_json(0.30, 0.50, 0.06, 0.34, 0.06)))
    small.write_text(json.dumps(_probe_json(0.30, 0.55, 0.06, 0.34, 0.06)))
    args = argparse.Namespace(
        tag="wiki",
        probe=[f"5000={big}", f"1000={small}"],
        positions="positions.npy",
        eval_tokens="eval.npy",
        out_json=str(tmp_path / "verdict.json"),
        out_md="",
    )
    assert ALIGNED.cmd_aggregate(args) == 0
    out = json.loads((tmp_path / "verdict.json").read_text())
    assert out["verdict"] == "ROWS_CARRY_THE_TRIGRAM"
    assert out["provenance"]["primary_k"] == 5000
    assert (tmp_path / "verdict.md").exists()


def test_aggregate_rejects_a_malformed_probe_spec(tmp_path: Path):
    args = argparse.Namespace(
        tag="wiki",
        probe=["not-a-spec"],
        positions="",
        eval_tokens="",
        out_json=str(tmp_path / "verdict.json"),
        out_md="",
    )
    with pytest.raises(SystemExit, match="K=path"):
        ALIGNED.cmd_aggregate(args)
