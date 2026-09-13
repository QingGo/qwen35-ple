"""Tests for the Stage 1.5c driver script's CPU path (no torch, no GPU).

The GPU stages cannot run in CI; what *can* be tested is the part where a silent
misalignment would be invisible -- combining the two ``.npz`` files, applying the
pre-registered rule and rendering the Markdown.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "round168_margin_distribution",
        REPO_ROOT / "scripts" / "round168_margin_distribution.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _write_inputs(tmp_path: Path, tag: str, *, n: int = 400, positive_tail: bool = True):
    rng = np.random.default_rng(0)
    cnt = rng.normal(5.0, 0.5, size=n)
    final = cnt + rng.normal(-1.0, 0.5, size=n)  # backbone better on average
    lens = cnt + rng.normal(-1.5, 0.5, size=n)   # injection point worse
    if positive_tail:
        final[:40] = cnt[:40] + 2.0
        lens[:40] = cnt[:40] + 3.0
    ctx = rng.integers(0, 50, size=n)
    tgt = rng.integers(0, 500, size=n)
    positions = np.arange(2, n, dtype=np.int64)
    for arr in (cnt, final, lens, ctx, tgt):
        arr[:2] = np.nan if arr.dtype.kind == "f" else 0
    np.savez_compressed(
        tmp_path / f"{tag}-counts.npz",
        cnt_nll=cnt.astype(np.float32),
        top_level=np.zeros(n, dtype=np.int8),
        context_counts=ctx.astype(np.int64),
        target_counts=tgt.astype(np.int64),
        positions=positions,
        train_npy="train.npy",
        eval_npy="eval.npy",
        order=np.int64(3),
        uniform_vocab=np.int64(248047),
        smoothing="mkn",
        n_train_tokens=np.int64(1000),
        n_eval_tokens=np.int64(n),
    )
    np.savez_compressed(
        tmp_path / f"{tag}-backbone.npz",
        bb_final=final.astype(np.float32),
        bb_lens=lens.astype(np.float32),
        model="Qwen3.5-0.8B",
        layer=np.int64(2),
        lens_hidden_index=np.int64(3),
        backbone_dtype="float32",
        chunk_tokens=np.int64(4096),
        n_eval_tokens=np.int64(n),
        eval_npy="eval.npy",
    )


def _args(tmp_path: Path, tag: str, min_ctx: int = 5) -> argparse.Namespace:
    return argparse.Namespace(
        workdir=str(tmp_path), tag=tag, min_context_count=min_ctx,
    )


def test_analyze_writes_json_and_markdown(tmp_path):
    _write_inputs(tmp_path, "wiki")
    assert MOD.stage_analyze(_args(tmp_path, "wiki")) == 0
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    assert out["primary_frame"] == "lens"
    assert set(out["frames"]) == {"lens", "final"}
    assert (tmp_path / "wiki-margin.md").exists()


def test_analyze_uses_both_frames_independently(tmp_path):
    _write_inputs(tmp_path, "wiki")
    MOD.stage_analyze(_args(tmp_path, "wiki"))
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    lens = out["frames"]["lens"]
    final = out["frames"]["final"]
    assert lens["mean_bb_nll"] != pytest.approx(final["mean_bb_nll"])
    assert out["verdict"] == lens["verdict"]


def test_analyze_aligns_on_the_count_model_position_set(tmp_path):
    _write_inputs(tmp_path, "wiki", n=50)
    MOD.stage_analyze(_args(tmp_path, "wiki"))
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    # positions 0 and 1 are outside the trigram model's scored set
    assert out["n_scored_positions"] == 48
    assert out["coverage"] == pytest.approx(48 / 50)


def test_analyze_reports_positive_verdict_when_tail_exists(tmp_path):
    _write_inputs(tmp_path, "wiki", positive_tail=True)
    MOD.stage_analyze(_args(tmp_path, "wiki"))
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    assert out["frames"]["lens"]["positive_share"] > 0
    assert out["verdict"]["label"] != "NO_POSITIVE_POSITION"


def test_analyze_reports_no_positive_position_when_backbone_always_wins(tmp_path):
    _write_inputs(tmp_path, "wiki", positive_tail=False)
    MOD.stage_analyze(_args(tmp_path, "wiki"))
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    assert out["frames"]["lens"]["positive_share"] == 0.0
    assert out["verdict"]["label"] == "NO_POSITIVE_POSITION"


def test_analyze_rejects_mismatched_array_lengths(tmp_path):
    _write_inputs(tmp_path, "wiki", n=40)
    counts = dict(np.load(tmp_path / "wiki-counts.npz"))
    counts["cnt_nll"] = counts["cnt_nll"][:30]
    counts["positions"] = counts["positions"][:30]
    np.savez_compressed(tmp_path / "wiki-counts.npz", **counts)
    with pytest.raises(SystemExit, match="length mismatch"):
        MOD.stage_analyze(_args(tmp_path, "wiki"))


def test_markdown_contains_both_curves_and_the_cross_tab(tmp_path):
    _write_inputs(tmp_path, "wiki")
    MOD.stage_analyze(_args(tmp_path, "wiki"))
    md = (tmp_path / "wiki-margin.md").read_text()
    assert "Oracle gain curve" in md
    assert "ctx_count>=5" in md
    assert "Cross-tab" in md
    assert "target \\ ctx" in md


def test_min_context_count_is_recorded_and_used(tmp_path):
    _write_inputs(tmp_path, "wiki")
    MOD.stage_analyze(_args(tmp_path, "wiki", min_ctx=10))
    out = json.loads((tmp_path / "wiki-margin.json").read_text())
    assert out["min_context_count"] == 10
    labels = {e["label"] for e in out["frames"]["lens"]["gain_curve"]}
    assert "ctx_count>=10" in labels


def test_paths_helper_names_are_stable(tmp_path):
    p = MOD._paths(tmp_path, "stem")
    assert p["counts"].name == "stem-counts.npz"
    assert p["json"].name == "stem-margin.json"
