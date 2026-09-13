"""End-to-end CPU test of the Stage 1.5e driver (no torch, no GPU).

The GPU stages cannot run in CI.  What is testable here is the wiring where a
sign error would be invisible: NLL -> probability, the choice of decision
quantity, and the Markdown report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module("round168_fusion_probe", "round168_fusion_probe.py")

N = 3000


def _write_npz(tmp_path: Path, tag: str, *, rescue: bool = True):
    """``p_bb`` collapses exactly where ``p_cnt`` rescues (or never does)."""
    rng = np.random.default_rng(0)
    z = rng.random(N) < 0.5
    if rescue:
        lens_nll = np.where(z, 1.0, 9.0)          # backbone often lost
        cnt_nll = np.where(z, 5.0, 1.5)           # count model rescues
    else:
        lens_nll = np.where(rng.random(N) < 0.6, 1.0, 9.0)
        cnt_nll = np.full(N, 6.0)                 # constant: shrinkage only
    final_nll = lens_nll - 0.5
    ctx = rng.integers(0, 60, size=N)
    positions = np.arange(3, N, dtype=np.int64)
    np.savez_compressed(
        tmp_path / f"{tag}-counts.npz",
        cnt_nll=cnt_nll.astype(np.float32),
        context_counts=ctx.astype(np.int64),
        target_counts=rng.integers(0, 500, size=N).astype(np.int64),
        positions=positions,
        train_npy="train.npy",
        eval_npy="eval.npy",
    )
    np.savez_compressed(
        tmp_path / f"{tag}-backbone.npz",
        bb_final=final_nll.astype(np.float32),
        bb_lens=lens_nll.astype(np.float32),
    )


def _args(tmp_path: Path, tag: str, n_perm: int = 4) -> argparse.Namespace:
    return argparse.Namespace(
        tag=tag, workdir=str(tmp_path), output=None, n_perm=n_perm,
    )


def test_main_writes_json_and_markdown(tmp_path):
    _write_npz(tmp_path, "wiki")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "4"]
    try:
        assert MOD.main() == 0
    finally:
        sys.argv = argv
    out = json.loads((tmp_path / "wiki-fusion.json").read_text())
    assert out["n_positions"] == N - 3
    assert set(out["frames"]) == {"lens", "final"}
    assert (tmp_path / "wiki-fusion.md").exists()


def test_rescue_structure_is_confirmed_end_to_end(tmp_path):
    _write_npz(tmp_path, "wiki", rescue=True)
    import sys
    argv = sys.argv
    sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "4"]
    try:
        MOD.main()
    finally:
        sys.argv = argv
    out = json.loads((tmp_path / "wiki-fusion.json").read_text())
    assert out["verdict"]["label"] == "COMPLEMENTARITY_CONFIRMED"
    assert out["frames"]["lens"]["global"]["verdict"]["excess"] > 0.1


def test_constant_counter_is_absent_end_to_end(tmp_path):
    _write_npz(tmp_path, "wiki", rescue=False)
    import sys
    argv = sys.argv
    sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "4"]
    try:
        MOD.main()
    finally:
        sys.argv = argv
    out = json.loads((tmp_path / "wiki-fusion.json").read_text())
    lens = out["frames"]["lens"]
    # a large raw gain (recalibration) but no excess
    assert lens["global"]["real"]["gain"] > 0.0
    assert abs(lens["global"]["verdict"]["excess"]) < 0.01
    assert out["verdict"]["label"] == "COMPLEMENTARITY_ABSENT"


def test_report_marks_the_excess_as_the_decision_quantity(tmp_path):
    _write_npz(tmp_path, "wiki")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "4"]
    try:
        MOD.main()
    finally:
        sys.argv = argv
    md = (tmp_path / "wiki-fusion.md").read_text()
    assert "Per context-frequency band" in md
    assert "only the **excess** carries" in md
    assert "oracle gain" in md


def test_band_population_covers_every_scored_position(tmp_path):
    _write_npz(tmp_path, "wiki")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "2"]
    try:
        MOD.main()
    finally:
        sys.argv = argv
    out = json.loads((tmp_path / "wiki-fusion.json").read_text())
    assert sum(out["band_population"].values()) == out["n_positions"]


def test_missing_backbone_file_fails_loudly(tmp_path):
    _write_npz(tmp_path, "wiki")
    (tmp_path / "wiki-backbone.npz").unlink()
    with pytest.raises(FileNotFoundError):
        import sys
        argv = sys.argv
        sys.argv = ["x", "--tag", "wiki", "--workdir", str(tmp_path), "--n-perm", "2"]
        try:
            MOD.main()
        finally:
            sys.argv = argv
