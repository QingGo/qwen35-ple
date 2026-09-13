"""End-to-end CPU test of the Stage 1.5f driver (no torch, no GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from qwen35_ple.addressing import structured_stream

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "round168_addressing_resolution",
        REPO_ROOT / "scripts" / "round168_addressing_resolution.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _run(tmp_path: Path, tag: str, tokens: np.ndarray) -> dict:
    npy = tmp_path / f"{tag}.npy"
    np.save(npy, tokens)
    argv = sys.argv
    sys.argv = [
        "x", "--tag", tag, "--eval-npy", str(npy),
        "--workdir", str(tmp_path), "--max-tokens", "6000",
    ]
    try:
        assert MOD.main() == 0
    finally:
        sys.argv = argv
    return json.loads((tmp_path / f"{tag}-addressing.json").read_text())


def test_real_spec_is_injective_on_a_stream(tmp_path):
    out = _run(tmp_path, "wiki", structured_stream(repeats=600))
    assert out["validation"]["real_injective"] is True
    assert out["validation"]["real_top1_gap"] == 0.0
    assert out["conclusion"].startswith("ADDRESSING_IS_LOSSLESS")


def test_control_registers_a_loss(tmp_path):
    out = _run(tmp_path, "wiki", structured_stream(repeats=600))
    assert out["validation"]["control_lossy"] is True
    assert out["validation"]["control_lost_tuples"] > 0
    assert out["validation"]["resolution_sensitive"] is True


def test_applicability_arm_shows_top1_works_on_a_structured_stream(tmp_path):
    out = _run(tmp_path, "wiki", structured_stream(repeats=600))
    ap = out["applicability"]
    assert ap["crushed_lossy"] is True
    assert ap["crushed_top1_gap"] > 0.05
    assert ap["real_top1_gap"] == 0.0


def test_report_states_the_corpus_dependence_of_top1(tmp_path):
    _run(tmp_path, "wiki", structured_stream(repeats=600))
    md = (tmp_path / "wiki-addressing.md").read_text()
    assert "Top-1 needs a structured corpus" in md
    assert "Two-way validation" in md


def test_arms_are_reported_for_every_crush_factor(tmp_path):
    out = _run(tmp_path, "wiki", structured_stream(repeats=600))
    names = [a["arm"] for a in out["arms"]]
    assert names[0] == "real"
    assert len(names) == 1 + len(out["crush_factors"])
    assert all(a["total_rows"] > 0 for a in out["arms"])
