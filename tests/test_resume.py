"""Tests for fine-grained Phase 0/2 resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

from qwen35_ple.eval.resume import (
    load_partial_results,
    merge_results,
    partial_result_path,
    write_partial_result,
)


def test_partial_path_and_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "phase1-PURE_WIKI.json"
    partial_dir = tmp_path / "partial"
    path = partial_result_path(partial_dir, output, "real", 0)
    assert path.name == "phase1-PURE_WIKI-real-seed0.json"

    result = {"mode": "real", "seed": 0, "val_loss": 1.0}
    written = write_partial_result(partial_dir, output, result)
    assert written == path
    loaded = load_partial_results(partial_dir, output)
    assert loaded[("real", 0)]["val_loss"] == 1.0


def test_load_ignores_foreign_and_corrupt(tmp_path: Path) -> None:
    output = tmp_path / "phase1-PURE_WIKI.json"
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    (partial_dir / "phase1-PURE_FINEWEB-real-seed0.json").write_text(
        json.dumps({"mode": "real", "seed": 0}), encoding="utf-8"
    )
    (partial_dir / "phase1-PURE_WIKI-real-seed1.json").write_text(
        "{not json", encoding="utf-8"
    )
    assert load_partial_results(partial_dir, output) == {}


def test_merge_results_orders_by_seed_then_mode(tmp_path: Path) -> None:
    existing = {
        ("control", 0): {"mode": "control", "seed": 0},
        ("real", 0): {"mode": "real", "seed": 0},
    }
    new = [
        {"mode": "no-reader", "seed": 1},
        {"mode": "real", "seed": 1},
    ]
    merged = merge_results(
        existing,
        new,
        modes=("real", "control", "no-reader"),
        seeds=(0, 1),
    )
    assert [(item["mode"], item["seed"]) for item in merged] == [
        ("real", 0),
        ("control", 0),
        ("real", 1),
        ("no-reader", 1),
    ]
