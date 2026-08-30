"""Tests for the A0/A1 evaluation comparison protocol."""

from __future__ import annotations

import json

import pytest

from qwen35_ple.eval.protocol import load_comparison


def test_compare_a0_a1(tmp_path):
    a0 = tmp_path / "a0.json"
    a1 = tmp_path / "a1.json"
    a0.write_text(
        json.dumps({"model": "A0", "metrics": {"knowledge": 0.50, "longctx": 0.40}}),
        encoding="utf-8",
    )
    a1.write_text(
        json.dumps({"model": "A1", "metrics": {"knowledge": 0.60, "longctx": 0.38}}),
        encoding="utf-8",
    )

    comp = load_comparison(a0, a1)
    assert comp.delta("knowledge") == pytest.approx(0.10)
    assert comp.relative_delta("knowledge") == pytest.approx(0.20)
    assert "A0" in comp.to_report()
    assert "A1" in comp.to_report()
