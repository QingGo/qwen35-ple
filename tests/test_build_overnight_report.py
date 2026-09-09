"""Tests for the overnight report formatting helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_overnight_report.py"
    spec = importlib.util.spec_from_file_location("build_overnight_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_metric_uses_task_aware_default():
    module = _load_module()
    entry = {
        "tasks": {
            "boolq": {"extracted_exact": 0.5, "extracted_contains": 0.9},
            "triviaqa": {"extracted_exact": 0.1, "extracted_contains": 0.7},
        }
    }
    assert module._task_metric(entry, "boolq") == 0.5
    assert module._task_metric(entry, "triviaqa") == 0.7
    assert module._task_metric(entry, "nq") is None


def test_fmt_handles_nan_and_none():
    module = _load_module()
    assert module._fmt(None) == "N/A"
    assert module._fmt(float("nan")) == "N/A"
    assert module._fmt(0.123456) == "0.1235"
