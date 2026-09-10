"""Tests for the offline oracle routing upper-bound analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_oracle_upper_bound.py"
    spec = importlib.util.spec_from_file_location("analyze_oracle_upper_bound", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(task: str, gold: str, prediction: str) -> dict:
    return {
        "task": task,
        "question": f"q-{task}-{gold}",
        "answer": gold,
        "generated": prediction,
    }


def test_iter_mode_rows_reads_summary_details():
    module = _load_module()
    data = {
        "summary": {
            "real": {
                "details": [
                    {
                        "seed": 0,
                        "qa_exact": {
                            "answers": [_row("triviaqa", "Paris", "Paris")]
                        },
                    }
                ]
            }
        }
    }
    rows = module._iter_mode_rows(data)
    assert ("real", 0) in rows
    assert rows[("real", 0)][0]["answer"] == "Paris"


def test_iter_mode_rows_aliases_full_to_no_reader():
    module = _load_module()
    data = {
        "results": [
            {
                "mode": "full",
                "seed": 0,
                "qa_exact": {"answers": [_row("boolq", "yes", "yes")]},
            }
        ]
    }
    rows = module._iter_mode_rows(data)
    assert ("no-reader", 0) in rows
    assert ("full", 0) not in rows


def test_unique_counts_empty_when_modes_missing():
    module = _load_module()
    rows_by_mode = {"real": [_row("boolq", "yes", "yes")]}
    assert module._unique_counts(rows_by_mode, [0], "v2", "auto") == {}


def test_summarize_oracle_upper_bounds():
    module = _load_module()
    rows_by_mode = {
        "real": [
            _row("boolq", "yes", "yes"),
            _row("boolq", "no", "yes"),
            _row("triviaqa", "Paris", "Paris"),
            _row("triviaqa", "Jupiter", "Jupiter"),
            _row("nq", "Au", "Au"),
            _row("nq", "yen", "dollar"),
        ],
        "control": [
            _row("boolq", "yes", "no"),
            _row("boolq", "no", "no"),
            _row("triviaqa", "Paris", "Mars"),
            _row("triviaqa", "Jupiter", "Jupiter"),
            _row("nq", "Au", "Ag"),
            _row("nq", "yen", "yen"),
        ],
        "no-reader": [
            _row("boolq", "yes", "yes"),
            _row("boolq", "no", "no"),
            _row("triviaqa", "Paris", "London"),
            _row("triviaqa", "Jupiter", "Saturn"),
            _row("nq", "Au", "Au"),
            _row("nq", "yen", "euro"),
        ],
    }
    report = module._summarize(rows_by_mode, protocol="v2", metric="auto")
    assert report["overall"]["always_real"] == 4 / 6
    assert report["overall"]["always_control"] == 3 / 6
    assert report["overall"]["always_no_reader"] == 3 / 6
    assert report["oracle_real_no"]["accuracy"] == 5 / 6
    assert report["oracle_all"]["accuracy"] == 1.0
    assert report["per_task"]["boolq"]["always_no_reader"] == 1.0
    assert report["per_task"]["triviaqa"]["always_real"] == 1.0
    assert report["oracle_all"]["per_task"]["nq"] == 1.0
    assert report["oracle_real_no"]["choice_counts"]["real"] >= 1
    assert report["oracle_real_no"]["choice_counts"]["no-reader"] >= 1


def test_oracle_all_ignores_all_wrong_items():
    module = _load_module()
    rows_by_mode = {
        "real": [_row("triviaqa", "Paris", "Mars")],
        "control": [_row("triviaqa", "Paris", "Venus")],
        "no-reader": [_row("triviaqa", "Paris", "London")],
    }
    report = module._summarize(rows_by_mode, protocol="v2", metric="auto")
    assert report["oracle_all"]["accuracy"] == 0.0
