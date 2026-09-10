"""Tests for the gold-answer NLL summary helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_gold_nll.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_gold_nll", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(task: str, question: str, nll: float, field: str = "qa_gold") -> dict:
    key = "gold_nll" if field == "qa_gold" else "loss"
    return {"task": task, "question": question, "answer": "a", key: nll}


def test_extract_items_prefers_qa_gold():
    module = _load_module()
    data = {
        "results": [
            {
                "qa_gold": {
                    "answers": [
                        _item("boolq", "q1", 1.5),
                        _item("triviaqa", "q2", 2.5),
                    ]
                }
            }
        ]
    }
    items, field = module._extract_items(data)
    assert field == "qa_gold"
    assert [item["nll"] for item in items] == [1.5, 2.5]


def test_extract_items_falls_back_to_legacy_qa():
    module = _load_module()
    data = {"results": [{"qa": {"answers": [_item("nq", "q", 3.0, "qa")]}}]}
    items, field = module._extract_items(data)
    assert field == "qa"
    assert items[0]["nll"] == 3.0


def test_paired_delta_direction(tmp_path: Path):
    module = _load_module()
    def norm(task: str, question: str, nll: float) -> dict:
        return {"task": task, "question": question, "answer": "a", "nll": nll}

    baseline = [norm("boolq", "q1", 5.0), norm("triviaqa", "q2", 6.0)]
    better = [norm("boolq", "q1", 4.0), norm("triviaqa", "q2", 7.0)]
    markdown = module.build_report(
        {"baseline": baseline, "run": better}, "baseline", "title"
    )
    assert "Paired against" in markdown
    # mean delta = ((5-4) + (6-7)) / 2 = 0.0
    assert "| run | 1.0000 | -1.0000 | N/A | 0.0000 |" in markdown


def test_cli_writes_outputs(tmp_path: Path):
    module = _load_module()
    baseline_path = tmp_path / "baseline.json"
    run_path = tmp_path / "run.json"
    baseline_path.write_text(
        json.dumps({"results": [{"qa_gold": {"answers": [_item("boolq", "q", 5.0)]}}]}),
        encoding="utf-8",
    )
    run_path.write_text(
        json.dumps({"results": [{"qa_gold": {"answers": [_item("boolq", "q", 4.0)]}}]}),
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    rc = module.main(
        [
            "--run",
            f"baseline={baseline_path}",
            "--run",
            f"run={run_path}",
            "--baseline",
            "baseline",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.with_suffix(".json").exists()
