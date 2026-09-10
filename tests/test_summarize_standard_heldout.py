"""Tests for the standard held-out summary helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_standard_heldout.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_standard_heldout", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(task: str, gold: str, generated: str) -> dict:
    return {
        "task": task,
        "question": f"q-{task}-{gold}",
        "answer": gold,
        "generated": generated,
    }


def _payload(answers: list[dict]) -> dict:
    return {"results": [{"qa_exact": {"answers": answers}}]}


def test_answers_from_results_and_summary():
    module = _load_module()
    answers = [_answer("boolq", "yes", "yes")]
    assert module._answers_from_payload(_payload(answers)) == answers
    summary_payload = {
        "summary": {
            "full": {
                "details": [{"seed": 0, "qa_exact": {"answers": answers}}]
            }
        }
    }
    assert module._answers_from_payload(summary_payload) == answers


def test_score_rows_uses_v2_extraction_metrics():
    module = _load_module()
    rows = [
        _answer("boolq", "yes", "yes"),
        _answer("boolq", "no", "yes"),
        _answer("triviaqa", "Paris", "Paris"),
        _answer("triviaqa", "Jupiter", "London"),
        _answer("nq", "Au", "dollar"),
    ]
    metrics = module.score_rows(rows)
    assert metrics["boolq"] == 0.5
    assert metrics["triviaqa"] == 0.5
    assert metrics["nq"] == 0.0
    assert metrics["mean"] == 2 / 5


def test_subset_per_task_keeps_first_items():
    module = _load_module()
    rows = [_answer("boolq", "yes", "yes")] * 3 + [
        _answer("boolq", "no", "no")
    ] * 2
    metrics = module.score_rows(rows, subset_per_task=2)
    assert metrics["boolq"] == 1.0
    assert metrics["mean"] == 1.0


def test_collect_arm_metrics_maps_full_to_no_reader(tmp_path: Path):
    module = _load_module()
    (tmp_path / "phase1-full.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "yes")])),
        encoding="utf-8",
    )
    (tmp_path / "phase1-real-seed0.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "no")])),
        encoding="utf-8",
    )
    (tmp_path / "phase1-real-seed1.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "yes")])),
        encoding="utf-8",
    )
    arms = module.collect_arm_metrics(tmp_path)
    assert [entry["seed"] for entry in arms["no-reader"]] == [0]
    assert [entry["seed"] for entry in arms["real"]] == [0, 1]
    assert arms["no-reader"][0]["boolq"] == 1.0
    assert arms["real"][0]["boolq"] == 0.0


def test_build_markdown_contains_factorial(tmp_path: Path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    chat_dir = tmp_path / "chat"
    raw_dir.mkdir()
    chat_dir.mkdir()
    (raw_dir / "phase1-full.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "no")])),
        encoding="utf-8",
    )
    (raw_dir / "phase1-real-seed0.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "yes")])),
        encoding="utf-8",
    )
    (chat_dir / "phase1-full.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "yes")])),
        encoding="utf-8",
    )
    (chat_dir / "phase1-real-seed0.json").write_text(
        json.dumps(_payload([_answer("boolq", "yes", "no")])),
        encoding="utf-8",
    )
    markdown = module.build_markdown(raw_dir, chat_dir)
    assert "## 2x2 train/eval format table" in markdown
    assert "| raw-trained reader | 1.0000 ± 0.0000 |" in markdown
    assert "| chat-trained reader | N/A (not run) | 0.0000 ± 0.0000 |" in markdown
