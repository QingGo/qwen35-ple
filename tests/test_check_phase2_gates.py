"""Tests for scripts/check_phase2_gates.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_phase2_gates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_phase2_gates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(
    name: str,
    *,
    real_ppl: float,
    control_ppl: float,
    no_ppl: float,
    real_task: float,
    control_task: float,
    no_task: float,
) -> dict:
    def mode(ppl: float, task: float) -> dict:
        return {
            "ppl_mean": ppl,
            "ppl_by_seed": [ppl, ppl, ppl],
            "tasks": {"boolq": {"extracted_exact": task}},
        }

    return {
        "corpus": name,
        "modes": {
            "real": mode(real_ppl, real_task),
            "control": mode(control_ppl, control_task),
            "no-reader": mode(no_ppl, no_task),
        },
    }


def test_ppl_gate_passes() -> None:
    module = _load_module()
    corpora = [
        _corpus(
            f"C{index}",
            real_ppl=1.0,
            control_ppl=2.0,
            no_ppl=3.0,
            real_task=0.7,
            control_task=0.6,
            no_task=0.2,
        )
        for index in range(6)
    ]
    assert module._ppl_gate(corpora)["pass"] is True


def test_ppl_gate_fails_when_real_worse_than_control() -> None:
    module = _load_module()
    corpora = [
        _corpus(
            f"C{index}",
            real_ppl=2.0 if index < 2 else 1.0,
            control_ppl=1.5 if index < 2 else 2.0,
            no_ppl=3.0,
            real_task=0.7,
            control_task=0.6,
            no_task=0.2,
        )
        for index in range(6)
    ]
    # Two of six corpora fail the 5/6 threshold.
    assert module._ppl_gate(corpora)["pass"] is False


def test_resolve_task_metric_auto() -> None:
    module = _load_module()
    assert module._resolve_task_metric("boolq", "auto") == "extracted_exact"
    assert module._resolve_task_metric("triviaqa", "auto") == "extracted_contains"
    assert module._resolve_task_metric("nq", "auto") == "extracted_contains"
    assert module._resolve_task_metric("boolq", "exact") == "exact"


def test_task_gate_candidate_and_regression() -> None:
    module = _load_module()
    corpora = [
        _corpus(
            "GOOD",
            real_ppl=1.0,
            control_ppl=2.0,
            no_ppl=3.0,
            real_task=0.7,
            control_task=0.6,
            no_task=0.2,
        ),
        _corpus(
            "BAD",
            real_ppl=1.0,
            control_ppl=2.0,
            no_ppl=3.0,
            real_task=0.1,
            control_task=0.3,
            no_task=0.4,
        ),
    ]
    gate = module._task_gate(corpora, "extracted_exact", 0.05)
    assert gate["candidates"][0]["corpus"] == "GOOD"
    assert gate["severe_regressions"][0]["corpus"] == "BAD"
    assert gate["pass"] is False
