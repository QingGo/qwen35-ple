#!/usr/bin/env python3
"""Build a compact morning report from the overnight queue outputs.

Reads the summary/gates/oracle/gate-stats JSONs produced by the queue and
writes a single markdown file with the numbers that matter for the pure-PLE
grafting decision.

Usage::

    PYTHONPATH=src python scripts/build_overnight_report.py \
      --root /root/autodl-tmp/qwen35-ple/outputs \
      --output /root/autodl-tmp/qwen35-ple/outputs/OVERNIGHT_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TASKS = ("boolq", "triviaqa", "nq")
MODES = ("real", "control", "no-reader")


def _load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _task_metric(entry: dict, task: str, metric: str = "auto") -> float | None:
    tasks = entry.get("tasks") or {}
    if task not in tasks:
        return None
    task_entry = tasks[task]
    if metric == "auto":
        metric = "extracted_exact" if task == "boolq" else "extracted_contains"
    value = task_entry.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _corpus_gates(root: Path, name: str) -> dict | None:
    data = _load(root / name / "gates-v2.json")
    if not data:
        return None
    corpora = data.get("corpora") or []
    return corpora[0] if corpora else None


def _corpus_summary(root: Path, name: str) -> dict | None:
    data = _load(root / name / "summary-v2.json")
    if isinstance(data, list) and data:
        return data[0]
    return None


def _task_table(root: Path, names: list[str]) -> list[str]:
    lines = [
        "| Config | Corpus | Mode | PPL | BoolQ ext-exact | TriviaQA ext-contains | NQ ext-contains |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name in names:
        corpus = _corpus_gates(root, name)
        summary = _corpus_summary(root, name)
        if not corpus:
            continue
        corpus_name = corpus.get("corpus", "?")
        for mode in MODES:
            entry = (corpus.get("modes") or {}).get(mode)
            if not entry:
                continue
            ppl = entry.get("ppl_mean")
            if ppl is None and summary:
                ppl = summary.get(f"{mode}_ppl")
            lines.append(
                f"| {name} | {corpus_name} | {mode} | {_fmt(ppl, 3)} | "
                f"{_fmt(_task_metric(entry, 'boolq'))} | "
                f"{_fmt(_task_metric(entry, 'triviaqa'))} | "
                f"{_fmt(_task_metric(entry, 'nq'))} |"
            )
    return lines


def _oracle_table(root: Path) -> list[str]:
    lines = [
        "| Oracle analysis | always real | always control | always no-reader | oracle(real/no) | oracle(all) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    oracle_dir = root / "oracle-analysis"
    if not oracle_dir.exists():
        return lines
    for path in sorted(oracle_dir.glob("*.json")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        agg = data.get("aggregate") or {}
        def m(key: str, agg=agg) -> str:
            entry = agg.get(key) or {}
            return _fmt(entry.get("mean") if isinstance(entry, dict) else entry)
        lines.append(
            f"| {path.stem} | {m('always_real')} | {m('always_control')} | "
            f"{m('always_no_reader')} | {m('oracle_real_no')} | {m('oracle_all')} |"
        )
    return lines


def _oracle_task_table(root: Path) -> list[str]:
    lines = [
        "| Oracle analysis | Task | real | control | no-reader | oracle(real/no) | oracle(all) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    oracle_dir = root / "oracle-analysis"
    if not oracle_dir.exists():
        return lines
    for path in sorted(oracle_dir.glob("*.json")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        per_task = (data.get("aggregate") or {}).get("per_task") or {}
        for task in TASKS:
            entry = per_task.get(task)
            if not isinstance(entry, dict):
                continue
            def m(key: str, entry=entry) -> str:
                sub = entry.get(key) or {}
                return _fmt(sub.get("mean") if isinstance(sub, dict) else sub)
            lines.append(
                f"| {path.stem} | {task} | {m('always_real')} | {m('always_control')} | "
                f"{m('always_no_reader')} | {m('oracle_real_no')} | {m('oracle_all')} |"
            )
    return lines


def _unique_table(root: Path) -> list[str]:
    lines = [
        "| Oracle analysis | Task | both real/no | real only | no-reader only | control only | all wrong |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    oracle_dir = root / "oracle-analysis"
    if not oracle_dir.exists():
        return lines
    for path in sorted(oracle_dir.glob("*.json")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        per_task = (data.get("aggregate") or {}).get("per_task") or {}
        for task in TASKS:
            counts = (per_task.get(task) or {}).get("unique_counts") or {}
            if not counts:
                continue
            lines.append(
                f"| {path.stem} | {task} | {counts.get('both_real_no', 0)} | "
                f"{counts.get('real_only', 0)} | {counts.get('no_reader_only', 0)} | "
                f"{counts.get('control_only', 0)} | {counts.get('both_wrong', 0)} |"
            )
    return lines


def _gate_table(root: Path) -> list[str]:
    lines = [
        "| Reader | Task | n | gate mean | gate max | open frac | tail-32 mean | body mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    gate_dir = root / "gate-stats"
    if not gate_dir.exists():
        return lines
    for path in sorted(gate_dir.glob("*.json")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        for task, entry in sorted((data.get("aggregate") or {}).items()):
            lines.append(
                f"| {path.stem} | {task} | {entry.get('n', '?')} | "
                f"{_fmt(entry.get('gate_mean'))} | {_fmt(entry.get('gate_max'))} | "
                f"{_fmt(entry.get('gate_open_frac'))} | "
                f"{_fmt(entry.get('gate_mean_tail32'))} | "
                f"{_fmt(entry.get('gate_mean_body'))} |"
            )
    return lines


def _context_table(root: Path) -> list[str]:
    lines = [
        "| Context arm | Mode | contains | extracted_exact | extracted_contains |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ("oracle-context/eval62", "oracle-context/150"):
        summary = _load(root / name / "summary-v2.json")
        if not isinstance(summary, list):
            continue
        for entry in summary:
            corpus = entry.get("corpus", "?")
            for mode in MODES:
                prefix = f"{mode}_"
                if f"{prefix}contains" not in entry:
                    continue
                lines.append(
                    f"| {name} ({corpus}) | {mode} | {_fmt(entry.get(prefix + 'contains'))} | "
                    f"{_fmt(entry.get(prefix + 'extracted_exact'))} | "
                    f"{_fmt(entry.get(prefix + 'extracted_contains'))} |"
                )
    return lines


def _task_counts(root: Path, name: str) -> dict[str, int]:
    """Return per-task answer counts for one config (first available mode)."""
    data = _load(root / name / "phase1-PURE_WIKI.json")
    if not isinstance(data, dict):
        return {}
    summary = data.get("summary") or {}
    for mode in MODES:
        entry = summary.get(mode) or {}
        details = entry.get("details") or []
        if not details:
            continue
        answers = ((details[0].get("qa_exact") or {}).get("answers")) or []
        counts: dict[str, int] = {}
        for row in answers:
            task = str(row.get("task", "unknown"))
            counts[task] = counts.get(task, 0) + 1
        return counts
    return {}


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def _ci_table(root: Path, names: list[str]) -> list[str]:
    lines = [
        "| Config | Task | real metric | 95% Wilson CI | n |",
        "|---|---|---:|---|---:|",
    ]
    for name in names:
        corpus = _corpus_gates(root, name)
        if not corpus:
            continue
        counts = _task_counts(root, name)
        for task in TASKS:
            entry = (corpus.get("modes") or {}).get("real")
            if not entry:
                continue
            value = _task_metric(entry, task)
            n = counts.get(task, 0)
            if value is None or n <= 0:
                continue
            lo, hi = _wilson_ci(value, n)
            lines.append(
                f"| {name} | {task} | {_fmt(value)} | "
                f"[{_fmt(lo)}, {_fmt(hi)}] | {n} |"
            )
    return lines


def _gate_ablation_notes(root: Path) -> list[str]:
    notes: list[str] = []
    rows: list[tuple[str, float | None, float | None]] = []
    for path in sorted(root.glob("gate-ablation-*")):
        corpus = _corpus_gates(root, path.name)
        if not corpus:
            continue
        real = (corpus.get("modes") or {}).get("real") or {}
        rows.append(
            (
                path.name,
                _task_metric(real, "boolq"),
                _task_metric(real, "triviaqa"),
            )
        )
    if not rows:
        return notes
    notes.append("- Gate ablation (BoolQ / TriviaQA real accuracy):")
    for name, boolq, trivia in rows:
        notes.append(f"  - {name}: BoolQ={_fmt(boolq)}, TriviaQA={_fmt(trivia)}")
    # If forcing the gate closed recovers BoolQ, the regression is gate-driven.
    closed = [r for r in rows if r[0].endswith("-0.0")]
    open_ = [r for r in rows if r[0].endswith("-1.0")]
    if closed and open_:
        for (cname, cb, ct), (oname, ob, ot) in zip(closed, open_):
            if cb is not None and ob is not None and cb > ob + 0.05:
                notes.append(
                    f"  - forcing gate closed improves BoolQ "
                    f"({cname} {_fmt(cb)} vs {oname} {_fmt(ob)}): "
                    "the BoolQ regression is gate-driven."
                )
            if ct is not None and ot is not None and ot > ct + 0.05:
                notes.append(
                    f"  - forcing gate open improves TriviaQA "
                    f"({oname} {_fmt(ot)} vs {cname} {_fmt(ct)}): "
                    "the TriviaQA gain is PLE-content-driven."
                )
    return notes


def _decision_notes(root: Path) -> list[str]:
    notes: list[str] = []
    layer2 = _load(root / "oracle-analysis" / "layer2-corpus-only.json")
    if isinstance(layer2, dict):
        agg = layer2.get("aggregate") or {}
        def mean(key: str) -> float | None:
            entry = agg.get(key) or {}
            return entry.get("mean") if isinstance(entry, dict) else entry
        real = mean("always_real")
        no = mean("always_no_reader")
        oracle_no = mean("oracle_real_no")
        oracle_all = mean("oracle_all")
        if real is not None and no is not None and oracle_no is not None:
            if oracle_no > no + 0.02:
                notes.append(
                    f"- layer2 oracle(real/no)={_fmt(oracle_no)} > no-reader={_fmt(no)}: "
                    "routing headroom exists, so a learned gate is scientifically justified."
                )
            else:
                notes.append(
                    f"- layer2 oracle(real/no)={_fmt(oracle_no)} ~= no-reader={_fmt(no)}: "
                    "routing cannot recover much; focus on reader/PLE content."
                )
            if real < no:
                notes.append(
                    f"- layer2 always-real={_fmt(real)} < no-reader={_fmt(no)}: "
                    "the bottleneck is harmful always-on injection, not missing PLE content."
                )
            if oracle_all is not None and oracle_all > max(real or 0, no or 0) + 0.02:
                notes.append(
                    f"- oracle(all)={_fmt(oracle_all)}: shuffled-control rows also carry "
                    "task signal, so part of the gain may be reader regularization/format."
                )
        per_task = (layer2.get("aggregate") or {}).get("per_task") or {}
        boolq = ((per_task.get("boolq") or {}).get("unique_counts")) or {}
        trivia = ((per_task.get("triviaqa") or {}).get("unique_counts")) or {}
        if boolq:
            notes.append(
                f"- BoolQ unique-correct: real-only={boolq.get('real_only', 0)}, "
                f"no-reader-only={boolq.get('no_reader_only', 0)}. "
                "If no-reader-only dominates, the gate should close on BoolQ."
            )
        if trivia:
            notes.append(
                f"- TriviaQA unique-correct: real-only={trivia.get('real_only', 0)}, "
                f"no-reader-only={trivia.get('no_reader_only', 0)}. "
                "If real-only dominates, the gate should open on TriviaQA."
            )
    sft = _load(root / "oracle-analysis" / "sft-mixed50.json")
    if isinstance(sft, dict):
        agg = sft.get("aggregate") or {}
        def mean2(key: str) -> float | None:
            entry = agg.get(key) or {}
            return entry.get("mean") if isinstance(entry, dict) else entry
        notes.append(
            f"- sft-mixed50: real={_fmt(mean2('always_real'))}, "
            f"no-reader={_fmt(mean2('always_no_reader'))}, "
            f"oracle(all)={_fmt(mean2('oracle_all'))}."
        )
    gate = _load(root / "gate-stats" / "layer2-real-seed0.json")
    if isinstance(gate, dict):
        agg = gate.get("aggregate") or {}
        b = agg.get("boolq") or {}
        t = agg.get("triviaqa") or {}
        if b and t:
            notes.append(
                f"- gate stats (layer2 real): BoolQ mean={_fmt(b.get('gate_mean'))}, "
                f"TriviaQA mean={_fmt(t.get('gate_mean'))}; "
                "a higher BoolQ gate would explain the format regression."
            )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/qwen35-ple/outputs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "OVERNIGHT_REPORT.md"

    configs = [
        "phase2-diagnostic-layer2",
        "baseline-layer2-eval62",
        "sft-only",
        "sft-mixed75",
        "sft-mixed50",
        "sft-mixed25",
        "sft-mixed50-gate01",
        "sft-mixed50-purecode",
    ]
    configs += [
        path.name
        for path in sorted(root.glob("gate-ablation-*"))
        if (path / "gates-v2.json").exists()
    ]
    lines: list[str] = [
        "# Overnight pure-PLE grafting report",
        "",
        "This file is generated automatically from the queue outputs.  The key",
        "question is whether a *fixed* PLE policy is enough, or whether a learned",
        "gate is required to recover the oracle upper bound.",
        "",
        "## 1. Task metrics (held-out 62-item eval set where applicable)",
        "",
    ]
    lines += _task_table(root, configs)
    lines += [
        "",
        "## 2. Oracle routing upper bounds",
        "",
    ]
    lines += _oracle_table(root)
    lines += ["", "### Per-task oracle", ""]
    lines += _oracle_task_table(root)
    lines += [
        "",
        "### Unique-correct counts",
        "",
    ]
    lines += _unique_table(root)
    lines += [
        "",
        "## 3. Oracle context (gold answer in prompt)",
        "",
    ]
    lines += _context_table(root)
    lines += [
        "",
        "## 4. Gate statistics",
        "",
    ]
    lines += _gate_table(root)
    lines += [
        "",
        "## 5. Uncertainty on the held-out eval (Wilson 95% CI)",
        "",
    ]
    lines += _ci_table(
        root,
        [
            "phase2-diagnostic-layer2",
            "baseline-layer2-eval62",
            "sft-only",
            "sft-mixed50",
            "sft-mixed50-gate01",
            "sft-mixed50-purecode",
        ],
    )
    lines += [
        "",
        "> On the 62-item held-out eval, BoolQ has n=22 and TriviaQA/NQ n=20.",
        "> Treat differences smaller than the CI width as directional, not",
        "> statistically established.",
        "",
        "## 6. Decision notes",
        "",
    ]
    notes = _decision_notes(root)
    notes += _gate_ablation_notes(root)
    lines += notes or ["- (not enough outputs yet)"]
    lines += [
        "",
        "## 7. How to read this",
        "",
        "| Observation | Interpretation |",
        "|---|---|",
        "| oracle(real/no) ~= no-reader | PLE content adds little; routing cannot help. |",
        "| oracle(real/no) > no-reader, always-real < no-reader | routing is the bottleneck; train the existing gate. |",
        "| oracle(real/no) ~= always-real | content/reader is the bottleneck; improve reader/PLE. |",
        "| SFT config real > layer2 real on BoolQ, TriviaQA stays high | answer-only SFT fixed the format/utility mismatch. |",
        "| gate-reg config open-frac drops on BoolQ without losing TriviaQA | gate regularization is the right lever. |",
        "",
        "> Oracle numbers are label-seeing optimistic upper bounds; use the held-out",
        "> 62-item eval set for any deployable claim.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
