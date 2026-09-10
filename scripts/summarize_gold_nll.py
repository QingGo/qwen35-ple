#!/usr/bin/env python3
"""Summarize teacher-forced gold-answer NLL runs.

Each ``--run LABEL=PATH`` points at a Phase 0 JSON containing either a new
``qa_gold`` block (``metrics.qa_<task>_nll``) or the legacy ``qa`` block
(``metrics.qa_<task>_loss``).  The report keeps two things separate:

* per-run absolute NLL per task (lower is better);
* paired deltas against a baseline, aligned by ``(task, question)``, so the
  seed-to-seed item variance cancels out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

TASKS = ("boolq", "triviaqa", "nq")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_items(data: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return ``(items, field)`` where each item has task/question/gold_nll."""
    results = data.get("results")
    entries: list[dict[str, Any]] = []
    if isinstance(results, list):
        entries = [entry for entry in results if isinstance(entry, dict)]
    if not entries:
        raise SystemExit("no results entries found")
    field = None
    answers: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry.get("qa_gold"), dict) and entry["qa_gold"].get("answers"):
            field = "qa_gold"
            answers.extend(entry["qa_gold"]["answers"])
        elif isinstance(entry.get("qa"), dict) and entry["qa"].get("answers"):
            field = "qa"
            answers.extend(entry["qa"]["answers"])
    if not answers:
        raise SystemExit("no qa_gold/qa answer rows found")
    items: list[dict[str, Any]] = []
    for row in answers:
        if field == "qa_gold":
            value = row.get("gold_nll")
        else:
            value = row.get("loss")
        if value is None:
            continue
        items.append(
            {
                "task": str(row.get("task", "unknown")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "nll": float(value),
            }
        )
    return items, str(field)


def _key(item: dict[str, Any]) -> tuple[str, str]:
    return (item["task"], item["question"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _run_table(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| Run | BoolQ NLL | TriviaQA NLL | NQ NLL | Overall NLL | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, items in runs.items():
        per_task = {task: [] for task in TASKS}
        for item in items:
            if item["task"] in per_task:
                per_task[item["task"]].append(item["nll"])
        cells = [_fmt(_mean(per_task[task])) for task in TASKS]
        lines.append(
            f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{_fmt(_mean([item['nll'] for item in items]))} | {len(items)} |"
        )
    return lines


def _paired_table(
    baseline_label: str,
    baseline: list[dict[str, Any]],
    runs: dict[str, list[dict[str, Any]]],
) -> list[str]:
    base_by_key = {_key(item): item["nll"] for item in baseline}
    lines = [
        "| Run | BoolQ ΔNLL | TriviaQA ΔNLL | NQ ΔNLL | Overall ΔNLL | SEM | win rate | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, items in runs.items():
        if label == baseline_label:
            continue
        deltas: list[float] = []
        per_task: dict[str, list[float]] = {task: [] for task in TASKS}
        wins = 0
        for item in items:
            base = base_by_key.get(_key(item))
            if base is None:
                continue
            delta = base - item["nll"]  # positive = run has lower (better) NLL
            deltas.append(delta)
            if item["task"] in per_task:
                per_task[item["task"]].append(delta)
            if delta > 0:
                wins += 1
        if not deltas:
            continue
        mean = _mean(deltas)
        var = _mean([(value - mean) ** 2 for value in deltas]) if len(deltas) > 1 else 0.0
        sem = math.sqrt(var / len(deltas)) if len(deltas) > 1 else 0.0
        lines.append(
            f"| {label} | {_fmt(_mean(per_task['boolq']))} | "
            f"{_fmt(_mean(per_task['triviaqa']))} | {_fmt(_mean(per_task['nq']))} | "
            f"{_fmt(mean)} | {_fmt(sem)} | {wins / len(deltas):.3f} | {len(deltas)} |"
        )
    return lines


def build_report(runs: dict[str, list[dict[str, Any]]], baseline: str, title: str) -> str:
    lines = [f"# {title}", "", "Positive paired ΔNLL means the run has a *lower* (better) gold-answer NLL than the baseline.", ""]
    lines += _run_table(runs)
    if baseline in runs:
        lines += ["", f"## Paired against `{baseline}`", ""]
        lines += _paired_table(baseline, runs[baseline], runs)
    return "\n".join(lines) + "\n"



def _pair_table(runs, pairs):
    by_label = {label: {_key(item): item["nll"] for item in items} for label, items in runs.items()}
    lines = [
        "| Pair (first vs second) | BoolQ D | TriviaQA D | NQ D | Overall D | SEM | win rate | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in pairs:
        if "=" not in spec:
            continue
        first, second = spec.split("=", 1)
        if first not in runs or second not in runs:
            continue
        deltas = []
        per_task = {task: [] for task in TASKS}
        wins = 0
        for item in runs[first]:
            other = by_label[second].get(_key(item))
            if other is None:
                continue
            delta = other - item["nll"]
            deltas.append(delta)
            if item["task"] in per_task:
                per_task[item["task"]].append(delta)
            if delta > 0:
                wins += 1
        if not deltas:
            continue
        mean = _mean(deltas)
        var = _mean([(value - mean) ** 2 for value in deltas]) if len(deltas) > 1 else 0.0
        sem = math.sqrt(var / len(deltas)) if len(deltas) > 1 else 0.0
        lines.append(
            f"| {first} vs {second} | {_fmt(_mean(per_task['boolq']))} | "
            f"{_fmt(_mean(per_task['triviaqa']))} | {_fmt(_mean(per_task['nq']))} | "
            f"{_fmt(mean)} | {_fmt(sem)} | {wins / len(deltas):.3f} | {len(deltas)} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pair", action="append", default=[], metavar="FIRST=SECOND")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--title", default="Gold-answer NLL (teacher-forced, format-insensitive)"
    )
    args = parser.parse_args(argv)

    runs: dict[str, list[dict[str, Any]]] = {}
    fields: dict[str, str] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        items, field = _extract_items(_load(Path(path)))
        runs[label] = items
        fields[label] = field
    if args.baseline not in runs:
        raise SystemExit(f"baseline {args.baseline!r} is not one of the runs")
    markdown = build_report(runs, args.baseline, args.title)
    if args.pair:
        markdown += "\n## Explicit pairs (positive = first label has lower NLL)\n\n"
        markdown += "\n".join(_pair_table(runs, args.pair)) + "\n"
    payload = {
        "baseline": args.baseline,
        "fields": fields,
        "runs": {
            label: {
                "n": len(items),
                "mean_nll": _mean([item["nll"] for item in items]),
                "per_task": {
                    task: _mean([item["nll"] for item in items if item["task"] == task])
                    for task in TASKS
                },
            }
            for label, items in runs.items()
        },
    }
    if args.output is None:
        sys.stdout.write(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        args.output.with_suffix(".json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
