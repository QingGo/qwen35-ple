#!/usr/bin/env python3
"""Summarize the standard held-out PLE graft evaluation.

Reads the Phase 0/1 JSON files produced for the standard
``data/qa-standard/eval.jsonl`` split and reports v2 answer-extraction metrics
for each evaluation format (raw completion prompt vs tokenizer chat template).

File naming convention:

* ``phase1-full.json`` is the no-reader arm (``mode=no-reader``);
* ``phase1-real-seed*.json`` is the trained-reader arm;
* ``phase1-control-seed*.json`` is the shuffled-row control arm.

Typical use::

    PYTHONPATH=src python scripts/summarize_standard_heldout.py \
      --raw-dir  outputs/qa-standard-eval-large-mixed50 \
      --chat-dir outputs/qa-standard-eval-chat-large \
      --output outputs/standard-heldout-summary.md

Pass ``--subset-per-task 100`` to restrict every arm to the first 100 items per
task, which makes a 1500-item run comparable with the 300-item chat ablation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TASKS = ("boolq", "triviaqa", "nq")
METRICS = {
    "boolq": "extracted_exact",
    "triviaqa": "extracted_contains",
    "nq": "extracted_contains",
}


def _load_answer_helpers():
    """Import the answer scorer with a helpful error for a missing PYTHONPATH."""
    try:
        from qwen35_ple.eval.answers import score_answer_v2
    except ImportError as exc:  # pragma: no cover - exercised by the CLI user
        raise SystemExit(
            "cannot import qwen35_ple.eval.answers; run with PYTHONPATH=src"
        ) from exc
    return score_answer_v2


def _answers_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the answer rows from either a results or summary Phase 0 JSON."""
    results = data.get("results")
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict):
                continue
            qa = entry.get("qa_exact")
            if isinstance(qa, dict) and isinstance(qa.get("answers"), list):
                return qa["answers"]
    summary = data.get("summary")
    if isinstance(summary, dict):
        for entry in summary.values():
            if not isinstance(entry, dict):
                continue
            for detail in entry.get("details", []) or []:
                if not isinstance(detail, dict):
                    continue
                qa = detail.get("qa_exact")
                if isinstance(qa, dict) and isinstance(qa.get("answers"), list):
                    return qa["answers"]
    return []


def _subset(rows: list[dict[str, Any]], per_task: int | None) -> list[dict[str, Any]]:
    if per_task is None:
        return rows
    counts: dict[str, int] = {}
    keep: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "unknown"))
        counts[task] = counts.get(task, 0) + 1
        if counts[task] <= per_task:
            keep.append(row)
    return keep


def score_rows(
    rows: Iterable[dict[str, Any]], subset_per_task: int | None = None
) -> dict[str, float]:
    """Score answer rows with the v2 extraction protocol.

    BoolQ uses ``extracted_exact`` and the open QA tasks use
    ``extracted_contains``.  The returned dict always contains ``mean`` plus
    every task that appeared in the rows.
    """
    score_answer_v2 = _load_answer_helpers()
    rows = _subset(list(rows), subset_per_task)
    grouped: dict[str, list[int]] = {}
    for row in rows:
        task = str(row.get("task", "unknown"))
        metric = METRICS.get(task)
        if metric is None:
            continue
        scores = score_answer_v2(
            str(row.get("generated", "")), str(row.get("answer", "")), task=task
        )
        grouped.setdefault(task, []).append(int(bool(scores[metric])))
    out: dict[str, float] = {}
    total = 0
    correct = 0
    for task in TASKS:
        values = grouped.get(task)
        if not values:
            continue
        out[task] = sum(values) / len(values)
        total += len(values)
        correct += sum(values)
    out["mean"] = (correct / total) if total else float("nan")
    return out


def _arm_from_filename(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if stem == "phase1-full":
        return ("no-reader", 0)
    if not stem.startswith("phase1-"):
        return None
    rest = stem[len("phase1-") :]
    if rest.startswith("real"):
        seed = 0
        if "-seed" in rest:
            try:
                seed = int(rest.rsplit("-seed", 1)[1])
            except ValueError:
                seed = 0
        return ("real", seed)
    if rest.startswith("control"):
        seed = 0
        if "-seed" in rest:
            try:
                seed = int(rest.rsplit("-seed", 1)[1])
            except ValueError:
                seed = 0
        return ("control", seed)
    if rest.startswith("no-reader"):
        seed = 0
        if "-seed" in rest:
            try:
                seed = int(rest.rsplit("-seed", 1)[1])
            except ValueError:
                seed = 0
        return ("no-reader", seed)
    return (rest, 0)


def collect_arm_metrics(
    directory: Path, subset_per_task: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Return ``mode -> [{seed, boolq, triviaqa, nq, mean}, ...]``."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("phase1*.json")):
        if any(part in path.name for part in ("backup", "partial")):
            continue
        arm = _arm_from_filename(path)
        if arm is None:
            continue
        mode, seed = arm
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = _answers_from_payload(data)
        if not rows:
            continue
        metrics = score_rows(rows, subset_per_task)
        metrics["seed"] = seed
        arms.setdefault(mode, []).append(metrics)
    for entries in arms.values():
        entries.sort(key=lambda item: item.get("seed", 0))
    return arms


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (float("nan"), float("nan"))
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return (mean, math.sqrt(var))


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _metrics_table(arms: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| Mode | Seed | BoolQ | TriviaQA | NQ | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    no_reader = arms.get("no-reader") or []
    for entry in no_reader:
        lines.append(
            f"| no-reader | {entry['seed']} | {_fmt(entry.get('boolq', float('nan')))} | "
            f"{_fmt(entry.get('triviaqa', float('nan')))} | "
            f"{_fmt(entry.get('nq', float('nan')))} | {_fmt(entry.get('mean', float('nan')))} |"
        )
    real = arms.get("real") or []
    for entry in real:
        lines.append(
            f"| real | {entry['seed']} | {_fmt(entry.get('boolq', float('nan')))} | "
            f"{_fmt(entry.get('triviaqa', float('nan')))} | "
            f"{_fmt(entry.get('nq', float('nan')))} | {_fmt(entry.get('mean', float('nan')))} |"
        )
    if real:
        mean, std = _mean_std([entry["mean"] for entry in real])
        task_means = {
            task: _mean_std([entry.get(task, float("nan")) for entry in real])[0]
            for task in TASKS
        }
        lines.append(
            f"| real mean | - | {_fmt(task_means['boolq'])} | "
            f"{_fmt(task_means['triviaqa'])} | {_fmt(task_means['nq'])} | "
            f"{_fmt(mean)} ± {_fmt(std)} |"
        )
        if no_reader:
            baseline = no_reader[0]
            baseline_mean = baseline.get("mean", float("nan"))
            deltas = {
                task: task_means[task] - baseline.get(task, float("nan"))
                for task in TASKS
            }
            lines.append(
                f"| real - no-reader | - | {_fmt(deltas['boolq'])} | "
                f"{_fmt(deltas['triviaqa'])} | {_fmt(deltas['nq'])} | "
                f"{_fmt(mean - baseline_mean)} |"
            )
    for mode in ("control",):
        for entry in arms.get(mode) or []:
            lines.append(
                f"| {mode} | {entry['seed']} | {_fmt(entry.get('boolq', float('nan')))} | "
                f"{_fmt(entry.get('triviaqa', float('nan')))} | "
                f"{_fmt(entry.get('nq', float('nan')))} | {_fmt(entry.get('mean', float('nan')))} |"
            )
    return lines


def _factorial_table(
    raw_arms: dict[str, list[dict[str, Any]]],
    chat_arms: dict[str, list[dict[str, Any]]],
) -> list[str]:
    def cell(arms: dict[str, list[dict[str, Any]]], mode: str) -> str:
        entries = arms.get(mode) or []
        if not entries:
            return "N/A"
        if mode == "no-reader":
            return _fmt(entries[0].get("mean", float("nan")))
        mean, std = _mean_std([entry["mean"] for entry in entries])
        return f"{_fmt(mean)} ± {_fmt(std)}"

    return [
        "| Train \\ Eval | Raw prompt | Chat template |",
        "|---|---:|---:|",
        f"| no reader | {cell(raw_arms, 'no-reader')} | {cell(chat_arms, 'no-reader')} |",
        f"| raw-trained reader | {cell(raw_arms, 'real')} | N/A (300-item ablation only) |",
        f"| chat-trained reader | N/A (not run) | {cell(chat_arms, 'real')} |",
    ]


def build_markdown(
    raw_dir: Path,
    chat_dir: Path | None,
    subset_per_task: int | None = None,
    title: str = "Standard held-out PLE graft evaluation",
) -> str:
    raw_arms = collect_arm_metrics(raw_dir, subset_per_task)
    lines = [
        f"# {title}",
        "",
        f"- raw arm directory: `{raw_dir}`",
        f"- subset per task: {subset_per_task if subset_per_task is not None else 'all'}",
        "",
        "## Raw completion prompt",
        "",
    ]
    lines += _metrics_table(raw_arms)
    if chat_dir is not None:
        chat_arms = collect_arm_metrics(chat_dir, subset_per_task)
        lines += ["", "## Chat template", ""]
        lines += _metrics_table(chat_arms)
        lines += [
            "",
            "## 2x2 train/eval format table (overall mean)",
            "",
        ]
        lines += _factorial_table(raw_arms, chat_arms)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--chat-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subset-per-task", type=int, default=None)
    parser.add_argument("--title", default="Standard held-out PLE graft evaluation")
    args = parser.parse_args(argv)

    markdown = build_markdown(
        args.raw_dir, args.chat_dir, args.subset_per_task, args.title
    )
    if args.output is None:
        sys.stdout.write(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
