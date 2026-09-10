#!/usr/bin/env python3
"""Summarize generation EM and gold NLL for a set of ``run_phase0`` arms.

``summarize_gold_nll.py`` covers gold-answer NLL only; this script adds the
generation side (exact match per task) and paired item-level comparisons, which
is what the adaptation 2x2 needs to decide whether a row's PLE effect is real.

Usage:
    python scripts/summarize_arms.py \
        --run lora-nople=outputs/round152/lora-nople.json \
        --run lora-real=outputs/round152/lora-real.json \
        --pair lora-real=lora-control \
        --output outputs/round152/summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TASKS = ("boolq", "triviaqa", "nq")


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data.get("results") or []
    if not results:
        raise SystemExit(f"{path}: no results")
    return {"path": str(path), "config": data.get("config") or {}, "result": results[0]}


def _key(item: dict[str, Any]) -> tuple[str, str]:
    question = item.get("question") or ""
    return (str(item.get("task") or ""), question.strip()[:200])


def _items(arm: dict[str, Any]) -> list[dict[str, Any]]:
    return (arm["result"].get("qa_exact") or {}).get("answers") or []


def _gold_items(arm: dict[str, Any]) -> list[dict[str, Any]]:
    return (arm["result"].get("qa_gold") or {}).get("answers") or []


def _em_table(arms: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Generation exact match (standard 1500)",
        "",
        "| Run | BoolQ | TriviaQA | NQ | Mean EM | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, arm in arms.items():
        metrics = (arm["result"].get("qa_exact") or {}).get("metrics") or {}
        lines.append(
            f"| {label} | {metrics.get('qa_boolq_em', float('nan')):.3f} | "
            f"{metrics.get('qa_triviaqa_em', float('nan')):.3f} | "
            f"{metrics.get('qa_nq_em', float('nan')):.3f} | "
            f"{metrics.get('qa_em_mean', float('nan')):.4f} | "
            f"{int(metrics.get('qa_n', 0))} |"
        )
    return lines


def _gold_table(arms: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Gold-answer NLL (lower is better)",
        "",
        "| Run | BoolQ | TriviaQA | NQ | Overall | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, arm in arms.items():
        metrics = (arm["result"].get("qa_gold") or {}).get("metrics") or {}
        if not metrics:
            lines.append(f"| {label} | - | - | - | - | 0 |")
            continue
        lines.append(
            f"| {label} | {metrics.get('qa_boolq_nll', float('nan')):.4f} | "
            f"{metrics.get('qa_triviaqa_nll', float('nan')):.4f} | "
            f"{metrics.get('qa_nq_nll', float('nan')):.4f} | "
            f"{metrics.get('qa_mean_nll', float('nan')):.4f} | "
            f"{int(metrics.get('qa_n', 0))} |"
        )
    return lines


def _paired(
    first: dict[str, Any], second: dict[str, Any], metric: str
) -> dict[str, Any]:
    """Paired item-level comparison; positive delta favours ``first``."""
    if metric == "em":
        left = {_key(i): i for i in _items(first)}
        right = {_key(i): i for i in _items(second)}
        lv = {k: float(bool(i.get("correct"))) for k, i in left.items()}
        rv = {k: float(bool(i.get("correct"))) for k, i in right.items()}
    else:
        left = {_key(i): i for i in _gold_items(first)}
        right = {_key(i): i for i in _gold_items(second)}
        lv = {k: float(i["gold_nll"]) for k, i in left.items() if "gold_nll" in i}
        rv = {k: float(i["gold_nll"]) for k, i in right.items() if "gold_nll" in i}
    keys = [k for k in lv if k in rv]
    if not keys:
        return {}
    # For NLL, lower is better, so the sign is flipped to keep "positive = first
    # is better" consistent across both metrics.
    sign = 1.0 if metric == "em" else -1.0
    diffs = [sign * (lv[k] - rv[k]) for k in keys]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    sem = math.sqrt(var / len(diffs))
    wins = sum(1 for d in diffs if d > 0)
    per_task: dict[str, Any] = {}
    for task in TASKS:
        tkeys = [k for k in keys if k[0] == task]
        if not tkeys:
            continue
        tvals = [sign * (lv[k] - rv[k]) for k in tkeys]
        per_task[task] = {
            "n": len(tvals),
            "mean_delta": sum(tvals) / len(tvals),
        }
    return {
        "n": len(keys),
        "mean_delta": mean,
        "sem": sem,
        "win_rate": wins / len(keys),
        "per_task": per_task,
    }


def build_report(
    arms: dict[str, dict[str, Any]], pairs: list[tuple[str, str]]
) -> tuple[str, dict[str, Any]]:
    lines = [
        "# Adaptation-row summary (generation + gold NLL)",
        "",
        "Positive paired deltas always mean the **first** run of the pair is better.",
        "",
    ]
    lines += _em_table(arms)
    lines += _gold_table(arms)
    detail: dict[str, Any] = {"arms": {}, "pairs": {}}
    for label, arm in arms.items():
        config = arm["config"]
        detail["arms"][label] = {
            "path": arm["path"],
            "mode": arm["result"].get("mode"),
            "seed": arm["result"].get("seed"),
            "ple_off": arm["result"].get("ple_off"),
            "gate_override": arm["result"].get("gate_override"),
            "lora": arm["result"].get("lora"),
            "backbone_dtype": config.get("backbone_dtype"),
            "train_final_loss": arm["result"].get("train_final_loss"),
            "gen_metrics": (arm["result"].get("qa_exact") or {}).get("metrics"),
            "gold_metrics": (arm["result"].get("qa_gold") or {}).get("metrics"),
        }
    if pairs:
        lines += [
            "",
            "## Paired item-level comparisons",
            "",
            "| Pair | Metric | n | Mean Δ | SEM | Win rate | BoolQ Δ | TriviaQA Δ | NQ Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for first, second in pairs:
            if first not in arms or second not in arms:
                continue
            for metric in ("em", "nll"):
                stats = _paired(arms[first], arms[second], metric)
                if not stats:
                    continue
                per_task = stats["per_task"]
                lines.append(
                    f"| {first} vs {second} | {metric} | {stats['n']} | "
                    f"{stats['mean_delta']:+.4f} | {stats['sem']:.4f} | "
                    f"{stats['win_rate']:.3f} | "
                    + " | ".join(
                        f"{per_task.get(t, {}).get('mean_delta', float('nan')):+.4f}"
                        for t in TASKS
                    )
                    + " |"
                )
                detail.setdefault("pairs", {})[f"{first} vs {second}"][metric] = stats
    return "\n".join(lines) + "\n", detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--pair", action="append", default=[], metavar="FIRST=SECOND")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--title", default="Adaptation-row summary")
    args = parser.parse_args()

    arms: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--run needs LABEL=PATH, got {spec!r}")
        arms[label] = _load(Path(path))
    pairs = []
    for spec in args.pair:
        first, _, second = spec.partition("=")
        if not second:
            raise SystemExit(f"--pair needs FIRST=SECOND, got {spec!r}")
        pairs.append((first, second))

    markdown, detail = build_report(arms, pairs)
    markdown = markdown.replace(
        "# Adaptation-row summary", f"# {args.title}", 1
    )
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(markdown)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
