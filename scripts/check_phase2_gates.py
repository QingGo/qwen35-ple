#!/usr/bin/env python3
"""Apply the pre-registered Phase 2 gates to real/control/no-reader results.

The gate is deliberately conservative:

* PPL gate: real < control and real < no-reader on enough corpora/seeds;
* task gate: at least one (corpus, task) where real > no-reader and
  real >= control, with no severe task-level regression elsewhere;
* no-reader is the zero-shot baseline; control is the shuffled-row reader.

Usage::

    python scripts/check_phase2_gates.py \
      --files artifacts/phase2-full-4090/phase1-*.json \
      --protocol v2 \
      --metric extracted_exact \
      --output outputs/gates-v2.json \
      --markdown outputs/gates-v2.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

from qwen35_ple.eval.answers import score_answer, score_answer_v2

MODES = ("real", "control", "no-reader")
TASKS = ("boolq", "triviaqa", "nq")


def _corpus_name(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("phase1-") if stem.startswith("phase1-") else stem


def _row_prediction(row: dict) -> str:
    for key in ("generated", "generated_text", "prediction", "pred", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _score_row(row: dict, protocol: str) -> dict[str, bool]:
    gold = str(row.get("answer", ""))
    prediction = _row_prediction(row)
    if protocol == "v2":
        return score_answer_v2(prediction, gold, task=row.get("task"))
    return score_answer(prediction, gold)


def _task_metrics(details: list[dict], protocol: str) -> dict[str, dict[str, float]]:
    """Return ``task -> {metric -> mean}`` over all QA answers in *details*."""
    buckets: dict[str, list[dict[str, bool]]] = {}
    for detail in details:
        qa = detail.get("qa_exact")
        if not isinstance(qa, dict):
            continue
        for row in qa.get("answers", []):
            if not isinstance(row, dict):
                continue
            task = str(row.get("task", "unknown"))
            buckets.setdefault(task, []).append(_score_row(row, protocol))
    out: dict[str, dict[str, float]] = {}
    for task, scores in buckets.items():
        out[task] = {
            metric: sum(int(score[metric]) for score in scores) / len(scores)
            for metric in (
                "exact",
                "contains",
                "extracted_exact",
                "extracted_contains",
            )
        }
    return out


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _corpus_metrics(path: Path, protocol: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    corpus: dict[str, Any] = {"corpus": _corpus_name(path), "modes": {}}
    for mode in MODES:
        entry = summary.get(mode, {})
        details = entry.get("details", [])
        if not isinstance(details, list):
            details = []
        ppls: list[float] = []
        for detail in details:
            loss = detail.get("val_loss")
            if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
                ppls.append(math.exp(float(loss)))
        corpus["modes"][mode] = {
            "n_seeds": len(details),
            "ppl_mean": _mean(ppls),
            "ppl_by_seed": ppls,
            "tasks": _task_metrics(details, protocol),
        }
    return corpus


def _task_value(corpus: dict[str, Any], mode: str, task: str, metric: str) -> float | None:
    tasks = corpus["modes"][mode]["tasks"]
    if task not in tasks:
        return None
    return tasks[task].get(metric)


def _ppl_gate(corpora: list[dict[str, Any]]) -> dict[str, Any]:
    corpora_real_lt_control = 0
    corpora_real_lt_no = 0
    seeds_real_lt_control = 0
    seeds_real_lt_no = 0
    seeds_total = 0
    for corpus in corpora:
        real = corpus["modes"]["real"]["ppl_mean"]
        control = corpus["modes"]["control"]["ppl_mean"]
        no_reader = corpus["modes"]["no-reader"]["ppl_mean"]
        if real is not None and control is not None and real < control:
            corpora_real_lt_control += 1
        if real is not None and no_reader is not None and real < no_reader:
            corpora_real_lt_no += 1
        real_seeds = corpus["modes"]["real"]["ppl_by_seed"]
        control_seeds = corpus["modes"]["control"]["ppl_by_seed"]
        no_seeds = corpus["modes"]["no-reader"]["ppl_by_seed"]
        for index, real_ppl in enumerate(real_seeds):
            seeds_total += 1
            if index < len(control_seeds) and real_ppl < control_seeds[index]:
                seeds_real_lt_control += 1
            if index < len(no_seeds) and real_ppl < no_seeds[index]:
                seeds_real_lt_no += 1
    corpora_total = len(corpora)
    return {
        "corpora_real_lt_control": corpora_real_lt_control,
        "corpora_real_lt_no_reader": corpora_real_lt_no,
        "corpora_total": corpora_total,
        "seeds_real_lt_control": seeds_real_lt_control,
        "seeds_real_lt_no_reader": seeds_real_lt_no,
        "seeds_total": seeds_total,
        "pass": bool(
            corpora_total > 0
            and corpora_real_lt_control >= max(1, math.ceil(corpora_total * 5 / 6))
            and corpora_real_lt_no == corpora_total
            and seeds_total > 0
            and seeds_real_lt_control >= math.ceil(seeds_total * 2 / 3)
            and seeds_real_lt_no == seeds_total
        ),
    }


def _resolve_task_metric(task: str, metric: str) -> str:
    """Resolve ``auto`` to the task-appropriate v2 metric.

    BoolQ is a yes/no task, so exact extracted yes/no is the right signal.
    TriviaQA / NQ answers are short spans but generations are often full
    sentences, so extracted contains is the right signal.
    """
    if metric != "auto":
        return metric
    return "extracted_exact" if task == "boolq" else "extracted_contains"


def _task_gate(
    corpora: list[dict[str, Any]],
    metric: str,
    regression_threshold: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    severe_regressions: list[dict[str, Any]] = []
    for corpus in corpora:
        for task in TASKS:
            task_metric = _resolve_task_metric(task, metric)
            real = _task_value(corpus, "real", task, task_metric)
            control = _task_value(corpus, "control", task, task_metric)
            no_reader = _task_value(corpus, "no-reader", task, task_metric)
            if real is None or control is None or no_reader is None:
                continue
            record = {
                "corpus": corpus["corpus"],
                "task": task,
                "metric": task_metric,
                "real": real,
                "control": control,
                "no_reader": no_reader,
            }
            if real > no_reader and real >= control:
                candidates.append(record)
            # A task where real is far below the best baseline is a severe
            # regression, even if the other baseline is also weak.
            if real < max(control, no_reader) - regression_threshold:
                severe_regressions.append(record)
    return {
        "metric": metric,
        "candidates": candidates,
        "severe_regressions": severe_regressions,
        "pass": bool(candidates) and not severe_regressions,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 gate report",
        "",
        f"- protocol: `{report['protocol']}`",
        f"- task metric: `{report['task_gate']['metric']}`",
        f"- overall pass: `{report['overall_pass']}`",
        "",
        "## PPL gate",
        "",
        (
            f"- real < control corpora: {report['ppl_gate']['corpora_real_lt_control']}"
            f" / {report['ppl_gate']['corpora_total']}"
        ),
        (
            f"- real < no-reader corpora: "
            f"{report['ppl_gate']['corpora_real_lt_no_reader']}"
            f" / {report['ppl_gate']['corpora_total']}"
        ),
        (
            f"- real < control seeds: {report['ppl_gate']['seeds_real_lt_control']}"
            f" / {report['ppl_gate']['seeds_total']}"
        ),
        (
            f"- real < no-reader seeds: {report['ppl_gate']['seeds_real_lt_no_reader']}"
            f" / {report['ppl_gate']['seeds_total']}"
        ),
        f"- pass: `{report['ppl_gate']['pass']}`",
        "",
        "## Task gate",
        "",
        "| Corpus | Task | metric | real | control | no-reader | candidate | severe |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    candidate_keys = {
        (item["corpus"], item["task"]) for item in report["task_gate"]["candidates"]
    }
    severe_keys = {
        (item["corpus"], item["task"])
        for item in report["task_gate"]["severe_regressions"]
    }
    for corpus in report["corpora"]:
        for task in TASKS:
            task_metric = _resolve_task_metric(task, report["task_gate"]["metric"])
            real = _task_value(corpus, "real", task, task_metric)
            control = _task_value(corpus, "control", task, task_metric)
            no_reader = _task_value(corpus, "no-reader", task, task_metric)
            if real is None or control is None or no_reader is None:
                continue
            key = (corpus["corpus"], task)
            lines.append(
                f"| {corpus['corpus']} | {task} | {task_metric} | {real:.3f} | "
                f"{control:.3f} | {no_reader:.3f} | {key in candidate_keys} | "
                f"{key in severe_keys} |"
            )
    lines += [
        "",
        f"- task gate pass: `{report['task_gate']['pass']}`",
        f"- overall pass: `{report['overall_pass']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--metric",
        choices=(
            "auto",
            "extracted_exact",
            "extracted_contains",
            "contains",
            "exact",
        ),
        default="auto",
        help=(
            "auto = extracted_exact for BoolQ and extracted_contains for open QA"
        ),
    )
    parser.add_argument("--regression-threshold", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.files:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    paths = sorted({path for path in paths if path.exists()})
    corpora = [_corpus_metrics(path, args.protocol) for path in paths]
    report = {
        "protocol": args.protocol,
        "files": [str(path) for path in paths],
        "corpora": corpora,
        "ppl_gate": _ppl_gate(corpora),
        "task_gate": _task_gate(corpora, args.metric, args.regression_threshold),
    }
    report["overall_pass"] = bool(
        report["ppl_gate"]["pass"] and report["task_gate"]["pass"]
    )

    print(
        json.dumps(
            {
                "protocol": args.protocol,
                "ppl_gate_pass": report["ppl_gate"]["pass"],
                "task_gate_pass": report["task_gate"]["pass"],
                "overall_pass": report["overall_pass"],
                "candidates": report["task_gate"]["candidates"],
                "severe_regressions": report["task_gate"]["severe_regressions"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[gates] wrote {output_path}")
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        print(f"[gates] wrote {markdown_path}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
