#!/usr/bin/env python3
"""Compute oracle routing upper bounds from Phase 0/1/2 matrix JSON files.

The existing matrix files contain the *same* QA items under three arms:

* ``real``      : frozen backbone + frozen PLE table + trained reader
* ``control``   : frozen backbone + shuffled PLE rows + trained reader
* ``no-reader`` : frozen backbone, no PLE injection

For every item we know which arm answered correctly.  A perfect binary router
(oracle gate) can therefore choose the best arm per item.  This script reports:

* ``always_real``, ``always_control``, ``always_no_reader``: fixed policies;
* ``oracle_real_no``: max(real, no-reader) per item;
* ``oracle_real_control``: max(real, control) per item;
* ``oracle_all``: max(real, control, no-reader) per item;
* ``oracle_choice_*``: how often each arm was chosen by the oracle.

It also reports per-task numbers, per-seed mean/std, and the utility gap between
the oracle and the best fixed policy.  The oracle is an *optimistic* upper bound
(it sees the gold answer), so it is useful for deciding whether routing can
help at all, not for claiming a deployable gain.

Usage::

    PYTHONPATH=src python scripts/analyze_oracle_upper_bound.py \
      --files artifacts/phase2-diagnostic-layer8/phase1-*.json \
      --protocol v2 --output outputs/oracle-layer8.json \
      --markdown outputs/oracle-layer8.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


def _load_answer_helpers():
    """Import the dependency-free answer utilities without importing torch.

    ``qwen35_ple/__init__.py`` imports the live-store stack (and therefore
    torch).  The answer scorer only needs ``qwen35_ple/eval/answers.py``, so
    load that file directly when the package import fails (e.g. local macOS
    Python 3.9 with an incompatible torch build).
    """
    try:
        from qwen35_ple.eval.answers import score_answer, score_answer_v2

        return score_answer, score_answer_v2
    except Exception:
        import importlib.util

        answers_path = Path(__file__).resolve().parents[1] / "src" / "qwen35_ple" / "eval" / "answers.py"
        spec = importlib.util.spec_from_file_location("_qwen35_ple_answers", answers_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.score_answer, module.score_answer_v2


score_answer, score_answer_v2 = _load_answer_helpers()

MODES = ("real", "control", "no-reader")

# Phase 0 writes a no-reader run as ``phase1-full.json`` (mode ``full``),
# while the oracle analysis historically keyed on ``no-reader``.  Normalize
# the aliases here so the standard held-out full eval can be analyzed without
# renaming artifacts.
MODE_ALIASES = {
    "full": "no-reader",
    "none": "no-reader",
    "no_reader": "no-reader",
    "baseline": "no-reader",
}


def _normalize_mode(mode: Any) -> str:
    raw = str(mode)
    return MODE_ALIASES.get(raw, raw)
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


def _score_row(row: dict, protocol: str, metric: str) -> bool:
    gold = str(row.get("answer", ""))
    prediction = _row_prediction(row)
    if protocol == "v2":
        scores = score_answer_v2(prediction, gold, task=row.get("task"))
    else:
        scores = score_answer(prediction, gold)
    if metric == "auto":
        # Task-aware default used by the Phase 2 gates: BoolQ is scored with
        # extracted exact match, open QA with extracted contains.
        if str(row.get("task", "")).lower() == "boolq":
            return bool(scores["extracted_exact"])
        return bool(scores["extracted_contains"])
    if metric not in scores:
        raise KeyError(f"unknown metric {metric!r}; choices: {sorted(scores)}")
    return bool(scores[metric])


def _iter_mode_rows(data: dict) -> dict[tuple[str, int], list[dict]]:
    """Return ``(mode, seed) -> answer rows`` from a Phase 0/1/2 JSON file."""
    out: dict[tuple[str, int], list[dict]] = {}
    summary = data.get("summary")
    if isinstance(summary, dict):
        for mode, entry in summary.items():
            if not isinstance(entry, dict):
                continue
            for detail in entry.get("details", []) or []:
                if not isinstance(detail, dict):
                    continue
                qa = detail.get("qa_exact")
                if not isinstance(qa, dict):
                    continue
                answers = qa.get("answers")
                if not isinstance(answers, list):
                    continue
                seed = int(detail.get("seed", 0))
                out[(_normalize_mode(mode), seed)] = answers
        if out:
            return out

    results = data.get("results")
    if isinstance(results, list):
        for detail in results:
            if not isinstance(detail, dict):
                continue
            qa = detail.get("qa_exact")
            if not isinstance(qa, dict):
                continue
            answers = qa.get("answers")
            if not isinstance(answers, list):
                continue
            out[
                (
                    _normalize_mode(detail.get("mode", "unknown")),
                    int(detail.get("seed", 0)),
                )
            ] = answers
    return out


def _align_items(
    rows_by_mode: dict[str, list[dict]],
) -> tuple[list[dict], list[str]]:
    """Return aligned gold rows and a validation warning list."""
    warnings: list[str] = []
    base_mode = next(iter(rows_by_mode))
    base = rows_by_mode[base_mode]
    n = len(base)
    for mode, rows in rows_by_mode.items():
        if len(rows) != n:
            warnings.append(f"mode {mode!r} has {len(rows)} rows, expected {n}")
            continue
        for idx, (a, b) in enumerate(zip(base, rows)):
            if str(a.get("question")) != str(b.get("question")):
                warnings.append(f"mode {mode!r} item {idx} question mismatch")
                break
    return base, warnings


def _accuracy(rows: list[dict], protocol: str, metric: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if _score_row(row, protocol, metric)) / len(rows)


def _accuracy_oracle(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.get("correct")) / len(rows)


def _oracle_rows(
    rows_by_mode: dict[str, list[dict]],
    arms: tuple[str, ...],
    protocol: str,
    metric: str,
) -> tuple[list[dict], dict[str, int]]:
    """Return per-item oracle rows and arm-choice counts for *arms*."""
    n = len(next(iter(rows_by_mode.values())))
    chosen_counts = {arm: 0 for arm in arms}
    out: list[dict] = []
    for idx in range(n):
        scores = {
            arm: _score_row(rows_by_mode[arm][idx], protocol, metric)
            for arm in arms
        }
        # Prefer real > control > no-reader on ties.  This tie-break is
        # conservative for the scientific question: it never credits the
        # oracle for choosing a cheaper arm when a fixed real policy already
        # succeeds.
        chosen = next((arm for arm in MODES if scores.get(arm)), arms[0])
        if any(scores.values()):
            chosen_counts[chosen] += 1
        out.append(
            {
                "task": str(rows_by_mode[arms[0]][idx].get("task", "unknown")),
                "correct": bool(any(scores.values())),
                **{f"correct_{arm}": scores[arm] for arm in arms},
            }
        )
    return out, chosen_counts


def _unique_counts(
    rows_by_mode: dict[str, list[dict]],
    indices: list[int],
    protocol: str,
    metric: str,
) -> dict[str, int]:
    """Count where real/no-reader are uniquely correct or both wrong."""
    if "real" not in rows_by_mode or "no-reader" not in rows_by_mode:
        return {}
    counts = {
        "both_real_no": 0,
        "real_only": 0,
        "no_reader_only": 0,
        "control_only": 0,
        "both_wrong": 0,
    }
    for idx in indices:
        real = _score_row(rows_by_mode["real"][idx], protocol, metric)
        no = _score_row(rows_by_mode["no-reader"][idx], protocol, metric)
        control = (
            _score_row(rows_by_mode["control"][idx], protocol, metric)
            if "control" in rows_by_mode
            else False
        )
        if real and no:
            counts["both_real_no"] += 1
        elif real and not no:
            counts["real_only"] += 1
        elif no and not real:
            counts["no_reader_only"] += 1
        elif control and not real and not no:
            counts["control_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _summarize(
    rows_by_mode: dict[str, list[dict]], protocol: str, metric: str
) -> dict[str, Any]:
    base_rows, warnings = _align_items(rows_by_mode)
    tasks = sorted({str(row.get("task", "unknown")) for row in base_rows})
    armsets: dict[str, tuple[str, ...]] = {
        "oracle_real_no": ("real", "no-reader"),
        "oracle_real_control": ("real", "control"),
        "oracle_all": ("real", "control", "no-reader"),
    }

    def mode_slice(mode: str, indices: list[int]) -> list[dict]:
        return [rows_by_mode[mode][idx] for idx in indices]

    def fixed_metrics(indices: list[int]) -> dict[str, float]:
        return {
            "always_real": _accuracy(mode_slice("real", indices), protocol, metric)
            if "real" in rows_by_mode
            else float("nan"),
            "always_control": _accuracy(mode_slice("control", indices), protocol, metric)
            if "control" in rows_by_mode
            else float("nan"),
            "always_no_reader": _accuracy(
                mode_slice("no-reader", indices), protocol, metric
            )
            if "no-reader" in rows_by_mode
            else float("nan"),
        }

    def task_slice(rows: list[dict], task: str) -> list[dict]:
        return [row for row in rows if str(row.get("task", "unknown")) == task]

    all_indices = list(range(len(base_rows)))
    out: dict[str, Any] = {
        "n_items": len(base_rows),
        "warnings": warnings,
        "overall": fixed_metrics(all_indices),
        "unique_counts": _unique_counts(
            rows_by_mode, all_indices, protocol, metric
        ),
        "per_task": {},
    }
    for task in tasks:
        task_indices = [
            idx
            for idx, row in enumerate(base_rows)
            if str(row.get("task", "unknown")) == task
        ]
        out["per_task"][task] = fixed_metrics(task_indices)
        out["per_task"][task]["unique_counts"] = _unique_counts(
            rows_by_mode, task_indices, protocol, metric
        )

    for name, arms in armsets.items():
        if not all(arm in rows_by_mode for arm in arms):
            continue
        oracle_rows, counts = _oracle_rows(rows_by_mode, arms, protocol, metric)
        out[name] = {
            "accuracy": _accuracy_oracle(oracle_rows),
            "choice_counts": counts,
            "choice_rate": {
                arm: (count / len(oracle_rows) if oracle_rows else float("nan"))
                for arm, count in counts.items()
            },
        }
        for task in tasks:
            sliced = task_slice(oracle_rows, task)
            out[name].setdefault("per_task", {})[task] = _accuracy_oracle(sliced)
    return out


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(var)


def _aggregate_seeds(per_seed: list[dict]) -> dict[str, Any]:
    """Aggregate the scalar metrics across seeds."""
    keys = [
        "always_real",
        "always_control",
        "always_no_reader",
    ]
    oracle_keys = [
        "oracle_real_no",
        "oracle_real_control",
        "oracle_all",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = [entry["overall"].get(key, float("nan")) for entry in per_seed]
        mean, std = _mean_std(vals)
        out[key] = {"mean": mean, "std": std, "per_seed": vals}
    for key in oracle_keys:
        vals = [
            entry.get(key, {}).get("accuracy", float("nan")) for entry in per_seed
        ]
        mean, std = _mean_std(vals)
        out[key] = {"mean": mean, "std": std, "per_seed": vals}
    # Per-task aggregation.
    tasks = sorted(
        {
            task
            for entry in per_seed
            for task in entry.get("per_task", {})
        }
    )
    out["per_task"] = {}
    for task in tasks:
        out["per_task"][task] = {}
        for key in keys:
            vals = [
                entry["per_task"].get(task, {}).get(key, float("nan"))
                for entry in per_seed
            ]
            mean, std = _mean_std(vals)
            out["per_task"][task][key] = {
                "mean": mean,
                "std": std,
                "per_seed": vals,
            }
        for key in oracle_keys:
            vals = [
                entry.get(key, {})
                .get("per_task", {})
                .get(task, float("nan"))
                for entry in per_seed
            ]
            mean, std = _mean_std(vals)
            out["per_task"][task][key] = {
                "mean": mean,
                "std": std,
                "per_seed": vals,
            }
        count_keys = (
            "both_real_no",
            "real_only",
            "no_reader_only",
            "control_only",
            "both_wrong",
        )
        out["per_task"][task]["unique_counts"] = {
            key: sum(
                entry.get("per_task", {})
                .get(task, {})
                .get("unique_counts", {})
                .get(key, 0)
                for entry in per_seed
            )
            for key in count_keys
        }
    count_keys = (
        "both_real_no",
        "real_only",
        "no_reader_only",
        "control_only",
        "both_wrong",
    )
    out["unique_counts"] = {
        key: sum(entry.get("unique_counts", {}).get(key, 0) for entry in per_seed)
        for key in count_keys
    }
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def _markdown(report: dict[str, Any], protocol: str, metric: str) -> str:
    lines = [
        "# Oracle routing upper bound",
        "",
        f"- protocol: `{protocol}`",
        f"- metric: `{metric}`",
        f"- items per seed: {report.get('n_items', 'N/A')}",
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines += ["## Alignment warnings", ""]
        lines += [f"- {w}" for w in warnings]
        lines.append("")
    lines += [
        "## Aggregate across seeds",
        "",
        "| Policy | Mean | Std | Per-seed |",
        "|---|---:|---:|---|",
    ]
    for key in (
        "always_real",
        "always_control",
        "always_no_reader",
        "oracle_real_no",
        "oracle_real_control",
        "oracle_all",
    ):
        entry = report.get("aggregate", {}).get(key)
        if not entry:
            continue
        lines.append(
            f"| {key} | {_fmt(entry.get('mean'))} | {_fmt(entry.get('std'))} | "
            f"{entry.get('per_seed')} |"
        )
    lines += ["", "## Per-task aggregate", ""]
    lines += [
        "| Task | real | control | no-reader | oracle(real/no) | oracle(all) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, entry in sorted(report.get("aggregate", {}).get("per_task", {}).items()):
        def m(key: str, entry=entry) -> str:
            sub = entry.get(key, {})
            return _fmt(sub.get("mean")) if isinstance(sub, dict) else "N/A"

        lines.append(
            f"| {task} | {m('always_real')} | {m('always_control')} | "
            f"{m('always_no_reader')} | {m('oracle_real_no')} | {m('oracle_all')} |"
        )
    lines += [
        "",
        "## Unique-correct counts (summed across seeds)",
        "",
        "| Task | both real/no | real only | no-reader only | control only | all wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, entry in sorted(report.get("aggregate", {}).get("per_task", {}).items()):
        counts = entry.get("unique_counts") or {}
        lines.append(
            f"| {task} | {counts.get('both_real_no', 0)} | "
            f"{counts.get('real_only', 0)} | {counts.get('no_reader_only', 0)} | "
            f"{counts.get('control_only', 0)} | {counts.get('both_wrong', 0)} |"
        )
    lines += [
        "",
        "## Per-corpus",
        "",
        "| Corpus | real | control | no-reader | oracle(real/no) | oracle(all) | real only | no-reader only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, entry in sorted(report.get("per_corpus", {}).items()):
        agg = entry.get("aggregate", {})
        counts = agg.get("unique_counts") or {}
        def m2(key: str, agg=agg) -> str:
            sub = agg.get(key, {})
            return _fmt(sub.get("mean")) if isinstance(sub, dict) else "N/A"

        lines.append(
            f"| {corpus} | {m2('always_real')} | {m2('always_control')} | "
            f"{m2('always_no_reader')} | {m2('oracle_real_no')} | {m2('oracle_all')} | "
            f"{counts.get('real_only', 0)} | {counts.get('no_reader_only', 0)} |"
        )
    lines += [
        "",
        "> Oracle is an optimistic, label-seeing upper bound.  It answers whether",
        "> *any* binary router could improve on the fixed policies, not whether a",
        "> learned gate will achieve that gain.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="phase1-*.json files (globs are expanded by the shell)",
    )
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--metric",
        choices=(
            "auto",
            "exact",
            "contains",
            "extracted_exact",
            "extracted_contains",
        ),
        default="auto",
    )
    parser.add_argument("--output", default="outputs/oracle-upper-bound.json")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    files: list[Path] = []
    for pattern in args.files:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        files.extend(matches or [Path(pattern)])
    if not files:
        raise SystemExit("no input files")

    per_corpus: dict[str, Any] = {}
    all_seed_reports: list[dict] = []
    for path in files:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows_by_key = _iter_mode_rows(data)
        if not rows_by_key:
            raise SystemExit(f"{path}: no qa_exact rows found")
        seeds = sorted({seed for (_, seed) in rows_by_key})
        per_seed: list[dict] = []
        for seed in seeds:
            rows_by_mode = {
                mode: rows
                for (mode, s), rows in rows_by_key.items()
                if s == seed and mode in MODES
            }
            if not rows_by_mode:
                continue
            report = _summarize(rows_by_mode, args.protocol, args.metric)
            report["seed"] = seed
            per_seed.append(report)
            all_seed_reports.append(report)
        aggregate = _aggregate_seeds(per_seed) if per_seed else {}
        per_corpus[_corpus_name(path)] = {
            "path": str(path),
            "seeds": seeds,
            "per_seed": per_seed,
            "aggregate": aggregate,
        }

    overall = _aggregate_seeds(all_seed_reports) if all_seed_reports else {}
    report: dict[str, Any] = {
        "protocol": args.protocol,
        "metric": args.metric,
        "files": [str(path) for path in files],
        "per_corpus": per_corpus,
        "aggregate": overall,
        "n_items": all_seed_reports[0].get("n_items") if all_seed_reports else None,
        "warnings": sorted(
            {w for entry in all_seed_reports for w in entry.get("warnings", [])}
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown(report, args.protocol, args.metric), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
