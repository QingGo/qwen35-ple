#!/usr/bin/env python3
"""Re-score generated QA answers with the improved Phase 1 evaluation protocol.

The original Phase 0 reports used strict exact-match on the raw generation.
This script reads a results JSON produced by ``run_phase0.py --qa-exact-match``
(or any file with ``qa_exact.answers``) and computes:

* exact:  raw normalized equality against the gold answer;
* contains: gold answer appears anywhere in the normalized generation;
* extracted_exact / extracted_contains: after extracting a concise answer
  candidate with ``qwen35_ple.eval.answers.extract_answer``.

It also writes a per-task summary and a human-readable markdown report so the
same run can be compared under both strict and lenient protocols.

Usage::

    python scripts/evaluate_generated_answers.py \
      --results outputs/phase0-M1-seed0.json \
      --output outputs/phase1-answer-report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen35_ple.eval.answers import score_answer


def _load_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--results must be a JSON object")
    return data


def _iter_answer_rows(results: dict):
    """Yield (row, mode) pairs from either a flat or Phase-0 summary JSON."""
    qa_exact = results.get("qa_exact")
    if isinstance(qa_exact, dict) and isinstance(qa_exact.get("answers"), list):
        for row in qa_exact["answers"]:
            yield row, results.get("mode")
        return

    summary = results.get("summary")
    if isinstance(summary, dict):
        for mode, entry in summary.items():
            details = entry.get("details") if isinstance(entry, dict) else None
            if not isinstance(details, list):
                continue
            for detail in details:
                detail_qa = detail.get("qa_exact") if isinstance(detail, dict) else None
                if isinstance(detail_qa, dict) and isinstance(detail_qa.get("answers"), list):
                    for row in detail_qa["answers"]:
                        yield row, mode
        return

    # Fallback: top-level list of answer rows.
    if isinstance(results.get("answers"), list):
        for row in results["answers"]:
            yield row, results.get("mode")
        return

    raise SystemExit(
        "could not find qa_exact.answers in results; "
        "run with --qa-exact-match or provide a compatible file"
    )


def _prediction(row: dict) -> str:
    for key in ("generated", "generated_text", "prediction", "pred", "text"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    return ""


def _aggregate(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    metric_names = ["exact", "contains", "extracted_exact", "extracted_contains"]
    total = len(rows)
    for metric in metric_names:
        counts[metric] = sum(1 for r in rows if r.get(metric))
    metrics = {
        name: (counts[name] / total if total else float("nan"))
        for name in metric_names
    }
    mean_len = (
        sum(r.get("generated_len", 0) for r in rows) / total if total else float("nan")
    )
    metrics["mean_generated_len"] = mean_len
    return metrics


def _per_task(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task", "qa")), []).append(row)
    out: dict[str, dict] = {}
    for task, task_rows in sorted(by_task.items()):
        agg = _aggregate(task_rows)
        out[task] = {
            "n": len(task_rows),
            **{k: round(v, 4) for k, v in agg.items()},
        }
    return out


def _per_mode(rows: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        mode = str(row.get("_mode") or row.get("mode") or "unknown")
        by_mode.setdefault(mode, []).append(row)
    out: dict[str, dict] = {}
    for mode, mode_rows in sorted(by_mode.items()):
        agg = _aggregate(mode_rows)
        out[mode] = {
            "n": len(mode_rows),
            **{k: round(v, 4) for k, v in agg.items()},
        }
    return out


def _report_markdown(
    results: dict, per_task: dict, per_mode: dict, metrics: dict | None = None
) -> str:
    lines = [
        "# Phase 1 Answer-Protocol Report",
        "",
        f"- model: {results.get('model', results.get('mode', 'unknown'))}",
        f"- mode: {results.get('mode', 'n/a')}",
        f"- seed: {results.get('seed', 'n/a')}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    overall = metrics if metrics is not None else results.get("metrics", {})
    for key in ["exact", "contains", "extracted_exact", "extracted_contains"]:
        if key in overall:
            lines.append(f"| {key} | {overall[key]:.4f} |")
    lines.append("")
    lines.append("## Per-task")
    lines.append("")
    lines.append("| Task | n | exact | contains | extracted_exact | extracted_contains |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task, entry in per_task.items():
        lines.append(
            f"| {task} | {entry['n']} | {entry['exact']:.4f} | "
            f"{entry['contains']:.4f} | {entry['extracted_exact']:.4f} | "
            f"{entry['extracted_contains']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-mode")
    lines.append("")
    lines.append("| Mode | n | exact | contains | extracted_exact | extracted_contains |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, entry in per_mode.items():
        lines.append(
            f"| {mode} | {entry['n']} | {entry['exact']:.4f} | "
            f"{entry['contains']:.4f} | {entry['extracted_exact']:.4f} | "
            f"{entry['extracted_contains']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="outputs/phase1-answer-report.json")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    results = _load_results(Path(args.results))
    scored: list[dict] = []
    for row, mode in _iter_answer_rows(results):
        gold = str(row.get("answer", ""))
        pred = _prediction(row)
        s = score_answer(pred, gold)
        scored.append(
            {
                **row,
                "_mode": mode,
                "generated_len": len(pred.split()),
                "exact": s["exact"],
                "contains": s["contains"],
                "extracted": s["extracted"],
                "extracted_exact": s["extracted_exact"],
                "extracted_contains": s["extracted_contains"],
            }
        )

    overall = _aggregate(scored)
    per_task = _per_task(scored)
    per_mode = _per_mode(scored)
    out = {
        "source": str(Path(args.results).resolve()),
        "n": len(scored),
        "metrics": {k: round(v, 6) for k, v in overall.items()},
        "per_task": per_task,
        "per_mode": per_mode,
        "rows": scored,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n": len(scored), "metrics": out["metrics"]}, indent=2))

    md_path = Path(args.markdown) if args.markdown else out_path.with_suffix(".md")
    md_path.write_text(
        _report_markdown(results, per_task, per_mode, out["metrics"]),
        encoding="utf-8",
    )
    print(f"[answer-eval] wrote {out_path}")
    print(f"[answer-eval] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
