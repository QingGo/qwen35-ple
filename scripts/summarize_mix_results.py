#!/usr/bin/env python3
"""Summarize Phase 0 three-arm results across M1-M5 mixed corpora.

Usage:

    python scripts/summarize_mix_results.py \
      --files outputs/phase0-M1-seed0.json outputs/phase0-M2-seed0.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ("real", "control", "no-reader")
TASKS = ("triviaqa", "nq", "boolq")


def load_summary(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stem = path.stem
    if stem.startswith("phase0-"):
        stem = stem[len("phase0-"):]
    if stem.endswith("-seed0"):
        stem = stem[: -len("-seed0")]
    name = stem
    summary = data.get("summary", {})
    # If the file is a single-mode run, still expose per-mode metrics from results.
    if not summary and data.get("results"):
        for res in data["results"]:
            mode = res.get("mode")
            if mode:
                qa = res.get("qa_exact") or {}
                summary[mode] = {
                    "val_loss_mean": res.get("val_loss"),
                    "val_ppl_mean": res.get("val_ppl"),
                    "qa_em_mean": (qa.get("metrics") or {}).get("qa_em_mean"),
                }
    return name, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    rows: list[dict] = []
    for file in args.files:
        name, summary = load_summary(Path(file))
        row: dict = {"mix": name}
        for mode in MODES:
            entry = summary.get(mode, {})
            row[f"{mode}_em"] = entry.get("qa_em_mean")
            row[f"{mode}_val_loss"] = entry.get("val_loss_mean")
            row[f"{mode}_ppl"] = entry.get("val_ppl_mean")
        # Per-task EM from the first real result if present.
        per_task: dict[str, float] = {}
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        for res in data.get("results", []):
            if res.get("mode") == "real" and res.get("qa_exact"):
                metrics = res["qa_exact"].get("metrics", {})
                for task in TASKS:
                    key = f"qa_{task}_em"
                    if key in metrics:
                        per_task[task] = metrics[key]
                break
        for task in TASKS:
            row[f"real_{task}_em"] = per_task.get(task)
        rows.append(row)

    print("\n=== Mix summary ===")
    header = (
        "mix        real_em  ctrl_em  no_em  real_loss  ctrl_loss  no_loss  "
        "real_trivia  real_nq  real_boolq"
    )
    print(header)
    for row in rows:
        def fmt(v, width=8):
            if v is None:
                return " " * width
            return f"{v:.3f}".rjust(width)

        print(
            f"{row['mix']:<10}"
            f"{fmt(row.get('real_em'))}"
            f"{fmt(row.get('control_em'))}"
            f"{fmt(row.get('no_reader_em'))}"
            f"{fmt(row.get('real_val_loss'), 10)}"
            f"{fmt(row.get('control_val_loss'), 10)}"
            f"{fmt(row.get('no_reader_val_loss'), 10)}"
            f"{fmt(row.get('real_triviaqa_em'))}"
            f"{fmt(row.get('real_nq_em'))}"
            f"{fmt(row.get('real_boolq_em'))}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n[summary] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
