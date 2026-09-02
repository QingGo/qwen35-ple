#!/usr/bin/env python3
"""Export Phase 0 result JSONs to article-friendly CSV/JSON tables.

This turns the raw ``outputs/phase0-*.json`` files into:

* ``train_loss.csv``: per-step training loss curves
* ``summary.csv``: per mix/mode/seed aggregate metrics
* ``per_question.csv``: per-question exact-match details
* ``summary.json``: machine-readable aggregate

Usage:

    python scripts/export_phase0_metrics.py \
      --files outputs/phase0-M1-seed0.json outputs/phase0-M2-seed0.json \
      --output-dir outputs/paper-metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _stem_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("phase0-"):
        stem = stem[len("phase0-"):]
    if stem.endswith("-seed0"):
        stem = stem[: -len("-seed0")]
    return stem


def _task_metrics(res: dict) -> dict[str, float | None]:
    qa = res.get("qa_exact") or {}
    metrics = qa.get("metrics") or {}
    return {
        "qa_em": metrics.get("qa_em_mean"),
        "trivia_em": metrics.get("qa_triviaqa_em"),
        "nq_em": metrics.get("qa_nq_em"),
        "boolq_em": metrics.get("qa_boolq_em"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/paper-metrics")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    summary_rows: list[dict] = []
    question_rows: list[dict] = []
    all_summary: dict = {"files": []}

    for path_str in args.files:
        path = Path(path_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        mix = _stem_name(path)
        all_summary["files"].append(
            {
                "file": str(path),
                "mix": mix,
                "config": data.get("config", {}),
            }
        )

        for res in data.get("results", []):
            mode = res.get("mode")
            seed = res.get("seed")
            if res.get("train_losses"):
                for step, loss in enumerate(res["train_losses"], start=1):
                    train_rows.append(
                        {
                            "mix": mix,
                            "mode": mode,
                            "seed": seed,
                            "step": step,
                            "train_loss": loss,
                        }
                    )
            for point in res.get("val_curve") or []:
                val_rows.append(
                    {
                        "mix": mix,
                        "mode": mode,
                        "seed": seed,
                        "step": point.get("step"),
                        "val_loss": point.get("val_loss"),
                    }
                )
            tm = _task_metrics(res)
            summary_rows.append(
                {
                    "mix": mix,
                    "mode": mode,
                    "seed": seed,
                    "val_loss": res.get("val_loss"),
                    "val_ppl": res.get("val_ppl"),
                    "qa_em": tm["qa_em"],
                    "trivia_em": tm["trivia_em"],
                    "nq_em": tm["nq_em"],
                    "boolq_em": tm["boolq_em"],
                }
            )
            qa = res.get("qa_exact") or {}
            for i, ans in enumerate(qa.get("answers") or []):
                question_rows.append(
                    {
                        "mix": mix,
                        "mode": mode,
                        "seed": seed,
                        "index": i,
                        "task": ans.get("task"),
                        "question": ans.get("question"),
                        "answer": ans.get("answer"),
                        "generated": ans.get("generated"),
                        "correct": ans.get("correct"),
                    }
                )

    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
        p = out_dir / name
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[export] {p} ({len(rows)} rows)")
        return p

    train_fields = ["mix", "mode", "seed", "step", "train_loss"]
    val_fields = ["mix", "mode", "seed", "step", "val_loss"]
    summary_fields = [
        "mix", "mode", "seed", "val_loss", "val_ppl",
        "qa_em", "trivia_em", "nq_em", "boolq_em",
    ]
    question_fields = [
        "mix", "mode", "seed", "index", "task",
        "question", "answer", "generated", "correct",
    ]

    write_csv("train_loss.csv", train_fields, train_rows)
    write_csv("val_curve.csv", val_fields, val_rows)
    write_csv("summary.csv", summary_fields, summary_rows)
    write_csv("per_question.csv", question_fields, question_rows)

    all_summary["summary"] = summary_rows
    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[export] {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
