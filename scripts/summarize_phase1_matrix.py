#!/usr/bin/env python3
"""Summarize Phase 2 / Phase 1 matrix outputs into a compact results table.

Each Phase 0 JSON produced by ``scripts/run_phase1_matrix.sh`` contains a
``summary`` with one entry per mode (real / control / no-reader) and per-seed
``details``.  This script aggregates:

* validation PPL;
* lenient contains EM (gold answer appears in the generated text);
* extracted exact EM and extracted contains EM from the improved protocol.

Use ``--protocol v2`` to score with the answer-marker / first-sentence
extractor in :mod:`qwen35_ple.eval.answers`.  The default ``v1`` keeps the
original last-sentence protocol for backward compatibility.

Usage::

    python scripts/summarize_phase1_matrix.py \
      --files outputs/phase1-PURE_WIKI.json outputs/phase1-PURE_FINEWEB.json ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from qwen35_ple.eval.answers import score_answer, score_answer_v2

MODES = ("real", "control", "no-reader")


def _corpus_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("phase1-"):
        return stem.removeprefix("phase1-")
    return stem


def _row_prediction(row: dict) -> str:
    for key in ("generated", "generated_text", "prediction", "pred", "text"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    return ""


def _score_answers(
    answers: list[dict], protocol: str = "v1"
) -> dict[str, float]:
    if not answers:
        return {
            "contains": float("nan"),
            "extracted_exact": float("nan"),
            "extracted_contains": float("nan"),
        }
    n = len(answers)
    contains = 0
    extracted_exact = 0
    extracted_contains = 0
    for row in answers:
        gold = str(row.get("answer", ""))
        pred = _row_prediction(row)
        if protocol == "v2":
            s = score_answer_v2(pred, gold, task=row.get("task"))
        else:
            s = score_answer(pred, gold)
        contains += int(s["contains"])
        extracted_exact += int(s["extracted_exact"])
        extracted_contains += int(s["extracted_contains"])
    return {
        "contains": contains / n,
        "extracted_exact": extracted_exact / n,
        "extracted_contains": extracted_contains / n,
    }


def _mode_metrics(
    details: list[dict], protocol: str = "v1"
) -> dict[str, float | None]:
    val_losses: list[float] = []
    answer_scores: list[dict[str, float]] = []
    for detail in details:
        val_loss = detail.get("val_loss")
        if isinstance(val_loss, (int, float)):
            val_losses.append(float(val_loss))
        qa = detail.get("qa_exact")
        if isinstance(qa, dict) and isinstance(qa.get("answers"), list):
            answer_scores.append(_score_answers(qa["answers"], protocol=protocol))
    out: dict[str, float | None] = {
        "val_ppl": None,
        "contains": None,
        "extracted_exact": None,
        "extracted_contains": None,
    }
    if val_losses:
        import math

        mean_loss = sum(val_losses) / len(val_losses)
        out["val_ppl"] = math.exp(mean_loss)
    if answer_scores:
        for key in ("contains", "extracted_exact", "extracted_contains"):
            vals = [s[key] for s in answer_scores if s[key] == s[key]]
            if vals:
                out[key] = sum(vals) / len(vals)
    return out


def _load_matrix(path: Path, protocol: str = "v1") -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    out: dict[str, dict] = {}
    for mode in MODES:
        entry = summary.get(mode)
        if not isinstance(entry, dict):
            continue
        details = entry.get("details", [])
        if not isinstance(details, list):
            details = [entry]
        out[mode] = _mode_metrics(details, protocol=protocol)
    return out


def _fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return " " * width
    if math.isnan(value):
        return " " * width
    return f"{value:.3f}".rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--protocol",
        choices=("v1", "v2"),
        default="v1",
        help="answer extraction protocol; v2 prefers explicit markers and first sentences",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for file in args.files:
        path = Path(file)
        matrix = _load_matrix(path, protocol=args.protocol)
        row: dict = {"corpus": _corpus_name(path)}
        for mode in MODES:
            m = matrix.get(mode, {})
            prefix = mode.replace("-", "_")
            row[f"{prefix}_ppl"] = m.get("val_ppl")
            row[f"{prefix}_contains"] = m.get("contains")
            row[f"{prefix}_extracted_exact"] = m.get("extracted_exact")
            row[f"{prefix}_extracted_contains"] = m.get("extracted_contains")
        rows.append(row)

    print(f"=== Phase 1/2 matrix summary (protocol={args.protocol}) ===")
    print(
        "corpus        real_ppl ctrl_ppl no_ppl real_cont ctrl_cont no_cont "
        "real_extreal ctrl_extreal no_extreal"
    )
    for row in rows:
        print(
            f"{row['corpus']:<12}"
            f"{_fmt(row.get('real_ppl'))}"
            f"{_fmt(row.get('control_ppl'))}"
            f"{_fmt(row.get('no_reader_ppl'))}"
            f"{_fmt(row.get('real_contains'))}"
            f"{_fmt(row.get('control_contains'))}"
            f"{_fmt(row.get('no_reader_contains'))}"
            f"{_fmt(row.get('real_extracted_exact'))}"
            f"{_fmt(row.get('control_extracted_exact'))}"
            f"{_fmt(row.get('no_reader_extracted_exact'))}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[summary] wrote {out}")

    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Phase 1/2 Matrix Summary (protocol={args.protocol})",
            "",
            "| Corpus | Real PPL | Ctrl PPL | No-reader PPL | Real contains | Ctrl contains | No contains | Real ext-exact | Ctrl ext-exact | No ext-exact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['corpus']} | "
                f"{_fmt(row.get('real_ppl'), 6).strip()} | "
                f"{_fmt(row.get('control_ppl'), 6).strip()} | "
                f"{_fmt(row.get('no_reader_ppl'), 6).strip()} | "
                f"{_fmt(row.get('real_contains'), 6).strip()} | "
                f"{_fmt(row.get('control_contains'), 6).strip()} | "
                f"{_fmt(row.get('no_reader_contains'), 6).strip()} | "
                f"{_fmt(row.get('real_extracted_exact'), 6).strip()} | "
                f"{_fmt(row.get('control_extracted_exact'), 6).strip()} | "
                f"{_fmt(row.get('no_reader_extracted_exact'), 6).strip()} |"
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[summary] wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
