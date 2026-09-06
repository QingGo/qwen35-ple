#!/usr/bin/env python3
"""Analyze per-item CAP-1/formal eval JSONs across seeds/adapters.

Reads outputs from ``scripts/run_cap1_formal_eval.py`` and produces:

* per-run summary means;
* pairwise paired deltas vs a chosen base run;
* normal-approximation paired p-values;
* bootstrap 95% CIs for mean deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _hit_rate(rows: list[dict]) -> float | None:
    vals = [1.0 if r.get("first_hit") else 0.0 for r in rows]
    return float(np.mean(vals)) if vals else None


def _paired_test(a: list[float], b: list[float], seed: int = 0) -> dict:
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": None, "std": None, "t": None, "p": None, "ci95": None}
    arr = np.asarray(diffs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    stderr = std / math.sqrt(n)
    t = mean / stderr if stderr > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    rng = random.Random(seed)
    boot = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(float(np.mean(arr[idx])))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t": t,
        "p": p,
        "ci95": [float(lo), float(hi)],
        "wins": int(sum(1 for d in diffs if d > 0)),
        "losses": int(sum(1 for d in diffs if d < 0)),
    }


def _compare_cap1(base_rows: list[dict], other_rows: list[dict], seed: int) -> dict:
    rows = min(len(base_rows), len(other_rows))
    a = [base_rows[i]["answer_logprob"] for i in range(rows)]
    b = [other_rows[i]["answer_logprob"] for i in range(rows)]
    return _paired_test(b, a, seed=seed)


def _compare_formal(
    base_formal: dict[str, list[dict]],
    other_formal: dict[str, list[dict]],
    seed: int,
) -> dict:
    out = {}
    for family in sorted(set(base_formal) | set(other_formal)):
        ba = base_formal.get(family, [])
        oa = other_formal.get(family, [])
        rows = min(len(ba), len(oa))
        if rows == 0:
            continue
        a = [ba[i]["answer_logprob"] for i in range(rows)]
        b = [oa[i]["answer_logprob"] for i in range(rows)]
        out[family] = _paired_test(b, a, seed=seed)
    return out


def _summary(data: dict) -> dict:
    cap = data.get("cap1_per_item", [])
    formal = data.get("formal_per_item", {})
    return {
        "cap1": {
            "n": len(cap),
            "mean_answer_logprob": _mean([r["answer_logprob"] for r in cap]),
            "first_token_hit": _hit_rate(cap),
        },
        "formal": {
            fam: {
                "n": len(rows),
                "mean_answer_logprob": _mean([r["answer_logprob"] for r in rows]),
                "first_token_hit": _hit_rate(rows),
            }
            for fam, rows in sorted(formal.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--base-label", default=None, help="label to use as baseline for paired deltas")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/seed-peritem-analysis.json")
    parser.add_argument("--report", default="outputs/seed-peritem-analysis.md")
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        parser.error("--files and --labels must have the same length")
    data = {label: _load(Path(p)) for label, p in zip(args.labels, args.files, strict=True)}
    summaries = {label: _summary(d) for label, d in data.items()}

    base_label = args.base_label or args.labels[0]
    comparisons = {}
    for label, d in data.items():
        if label == base_label:
            continue
        base = data[base_label]
        comparisons[label] = {
            "cap1": _compare_cap1(
                base.get("cap1_per_item", []), d.get("cap1_per_item", []), args.seed
            ),
            "formal": _compare_formal(
                base.get("formal_per_item", {}), d.get("formal_per_item", {}), args.seed
            ),
        }

    result = {
        "schema": "seed-peritem-analysis-v1",
        "base_label": base_label,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Multi-seed per-item analysis",
        "",
        f"- Base: {base_label}",
        "",
        "## CAP-1 heldout",
        "",
        "| Run | N | Mean logprob | First-hit |",
        "|---|---:|---:|---:|",
    ]
    for label, s in summaries.items():
        lines.append(
            f"| {label} | {s['cap1']['n']} | {s['cap1']['mean_answer_logprob']:.4f} "
            f"| {s['cap1']['first_token_hit']:.4f} |"
        )
    lines += ["", "### Paired deltas vs base", "", "| Run | Δ mean | 95% CI | p | wins/losses |", "|---|---:|---:|---:|---:|"]
    for label, c in comparisons.items():
        v = c["cap1"]
        ci = v["ci95"]
        lines.append(
            f"| {label} | {v['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{v['p']:.4f} | {v['wins']}/{v['losses']} |"
        )

    # Formal benchmark tables.
    fams = sorted({fam for s in summaries.values() for fam in s["formal"]})
    if fams:
        lines += ["", "## Formal benchmarks", ""]
        for fam in fams:
            lines += [f"### {fam}", "", "| Run | N | Mean logprob | First-hit |", "|---|---:|---:|---:|"]
            for label, s in summaries.items():
                row = s["formal"].get(fam)
                if not row:
                    continue
                lines.append(
                    f"| {label} | {row['n']} | {row['mean_answer_logprob']:.4f} "
                    f"| {row['first_token_hit']:.4f} |"
                )
            lines += ["", "Paired deltas vs base:", "", "| Run | Δ mean | 95% CI | p | wins/losses |", "|---|---:|---:|---:|---:|"]
            for label, c in comparisons.items():
                v = c["formal"].get(fam)
                if not v or v["mean"] is None:
                    continue
                ci = v["ci95"]
                lines.append(
                    f"| {label} | {v['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                    f"{v['p']:.4f} | {v['wins']}/{v['losses']} |"
                )
            lines.append("")

    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[seed-analysis] wrote {out} and {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
