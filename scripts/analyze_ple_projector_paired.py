#!/usr/bin/env python3
"""Analyze paired PLE-projector experiments across seeds.

Reads experiment JSON files produced by ``scripts/train_ple_projector.py`` and
computes paired deltas on the same eval rows:

* fixed vs base
* projector vs fixed
* projector vs base

The main statistic is the per-token NLL difference; positive values mean lower
NLL / better calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _paired_ttest(deltas: np.ndarray) -> dict[str, float]:
    """One-sample paired t-test of the deltas against zero."""
    if len(deltas) < 2:
        return {"t": 0.0, "p": 1.0, "mean": float(np.mean(deltas)), "n": len(deltas)}
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1))
    se = std / math.sqrt(len(deltas))
    if se == 0:
        t = 0.0 if mean == 0 else math.copysign(float("inf"), mean)
    else:
        t = mean / se
    p = 2.0 * (1.0 - _normal_cdf(abs(t)))
    return {"t": t, "p": p, "mean": mean, "n": len(deltas), "se": se}


def _normal_cdf(x: float) -> float:
    # Abramowitz & Stegun approximation.
    if x < 0:
        return 1.0 - _normal_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327 * math.exp(-x * x / 2.0)
    p = d * t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return 1.0 - p


def _aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    fixed_delta = np.asarray(
        [r["base_nll"] - r["fixed_nll"] for r in rows], dtype=np.float64
    )
    proj_vs_fixed = np.asarray(
        [r["fixed_nll"] - r["proj_nll"] for r in rows], dtype=np.float64
    )
    proj_vs_base = np.asarray(
        [r["base_nll"] - r["proj_nll"] for r in rows], dtype=np.float64
    )
    fixed_hit = np.asarray([r["fixed_hit"] for r in rows], dtype=np.float64)
    proj_hit = np.asarray([r["proj_hit"] for r in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "fixed_vs_base_nll": _paired_ttest(fixed_delta),
        "proj_vs_fixed_nll": _paired_ttest(proj_vs_fixed),
        "proj_vs_base_nll": _paired_ttest(proj_vs_base),
        "fixed_hit": float(np.mean(fixed_hit)) if len(fixed_hit) else 0.0,
        "proj_hit": float(np.mean(proj_hit)) if len(proj_hit) else 0.0,
        "hit_diff_proj_vs_fixed": (
            float(np.mean(proj_hit - fixed_hit)) if len(fixed_hit) else 0.0
        ),
    }


def _task_stat(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for task in sorted({r["task"] for r in rows}):
        out[task] = _aggregate_rows([r for r in rows if r["task"] == task])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/ple-projector-paired-analysis.json")
    args = parser.parse_args()

    seeds = []
    for path in args.files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        paired = data.get("paired_eval")
        if not paired:
            print(f"[paired] {path}: no paired_eval, skip", flush=True)
            continue
        rows = paired.get("rows", [])
        entry = {
            "path": path,
            "seed": data.get("config", {}).get("seed"),
            "aggregate": _aggregate_rows(rows),
            "per_task": _task_stat(rows),
        }
        seeds.append(entry)
        print(
            f"[paired] {path}: n={len(rows)} "
            f"proj_vs_fixed_mean={entry['aggregate']['proj_vs_fixed_nll']['mean']:.4f} "
            f"p={entry['aggregate']['proj_vs_fixed_nll']['p']:.4f}",
            flush=True,
        )

    if not seeds:
        print("[paired] no seeds", flush=True)
        return 1

    proj_means = [s["aggregate"]["proj_vs_fixed_nll"]["mean"] for s in seeds]
    fixed_means = [s["aggregate"]["fixed_vs_base_nll"]["mean"] for s in seeds]
    base_means = [s["aggregate"]["proj_vs_base_nll"]["mean"] for s in seeds]
    summary = {
        "n_seeds": len(seeds),
        "fixed_vs_base_nll_mean": float(np.mean(fixed_means)),
        "proj_vs_fixed_nll_mean": float(np.mean(proj_means)),
        "proj_vs_base_nll_mean": float(np.mean(base_means)),
        "proj_vs_fixed_std": (
            float(np.std(proj_means, ddof=1)) if len(proj_means) > 1 else 0.0
        ),
        "proj_vs_fixed_positive_seeds": int(np.sum(np.asarray(proj_means) > 0)),
        "seeds": seeds,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_seeds": summary["n_seeds"],
                "fixed_vs_base_nll_mean": summary["fixed_vs_base_nll_mean"],
                "proj_vs_fixed_nll_mean": summary["proj_vs_fixed_nll_mean"],
                "proj_vs_base_nll_mean": summary["proj_vs_base_nll_mean"],
                "proj_vs_fixed_positive_seeds": summary["proj_vs_fixed_positive_seeds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
