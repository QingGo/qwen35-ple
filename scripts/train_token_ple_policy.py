#!/usr/bin/env python3
"""Train a small token-level PLE fusion policy from per-token observations.

The observations are produced by ``scripts/run_ple_baseline_ablation.py`` when
``each result carries ``per_item`` rows.  For each token we know whether the
calibrated PLE fusion improved the teacher-forced log-probability, together
with n-gram/token features:

* matched_order
* base_entropy
* memory_entropy
* density_ratio
* base_top1_prob
* memory_top1_prob
* memory_top1_agree_base

A logistic regression is trained to predict ``PLE helps`` from those features.
This is a simple data-driven replacement for hand-written query-level rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = [
    "matched_order",
    "base_entropy",
    "memory_entropy",
    "density_ratio",
    "base_top1_prob",
    "memory_top1_prob",
    "memory_top1_agree_base",
]


def _load_samples(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for res in data.get("results", {}).values():
            for row in res.get("per_item", []):
                if row.get("ple_logprob") is None or row.get("base_logprob") is None:
                    continue
                if row.get("density_ratio") is None:
                    continue
                xs.append([float(row[f]) for f in FEATURES])
                ys.append(1 if row["ple_logprob"] > row["base_logprob"] else 0)
    if not xs:
        raise ValueError("no usable per-token samples found")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    pairs = 0
    for p in pos:
        pairs += int(np.sum(p > neg))
    return pairs / (len(pos) * len(neg))


def train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    steps: int = 500,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
    n, d = x.shape
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    xn = (x - mean) / std
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(steps):
        z = xn @ w + b
        p = _sigmoid(z)
        grad_w = xn.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    pred = _sigmoid(xn @ w + b)
    acc = float(np.mean((pred >= 0.5).astype(int) == y))
    auroc = _auc(pred, y)
    return w, b, mean, std, acc, auroc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/token-ple-policy.json")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    x, y = _load_samples([Path(p) for p in args.files])
    w, b, mean, std, acc, auroc = train(
        x, y, lr=args.lr, steps=args.steps, l2=args.l2, seed=args.seed
    )
    result = {
        "schema": "token-ple-policy-v1",
        "feature_names": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "metrics": {
            "n": len(y),
            "positive_rate": float(y.mean()),
            "accuracy": acc,
            "auc": auroc,
        },
        "config": vars(args),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"[token-policy] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
