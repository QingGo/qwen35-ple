#!/usr/bin/env python3
"""Estimate tighter PLE memory bounds from collected (H, E, Y) arrays.

This script computes the first two levels of the bound ladder described in
``docs/round-56-tighter-bounds.md``:

* B0: full linear conditional-information upper bound;
* B1(r): linear/PLS rank-r upper bound.

It does not require the backbone or EngramDB; it only needs saved NumPy arrays:

* ``--h``: hidden states [N, d_model]
* ``--e``: PLE features [N, d_mem]
* ``--y``: targets [N, d_y] or [N]

Usage::

    python scripts/estimate_ple_bounds.py \
        --h outputs/h.npy \
        --e outputs/e.npy \
        --y outputs/y.npy \
        --output outputs/ple-bounds.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_array(path: str, name: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {arr.shape}")
    return arr.astype(np.float64)


def _center(train: np.ndarray) -> np.ndarray:
    return train - train.mean(axis=0, keepdims=True)


def _ridge_coef(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = x.shape[1]
    xtx = x.T @ x + lam * np.eye(d)
    xty = x.T @ y
    return np.linalg.solve(xtx, xty)


def _delta_r2_by_rank(
    e_perp: np.ndarray, r: np.ndarray, y: np.ndarray, max_rank: int
) -> tuple[list[float], list[float]]:
    """Return (delta_r2_per_rank, cumulative_delta_r2_per_rank)."""
    # Cross-covariance between E_perp and residual R.
    m = e_perp.T @ r
    u, _, _vt = np.linalg.svd(m, full_matrices=False)
    y_ss = float(np.sum(y**2))
    if y_ss <= 0:
        return [], []

    per_rank: list[float] = []
    cum: list[float] = []
    current = 0.0
    k = min(max_rank, u.shape[1])
    for i in range(k):
        direction = u[:, i][:, None]  # [d_mem, 1]
        score = e_perp @ direction     # [N, 1]
        coef = np.linalg.lstsq(score, r, rcond=None)[0]
        pred = score @ coef
        explained = float(np.sum(pred**2)) / y_ss
        current += explained
        per_rank.append(explained)
        cum.append(current)
    return per_rank, cum


def _mi_approx(delta_r2: float) -> float:
    if delta_r2 <= 0:
        return 0.0
    if delta_r2 >= 1.0:
        return float("inf")
    return -0.5 * math.log(1.0 - delta_r2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", required=True)
    parser.add_argument("--e", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=10.0)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--output", default="outputs/ple-bounds.json")
    args = parser.parse_args()

    h = _load_array(args.h, "H")
    e = _load_array(args.e, "E")
    y = _load_array(args.y, "Y")
    n = min(len(h), len(e), len(y))
    h = h[:n]
    e = e[:n]
    y = y[:n]

    hc = _center(h)
    ec = _center(e)
    yc = _center(y)

    # E_perp = E - P_H E
    beta_e = _ridge_coef(hc, ec, args.lam)
    e_perp = ec - hc @ beta_e

    # R = Y - P_H Y
    beta_y = _ridge_coef(hc, yc, args.lam)
    r = yc - hc @ beta_y

    # Full linear bound.
    beta_r = _ridge_coef(e_perp, r, args.lam)
    r_hat_full = e_perp @ beta_r
    y_ss = float(np.sum(yc**2))
    delta_r2_full = float(np.sum(r_hat_full**2)) / y_ss if y_ss > 0 else 0.0

    per_rank, cum = _delta_r2_by_rank(e_perp, r, yc, args.max_rank)

    result = {
        "n_samples": int(n),
        "d_h": int(h.shape[1]),
        "d_e": int(e.shape[1]),
        "d_y": int(y.shape[1]),
        "lambda": args.lam,
        "delta_r2_full": delta_r2_full,
        "B0_linear_approx_nats": _mi_approx(delta_r2_full),
        "rank_curve": {
            "cumulative_delta_r2": cum,
            "per_rank_delta_r2": per_rank,
            "B1_linear_approx_nats": [_mi_approx(v) for v in cum],
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rank_curve"}, indent=2))
    print(f"[bounds] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
