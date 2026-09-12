#!/usr/bin/env python3
"""Round 164: characterise the injected vector from the saved per-item vectors.

Reads ``outputs/round164/vectors/*.npy`` (written by
``scripts/round164_window_dominance.py``) and reports both halves of round 164:

* **Pre-registered** -- the within-group mean pairwise cosines and the
  permutation test that ``round164_window_dominance.py`` already ran.  Recomputed
  here so the frozen verdict can be checked against the offline copy.
* **Exploratory** -- what the vectors look like once the pre-registered test has
  failed: effective dimensionality (participation ratio) of ``c`` and ``h``, a
  random-linear-map baseline for that ratio, the cross-template angle, and how
  much the injected vector depends on the rows at all (``wiki`` vs ``wiki-shuf``,
  the same checkpoint with permuted rows).

The exploratory half is labelled as such in the write-up
(``docs/round-164-window-composition-results.md`` section 3.2): it was chosen
after seeing the result and must not be reported as pre-registered.

Usage::

    python scripts/analyze_round164_vectors.py --dir outputs/round164
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_PERMUTATIONS = 1000


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mean_pairwise_cosine(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Mean cosine over pairs of rows.

    With ``b`` given, the mean over all cross pairs.  With ``b=None``, the mean
    over pairs *within* ``a``, **excluding the diagonal** -- the diagonal is
    cosine 1 by construction and including it biases the mean up by
    ``(1 - S) / n``, which is $7e-5$ for a 200-row group near $0.986$ but only
    $3e-6$ for one near $0.9994$.  The online statistic excludes it, so this
    must too, or the offline recomputation silently disagrees with the frozen
    numbers in exactly the regime the write-up cares about.
    """
    ua = unit(np.atleast_2d(a))
    if b is None:
        gram = ua @ ua.T
        iu = np.triu_indices(gram.shape[0], k=1)
        return float(np.mean(gram[iu]))
    return float(np.mean(ua @ unit(np.atleast_2d(b)).T))


def participation_ratio(x: np.ndarray) -> tuple[float, float]:
    """(PR, top eigenvalue share) of the covariance of the rows of ``x``.

    PR = (sum lambda)^2 / sum lambda^2.  A rank-1 cloud gives 1.0; an isotropic
    cloud in d dimensions gives d.
    """
    centred = x - x.mean(0)
    w = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    w = w[w > 0]
    return float(w.sum() ** 2 / (w**2).sum()), float(w.max() / w.sum())


def permutation_p(mat: np.ndarray, is_boolq: np.ndarray, observed: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_PERMUTATIONS):
        p = rng.permutation(is_boolq)
        delta = mean_pairwise_cosine(mat[p]) - mean_pairwise_cosine(mat[~p])
        if delta >= observed:
            hits += 1
    return (hits + 1) / (N_PERMUTATIONS + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="outputs/round164")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    root = Path(args.dir)
    items = [json.loads(line) for line in (root / "items-eval-600b.jsonl").read_text().splitlines() if line.strip()]
    tasks = np.array([it["task"] for it in items])
    is_boolq = tasks == "boolq"
    frozen = json.loads((root / "window-dominance.json").read_text())

    report: dict = {"pre_registered": {}, "exploratory": {}}

    print("=== pre-registered (recomputed from the saved vectors) ===")
    for name in ("wiki", "wiki-shuf"):
        C = np.load(root / "vectors" / f"{name}-c.npy")
        H = np.load(root / "vectors" / f"{name}-h.npy")
        s_c_b, s_c_s = mean_pairwise_cosine(C[is_boolq]), mean_pairwise_cosine(C[~is_boolq])
        s_h_b, s_h_s = mean_pairwise_cosine(H[is_boolq]), mean_pairwise_cosine(H[~is_boolq])
        ds_c, ds_h = s_c_b - s_c_s, s_h_b - s_h_s
        p_c = permutation_p(C, is_boolq, ds_c, args.seed)
        rec = {
            "S_c_boolq": s_c_b, "S_c_short": s_c_s, "dS_c": ds_c, "p_c": p_c,
            "S_h_boolq": s_h_b, "S_h_short": s_h_s, "dS_h": ds_h,
            "verdict": frozen["per_reader"][name]["verdict"],
        }
        report["pre_registered"][name] = rec
        print(f"  {name:10s} dS_c={ds_c:+.4f} p_c={p_c:.4f}  dS_h={ds_h:+.4f}  -> {rec['verdict']}")

    print()
    print("=== exploratory (post hoc; see results doc 3.2) ===")
    W = np.load(root / "vectors" / "wiki-c.npy")
    S = np.load(root / "vectors" / "wiki-shuf-c.npy")
    H = np.load(root / "vectors" / "wiki-h.npy")

    pr_h, top_h = participation_ratio(H)
    pr_c, top_c = participation_ratio(W)
    rng = np.random.default_rng(args.seed)
    random_pr = [participation_ratio(H @ (rng.normal(size=(H.shape[1], H.shape[1])) / np.sqrt(H.shape[1])))[0] for _ in range(3)]

    cos_ws = np.sum(unit(W) * unit(S), axis=1)
    report["exploratory"] = {
        "PR_c": pr_c, "top_eig_share_c": top_c,
        "PR_h": pr_h, "top_eig_share_h": top_h,
        "PR_h_after_random_linear_map": random_pr,
        "variance_suppression_factor": (1 - top_c) / (1 - top_h),
        "cos_real_vs_shuffled_rows": {
            "mean": float(cos_ws.mean()), "median": float(np.median(cos_ws)), "min": float(cos_ws.min()),
        },
        "norm_ratio_shuf_over_real": float(np.mean(np.linalg.norm(S, axis=1) / np.linalg.norm(W, axis=1))),
        "cross_template_cosine_boolq_vs_short": mean_pairwise_cosine(W[is_boolq], W[~is_boolq]),
        "within_task_cosine_c": {t: mean_pairwise_cosine(W[tasks == t]) for t in sorted(set(tasks))},
    }
    e = report["exploratory"]
    print(f"  PR(c)={e['PR_c']:.3f} (top share {e['top_eig_share_c']:.4f})   PR(h)={e['PR_h']:.3f} (top share {e['top_eig_share_h']:.4f})")
    print(f"  PR(h @ random linear map) = {['%.3f' % v for v in random_pr]}  <- a generic map does not collapse h")
    print(f"  non-dominant variance kept: {e['variance_suppression_factor']:.2e} of h's "
          f"(i.e. suppressed {1/e['variance_suppression_factor']:.0f}x)")
    print(f"  cos(c_real, c_shuf) = {e['cos_real_vs_shuffled_rows']['mean']:.4f}  (rows barely matter)")
    print(f"  cross-template cos(boolq | short) = {e['cross_template_cosine_boolq_vs_short']:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[round164] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
