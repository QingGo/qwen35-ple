#!/usr/bin/env python3
"""Round 166 addendum: are the directional lexical predictions length-confounded?

``classify_format.domain_fingerprint`` counts *how many distinct* markers occur in a
generation, not how often they occur and not per token::

    prose = sum(1 for m in _PROSE_MARKERS if m in low)      # 0..6

A longer generation is therefore more likely to contain each marker, so an arm
that generates more tokens accumulates higher marker counts regardless of the
corpus behind its table.  Round 162's two directional predictions compare marker
counts across arms, so if the arms differ in generated length the comparison is
confounded.

This script quantifies that for every regime we have, and re-runs the two
predictions with generated length held fixed, by stratifying on the token count
and averaging the within-stratum differences (the Mantel-Haenszel-style
estimator for a paired contrast with a discrete confounder).

Usage::

    python scripts/audit_directional_length_confound.py \
      --regime 0.8B-nosft=outputs/round162-0.8B-nosft600 \
      --regime 4B-nosft=outputs/round166-4B-nosft \
      --output  outputs/round166-length-confound.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classify_format import domain_fingerprint

ARMS = ("wiki", "code", "stem")
PREDICTIONS = (
    # (name, arm_a, arm_b, marker): predict mean_a > mean_b
    ("code>wiki:code", "code", "wiki", "code"),
    ("wiki>code:prose", "wiki", "code", "prose"),
    ("stem>wiki:math", "stem", "wiki", "math"),
)


def _load_arm(d: Path, arm: str) -> dict:
    payload = json.loads((d / f"arm-{arm}.json").read_text())
    return payload["results"][0]["qa_exact"]["answers"]


def _exact_paired_p(diff: np.ndarray) -> float:
    """One-sided sign test p-value for mean(diff) > 0 (ties dropped)."""
    nz = diff[diff != 0]
    if nz.size == 0:
        return 1.0
    wins = int((nz > 0).sum())
    # Exact binomial tail at p=0.5, computed with integer arithmetic.
    from math import comb
    n = nz.size
    tail = sum(comb(n, k) for k in range(wins, n + 1))
    return float(tail / 2**n)


def analyse(d: Path) -> dict:
    data = {a: _load_arm(d, a) for a in ARMS}
    out: dict = {"regime_dir": str(d), "arms": {}}

    for a in ARMS:
        ans = data[a]
        n = np.array([x["n_generated"] for x in ans], dtype=np.int64)
        prose = np.array([domain_fingerprint(x["generated"])["prose"] for x in ans], dtype=float)
        code = np.array([domain_fingerprint(x["generated"])["code"] for x in ans], dtype=float)
        math = np.array([domain_fingerprint(x["generated"])["math"] for x in ans], dtype=float)
        out["arms"][a] = {
            "n": len(ans),
            "mean_n_generated": float(n.mean()),
            "pct_at_32_cap": float(100.0 * (n >= 32).mean()),
            "mean_prose": float(prose.mean()),
            "mean_code": float(code.mean()),
            "mean_math": float(math.mean()),
            "corr_len_prose": float(np.corrcoef(n, prose)[0, 1]) if n.std() > 0 else None,
            "corr_len_code": float(np.corrcoef(n, code)[0, 1]) if n.std() > 0 else None,
        }

    out["predictions"] = {}
    for name, arm_a, arm_b, marker in PREDICTIONS:
        A, B = data[arm_a], data[arm_b]
        assert len(A) == len(B), "arms must be paired over the same items"
        # Both arms answer the same file in the same order, so index i is the
        # same item in both and the contrast below is paired.
        m_a = np.array([domain_fingerprint(x["generated"])[marker] for x in A], dtype=float)
        m_b = np.array([domain_fingerprint(x["generated"])[marker] for x in B], dtype=float)
        n_a = np.array([x["n_generated"] for x in A], dtype=np.int64)
        n_b = np.array([x["n_generated"] for x in B], dtype=np.int64)

        raw = m_a - m_b
        # The confound is that a longer generation collects more distinct
        # markers.  Stacking the two arms and regressing marker on length would
        # need item fixed effects; the paired equivalent is to regress the
        # *difference* on the *difference* in length:
        #
        #     d_i = c + b * l_i + e_i,   d_i = m_a - m_b,  l_i = n_a - n_b
        #
        # c is the arm difference at equal generated length, and b is the
        # markers-per-token slope.  If the raw effect is entirely length, b
        # absorbs it and c goes to zero.
        d = raw
        l = (n_a - n_b).astype(float)
        X = np.column_stack([np.ones_like(l), l])
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        c_hat, b_hat = float(coef[0]), float(coef[1])
        resid = d - X @ coef
        dof = max(1, len(d) - 2)
        s2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se_c = float(np.sqrt(s2 * xtx_inv[0, 0]))
        se_b = float(np.sqrt(s2 * xtx_inv[1, 1]))

        def _t_p(est: float, se: float) -> float:
            """One-sided p for est > 0 under a normal approximation."""
            if se <= 0:
                return 1.0
            from math import erfc, sqrt
            return 0.5 * erfc(est / (se * sqrt(2.0)))

        # Support: how much of each arm's length range do the two share?
        lo = max(n_a.min(), n_b.min())
        hi = min(n_a.max(), n_b.max())
        in_a = int(((n_a >= lo) & (n_a <= hi)).sum())
        in_b = int(((n_b >= lo) & (n_b <= hi)).sum())
        both = (n_a >= lo) & (n_a <= hi) & (n_b >= lo) & (n_b <= hi)
        n_both = int(both.sum())
        restricted = raw[both]
        restricted_p = _exact_paired_p(restricted) if n_both else float("nan")

        out["predictions"][name] = {
            "n_pairs": len(raw),
            "mean_n_generated_a": float(n_a.mean()),
            "mean_n_generated_b": float(n_b.mean()),
            "length_a_quartiles": [float(x) for x in np.percentile(n_a, [25, 50, 75])],
            "length_b_quartiles": [float(x) for x in np.percentile(n_b, [25, 50, 75])],
            "raw_mean_diff": float(raw.mean()),
            "raw_one_sided_p": _exact_paired_p(raw),
            "markers_per_token_slope": b_hat,
            "markers_per_token_slope_se": se_b,
            "length_adjusted_diff": c_hat,
            "length_adjusted_se": se_c,
            "length_adjusted_one_sided_p": _t_p(c_hat, se_c),
            "overlap_lo": int(lo),
            "overlap_hi": int(hi),
            "n_a_in_overlap": in_a,
            "n_b_in_overlap": in_b,
            "n_pairs_in_overlap": n_both,
            "restricted_mean_diff": float(restricted.mean()) if n_both else None,
            "restricted_one_sided_p": restricted_p,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    report = {"regimes": {}}
    for spec in args.regime:
        name, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"--regime must be NAME=DIR, got {spec!r}")
        report["regimes"][name] = analyse(Path(d))

    # Pooled view for whichever regimes are present.
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for name, r in report["regimes"].items():
        print(f"=== {name} ===")
        for a, s in r["arms"].items():
            print(
                f"  {a:6s} n_gen={s['mean_n_generated']:6.2f} (at cap {s['pct_at_32_cap']:5.1f}%) "
                f"prose={s['mean_prose']:.3f} code={s['mean_code']:.3f} "
                f"corr(len,prose)={s['corr_len_prose']:+.3f}"
            )
        for p, s in r["predictions"].items():
            print(
                f"  {p:16s} len {s['mean_n_generated_a']:5.1f} vs {s['mean_n_generated_b']:5.1f}\n"
                f"      raw diff {s['raw_mean_diff']:+.4f} (p={s['raw_one_sided_p']:.4g})\n"
                f"      length-adjusted {s['length_adjusted_diff']:+.4f} "
                f"+- {s['length_adjusted_se']:.4f} (p={s['length_adjusted_one_sided_p']:.4g}), "
                f"slope {s['markers_per_token_slope']:+.4f}/token\n"
                f"      overlap [{s['overlap_lo']},{s['overlap_hi']}] holds "
                f"{s['n_pairs_in_overlap']}/{s['n_pairs']} pairs; restricted diff "
                f"{s['restricted_mean_diff'] if s['restricted_mean_diff'] is None else round(s['restricted_mean_diff'], 4)}"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
