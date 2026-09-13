#!/usr/bin/env python3
"""Round 168 Stage 1.5e: does the trigram carry information the backbone lacks?

Stage 1.5c measures the value of a *prediction-replacing* memory (``margin``).
That is not the quantity the ceiling argument is about, and the difference is
not academic: a predictor that is individually much worse can still know
something the backbone does not, and the forty rounds of null results all asked
the read-out to *reproduce* the backbone rather than to *add* to it.

This script asks the complementary question directly, on the two per-position
NLL arrays Stage 1.5c already stores -- so it needs **no GPU pass at all**:

    p_mix(w) = lam * p_bb(w) + (1 - lam) * p_cnt(w)

is a proper distribution (no partition function), so its probability at the
observed token is just the mixture of the two scalar probabilities.  ``lam`` is
fitted out-of-fold, the whole procedure is repeated with ``p_cnt`` permuted as
the null, and the per-context-frequency-band version localises any effect.

Read :mod:`qwen35_ple.fusion_probe` -- in particular its "shrinkage trap"
section -- before reading the numbers.  The decision quantity is the **excess
over the permuted null**, not the raw gain.

Pre-registration: ``docs/round-168-stage1.5e-preregistration.md``

Usage::

    python scripts/round168_fusion_probe.py --tag wiki \
        --workdir outputs/round168/margin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen35_ple.fusion_probe import (
    cross_fitted_gain,
    oracle_mixture_gain,
    permuted_gain,
    verdict_from_gain,
)
from qwen35_ple.margin import bucket_labels

#: The injection-point lens is the frame the decoder-independent bound is stated
#: in, so it is the primary; the final layer is the end-to-end view.
PRIMARY_FRAME = "lens"
FRAMES = ("lens", "final")
N_PERM = 16


def log(msg: str) -> None:
    print(f"[r168-1.5e] {msg}", flush=True)


def _analysis(
    name: str, p_bb: np.ndarray, p_cnt: np.ndarray, groups: np.ndarray,
    labels: list[str], n_perm: int,
) -> dict[str, Any]:
    real = cross_fitted_gain(p_bb, p_cnt)
    null = permuted_gain(p_bb, p_cnt, n_perm=n_perm)
    verdict = verdict_from_gain(
        real.gain, null.gain, null_std=null.null_std, n_perm=null.n_perm,
    )
    real_g = cross_fitted_gain(p_bb, p_cnt, groups=groups)
    null_g = permuted_gain(p_bb, p_cnt, groups=groups, n_perm=n_perm)
    # Only report bands that actually occur; an empty band is not a result.
    present = {labels[int(v)] for v in np.unique(groups)}
    per_group = {}
    for key, row in real_g.per_group.items():
        if key not in present:
            continue
        nrow = null_g.per_group.get(key, {})
        per_group[key] = {
            **row,
            "null_gain": nrow.get("gain"),
            "excess": row["gain"] - nrow.get("gain", 0.0),
            "null_std": null_g.null_std,
        }
    return {
        "frame": name,
        "global": {
            "real": real.to_dict(),
            "null": null.to_dict(),
            "verdict": verdict,
        },
        "by_context_band": {
            "grouped_gain": real_g.to_dict(),
            "grouped_null_gain": null_g.to_dict(),
            "per_group": per_group,
            "pooled_excess": real_g.pooled_gain - null_g.pooled_gain,
        },
        "oracle": oracle_mixture_gain(p_bb, p_cnt),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", default="outputs/round168/margin")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    work = Path(args.workdir)
    counts = np.load(work / f"{args.tag}-counts.npz")
    back = np.load(work / f"{args.tag}-backbone.npz")
    cnt_nll = np.asarray(counts["cnt_nll"], dtype=np.float64)
    ctx_counts = np.asarray(counts["context_counts"], dtype=np.int64)
    lens_nll = np.asarray(back["bb_lens"], dtype=np.float64)
    final_nll = np.asarray(back["bb_final"], dtype=np.float64)
    if not (cnt_nll.shape == lens_nll.shape == final_nll.shape):
        raise SystemExit(
            f"length mismatch: cnt {cnt_nll.shape} lens {lens_nll.shape} "
            f"final {final_nll.shape}"
        )
    scored = np.zeros(cnt_nll.shape, dtype=bool)
    scored[np.asarray(counts["positions"], dtype=np.int64)] = True
    valid = (
        scored & np.isfinite(cnt_nll) & np.isfinite(lens_nll) & np.isfinite(final_nll)
    )
    n = int(np.count_nonzero(valid))
    if n == 0:
        raise SystemExit("no positions scored by both stages")
    log(f"{n:,} positions scored by both stages")

    p_cnt = np.exp(-cnt_nll[valid])
    groups, labels = bucket_labels(ctx_counts[valid])
    out: dict[str, Any] = {
        "tag": args.tag,
        "train_npy": str(counts["train_npy"]),
        "eval_npy": str(counts["eval_npy"]),
        "n_positions": n,
        "n_perm": int(args.n_perm),
        "primary_frame": PRIMARY_FRAME,
        "band_labels": labels,
        "band_population": {
            labels[i]: int(np.count_nonzero(groups == i)) for i in range(len(labels))
        },
        "frames": {},
        "assumptions": [
            (
                "The decision quantity is the EXCESS of the out-of-fold mixture "
                "gain over the permuted null, not the raw gain: mixing an "
                "overconfident logit-lens distribution with anything moderate "
                "buys a large Jensen gain with no information transfer "
                "(see fusion_probe's 'shrinkage trap')."
            ),
            (
                "The null is conservative: it models mixing with an independent "
                "copy of the same marginal, which is easier than mixing with a "
                "correlated copy, so when the two models agree the real excess "
                "can be negative even where some complementarity exists."
            ),
            (
                "The family is linear in probability.  A log-linear (product) "
                "fusion is strictly more powerful, so ABSENT rules out the "
                "linear family, not all fusion."
            ),
            (
                "p_bb and p_cnt are probabilities of the OBSERVED token, which is "
                "all a proper mixture needs -- but it means this instrument "
                "cannot see token-identity information that cancels in the "
                "scalar summary."
            ),
        ],
    }
    for name, nll in (("lens", lens_nll), ("final", final_nll)):
        p_bb = np.exp(-nll[valid])
        out["frames"][name] = _analysis(
            name, p_bb, p_cnt, groups, labels, int(args.n_perm),
        )
        v = out["frames"][name]["global"]["verdict"]
        log(f"frame {name}: excess {v['excess']:+.4f} nats "
            f"(raw {v['gain']:.4f}, null {v['null_gain']:.4f}, "
            f"null_std {v['null_std']:.4f}) -> {v['label']}")

    primary = out["frames"][PRIMARY_FRAME]
    out["verdict"] = primary["global"]["verdict"]
    outpath = Path(args.output) if args.output else work / f"{args.tag}-fusion.json"
    outpath.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = outpath.with_suffix(".md")
    md.write_text(render_markdown(out), encoding="utf-8")
    log(f"wrote {outpath} and {md}")
    return 0


def render_markdown(out: dict[str, Any]) -> str:
    p = out["frames"][out["primary_frame"]]
    f = out["frames"]["final"]
    lines = [f"# Stage 1.5e fusion probe — {out['tag']}", ""]
    lines.append(f"* positions: {out['n_positions']:,}")
    lines.append(f"* count training stream: `{out['train_npy']}`")
    lines.append(f"* eval stream: `{out['eval_npy']}`")
    lines.append(f"* permutations averaged for the null: {out['n_perm']}")
    lines.append("")
    lines.append("## Verdict (primary frame = injection-point lens)")
    lines.append("")
    lines.append(f"**{p['global']['verdict']['label']}**")
    lines.append("")
    lines.append(f"> {p['global']['verdict']['reason']}")
    lines.append("")
    lines.append(f"Final-layer frame: **{f['global']['verdict']['label']}** — "
                 f"{f['global']['verdict']['reason']}")
    lines.append("")
    lines.append("## Global numbers (nats)")
    lines.append("")
    lines.append("| frame | raw gain | null gain | null std | **excess** | lam | oracle gain |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in FRAMES:
        d = out["frames"][name]
        g = d["global"]
        lines.append(
            f"| {name} | {g['real']['gain']:.4f} | {g['null']['gain']:.4f} | "
            f"{g['null']['null_std']:.4f} | **{g['verdict']['excess']:+.4f}** | "
            f"{g['real']['lam']:.3f} | {d['oracle']['gain']:.4f} |"
        )
    lines.append("")
    lines.append("The **raw gain** is dominated by recalibration of an "
                 "overconfident distribution; only the **excess** carries "
                 "information about the pairing.")
    lines.append("")
    lines.append("## Per context-frequency band (primary frame)")
    lines.append("")
    lines.append("| ctx count | n | lam | gain | null gain | **excess** | population |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, row in p["by_context_band"]["per_group"].items():
        pop = out["band_population"].get(label, 0)
        lines.append(
            f"| {label} | {row['n']:,} | {row['lam_mean']:.3f} | "
            f"{row['gain']:.4f} | {row['null_gain']:.4f} | "
            f"**{row['excess']:+.4f}** | {pop:,} |"
        )
    lines.append("")
    lines.append(f"Equal-weight pooled excess: "
                 f"**{p['by_context_band']['pooled_excess']:+.4f}** nats")
    lines.append("")
    lines.append("## Assumptions / honesty")
    lines.append("")
    for a in out["assumptions"]:
        lines.append(f"* {a}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
