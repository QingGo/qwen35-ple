#!/usr/bin/env python
"""Audit a hard row mask against the per-position record.

Phase 5 pre-registered an "identity": freezing every row whose *training* count is
below N should make the total delta equal the sum of the >=N bands read off the
full-training arm.  It did not.  Measured +0.0046840290 against a predicted
+0.0074002256 -- 13.6x the frozen tolerance -- so the pre-registration's own rule
fired and arm B was skipped.

A mismatch like that has two very different readings, and the whole value of the
run depends on telling them apart:

    (a) the mask did not do what it said  -> implementation bug, re-run
    (b) the mask did exactly what it said -> the "identity" was never one

So this tool asks four questions, each with its own independent evidence:

 1. ``did the mask bite?``    training count >= N  vs  the bank row actually
    differing from E0.  These are computed from different files -- the trainer's
    count array and the written bank -- so agreement is a real cross-check, not a
    restatement of the metadata.
 2. ``was the arithmetic right?``  recompute every band sum from the full arm and
    compare against the frozen table.  If these reproduce to ten decimals then the
    decomposition of *that* run is exact and the error is in the premise.
 3. ``what did the mask cost?``  the same positions scored under both arms.
 4. ``where is the hidden variable?``  positions whose row is BIT-IDENTICAL to E0
    must have delta exactly 0 under the masked arm -- the injected vector is the
    same vector.  Whatever is left is coupling that does not travel through the
    row table at all.

The one-line diagnosis this tool exists to produce: **the rows are independent in
the read and coupled in the gradient**, because the autoregressive hidden state
carries every position into every later one.

IO lives in ``main``; everything above it is pure so the tests can drive the
arithmetic without a GPU box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANDS = (1, 2, 5, 10, 20, 50, 100)


def keep_inverse(keep: np.ndarray) -> np.ndarray:
    """Map a full-table row id onto its slot inside the keep-subset bank.

    The bank on disk holds only the rows the eval stream can read, so the
    ``snapshot_index`` recorded per position -- a full-table id -- has to be
    translated before it can index the bank.  Ids outside ``keep`` map to -1 so a
    caller can count them instead of silently wrapping to the wrong row.
    """
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        return np.zeros(0, dtype=np.int64)
    if keep.min() < 0:
        raise ValueError("keep index contains negative row ids")
    inv = np.full(int(keep.max()) + 1, -1, dtype=np.int64)
    inv[keep] = np.arange(keep.size, dtype=np.int64)
    return inv


def rows_changed(e0_keep: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Per bank row: does it differ from the E0 value anywhere?

    NaN is treated as equal to NaN.  The banks are finite in practice, but a
    NaN-poisoned row would otherwise be reported as changed under every mask and
    would wreck the "exactly zero" test below, so the comparison is explicit.
    """
    a = np.asarray(e0_keep)
    b = np.asarray(bank)
    if a.shape != b.shape:
        raise ValueError(f"E0 keep-subset {a.shape} and bank {b.shape} must match")
    diff = a != b
    if np.issubdtype(a.dtype, np.floating):
        diff &= ~(np.isnan(a) & np.isnan(b))
    return diff.any(axis=1)


def crosstab(selected: np.ndarray, changed: np.ndarray) -> dict:
    """Agreement between "the mask says train it" and "its value moved"."""
    selected = np.asarray(selected, dtype=bool)
    changed = np.asarray(changed, dtype=bool)
    if selected.shape != changed.shape:
        raise ValueError(f"{selected.shape} and {changed.shape} must match")
    both = int((selected & changed).sum())
    selected_only = int((selected & ~changed).sum())
    changed_only = int((~selected & changed).sum())
    neither = int((~selected & ~changed).sum())
    n = int(selected.size)
    return {
        "n": n,
        "both": both,
        "selected_only": selected_only,
        "changed_only": changed_only,
        "neither": neither,
        "agreement": (n - selected_only - changed_only) / n if n else 1.0,
    }


def band_sums(
    delta: np.ndarray, counts: np.ndarray, total: int, thresholds=BANDS
) -> list[dict]:
    """The pre-registration's prediction: ``sum(delta[counts >= N]) / total``.

    ``total`` is the number of scored positions, passed separately because it is
    the denominator the frozen table used and it is not in general ``delta.size``
    once a caller subsets the record.
    """
    delta = np.asarray(delta, dtype=np.float64)
    counts = np.asarray(counts)
    if delta.shape != counts.shape:
        raise ValueError(f"delta {delta.shape} and counts {counts.shape} must match")
    if total <= 0:
        raise ValueError("total must be positive")
    out = []
    for n_thr in thresholds:
        sel = counts >= n_thr
        out.append(
            {
                "threshold": int(n_thr),
                "n": int(sel.sum()),
                "share": float(sel.mean()) if sel.size else 0.0,
                "sum_over_total": float(delta[sel].sum() / total),
            }
        )
    return out


def region(delta_a: np.ndarray, delta_b: np.ndarray, sel: np.ndarray) -> dict:
    """Mean delta in one region under both arms, and the masked arm's cost there."""
    a = np.asarray(delta_a, dtype=np.float64)[sel]
    b = np.asarray(delta_b, dtype=np.float64)[sel]
    return {
        "n": int(a.size),
        "mean_a": float(a.mean()) if a.size else 0.0,
        "mean_b": float(b.mean()) if b.size else 0.0,
        "mean_difference": float((b - a).mean()) if a.size else 0.0,
    }


def noise_floor(delta: np.ndarray, untouched: np.ndarray) -> dict:
    """Delta on positions whose injected row is bit-identical to E0.

    It must be exactly zero.  Anything else is coupling that bypasses the row
    table -- the backward pass, through a hidden state every earlier position
    wrote to.
    """
    d = np.asarray(delta, dtype=np.float64)[untouched]
    if d.size == 0:
        return {"n": 0, "mean": 0.0, "absmax": 0.0, "exact_zero_frac": 1.0}
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "absmax": float(np.abs(d).max()),
        "exact_zero_frac": float((d == 0).mean()),
    }


def render(audit: dict) -> str:
    """Plain-text report, ordered as the four questions."""
    out: list[str] = []
    w = out.append
    g = audit
    w(f"positions {g['n_positions']}   keep {g['n_keep']}   unmapped {g['n_unmapped']}")
    w(
        "train count: {shape}  max {max_count}  rows>=10 {ge10}  rows>=50 {ge50}".format(
            **g["train_count"]
        )
    )
    w(f"bank rows differing from E0: {g['bank_rows_changed']} of {g['bank_rows']}")
    w(
        "positions whose row actually changed: {n} ({share:.1%})".format(
            n=g["positions_changed"], share=g["positions_changed"] / max(g["n_positions"], 1)
        )
    )

    w("")
    w("=== 1. did the mask bite? ===")
    w(f"  {'N':>4} {'tc>=N':>8} {'share':>7} {'agree':>9} {'both':>8} {'sel-only':>9} {'chg-only':>9}")
    for row in g["crosstab"]:
        w(
            "  {threshold:>4} {n:>8} {share:>6.1%} {agreement:>9.6f} {both:>8} "
            "{selected_only:>9} {changed_only:>9}".format(**row)
        )

    w("")
    w("=== 2. was the arithmetic right? ===")
    w(f"  {'N':>4} {'positions':>10} {'recomputed':>16} {'predicted':>16} {'match':>6}")
    for row in g["bands"]:
        pred = row["predicted"]
        mark = "yes" if pred is not None and abs(pred - row["sum_over_total"]) <= 1e-9 else (
            "-" if pred is None else "NO"
        )
        w(
            "  {threshold:>4} {n:>10} {sum_over_total:>+16.10f} {pred:>16} {mark:>6}".format(
                pred=("(none)" if pred is None else f"{pred:+.10f}"), mark=mark, **row
            )
        )

    w("")
    w("=== 3. what did the mask cost? ===")
    w(f"  {'region':<18} {'n':>8} {'share':>7} {'mean A':>11} {'mean B':>11} {'B-A':>11}")
    for row in g["regions"]:
        w(
            "  {name:<18} {n:>8} {share:>6.1%} {mean_a:>+11.6f} {mean_b:>+11.6f} "
            "{mean_difference:>+11.6f}".format(**row)
        )

    w("")
    w("=== 4. where is the hidden variable? ===")
    nf = g["noise_floor"]
    w(f"  positions whose row is BIT-IDENTICAL to E0: {nf['n']}")
    w(f"    delta must be exactly 0 -> mean {nf['mean']:+.4e}  max|.| {nf['absmax']:.4e}  "
      f"exact-zero frac {nf['exact_zero_frac']:.6f}")
    if nf["exact_zero_frac"] < 1.0:
        w("    NOT zero: the coupling does not travel through the row table.")
    w(f"  predicted (>=N band sum of the full arm) = {g['predicted_total']:+.10f}")
    w(f"  measured  (masked arm total)             = {g['measured_total']:+.10f}")
    w(f"  shortfall                                = {g['shortfall']:+.10f}")
    if g["tolerance"] is not None:
        w(
            f"  |shortfall| = {abs(g['shortfall']):.4e} = "
            f"{abs(g['shortfall']) / g['tolerance']:.2f}x tolerance {g['tolerance']:g}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", required=True, type=Path)
    ap.add_argument("--masked-bank", required=True, type=Path,
                    help="the bank the masked arm trained (keep-subset of rows)")
    ap.add_argument("--masked-deltas", required=True, type=Path)
    ap.add_argument("--baseline-deltas", required=True, type=Path,
                    help="the unmasked arm's per-position record, same positions")
    ap.add_argument("--e0", type=Path, default=None, help="default: <snap>/E0.npy")
    ap.add_argument("--keep-index", type=Path, default=None,
                    help="default: <snap>/keep-wiki-eval.npy")
    ap.add_argument("--train-count", type=Path, default=None,
                    help="default: <snap>/trigram-train-count.npy")
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=None,
                    help="the pre-registered tolerance, for the verdict line")
    ap.add_argument("--predicted", type=float, default=None,
                    help="the pre-registered prediction; default: recomputed band sum")
    ap.add_argument("--out", type=Path, default=None, help="write the audit as JSON")
    args = ap.parse_args()

    snap = args.snapshot_dir
    e0_path = args.e0 or snap / "E0.npy"
    keep_path = args.keep_index or snap / "keep-wiki-eval.npy"
    count_path = args.train_count or snap / "trigram-train-count.npy"

    masked = np.load(args.masked_deltas)
    base = np.load(args.baseline_deltas)
    for key in ("delta", "snapshot_index", "trigram_code"):
        if key not in masked.files or key not in base.files:
            raise SystemExit(f"both records need a {key!r} array")
    if not np.array_equal(masked["trigram_code"], base["trigram_code"]):
        raise SystemExit("the two records are not the same positions (trigram_code differs)")

    si = masked["snapshot_index"].astype(np.int64)
    delta_b = masked["delta"].astype(np.float64)
    delta_a = base["delta"].astype(np.float64)
    n = int(si.size)

    keep = np.load(keep_path).astype(np.int64)
    slot = keep_inverse(keep)[si]
    unmapped = int((slot < 0).sum())

    counts_all = np.load(count_path)
    # The count array is indexed by full-table row id when it covers the whole
    # table, and by keep slot when it was already subset.  The bank's own row
    # count decides which.
    bank = np.load(args.masked_bank, mmap_mode="r")
    counts = counts_all[si] if counts_all.size > bank.shape[0] else counts_all[slot]

    e0 = np.load(e0_path, mmap_mode="r")
    if e0.shape[0] < int(keep.max()) + 1:
        raise SystemExit(f"E0 has {e0.shape[0]} rows, too few for keep max {int(keep.max())}")
    changed = rows_changed(np.asarray(e0[keep]), np.asarray(bank))
    if unmapped:
        # An unmapped position means the record and the keep index disagree about
        # which rows exist.  Zeroing it here would quietly void question 4, which
        # is the one question this tool exists to answer.
        raise SystemExit(
            f"{unmapped} of {n} positions index a row the keep index does not contain"
        )
    pos_changed = changed[slot]

    sel = counts >= args.threshold
    bands = band_sums(delta_a, counts, n)
    # The pre-registration froze one number at one threshold.  Every band is
    # recomputed here; --predicted only decides whether that one stored number is
    # compared against the recomputation or silently replaced by it.
    for b in bands:
        b["predicted"] = (
            float(args.predicted)
            if args.predicted is not None and b["threshold"] == args.threshold
            else None
        )
    predicted = (
        float(args.predicted)
        if args.predicted is not None
        else next(b["sum_over_total"] for b in bands if b["threshold"] == args.threshold)
    )
    measured = float(delta_b.mean())

    regions = []
    for name, mask in ((f"high tc>={args.threshold}", sel),
                       (f"low  tc<{args.threshold}", ~sel),
                       ("ALL", np.ones(n, dtype=bool))):
        row = region(delta_a, delta_b, mask)
        row["name"] = name
        row["share"] = float(mask.mean()) if n else 0.0
        regions.append(row)

    audit = {
        "n_positions": n,
        "n_keep": int(keep.size),
        "n_unmapped": unmapped,
        "train_count": {
            "shape": str(tuple(counts_all.shape)),
            "max_count": int(counts_all.max()) if counts_all.size else 0,
            "ge10": int((counts_all >= 10).sum()),
            "ge50": int((counts_all >= 50).sum()),
        },
        "bank_rows": int(changed.size),
        "bank_rows_changed": int(changed.sum()),
        "positions_changed": int(pos_changed.sum()),
        "threshold": args.threshold,
        "crosstab": [
            dict(crosstab(counts >= t, pos_changed), threshold=int(t),
                 share=float((counts >= t).mean()))
            for t in BANDS
        ],
        "bands": bands,
        "regions": regions,
        "noise_floor": noise_floor(delta_b, ~pos_changed),
        "predicted_total": float(predicted),
        "measured_total": measured,
        "shortfall": measured - float(predicted),
        "tolerance": args.tolerance,
    }
    print(render(audit))
    if args.out:
        args.out.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
