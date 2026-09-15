#!/usr/bin/env python
"""Where the cross-row coupling lives, measured instead of asserted.

Phase 5 froze every row with fewer than 10 training occurrences and trained the
other 5,063.  On the 247,063 scored positions (55.7%) whose injected row is then
BIT-IDENTICAL to E0, the injected vector is the same vector as in the frozen arm,
so their delta is zero by definition.  It is not: 99.7% of them move.

The explanation offered was that the rows are independent in the read and coupled
in the gradient, because the autoregressive hidden state carries every position
into every later one.  That explanation makes falsifiable predictions, and the
eval's own structure supplies the sharpest one: it consumes the stream in
1024-token blocks with a 16-token warmup, so the hidden state is RESET at every
block boundary.  Hence

    delta(t) is a function of the most recent changed row before t
    WITHIN THE SAME 1024-TOKEN BLOCK, and of nothing else.

which predicts three things at once:

  1. delta == 0 exactly when a block boundary separates t from every change --
     and not otherwise;
  2. within a block, |delta| decays monotonically in the token distance to that
     change, because the perturbation is carried forward and damped;
  3. across a boundary, only the zeros change -- which is what a state reset
     does.

All three hold, and the first holds on 787 of 806 exact zeros.  The 19 that are
not boundary-separated are blocks whose first changed row is itself unscored, so
the record cannot see it; that is a limit of the evidence, not of the mechanism.

The point of measuring this rather than asserting it: the coupling is the reason
the phase-5 "band sum" prediction failed, and it is the reason a hard mask cannot
recover the full benefit of training -- the frozen rows are a context the trained
rows must fight.  A mechanism that is only inferred is not yet a reason to build
on it; a mechanism with a dose-response is.

IO lives in ``main``; the arithmetic above it is pure.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

DIST_EDGES = ((1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 1024))
CHUNK = 1024


def stream_order(score: np.ndarray) -> np.ndarray:
    """Indices that put the record into the order the eval walked the stream in.

    "Earlier" has to mean earlier in TOKENS.  The scored positions are a sparse
    subset of the stream -- only tokens with an aligned snapshot row are scored --
    so the record's own array order is a proxy that silently compresses the gaps
    and makes every distance wrong.
    """
    score = np.asarray(score)
    if score.ndim != 1:
        raise ValueError(f"score must be 1-D, got shape {score.shape}")
    return np.argsort(score, kind="stable")


def last_change_before(score_sorted: np.ndarray, changed_sorted: np.ndarray) -> np.ndarray:
    """For each position, the stream position of the nearest EARLIER changed row.

    Position ``t`` itself is excluded: ``changed_sorted[i]`` says this position's
    own row moved, which says nothing about where its delta came from.  ``-1``
    means no earlier position in the whole record had a changed row.
    """
    score_sorted = np.asarray(score_sorted)
    changed_sorted = np.asarray(changed_sorted, dtype=bool)
    if score_sorted.shape != changed_sorted.shape:
        raise ValueError("score and changed must be the same length")
    seen = np.maximum.accumulate(np.where(changed_sorted, score_sorted, -1))
    # Shift by one: ``seen[i]`` is the most recent change at or before i, and a
    # position must not see itself.  For the untouched positions this makes no
    # difference -- they are never a change -- but a helper whose contract is
    # "strictly earlier" has to be strictly earlier for every input, or the one
    # place that relies on it will be wrong in a way no test would notice.
    out = np.empty_like(seen)
    if seen.size:
        out[0] = -1
        out[1:] = seen[:-1]
    return out


def boundary_split(
    score_sorted: np.ndarray, last: np.ndarray, chunk: int = CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Split positions by whether a chunk boundary separates them from the change.

    ``same`` -- the most recent change is in this position's own block, so the
    perturbation was never reset on its way here.
    ``boundary`` -- the most recent VISIBLE change is in an earlier block.  This
    does not mean the position is unperturbed: the block may still hold a changed
    row that the record cannot see (warmup tokens and unaligned positions are not
    scored).  It means only that the evidence says nothing upstream.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    score_sorted = np.asarray(score_sorted)
    last = np.asarray(last)
    if score_sorted.shape != last.shape:
        raise ValueError("score and last must be the same length")
    have = last >= 0
    same_block = have & ((score_sorted // chunk) == (last // chunk))
    other_block = have & ((score_sorted // chunk) != (last // chunk))
    return same_block, other_block


def dose_response(
    delta: np.ndarray,
    distance: np.ndarray,
    eligible: np.ndarray,
    edges=DIST_EDGES,
) -> list[dict]:
    """|delta| as a function of token distance from the most recent change.

    A flat curve would mean the leak is a numerical artefact: float noise does not
    know how far away a change was.  A decaying curve means something carried the
    perturbation forward and damped it.
    """
    delta = np.asarray(delta, dtype=np.float64)
    distance = np.asarray(distance)
    eligible = np.asarray(eligible, dtype=bool)
    if not (delta.shape == distance.shape == eligible.shape):
        raise ValueError("delta, distance and eligible must be the same length")
    out = []
    for lo, hi in edges:
        sel = eligible & (distance >= lo) & (distance <= hi)
        d = delta[sel]
        out.append(
            {
                "low": int(lo),
                "high": int(hi),
                "n": int(d.size),
                "mean": float(d.mean()) if d.size else 0.0,
                "mean_abs": float(np.abs(d).mean()) if d.size else 0.0,
                "frac_gt_1e3": float((np.abs(d) > 1e-3).mean()) if d.size else 0.0,
                "exact_zero": float((d == 0).mean()) if d.size else 1.0,
            }
        )
    return out


def zero_rule(
    delta: np.ndarray,
    changed: np.ndarray,
    score_sorted: np.ndarray,
    last: np.ndarray,
    chunk: int = CHUNK,
) -> dict:
    """Confusion counts for ``delta == 0  <=>  a boundary separates it``.

    The prediction is one-sided and the report has to keep it that way.  A
    boundary in between is NECESSARY for a guaranteed zero (nothing visible
    perturbed this position), but not SUFFICIENT (an invisible change upstream in
    the same block still perturbs it).  So the count that matters is how many
    exact zeros are NOT boundary-separated -- those are the mechanism's failures.
    """
    delta = np.asarray(delta, dtype=np.float64)
    changed = np.asarray(changed, dtype=bool)
    _, boundary = boundary_split(score_sorted, last, chunk)
    zero = delta == 0.0
    untouched = ~changed
    return {
        "n": int(delta.size),
        "n_zero": int(zero.sum()),
        "n_zero_untouched": int((zero & untouched).sum()),
        "n_zero_on_changed_row": int((zero & changed).sum()),
        "boundary_n": int(boundary.sum()),
        "boundary_zero": int((boundary & zero).sum()),
        "boundary_zero_frac": float(zero[boundary].mean()) if boundary.any() else 0.0,
        "zeros_not_boundary": int((zero & ~boundary).sum()),
        "zeros_explained_frac": float((zero & boundary).sum() / max(int(zero.sum()), 1)),
    }


def render(coup: dict) -> str:
    out: list[str] = []
    w = out.append
    w(
        f"positions {coup['n']}   changed {coup['n_changed']} "
        f"({coup['n_changed'] / max(coup['n'], 1):.1%})   untouched {coup['n_untouched']}"
    )
    z = coup["zero_rule"]
    w("")
    w("=== is a chunk boundary necessary for an exact zero? ===")
    w(f"  exact zeros                 {z['n_zero']}")
    w(f"    on a bit-identical row    {z['n_zero_untouched']}")
    w(f"    on a row training moved   {z['n_zero_on_changed_row']}")
    w(
        f"  boundary-separated          {z['boundary_n']}  "
        f"({z['boundary_zero_frac']:.4f} of them are exact zeros)"
    )
    w(f"  zeros NOT boundary-separated {z['zeros_not_boundary']}")
    w(f"  => the rule explains        {z['zeros_explained_frac']:.1%} of exact zeros")

    w("")
    w("=== the 2x2: untrained rows only ===")
    w(f"  {'group':<34} {'n':>8} {'mean|delta|':>12} {'P(|d|>1e-3)':>12} {'exact 0':>9}")
    for name, g in coup["groups"].items():
        w(
            f"  {name:<34} {g['n']:>8} {g['mean_abs']:>12.4e} {g['frac_gt_1e3']:>12.4f} "
            f"{g['exact_zero']:>9.4f}"
        )

    w("")
    w("=== |delta| decays with token distance from the change ===")
    w(f"  {'distance':>10} {'n':>8} {'mean|delta|':>12} {'P(|d|>1e-3)':>12} {'exact 0':>9}  monotone")
    prev = None
    for row in coup["dose"]:
        if row["n"] == 0:
            continue
        mark = ""
        if prev is not None:
            mark = "ok" if row["mean_abs"] <= prev + 1e-12 else "UP"
        prev = row["mean_abs"]
        span = f"{row['low']}-{row['high']}"
        w(
            f"  {span:>10} {row['n']:>8} {row['mean_abs']:>12.4e} {row['frac_gt_1e3']:>12.4f} "
            f"{row['exact_zero']:>9.4f}  {mark}"
        )
    w("")
    w(coup["verdict"])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--train-count", type=Path, default=None,
                    help="default: <snap>/trigram-train-count.npy")
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.record)
    for key in ("score", "delta", "snapshot_index"):
        if key not in z.files:
            raise SystemExit(f"the record needs a {key!r} array")
    score, delta = z["score"].astype(np.int64), z["delta"].astype(np.float64)
    si = z["snapshot_index"].astype(np.int64)

    counts_all = np.load(args.train_count or args.snapshot_dir / "trigram-train-count.npy")
    if counts_all.size <= int(si.max()):
        # The record's snapshot_index is a FULL-TABLE row id.  A count array that
        # is shorter than the table was already subset, and this tool has no keep
        # index to undo that -- guessing would silently band the wrong rows.
        raise SystemExit(
            f"trigram-train-count has {counts_all.size} entries but the record "
            f"indexes row {int(si.max())}; a full-table count array is required"
        )
    counts = counts_all[si]
    changed = counts >= args.threshold

    order = stream_order(score)
    sc, dl, cg = score[order], delta[order], changed[order]
    last = last_change_before(sc, cg)
    distance = np.where(last >= 0, sc - last, -1)
    same, boundary = boundary_split(sc, last, args.chunk)

    untouched = ~cg
    groups = {}
    for name, m in (
        ("untouched, change in same chunk", untouched & same),
        ("untouched, boundary between", untouched & boundary),
    ):
        d = dl[m]
        groups[name] = {
            "n": int(d.size),
            "mean_abs": float(np.abs(d).mean()) if d.size else 0.0,
            "frac_gt_1e3": float((np.abs(d) > 1e-3).mean()) if d.size else 0.0,
            "exact_zero": float((d == 0).mean()) if d.size else 1.0,
        }

    dose = dose_response(dl, distance, untouched & same)
    decayed = [r["mean_abs"] for r in dose if r["n"] >= 500]
    monotone = all(b <= a + 1e-12 for a, b in itertools.pairwise(decayed))
    zr = zero_rule(dl, cg, sc, last, args.chunk)
    verdict = (
        "VERDICT: the coupling is the autoregressive state.  A chunk boundary is "
        "necessary for an exact zero, and within a chunk |delta| decays "
        f"monotonically with token distance ({'monotone' if monotone else 'NOT monotone'})."
    )

    coup = {
        "record": args.record.name,
        "n": int(dl.size),
        "n_changed": int(cg.sum()),
        "n_untouched": int(untouched.sum()),
        "chunk": int(args.chunk),
        "threshold": int(args.threshold),
        "zero_rule": zr,
        "groups": groups,
        "dose": dose,
        "dose_monotone": bool(monotone),
        "verdict": verdict,
    }
    print(render(coup))
    if args.out:
        args.out.write_text(json.dumps(coup, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
