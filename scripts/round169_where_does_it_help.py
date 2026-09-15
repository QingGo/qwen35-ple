#!/usr/bin/env python
"""Where does the graft's injection actually help?

Every result this project has about the memory is an AGGREGATE: "frozen rows beat
the pure backbone by 0.17593 nats".  An aggregate cannot say whether that comes
from a uniform calibration shift, from a handful of positions, or from the tail
the bound actually permits -- and those three readings imply completely different
next moves.  The per-position record makes the question answerable without any
GPU: it carries ``nll_none``, ``nll_frozen`` and ``nll_trained`` for every scored
position, so the injection's per-position gain is directly computable.

Two decompositions, both read off the same record:

1. BY BACKBONE SURPRISAL (``nll_none``).  The bound is conditional on ``h_t``:
   ``I(Y; e | h_t) <= I(Y; w_t | h_t)``.  A memory channel can only be worth
   something where the backbone is NOT already certain, so the gain should vanish
   as ``nll_none -> 0``.  If instead the gain is flat in surprisal, the injection
   is acting as a global calibration shift rather than as content.

2. BY ROW EVIDENCE (the training count of the row that was injected).  Under the
   bound the permitted prior is the corpus's rare n-gram statistics, so the value
   should track how well that row is estimated.  A row seen once is a one-sample
   estimate of the prior, not the prior.

The two are crossed, because either alone is confounded: rare rows occur at
high-surprisal positions by construction (Zipf), so a surprisal-only table cannot
separate "the memory helps on surprises" from "the memory helps on rare rows".

Sign convention throughout: ``gain = nll_none - nll_frozen``, so POSITIVE means
the injection HELPED.  That is the same direction as the project's
``delta = frozen - trained``, which is about training instead of injection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURPRISAL_EDGES = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
EVIDENCE_EDGES = [1, 2, 5, 10, 50, 200, np.inf]


def _label_band(lo: float, hi: float, fmt: str = "{:g}") -> str:
    if np.isinf(hi):
        return f"[{fmt.format(lo)},+)"
    return f"[{fmt.format(lo)},{fmt.format(hi)})"


def cross_table(surprisal, evidence, gain, s_edges=None, e_edges=None) -> dict:
    """Mean gain and counts over a surprisal x evidence grid.

    Returns ``{"rows": [...], "cols": [...], "mean": [[...]], "n": [[...]]}`` with
    ``None`` where a cell is empty, so a caller can print it without guessing which
    cells exist.
    """
    surprisal = np.asarray(surprisal, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.int64)
    gain = np.asarray(gain, dtype=np.float64)
    if not (surprisal.shape == evidence.shape == gain.shape):
        raise ValueError("surprisal, evidence and gain must be parallel")
    s_edges = list(SURPRISAL_EDGES if s_edges is None else s_edges)
    e_edges = list(EVIDENCE_EDGES if e_edges is None else e_edges)

    rows = [_label_band(s_edges[i], s_edges[i + 1], "{:g}") for i in range(len(s_edges) - 1)]
    cols = [_label_band(e_edges[i], e_edges[i + 1], "{:g}") for i in range(len(e_edges) - 1)]
    mean = [[None] * len(cols) for _ in rows]
    n = [[0] * len(cols) for _ in rows]

    for i in range(len(s_edges) - 1):
        for j in range(len(e_edges) - 1):
            m = (
                (surprisal >= s_edges[i])
                & (surprisal < s_edges[i + 1])
                & (evidence >= e_edges[j])
                & (evidence < e_edges[j + 1])
            )
            k = int(m.sum())
            n[i][j] = k
            if k:
                mean[i][j] = float(gain[m].mean())
    return {"rows": rows, "cols": cols, "mean": mean, "n": n}


def gate_curve(surprisal, gain, thresholds=None) -> list[dict]:
    """Total gain if the injection is applied only where the backbone is unsure.

    A gate that reads ``h_t`` does not escape the bound -- the bound is already
    conditional on ``h_t`` -- so suppressing an injection that HURTS is free
    value.  If the gain is negative wherever the backbone is confident, the
    channel is paying for that whole region on every position and buying nothing.

    ``surprisal`` is the backbone's own NLL at the position, which an
    inference-time gate CANNOT see: it costs the realised next token.  So this
    curve is an UPPER BOUND on a realisable gate.  The realisable quantity it
    stands in for is the backbone's predictive entropy at that position, which is
    a function of ``h_t`` alone.
    """
    surprisal = np.asarray(surprisal, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if surprisal.shape != gain.shape:
        raise ValueError("surprisal and gain must be parallel")
    n = gain.size
    grid = thresholds if thresholds is not None else [
        0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
    ]
    out = []
    for tau in grid:
        keep = surprisal >= tau
        out.append({
            "tau": float(tau),
            "kept_frac": float(keep.mean()) if n else None,
            "gain_mean": float(gain[keep].sum() / n) if n else None,
            "gain_kept": float(gain[keep].mean()) if keep.any() else None,
        })
    return out


def concentration(gain, fracs=(0.01, 0.05, 0.10, 0.25, 0.50)) -> dict:
    """How much of the total gain the largest-gain positions carry.

    A channel whose value is spread evenly and a channel whose value is carried by
    1% of positions look identical in a mean; they are completely different as
    engineering targets.
    """
    gain = np.asarray(gain, dtype=np.float64)
    total = float(gain.sum())
    order = np.argsort(gain)[::-1]
    out: dict = {"total": total, "n": int(gain.size), "share": {}}
    for f in fracs:
        k = max(1, round(f * gain.size))
        out["share"][f"{f:.0%}"] = (float(gain[order[:k]].sum() / total) if total else None)
    out["negative_share"] = float((gain < 0).mean())
    out["negative_sum"] = float(gain[gain < 0].sum())
    out["positive_sum"] = float(gain[gain > 0].sum())
    return out


def render(surprisal, evidence, gain) -> str:
    tab = cross_table(surprisal, evidence, gain)
    lines = ["| backbone surprisal \\ row evidence | " + " | ".join(tab["cols"]) + " |",
             "|---" * (len(tab["cols"]) + 1) + "|"]
    for i, row in enumerate(tab["rows"]):
        cells = []
        for j in range(len(tab["cols"])):
            m, k = tab["mean"][i][j], tab["n"][i][j]
            cells.append("--" if m is None else f"{m:+.4f} (n={k:,})")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True, help="*.deltas.npz from round169_eval_rows.py")
    ap.add_argument("--snapshot-dir", required=True, help="holds trigram-train-count.npy")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    with np.load(args.record) as z:
        rec = {k: z[k] for k in z.files}
    count = np.load(Path(args.snapshot_dir) / "trigram-train-count.npy")
    evidence = count[np.asarray(rec["snapshot_index"], dtype=np.int64)]

    none = rec["nll_none"].astype(np.float64)
    frozen = rec["nll_frozen"].astype(np.float64)
    gain = none - frozen

    print(f"# {Path(args.record).stem}")
    print()
    print(f"scored positions: {gain.size:,}")
    print(f"injection gain (none - frozen): {gain.mean():+.5f} nats "
          f"-> total {gain.sum():+.1f} nats over the stream")
    print()
    print("## by backbone surprisal only")
    print()
    print(render(none, np.ones_like(evidence), gain).replace(" | [1,+) |", " | all |"))
    print()
    print("## by row evidence only")
    print()
    print(render(np.zeros_like(none), evidence, gain))
    print()
    print("## crossed")
    print()
    print(render(none, evidence, gain))
    print()

    conc = concentration(gain)
    print("## concentration")
    print()
    for k, v in conc["share"].items():
        print(f"  top {k:>4} of positions carry {v:6.1%} of the total gain")
    print(f"  positions where the injection HURT: {conc['negative_share']:.1%} "
          f"(sum {conc['negative_sum']:+.1f})")
    print(f"  positions where it HELPED:          {1 - conc['negative_share']:.1%} "
          f"(sum {conc['positive_sum']:+.1f})")
    print()
    print("## gate on the backbone's own surprisal (UPPER BOUND on a realisable gate)")
    print()
    curve = gate_curve(none, gain)
    print("| tau | kept | mean gain over ALL positions | mean gain where kept |")
    print("|---|---|---|---|")
    for row in curve:
        print(f"| >= {row['tau']:g} | {row['kept_frac']:.1%} | "
              f"**{row['gain_mean']:+.5f}** | {row['gain_kept']:+.4f} |")

    result = {
        "record": Path(args.record).name,
        "n": int(gain.size),
        "gain_mean": float(gain.mean()),
        "gain_total": float(gain.sum()),
        "by_surprisal": cross_table(none, np.ones_like(evidence), gain),
        "by_evidence": cross_table(np.zeros_like(none), evidence, gain),
        "crossed": cross_table(none, evidence, gain),
        "concentration": conc,
        "gate_curve": curve,
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\n[where] wrote {args.out_json}")
    if args.out_md:
        Path(args.out_md).write_text(render(none, evidence, gain) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
