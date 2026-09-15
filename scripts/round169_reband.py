#!/usr/bin/env python
"""Re-band a B1 evaluation by the frequency of the trigram that was ACTUALLY injected.

Why this exists
---------------
B1's published band table (and round 168's section-4 frequency prediction) bands
each scored position by ``context_counts`` from the margin artifact.  That
variable is not the frequency of the injected row's trigram.  Measured, not
argued -- see ``docs/round-169-band-variable-off-by-one.md``:

    where both are > 0:  context_counts[t] == count(trigram ending at t-1)
                        in 297,998 / 297,998 cases  (100.0%)
    context_counts[t] == 0  <=>  count(trigram ending at t-1) < 1
                        with zero disagreements in either direction

Meanwhile the row injected at stream position ``t`` is the row for the trigram
ENDING AT ``t``: ``codes[i]`` describes the trigram ending at ``i + 2``, the
snapshot is built with ``seen_pos = flatnonzero(hit_all) + 2``, and the eval's
own consistency check (``max|diff| = 0`` against ``fetch_e_t(all_rowids[probe])``)
confirms that ``E[t]`` is the shard table's row for position ``t``.  The graft
feeds the hidden state at ``t`` to predict ``t + 1``, so "ending at ``t``" is the
only self-consistent choice.

So the band variable is one token behind the row it is supposed to describe.  A
position whose PREVIOUS trigram is unseen (``context_counts == 0``) is not a
position that inherited a row it never voted for; its own trigram is in the
snapshot by construction.  ``trigram_codes`` is an injection
(``(a*V + b)*V + c``), not a hash, so there are no collisions to inherit
through either.

The fix is exact and cheap: the per-position record carries ``snapshot_index``,
and ``trigram-train-count.npy`` holds the exact training count of that snapshot
row's trigram.  No re-training and no re-evaluation are needed once the record
exists -- which is why ``round169_eval_rows.py`` now writes it.

Sign convention: ``delta = frozen - trained``; positive means training helped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (lo, hi); the last band is open-ended.  Same edges as the published table, so
# the correction can be read side by side with it.
BANDS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 49), (50, 199), (200, 10**9)]


def masked_prediction(counts: np.ndarray, delta: np.ndarray, threshold: int) -> dict:
    """The aggregate a hard-masked arm must produce, read off the frozen table.

    A hard mask holds every row below ``threshold`` at its frozen ``E0`` value.
    For those positions the eval substitutes the same vector in both arms, so
    ``delta = frozen - trained`` is EXACTLY zero, not approximately.  And because
    ``trigram_codes`` is an injection rather than a hash, each distinct trigram
    owns its own row, so holding the low-count rows cannot change the trajectory
    of the high-count ones -- their gradients never mix.

    So this is not a forecast with error bars: it is an identity.  That is what
    makes the masked arm worth running as a pre-registered test rather than as a
    search -- the number is written down before the GPU starts, and either the run
    reproduces it or the band decomposition is wrong.
    """
    counts = np.asarray(counts, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)
    if counts.shape != delta.shape:
        raise ValueError(f"counts {counts.shape} and delta {delta.shape} must be parallel")
    n = int(delta.size)
    keep = counts >= int(threshold)
    kept = int(keep.sum())
    return {
        "threshold": int(threshold),
        "n_scored": n,
        "n_trained": kept,
        "frac_trained": (kept / n) if n else None,
        "kept_band_delta": float(delta[keep].mean()) if kept else None,
        "predicted_delta": float(delta[keep].sum() / n) if n else None,
        "n_held_frozen": n - kept,
    }


def band_label(lo: int, hi: int) -> str:
    return f"[{lo},{'+' if hi >= 10**9 else hi}]"


def paired_stats(delta: np.ndarray) -> dict:
    """Mean / SE / t of a paired difference (same shape as the eval's own)."""
    n = int(delta.size)
    if n < 2:
        return {"n": n, "mean": float(delta.mean()) if n else None, "se": None, "t": None}
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "se": se, "t": mean / se if se > 0 else None}


def band_table(counts: np.ndarray, delta: np.ndarray, nll_frozen=None, nll_trained=None) -> dict:
    """Mean delta per count band, plus the in-band centre of mass of the counts."""
    counts = np.asarray(counts, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)
    if counts.shape != delta.shape:
        raise ValueError(f"counts {counts.shape} and delta {delta.shape} must be parallel")
    out: dict = {}
    for lo, hi in BANDS:
        m = (counts >= lo) & (counts <= hi)
        if not m.any():
            continue
        entry = {**paired_stats(delta[m]), "count_lo": lo, "count_hi": None if hi >= 10**9 else hi}
        if nll_frozen is not None:
            entry["mean_nll_frozen"] = float(np.asarray(nll_frozen)[m].mean())
        if nll_trained is not None:
            entry["mean_nll_trained"] = float(np.asarray(nll_trained)[m].mean())
        out[band_label(lo, hi)] = entry
    return out


def own_counts(record: dict, train_count: np.ndarray) -> np.ndarray:
    """Exact training count of each scored position's own injected trigram."""
    snap = np.asarray(record["snapshot_index"], dtype=np.int64)
    tc = np.asarray(train_count)
    if snap.size and (snap.min() < 0 or snap.max() >= tc.size):
        raise ValueError(
            f"snapshot_index out of range for a {tc.size}-row count table "
            f"(min {snap.min()}, max {snap.max()})"
        )
    return tc[snap]


def render_markdown(tag: str, own: dict, ctx: dict, extra: dict) -> str:
    lines = [f"## {tag}", ""]
    if "overall" in extra:
        o = extra["overall"]
        lines += [
            f"* scored positions: **{o['n']:,}**",
            f"* overall delta: **{o['mean']:+.5f}** (t = {o['t']:+.2f})",
            (
                f"* own-count minimum: **{extra['own_min']}** "
                "(1 is the floor: a position is scored only if its own trigram is in the snapshot)"
            ),
            (
                f"* positions with the OLD band variable at 0: **{extra['ctx_zero']:,}** "
                f"({extra['ctx_zero_frac']:.1%}) -- an artifact, "
                "not a set of row-inheriting positions"
            ),
            "",
        ]
    for name, table in (("by OWN trigram count (corrected)", own), ("by context_counts (as published)", ctx)):
        lines += [f"### {name}", "", "| band | n | delta | t |", "|---|---|---|---|"]
        for k, v in table.items():
            t = "n/a" if v["t"] is None else f"{v['t']:+.2f}"
            lines.append(f"| {k} | {v['n']:,} | {v['mean']:+.5f} | {t} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True, help="*.deltas.npz written by round169_eval_rows.py")
    ap.add_argument("--snapshot-dir", required=True, help="holds trigram-train-count.npy")
    ap.add_argument("--tag", default=None, help="label for the output (default: record stem)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument(
        "--predict-thresholds", default="",
        help="comma-separated row-count thresholds; prints, for each, the aggregate "
             "a hard-masked arm must produce. This is an identity, not a forecast: "
             "held rows contribute exactly zero and rows do not mix.",
    )
    args = ap.parse_args()

    rec_path = Path(args.record)
    with np.load(rec_path) as z:
        record = {k: z[k] for k in z.files}
    tc = np.load(Path(args.snapshot_dir) / "trigram-train-count.npy")

    delta = record["delta"].astype(np.float64)
    own = own_counts(record, tc)
    tag = args.tag or rec_path.name.replace(".deltas.npz", "")

    own_tab = band_table(own, delta, record.get("nll_frozen"), record.get("nll_trained"))
    ctx_tab = {}
    if "context_count" in record:
        ctx_tab = band_table(
            record["context_count"], delta, record.get("nll_frozen"), record.get("nll_trained")
        )
    extra = {
        "overall": paired_stats(delta),
        "own_min": int(own.min()) if own.size else None,
        "own_mean": float(own.mean()) if own.size else None,
        "ctx_zero": int((record["context_count"] == 0).sum()) if "context_count" in record else 0,
        "ctx_zero_frac": (
            float((record["context_count"] == 0).mean()) if "context_count" in record else 0.0
        ),
    }
    result = {"tag": tag, "record": rec_path.name, "by_own_count": own_tab,
              "by_context_count": ctx_tab, **extra}

    if args.predict_thresholds:
        preds = []
        for tok in args.predict_thresholds.split(","):
            tok = tok.strip()
            if not tok:
                continue
            p = masked_prediction(own, delta, int(tok))
            preds.append(p)
            print(
                f"[mask] threshold >= {p['threshold']:>6}: trains {p['n_trained']:,} / "
                f"{p['n_scored']:,} positions ({p['frac_trained']:.1%}), "
                f"kept-band delta {p['kept_band_delta']:+.5f} "
                f"-> predicted aggregate delta {p['predicted_delta']:+.5f}"
            )
        result["masked_predictions"] = preds
        print()

    print(render_markdown(tag, own_tab, ctx_tab, extra))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[reband] wrote {args.out_json}")
    if args.out_md:
        Path(args.out_md).write_text(render_markdown(tag, own_tab, ctx_tab, extra) + "\n")
        print(f"[reband] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
