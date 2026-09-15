#!/usr/bin/env python
"""The top-1 gate: does the NLL gain reach a decision?

Every number this project has produced is mean negative log-likelihood, and NLL
is a LOG quantity dominated by positions where the model was very wrong.  The
injection's gain is concentrated exactly there: at backbone surprisal >= 8 nats
the frozen rows beat the pure backbone by +1.925 nats per position.  A +1.925 nat
gain means the model put e^1.925 = 6.9x more probability on the realised token --
but at 8 nats the base probability was e^-8 = 3.4e-4, so the realised token is
still the wrong argmax either way.  A 6.9x improvement on a 0.03% probability is
a log-loss improvement and may be nothing else.

So the question is not rhetorical and it cannot be settled by more NLL: does the
injection change WHAT THE MODEL WOULD HAVE SAID?  The eval now records per-position
top-1 for all three arms, so this is one paired test on the positions where the
gain lives.

THE VERDICT RULE IS FIXED BEFORE THE RUN, and it is deliberately harsh:

    GAIN  iff  acc(frozen) - acc(none) > 0  AND  t >= 3   on surprisal >= tau
    FLAT  otherwise

``tau`` defaults to 2.0 nats, which is where the per-band gain first turns
positive ([-0.0420 at [1,2), +0.0645 at [2,4)) and where the oracle surprisal gate
peaks.  A FLAT verdict stops the whole NLL-optimisation line: if a 6.9x
probability improvement on the tail does not move the argmax, then the currency
this project has been optimising is not the currency of any capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("none", "frozen", "trained")


def paired_top1(hit_a: np.ndarray, hit_b: np.ndarray) -> dict:
    """Paired top-1 comparison of two arms on the same positions."""
    a = np.asarray(hit_a, dtype=np.float64)
    b = np.asarray(hit_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"arms are not parallel: {a.shape} vs {b.shape}")
    n = int(a.size)
    if n == 0:
        return {"n": 0, "acc_a": None, "acc_b": None, "delta": None, "se": None, "t": None}
    diff = b - a
    out = {"n": n, "acc_a": float(a.mean()), "acc_b": float(b.mean()), "delta": float(diff.mean())}
    if n >= 2:
        se = float(diff.std(ddof=1) / np.sqrt(n))
        out["se"] = se
        out["t"] = (out["delta"] / se) if se > 0 else None
    else:
        out["se"] = out["t"] = None
    return out


def verdict(delta, t, *, t_min: float = 3.0) -> str:
    """GAIN only on a positive improvement that a paired test separates from noise.

    Two-sided and deliberately blunt: the cost of a false GAIN is that the project
    keeps spending GPU on a currency that does not reach a decision.
    """
    if delta is None or not np.isfinite(delta) or delta <= 0:
        return "FLAT"
    if t is None or not np.isfinite(t) or t < t_min:
        return "FLAT"
    return "GAIN"


def gate(record: dict, tau: float) -> dict:
    """Top-1 for every arm, pooled over the positions where the gain lives."""
    if "nll_none" not in record:
        raise ValueError("record has no nll_none; cannot select the tail")
    surprisal = np.asarray(record["nll_none"], dtype=np.float64)
    tail = surprisal >= float(tau)
    if not tail.any():
        raise ValueError(f"no positions with backbone surprisal >= {tau}")

    arms = {a: np.asarray(record[f"hit_{a}"], dtype=np.float64) for a in ARMS if f"hit_{a}" in record}
    if "none" not in arms or "frozen" not in arms:
        raise ValueError("record is missing hit_none / hit_frozen")

    out = {
        "tau": float(tau),
        "n_total": int(surprisal.size),
        "n_tail": int(tail.sum()),
        "frac_tail": float(tail.mean()),
        "all_positions": {a: float(arms[a].mean()) for a in arms},
        "tail": {},
    }
    for a in arms:
        out["tail"][f"acc_{a}"] = float(arms[a][tail].mean())
    for a in arms:
        if a == "none":
            continue
        out["tail"][a] = paired_top1(arms["none"][tail], arms[a][tail])
    # the headline: does the injection move the argmax where the nats are
    key = out["tail"].get("frozen") or {}
    out["headline"] = {"delta": key.get("delta"), "t": key.get("t")}
    out["verdict"] = verdict(key.get("delta"), key.get("t"))
    return out


def render(g: dict) -> str:
    lines = [
        f"# top-1 gate (tail = backbone surprisal >= {g['tau']:g} nats)",
        "",
        (
            f"tail holds **{g['n_tail']:,}** of {g['n_total']:,} scored positions "
            f"({g['frac_tail']:.1%})"
        ),
        "",
        "| arm | top-1 (all) | top-1 (tail) |",
        "|---|---|---|",
    ]
    for a in ARMS:
        if a in g["all_positions"]:
            lines.append(f"| {a} | {g['all_positions'][a]:.4f} | {g['tail'][f'acc_{a}']:.4f} |")
    lines += ["", "| comparison | delta (top-1) | SE | t |", "|---|---|---|---|"]
    for a in ("frozen", "trained"):
        c = g["tail"].get(a)
        if not c:
            continue
        t = "n/a" if c["t"] is None else f"{c['t']:+.2f}"
        se = "n/a" if c["se"] is None else f"{c['se']:.5f}"
        lines.append(f"| {a} − none | {c['delta']:+.5f} | {se} | {t} |")
    lines += ["", f"**VERDICT: {g['verdict']}**", ""]
    if g["verdict"] == "FLAT":
        lines.append(
            "> The injection's NLL gain does not move the argmax where the gain is.\n"
            "> On this rule that stops the NLL-optimisation line, not the project."
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--t-min", type=float, default=3.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    with np.load(args.record) as z:
        rec = {k: z[k] for k in z.files}
    g = gate(rec, args.tau)
    g["t_min"] = args.t_min
    g["verdict"] = verdict(g["headline"]["delta"], g["headline"]["t"], t_min=args.t_min)
    text = render(g)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        Path(args.out_md).write_text(text + "\n")
    for f in (args.out_json, args.out_md):
        if f:
            print(f"[gate] wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
