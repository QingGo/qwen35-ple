#!/usr/bin/env python
"""Can a DEPLOYABLE gate recover any of the oracle's 2.02x?

The memory's value is being left on the table.  Injecting the frozen rows
everywhere buys +0.175931 nats; injecting only at the positions where it helps
buys +0.355047, which is 2.02x.  The whole gap is "not knowing when to trust the
memory".

Every gate variable this project has measured so far fails one of two ways:

  * **oracle.**  Surprisal of the realised token needs the answer to be known
    before it can be computed, so the tau=2.25 surprisal gate (1.18x) cannot be
    run at inference at all.
  * **worthless.**  The row's training count is observable by table lookup, and
    measured at <= 1.00x at every threshold -- because the injection's gain is
    POSITIVE in every count band, including trigrams seen exactly once (+0.0949).

This tool asks the question those two left open, using the model's own
confidences as features.  They exist at the moment the decision has to be made.

THE PROTOCOL IS THE WHOLE ARGUMENT, so it is fixed here and not chosen per run:

  * **Folds are contiguous blocks of the stream, never a random split.**  The
    coupling audit (docs/round-169-coupling-mechanism.md) showed a changed row
    perturbs every later position IN ITS 1024-TOKEN CHUNK, with |delta| decaying
    monotonically in token distance.  Adjacent positions therefore share both
    their features and their labels.  A random split would train on a position
    and test on its neighbour and report a multiplier that means nothing.
  * **Standardisation statistics come from the training folds only.**
  * **The reported number is the held-out one.**  The train/test gap is printed
    next to it, because a gap is the first thing a leak looks like.

`ORACLE_FIELDS` is enforced, not documented: a gate that sees `nll_none` can
predict its own label, and a study that leaks is worse than no study, because it
reads like a result.

Pure functions above ``main``; IO only in ``main``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# A field is oracle if computing it requires knowing the token that was actually
# next.  `nll_*` is the surprisal of that token and `delta` is a difference of
# two of them.  `hit_*` is the same leak wearing a boolean.
ORACLE_FIELDS = frozenset(
    {"nll_none", "nll_frozen", "nll_trained", "delta", "hit_none", "hit_frozen", "hit_trained"}
)

# Every feature here is computable before the next token is emitted:
#   *_none / *_frozen  the two arms' output distributions (both forwards are run
#                      anyway when the memory is used)
#   ent_shift          did the memory sharpen or flatten the distribution
#   agree              do the two arms pick the same argmax
#   context_count      the trigram's count in the eval stream, a table lookup
#   has_train_row      whether this trigram has an entry that training could move
DEFAULT_FEATURES = (
    "ent_none",
    "maxp_none",
    "logmargin_none",
    "ent_frozen",
    "maxp_frozen",
    "logmargin_frozen",
    "ent_shift",
    "agree",
    "context_count",
    "has_train_row",
)


def observable_features(rec: dict, names=DEFAULT_FEATURES) -> tuple[np.ndarray, list[str]]:
    """Build the gate's feature matrix, refusing anything that leaks the answer.

    The refusal is the point.  `nll_none > 2` predicts `nll_none - nll_frozen > 0`
    extremely well and is also, by construction, unavailable at inference -- a
    gate built on it would look like the best result in the project and be
    unrunnable.
    """
    names = list(names)
    bad = sorted(set(names) & ORACLE_FIELDS)
    if bad:
        raise ValueError(
            f"these features are oracle (they need the realised token): {bad}. "
            "A gate may not use them."
        )
    cols = []
    for nm in names:
        if nm == "ent_shift":
            if "ent_none" not in rec or "ent_frozen" not in rec:
                raise ValueError("ent_shift needs ent_none and ent_frozen")
            col = np.asarray(rec["ent_none"], dtype=np.float64) - np.asarray(
                rec["ent_frozen"], dtype=np.float64
            )
        elif nm == "agree":
            if "top1_none" not in rec or "top1_frozen" not in rec:
                raise ValueError("agree needs top1_none and top1_frozen")
            col = (
                np.asarray(rec["top1_none"]) == np.asarray(rec["top1_frozen"])
            ).astype(np.float64)
        else:
            if nm not in rec:
                raise ValueError(f"the record has no field {nm!r}")
            col = np.asarray(rec[nm], dtype=np.float64)
        if col.ndim != 1:
            raise ValueError(f"feature {nm!r} is not 1-D (shape {col.shape})")
        cols.append(col)
    if not cols:
        raise ValueError("no features requested")
    n = cols[0].size
    for nm, col in zip(names, cols):
        if col.size != n:
            raise ValueError(f"feature {nm!r} has {col.size} values, expected {n}")
    X = np.column_stack(cols)
    if not np.isfinite(X).all():
        bad_cols = [nm for nm, c in zip(names, cols) if not np.isfinite(c).all()]
        raise ValueError(f"non-finite feature values in {bad_cols}")
    return X, names


def stream_folds(score: np.ndarray, k: int = 10) -> np.ndarray:
    """Assign each position to one of k CONTIGUOUS blocks of the stream.

    Contiguity is a correctness requirement, not a preference: positions inside a
    1024-token chunk share the perturbation from any changed row in that chunk, so
    neighbouring positions have correlated features AND correlated labels.
    """
    score = np.asarray(score)
    if score.ndim != 1 or score.size == 0:
        raise ValueError("score must be a non-empty 1-D array")
    if k < 2:
        raise ValueError("need at least 2 folds")
    order = np.argsort(score, kind="stable")
    rank = np.empty(score.size, dtype=np.int64)
    rank[order] = np.arange(score.size)
    return np.minimum(rank * k // score.size, k - 1)


def gate_value(gain: np.ndarray, decision: np.ndarray, total: int | None = None) -> float:
    """Mean nats recovered by injecting exactly where ``decision`` is true.

    Using the memory emits ``nll_frozen``; not using it emits ``nll_none``; so the
    improvement over the pure backbone is the sum of ``gain`` over the positions
    where the gate says yes, divided by the number of scored positions.
    """
    gain = np.asarray(gain, dtype=np.float64)
    decision = np.asarray(decision, dtype=bool)
    if gain.shape != decision.shape:
        raise ValueError("gain and decision must have the same length")
    n = int(total if total is not None else gain.size)
    if n <= 0:
        raise ValueError("total must be positive")
    return float(gain[decision].sum() / n)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using the TRAIN fold's statistics only."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train - mu) / sd, (other - mu) / sd


def logistic_fit(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1e-3, iters: int = 250, lr: float = 0.5
) -> np.ndarray:
    """L2 logistic regression by gradient descent, on standardised features.

    No sklearn in this environment, and a 12-line optimiser that can be read is
    the right size for a study whose conclusion is a ratio.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and y must be (n,)")
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    w = np.zeros(Xb.shape[1])
    n = Xb.shape[0]
    reg = np.ones_like(w)
    reg[0] = 0.0  # never penalise the bias
    for _ in range(iters):
        z = np.clip(Xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y) / n + l2 * reg * w
        if np.abs(g).max() < 1e-9:
            break
        w -= lr * g
    return w


def ridge_fit(X: np.ndarray, gain: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    """Least squares of the GAIN ITSELF, not of its sign.

    This distinction is the whole reason this function exists.  The logistic gate
    above predicts ``gain > 0`` and decides at 0.5, which maximises CLASSIFICATION
    ACCURACY -- and that is not the quantity the project cares about.  What matters
    is the SUM of the gain over the positions where the gate says yes.  A position
    with a 45% chance of a +3.0 nat gain and a 55% chance of a -1.0 nat loss has
    positive expected value and a classifier will confidently discard it.

    Solving for the conditional mean and injecting when it is positive is the
    decision rule that matches the objective.

    Normal equations: with d ~ 12 features the 12x12 Gram matrix is trivial even
    at a million rows, so no iterative solver is needed.
    """
    X = np.asarray(X, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if X.ndim != 2 or gain.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and gain must be (n,)")
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    reg = np.eye(Xb.shape[1]) * l2
    reg[0, 0] = 0.0  # never penalise the bias
    gram = Xb.T @ Xb + reg
    return np.linalg.solve(gram, Xb.T @ gain)


def ridge_apply(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The predicted gain, so the caller can compare it against zero."""
    Xb = np.column_stack([np.ones(np.asarray(X).shape[0]), np.asarray(X, dtype=np.float64)])
    return Xb @ w


def logistic_apply(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(np.asarray(X).shape[0]), np.asarray(X, dtype=np.float64)])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30.0, 30.0)))


def best_threshold(
    feature: np.ndarray, gain: np.ndarray, train: np.ndarray, *, higher_is_better: bool = True
) -> tuple[float, float]:
    """The single-feature threshold that recovers the most nats on the TRAIN fold.

    Evaluated by one sort and a suffix sum rather than a grid of thresholds.  The
    grid version cost 220 billion comparisons at this record's size (11 features x
    10 folds x 200 thresholds x 1.0M rows), which is the difference between a study
    that runs in twenty seconds and one nobody re-runs.  The optimum of
    ``mean(gain[f >= t])`` over t is attained at one of the observed values of f,
    so the sort loses nothing.

    Ties are handled by cutting at the FIRST occurrence of each distinct value:
    ``f >= t`` keeps the whole tie group, so a cut inside a group is not a
    different rule and must not be scored as one.
    """
    f = np.asarray(feature, dtype=np.float64)
    g = np.asarray(gain, dtype=np.float64)
    if f.shape != g.shape:
        raise ValueError("feature and gain must be the same length")
    train = np.asarray(train, dtype=bool)
    f, g = f[train], g[train]
    if f.size == 0:
        raise ValueError("empty training fold")
    if not higher_is_better:
        t, val = best_threshold(-f, g, np.ones(f.size, dtype=bool))
        return -t, val
    order = np.argsort(f, kind="stable")
    fs, gs = f[order], g[order]
    suffix = np.concatenate([np.cumsum(gs[::-1])[::-1], [0.0]])
    first = np.ones(fs.size, dtype=bool)
    first[1:] = fs[1:] != fs[:-1]
    cuts = np.flatnonzero(first)
    vals = suffix[cuts] / fs.size
    best = int(np.argmax(vals))
    return float(fs[cuts[best]]), float(vals[best])


def two_regime_report(
    gain: np.ndarray, has_train_row: np.ndarray, decision: np.ndarray | None = None
) -> dict:
    """Split the injection's value by whether the trigram was seen in training.

    Every previous eval scored only the `True` side (39% of the stream), because
    `score` was restricted to positions with an E0 row.  The other 61% still
    receive a shard-table row and still feed the autoregressive state; they were
    simply never scored.
    """
    gain = np.asarray(gain, dtype=np.float64)
    row = np.asarray(has_train_row, dtype=bool)
    if gain.shape != row.shape:
        raise ValueError("gain and has_train_row must have the same length")
    out = {}
    for name, m in (("seen", row), ("unseen", ~row)):
        d = decision[m] if decision is not None else np.ones(int(m.sum()), dtype=bool)
        gg = gain[m]
        out[name] = {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "gain": float(gg.mean()) if gg.size else 0.0,
            "total": float(gg.sum()),
            "gated_total": float(gg[d].sum()) if gg.size else 0.0,
            "frac_positive": float((gg > 0).mean()) if gg.size else 0.0,
        }
    return out


def render(study: dict) -> str:
    o: list[str] = []
    w = o.append
    g = study
    w(f"positions {g['n']:,}   features {len(g['features'])}   folds {g['k_folds']} (contiguous)")
    u = g["unconditional"]
    w("")
    w("=== baselines ===")
    w(f"  unconditional inject everywhere      {u:+.6f}   1.00x")
    w(f"  oracle sign (upper bound, unrunnable) {g['oracle_sign']:+.6f}   "
      f"{g['oracle_sign'] / u:.2f}x")
    w(f"  oracle surprisal tau={g['oracle_tau']:g} (unrunnable)   {g['oracle_surprisal']:+.6f}   "
      f"{g['oracle_surprisal'] / u:.2f}x")
    w("")
    w("=== DEPLOYABLE gates, held out on contiguous folds ===")
    w(f"  {'gate':<28} {'keep':>7} {'test gain':>12} {'mult':>7} {'train mult':>11}")
    for row in g["gates"]:
        w(f"  {row['name']:<28} {row['keep']:>6.1%} {row['test_value']:>+12.6f} "
          f"{row['test_multiple']:>6.2f}x {row['train_multiple']:>10.2f}x")
    w("")
    w("=== by regime ===")
    for name, r in g["regimes"].items():
        w(f"  {name:<7} n {r['n']:>9,} ({r['share']:>5.1%})  gain {r['gain']:>+9.6f}  "
          f"total {r['total']:>+10.0f}  frac positive {r['frac_positive']:.3f}")
    if g.get("note"):
        w("")
        w(g["note"])
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate-record", required=True, type=Path,
                    help="the all-positions record written by --gate-record")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--k-folds", type=int, default=10)
    ap.add_argument("--max-fit-rows", type=int, default=300_000,
                    help="subsample the training folds when fitting the logistic gate. "
                         "11 features do not need a million rows, and the held-out "
                         "evaluation is unaffected because it uses every test row.")
    ap.add_argument("--oracle-tau", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.gate_record)
    rec = {k: z[k] for k in z.files}
    for need in ("score", "nll_none", "nll_frozen", "has_train_row"):
        if need not in rec:
            raise SystemExit(f"the gate record has no {need!r}")
    score = np.asarray(rec["score"], dtype=np.int64)
    gain = np.asarray(rec["nll_none"], dtype=np.float64) - np.asarray(
        rec["nll_frozen"], dtype=np.float64
    )
    has_row = np.asarray(rec["has_train_row"], dtype=bool)
    names = [s for s in args.features.split(",") if s]
    X, names = observable_features(rec, names)
    folds = stream_folds(score, args.k_folds)

    uncond = float(gain.mean())
    oracle_sign = float(np.abs(gain).sum() / gain.size)
    tail = np.asarray(rec["nll_none"], dtype=np.float64) >= args.oracle_tau
    oracle_surprisal = gate_value(gain, tail)

    # ---- held-out evaluation of every gate --------------------------------- #
    gates: list[dict] = []

    # Each single feature gets its threshold chosen on the training folds of every
    # fold and applied to the held-out one, so the reported keep-rate and value are
    # both out-of-sample.
    for i, nm in enumerate(names):
        col = X[:, i]
        test_dec = np.zeros(gain.size, dtype=bool)
        train_num = 0.0
        train_den = 0
        for k in range(args.k_folds):
            te = folds == k
            tr = ~te
            t, _ = best_threshold(col, gain, tr)
            dec = col >= t
            test_dec |= dec & te
            train_num += float(gain[tr & dec].sum())
            train_den += int(tr.sum())
        gates.append({
            "name": f"threshold: {nm}",
            "keep": float(test_dec.mean()),
            "test_value": gate_value(gain, test_dec),
            "train_multiple": (train_num / max(train_den, 1)) / uncond if uncond else float("nan"),
            "test_multiple": gate_value(gain, test_dec) / uncond if uncond else float("nan"),
            "kind": "single-feature threshold",
        })

    # ---- the learned gate --------------------------------------------------- #
    test_dec = np.zeros(gain.size, dtype=bool)
    train_vals = []
    coefs = []
    y = (gain > 0).astype(np.float64)
    for k in range(args.k_folds):
        te = folds == k
        tr = ~te
        Xtr, Xte = standardize(X[tr], X[te])
        if Xtr.shape[0] > args.max_fit_rows:
            pick = np.random.default_rng(0).choice(Xtr.shape[0], args.max_fit_rows, replace=False)
            fit_X, fit_y = Xtr[pick], y[tr][pick]
        else:
            fit_X, fit_y = Xtr, y[tr]
        w = logistic_fit(fit_X, fit_y)
        dec_te = logistic_apply(w, Xte) >= 0.5
        test_dec[te] = dec_te
        train_vals.append(float(gain[tr][logistic_apply(w, Xtr) >= 0.5].sum() / max(tr.sum(), 1)))
        coefs.append(w[1:])
    gates.append({
        "name": "logistic P(gain>0) >= 0.5",
        "keep": float(test_dec.mean()),
        "test_value": gate_value(gain, test_dec),
        "train_multiple": float(np.mean(train_vals)) / uncond if uncond else float("nan"),
        "test_multiple": gate_value(gain, test_dec) / uncond if uncond else float("nan"),
        "kind": "classification objective",
    })

    # ---- the VALUE gate: inject where the predicted gain is positive --------- #
    # Same features, same folds, same standardisation; only the objective differs.
    # This is the rule that matches what the project is trying to maximise.
    test_dec = np.zeros(gain.size, dtype=bool)
    train_vals = []
    ridge_coefs = []
    for k in range(args.k_folds):
        te = folds == k
        tr = ~te
        Xtr, Xte = standardize(X[tr], X[te])
        if Xtr.shape[0] > args.max_fit_rows:
            pick = np.random.default_rng(0).choice(Xtr.shape[0], args.max_fit_rows, replace=False)
            fit_X, fit_g = Xtr[pick], gain[tr][pick]
        else:
            fit_X, fit_g = Xtr, gain[tr]
        w = ridge_fit(fit_X, fit_g)
        test_dec[te] = ridge_apply(w, Xte) > 0.0
        train_vals.append(
            float(gain[tr][ridge_apply(w, Xtr) > 0.0].sum() / max(tr.sum(), 1))
        )
        ridge_coefs.append(w[1:])
    gates.append({
        "name": "ridge E[gain] > 0 (value)",
        "keep": float(test_dec.mean()),
        "test_value": gate_value(gain, test_dec),
        "train_multiple": float(np.mean(train_vals)) / uncond if uncond else float("nan"),
        "test_multiple": gate_value(gain, test_dec) / uncond if uncond else float("nan"),
        "kind": "value objective",
    })
    study_coefs = {nm: float(np.mean([c[i] for c in ridge_coefs]))
                   for i, nm in enumerate(names)}
    gates.sort(key=lambda r: -r["test_multiple"])

    study = {
        "record": args.gate_record.name,
        "n": int(gain.size),
        "features": names,
        "k_folds": args.k_folds,
        "unconditional": uncond,
        "oracle_sign": oracle_sign,
        "oracle_surprisal": oracle_surprisal,
        "oracle_tau": args.oracle_tau,
        "gates": gates,
        "regimes": two_regime_report(gain, has_row),
        "logistic_coef_mean": {nm: float(np.mean([c[i] for c in coefs]))
                               for i, nm in enumerate(names)},
        "ridge_coef_mean": study_coefs,
        "note": (
            "Every gate above uses only variables available before the next token is "
            "emitted. The two oracle rows are shown for scale, not as candidates: "
            "surprisal of the realised token cannot be computed at inference, which is "
            "why the tau gate was never runnable."
        ),
    }
    print(render(study))
    if args.out:
        args.out.write_text(json.dumps(study, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
