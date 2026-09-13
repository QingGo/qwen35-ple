"""Stage 1.5e: does the trigram carry information the backbone does not?

Why this module exists
----------------------
Stage 1.5c measures

    margin(t) = L_bb(t) - L_cnt(t)

which is the value of a **prediction-replacing** memory (read the trigram's
answer out and use it).  But the quantity the ceiling argument actually cares
about is the *complementary* one:

    I(Y ; w_t | h_t) = H(Y | h_t) - H(Y | h_t, w_t)

and those two are not the same thing.  ``margin`` can be negative everywhere
while ``I(Y;w_t|h_t)`` is large: a predictor that is individually much worse can
still know something the backbone does not.  Closing the line on a negative
``margin`` alone would therefore be a category error -- the same mistake as the
forty rounds that asked a reader to *reproduce* the backbone's prediction
instead of to *add* to it.

The instrument here is the classical held-out interpolation weight (Jelinek &
Mercer): form the proper mixture

    p_mix(w) = lam * p_bb(w) + (1 - lam) * p_cnt(w)

which needs **no partition function**, so it is computable from the two scalar
probabilities at the observed token -- i.e. from the two NLL arrays that Stage
1.5c already stores.  ``lam`` is fitted out-of-fold, and the whole procedure is
repeated with ``p_cnt`` permuted as the null.

The shrinkage trap (why the naive reading of a mixture gain is wrong)
--------------------------------------------------------------------
Because the mixture is linear in probability *space*,

    E[p_mix(y)] = lam * E[p_bb(y)] + (1 - lam) * E[p_cnt(y)]

exactly, whatever the joint distribution is.  So a gain in **mean NLL** cannot
come from the marginals: it can only come from the *shape* of the joint, through
the concavity of ``log``.  The dominant such effect is **recalibration**: if the
backbone's logit-lens distribution is overconfident (many positions at 1e-3 and
a few at 0.9), then mixing it with *anything* moderate raises the small values
and buys a large Jensen gain -- with no information transfer at all.

A first draft of this module was going to call that "complementarity".  It is
not.  The control is :func:`permuted_gain`: permuting ``p_cnt`` preserves its
marginal and destroys the position-level pairing, so the shrinkage part of the
gain survives permutation and the complementary part does not.  Concretely:

* ``p_cnt`` constant          -> large raw gain, **zero** excess over the null
  (pure recalibration).  Tested in ``test_constant_counter_distribution_is_shrinkage_only``.
* ``p_cnt`` rescues exactly the positions where ``p_bb`` collapses
  -> large raw gain **and** a positive excess.  Tested in
  ``test_rescue_structure_is_detected``.

The null is deliberately *conservative*: mixing with an independent copy of
``p_cnt`` is an easier task than mixing with a correlated one, so when the two
models agree (the usual case) the real excess can be negative even though some
complementarity exists.  A negative excess therefore means "no complementarity
beyond independent recalibration", not "no complementarity at all".

What each outcome licenses
--------------------------
* ``gain`` clearly above the permuted null  ->  complementarity beyond
  recalibration; the replacement framing was the problem, and the read-out's
  target becomes "reproduce this mixture", a concrete checkable objective.
* ``gain`` at or below the null  ->  the *linear-mixture* family adds nothing
  that recalibration alone does not.  This is stronger evidence than ``margin``
  alone (a fitted weight, a null, and a family upper bound), but it is still not
  a proof: a log-linear (product) family is strictly more powerful than its
  linear counterpart, and the pre-registration says so.

The primary number is therefore :func:`oracle_mixture_gain`: the per-position
best weight bounds what *any* weight schedule could do, so if even the oracle
cannot move the NLL the whole family is dead, and if the oracle can move it but
the fitted weight cannot, the missing ingredient is a *predictor of when to
trust the memory* -- which is a much more actionable finding than a null.

Per-group weights are supported because the design document's hypothesis is
that the useful region is a *narrow band* in n-gram frequency
(``docs/round-167-design-corpus-selection-by-marginal-advantage.md`` section 4):
one global ``lam`` averages that band away.

Torch-free and deterministic: unit-tested in ``tests/test_fusion_probe.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "COMPLEMENTARITY_ABSENT_NATS",
    "COMPLEMENTARITY_CONFIRMED_NATS",
    "LAMBDA_GRID",
    "NULL_MARGIN_NATS",
    "FusionResult",
    "cross_fitted_gain",
    "fit_mixture_weight",
    "mixture_nll",
    "nll_from_probs",
    "oracle_mixture_gain",
    "permuted_gain",
    "verdict_from_gain",
]

#: ``eps`` floor so a zero probability gives a finite (huge) NLL instead of inf.
EPS: float = 1e-12

#: Grid for the mixture weight.  Fine enough that the fitted value is not
#: quantised: the objective is smooth and unimodal in ``lam``.
LAMBDA_GRID: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 1.0, 1001))

#: Pre-registered thresholds (docs/round-168-stage1.5e-preregistration.md).
#: The project's measured graft effect sizes are 0.001-0.01 nats, so 0.02 nats
#: is above anything we have ever observed from a graft, and 0.005 is at the
#: top of that band.
COMPLEMENTARITY_CONFIRMED_NATS: float = 0.02
COMPLEMENTARITY_ABSENT_NATS: float = 0.005
#: The gain must beat the permuted-pairing null by this much.
NULL_MARGIN_NATS: float = 0.01


def nll_from_probs(probs: np.ndarray) -> np.ndarray:
    """``-log p`` with a floor, so a zero probability is finite."""
    p = np.asarray(probs, dtype=np.float64)
    return -np.log(np.maximum(p, EPS))


def mixture_nll(
    p_bb: np.ndarray, p_cnt: np.ndarray, lam: float,
) -> np.ndarray:
    """Per-position NLL of ``lam * p_bb + (1 - lam) * p_cnt`` at the true token.

    Both inputs are the probability each model assigns to the *observed* next
    token, so the mixture is a proper distribution and its probability of that
    token is the mixture of the two scalars -- no partition function, no full
    vocabulary pass.
    """
    bb = np.asarray(p_bb, dtype=np.float64)
    cnt = np.asarray(p_cnt, dtype=np.float64)
    if bb.shape != cnt.shape:
        raise ValueError(f"shape mismatch: p_bb {bb.shape} vs p_cnt {cnt.shape}")
    if bb.ndim != 1:
        raise ValueError(f"expected 1-D per-position probabilities, got {bb.ndim}-D")
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return nll_from_probs(lam * bb + (1.0 - lam) * cnt)


def fit_mixture_weight(
    p_bb: np.ndarray, p_cnt: np.ndarray,
    *, grid: Sequence[float] = LAMBDA_GRID,
) -> float:
    """``lam`` minimising the mean mixture NLL on the supplied positions."""
    bb = np.asarray(p_bb, dtype=np.float64)
    cnt = np.asarray(p_cnt, dtype=np.float64)
    if bb.size == 0:
        raise ValueError("cannot fit a mixture weight on zero positions")
    best_lam = 1.0
    best = float("inf")
    for lam in grid:
        value = float(mixture_nll(bb, cnt, lam).mean())
        if value < best:
            best = value
            best_lam = float(lam)
    return best_lam


def _fold_ids(n: int, folds: int) -> np.ndarray:
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    return np.arange(n, dtype=np.int64) % folds


@dataclass(frozen=True)
class FusionResult:
    """Out-of-fold mixture result, globally and optionally per group."""

    n: int
    base_nll: float
    fitted_nll: float
    gain: float
    lam: float
    folds: int
    per_group: dict[str, dict[str, float]] = field(default_factory=dict)
    pooled_gain: float = 0.0
    #: For a permuted null: how many permutations were averaged, and the spread
    #: of their gains.  A single permutation is not enough -- its finite-sample
    #: noise is the same order as the pre-registered thresholds.
    n_perm: int = 1
    null_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "base_nll": self.base_nll,
            "fitted_nll": self.fitted_nll,
            "gain": self.gain,
            "lam": self.lam,
            "folds": self.folds,
            "pooled_gain": self.pooled_gain,
            "per_group": self.per_group,
            "n_perm": self.n_perm,
            "null_std": self.null_std,
        }


def cross_fitted_gain(
    p_bb: np.ndarray, p_cnt: np.ndarray,
    *, groups: np.ndarray | None = None, folds: int = 2,
    grid: Sequence[float] = LAMBDA_GRID,
) -> FusionResult:
    """Fit ``lam`` on one fold, evaluate on the other; repeat, then pool.

    With ``groups`` supplied, a separate weight is fitted per group (still
    out-of-fold), which is what turns a global "does it help on average"
    question into a localised "where does it help" one.
    """
    bb = np.asarray(p_bb, dtype=np.float64)
    cnt = np.asarray(p_cnt, dtype=np.float64)
    if bb.shape != cnt.shape:
        raise ValueError(f"shape mismatch: p_bb {bb.shape} vs p_cnt {cnt.shape}")
    n = int(bb.size)
    if n == 0:
        raise ValueError("no positions")
    fold = _fold_ids(n, folds)
    base_nll = float(nll_from_probs(bb).mean())

    def _classes() -> list[tuple[str, np.ndarray]]:
        if groups is None:
            return [("all", np.ones(n, dtype=bool))]
        g = np.asarray(groups)
        if g.shape != bb.shape:
            raise ValueError(f"groups shape {g.shape} != p_bb {bb.shape}")
        return [(str(v), g == v) for v in np.unique(g)]

    fitted = np.empty(n, dtype=np.float64)
    weight_of: dict[str, list[float]] = {}
    for name, mask in _classes():
        weights: list[float] = []
        for f in range(folds):
            train = mask & (fold != f)
            test = mask & (fold == f)
            if not train.any():
                lam = 1.0
            else:
                lam = fit_mixture_weight(bb[train], cnt[train], grid=grid)
            weights.append(lam)
            if test.any():
                fitted[test] = mixture_nll(bb[test], cnt[test], lam)
        weight_of[name] = weights

    overall_gain = base_nll - float(fitted.mean())
    per_group: dict[str, dict[str, float]] = {}
    pooled_num = 0.0
    for name, mask in _classes():
        if not mask.any():
            continue
        g_base = float(nll_from_probs(bb[mask]).mean())
        g_fit = float(fitted[mask].mean())
        per_group[name] = {
            "n": int(mask.sum()),
            "base_nll": g_base,
            "fitted_nll": g_fit,
            "gain": g_base - g_fit,
            "lam_mean": float(np.mean(weight_of[name])),
        }
        pooled_num += g_base - g_fit
    pooled_gain = pooled_num / max(len(per_group), 1)

    return FusionResult(
        n=n,
        base_nll=base_nll,
        fitted_nll=float(fitted.mean()),
        # ``gain`` here is the oracles-free global number only when there is one
        # group; with groups it is the same quantity (mean over all positions),
        # which is what a reader should quote.  ``pooled_gain`` averages the
        # per-group gains equally, so a group cannot hide behind a large one.
        gain=overall_gain,
        lam=float(np.mean([w for ws in weight_of.values() for w in ws])),
        folds=folds,
        per_group=per_group,
        pooled_gain=pooled_gain,
    )


def _permute_within_groups(
    cnt: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator,
) -> np.ndarray:
    """Permute ``cnt``, but only *within* each group.

    Permuting globally would move a value from one group to another, so a group
    whose counter is constant would stop being constant under the null and
    acquire a spurious excess.  The null must preserve each group's marginal.
    """
    out = cnt.copy()
    if groups is None:
        return rng.permutation(out)
    g = np.asarray(groups)
    for value in np.unique(g):
        mask = g == value
        out[mask] = rng.permutation(out[mask])
    return out


def permuted_gain(
    p_bb: np.ndarray, p_cnt: np.ndarray,
    *, groups: np.ndarray | None = None, folds: int = 2,
    grid: Sequence[float] = LAMBDA_GRID, seed: int = 0, n_perm: int = 8,
) -> FusionResult:
    """The same procedure with ``p_cnt`` permuted: the pairing-breaking null.

    A real gain must exceed this, otherwise the fitted weight is just shrinking
    toward whichever model happens to have the flatter marginal.

    ``n_perm`` permutations are averaged.  One is not enough: the null is itself
    a finite-sample quantity, and with 20k positions its spread is the same
    order as the pre-registered thresholds -- a single permutation produced a
    spurious "MARGINAL" verdict in this module's own test suite.  ``null_std``
    reports the spread that averaging removed.
    """
    bb = np.asarray(p_bb, dtype=np.float64)
    cnt = np.asarray(p_cnt, dtype=np.float64)
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    rng = np.random.default_rng(seed)
    results = [
        cross_fitted_gain(
            bb, _permute_within_groups(cnt, groups, rng),
            groups=groups, folds=folds, grid=grid,
        )
        for _ in range(n_perm)
    ]
    gains = np.array([r.gain for r in results], dtype=np.float64)
    per_group: dict[str, dict[str, float]] = {}
    for name in results[0].per_group:
        per_group[name] = {
            key: float(np.mean([r.per_group[name][key] for r in results]))
            for key in results[0].per_group[name]
        }
    return FusionResult(
        n=results[0].n,
        base_nll=float(np.mean([r.base_nll for r in results])),
        fitted_nll=float(np.mean([r.fitted_nll for r in results])),
        gain=float(gains.mean()),
        lam=float(np.mean([r.lam for r in results])),
        folds=folds,
        per_group=per_group,
        pooled_gain=float(np.mean([r.pooled_gain for r in results])),
        n_perm=n_perm,
        null_std=float(gains.std(ddof=0)),
    )


def oracle_mixture_gain(
    p_bb: np.ndarray, p_cnt: np.ndarray,
    *, grid: Sequence[float] = LAMBDA_GRID,
) -> dict[str, float]:
    """Per-position best ``lam`` (oracle) -- the family's upper bound.

    Uses the true token to pick ``lam`` per position, so it is not achievable;
    it bounds what *any* mixture weight schedule could do.
    """
    bb = np.asarray(p_bb, dtype=np.float64)
    cnt = np.asarray(p_cnt, dtype=np.float64)
    best = np.full(bb.shape, np.inf, dtype=np.float64)
    best_lam = np.zeros(bb.shape, dtype=np.float64)
    for lam in grid:
        cur = mixture_nll(bb, cnt, lam)
        better = cur < best
        best = np.where(better, cur, best)
        best_lam = np.where(better, lam, best_lam)
    base = float(nll_from_probs(bb).mean())
    return {
        "n": float(bb.size),
        "base_nll": base,
        "oracle_nll": float(best.mean()),
        "gain": base - float(best.mean()),
        "mean_oracle_lam": float(best_lam.mean()),
        "share_lam_lt_one": float(np.count_nonzero(best_lam < 1.0) / max(bb.size, 1)),
    }


def verdict_from_gain(
    gain: float, null_gain: float, *,
    null_std: float = 0.0, n_perm: int = 1,
    confirmed: float = COMPLEMENTARITY_CONFIRMED_NATS,
    absent: float = COMPLEMENTARITY_ABSENT_NATS,
    null_margin: float = NULL_MARGIN_NATS,
) -> dict[str, object]:
    """Pre-registered decision rule for Stage 1.5e.

    The decision quantity is the **excess over the permuted null**, not the raw
    gain.  This is the fix for the shrinkage trap: in this module's own test
    suite a count model that was pure independent noise showed a raw gain of
    1.32 nats (recalibrating an overconfident backbone) and would have been
    labelled MARGINAL by a rule keyed on the raw gain.  Only the excess carries
    information about the *pairing*, i.e. about complementarity.

    The excess must additionally clear ``3`` standard errors of the null, so
    that averaging more permutations is rewarded and a one-permutation run
    cannot produce a confident verdict from noise.
    """
    excess = float(gain) - float(null_gain)
    se = float(null_std) / max(float(n_perm), 1.0) ** 0.5
    noise_floor = 3.0 * se
    detail = {
        "gain": float(gain),
        "null_gain": float(null_gain),
        "excess": excess,
        "null_std": float(null_std),
        "null_se": se,
        "noise_floor": noise_floor,
        "thresholds": {
            "confirmed_nats": confirmed,
            "absent_nats": absent,
            "null_margin_nats": null_margin,
        },
    }
    if excess <= max(absent, noise_floor):
        return {
            "label": "COMPLEMENTARITY_ABSENT",
            "reason": (
                f"excess over the permuted null {excess:+.4f} nats (raw gain "
                f"{gain:.4f}, null {null_gain:.4f}) is below max({absent}, "
                f"3*SE {noise_floor:.4f}): the linear-mixture family adds only "
                "recalibration, not information"
            ),
            **detail,
        }
    if excess > max(confirmed, noise_floor, null_margin):
        return {
            "label": "COMPLEMENTARITY_CONFIRMED",
            "reason": (
                f"excess over the permuted null {excess:+.4f} nats >= "
                f"max({confirmed}, {null_margin}, 3*SE {noise_floor:.4f})"
            ),
            **detail,
        }
    return {
        "label": "COMPLEMENTARITY_MARGINAL",
        "reason": (
            f"excess over the permuted null {excess:+.4f} nats lies between "
            f"{absent} and {confirmed} nats and clears 3*SE ({noise_floor:.4f})"
        ),
        **detail,
    }
